"""CHOLMOD-based sparse solves via JAX pure callbacks."""

import functools as ft
from typing import Optional

import jax
import numpy as np
from jax import custom_batching

from ._callback_utils import (
    assemble_batched_csr_arrays_np,
    build_csr_matrix,
    normalize_static_csr_vector,
    prepare_solver_vmap_inputs,
    prepare_batched_csr_payload,
    validate_single_csr_inputs,
)

# Keep callback behavior deterministic and avoid transform-dependent shape paths.
_CHOLMOD_VMAP_METHOD = "broadcast_all"


def _make_cholmod_state():
    return {
        "factor": None,
        "csr_columns": None,
        "csr_offsets": None,
        "n": None,
    }


def _factorize_and_solve_with_cache(state, *, lower, order, values, columns, offsets, rhs):
    from sksparse import cholmod

    n = int(rhs.shape[0])
    A_csr = build_csr_matrix(values, columns, offsets, n)
    A_csc = A_csr.tocsc()

    pattern_changed = (
        state["factor"] is None
        or state["n"] != n
        or state["csr_columns"] is None
        or state["csr_columns"].shape != columns.shape
        or not np.array_equal(state["csr_columns"], columns)
        or state["csr_offsets"] is None
        or state["csr_offsets"].shape != offsets.shape
        or not np.array_equal(state["csr_offsets"], offsets)
    )
    if pattern_changed:
        state["factor"] = cholmod.cho_factor(A_csc, lower=lower, order=order, sym_kind="sym")
        state["csr_columns"] = columns.copy()
        state["csr_offsets"] = offsets.copy()
        state["n"] = n
    else:
        state["factor"].factorize(A_csc)

    return np.asarray(state["factor"].solve(rhs), dtype=rhs.dtype)


@ft.lru_cache(maxsize=None)
def _get_cholmod_host_callback(
    lower: bool, order: Optional[str], cache_namespace: str, batched: bool
):
    """Create one stateful CHOLMOD callback for either single or batched solves."""
    del cache_namespace
    state = _make_cholmod_state()

    def _host_callback(csr_values, csr_columns, csr_offsets, b_values):
        values_np = np.asarray(csr_values)
        columns_np = np.asarray(csr_columns)
        offsets_np = np.asarray(csr_offsets)
        b_np = np.array(b_values, copy=True)

        if not batched:
            nnz = int(offsets_np[-1])
            return _factorize_and_solve_with_cache(
                state,
                lower=lower,
                order=order,
                values=values_np[:nnz],
                columns=columns_np[:nnz],
                offsets=offsets_np,
                rhs=b_np,
            )

        bdm_values, bdm_columns, bdm_offsets = assemble_batched_csr_arrays_np(
            data_blocks=values_np,
            single_indices=columns_np,
            single_indptr=offsets_np,
            block_ncols=b_np.shape[1],
        )
        x_np = _factorize_and_solve_with_cache(
            state,
            lower=lower,
            order=order,
            values=bdm_values,
            columns=bdm_columns,
            offsets=bdm_offsets,
            rhs=b_np.reshape(-1),
        )
        return x_np.reshape(b_np.shape)

    return _host_callback


@ft.partial(jax.jit, static_argnames=("lower", "order", "cache_namespace"))
def cholmod_solve_single(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    lower: bool = False,
    order: Optional[str] = "default",
    cache_namespace: str = "global",
):
    """Solve one SPD sparse linear system with CHOLMOD."""
    csr_offsets = normalize_static_csr_vector(csr_offsets)
    csr_columns = normalize_static_csr_vector(csr_columns)
    validate_single_csr_inputs(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        solver_name="CHOLMOD",
    )

    callback = _get_cholmod_host_callback(
        lower=lower, order=order, cache_namespace=cache_namespace, batched=False
    )
    result_shape = jax.ShapeDtypeStruct(b_values.shape, b_values.dtype)
    return jax.pure_callback(
        callback,
        result_shape,
        csr_values,
        csr_columns,
        csr_offsets,
        b_values,
    )


@ft.partial(jax.jit, static_argnames=("lower", "order", "cache_namespace", "vmap_method"))
def cholmod_solve_batched(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    lower: bool = False,
    order: Optional[str] = "default",
    cache_namespace: str = "global",
    vmap_method: str = _CHOLMOD_VMAP_METHOD,
):
    """Solve batched SPD sparse systems by assembling one host-side block diagonal."""
    csr_offsets = normalize_static_csr_vector(csr_offsets)
    csr_columns = normalize_static_csr_vector(csr_columns)

    batch_shape, total_batch, n, values_payload = prepare_batched_csr_payload(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        broadcast_values=True,
    )
    b_flat = b_values.reshape(total_batch, n)

    callback = _get_cholmod_host_callback(
        lower=lower, order=order, cache_namespace=cache_namespace, batched=True
    )
    result_shape = jax.ShapeDtypeStruct(b_flat.shape, b_flat.dtype)
    x_flat = jax.pure_callback(
        callback,
        result_shape,
        values_payload,
        csr_columns,
        csr_offsets,
        b_flat,
        vmap_method=vmap_method,
    )
    return x_flat.reshape(batch_shape + (n,))


@ft.lru_cache(maxsize=None)
def _get_cholmod_vmap_dispatch(
    lower: bool,
    order: Optional[str],
    cache_namespace: str,
    vmap_method: str,
):
    """Build a custom_vmap dispatcher bound to static solver config."""

    @custom_batching.custom_vmap
    def _dispatch(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
    ):
        if b_values.ndim != 1 or csr_values.ndim != 1:
            raise NotImplementedError("Direct batched calls are unsupported; use vmap.")

        return cholmod_solve_single(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            lower=lower,
            order=order,
            cache_namespace=cache_namespace,
        )

    @_dispatch.def_vmap
    def _cholmod_solve_vmap_rule(
        axis_size,
        in_batched,
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
    ):
        b_values, csr_values, csr_offsets, csr_columns = prepare_solver_vmap_inputs(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            in_batched,
            axis_size,
        )

        out = cholmod_solve_batched(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            lower=lower,
            order=order,
            cache_namespace=cache_namespace,
            vmap_method=vmap_method,
        )
        return out, True

    return _dispatch


def cholmod_solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    lower: bool = False,
    order: Optional[str] = "default",
    cache_namespace: str = "global",
    vmap_method: str = _CHOLMOD_VMAP_METHOD,
):
    """Public CHOLMOD solve API."""
    dispatch = _get_cholmod_vmap_dispatch(lower, order, cache_namespace, vmap_method)
    return dispatch(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
    )
