"""UMFPACK-based sparse solves via JAX pure callbacks.

This module mirrors the CHOLMOD batched API for general sparse systems while
using :func:`sksparse.umfpack.umf_solve` as the host-side direct solver.
"""

import functools as ft

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
_UMFPACK_VMAP_METHOD = "broadcast_all"


def _make_umfpack_state():
    return {
        "factor": None,
        "csr_columns": None,
        "csr_offsets": None,
        "n": None,
    }


def _factorize_and_solve_with_cache(state, *, trans, values, columns, offsets, rhs):
    from sksparse.umfpack import umf_factor

    n = int(rhs.shape[0])
    A_csc = build_csr_matrix(values, columns, offsets, n).tocsc()

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
        state["factor"] = umf_factor(A_csc)
        state["csr_columns"] = columns.copy()
        state["csr_offsets"] = offsets.copy()
        state["n"] = n
    else:
        state["factor"].factorize(A_csc)

    return np.asarray(state["factor"].solve(rhs, trans=trans), dtype=rhs.dtype)


@ft.lru_cache(maxsize=None)
def _get_umfpack_host_callback(
    trans: str, cache_namespace: str, batched: bool, shared_values: bool
):
    """Create one UMFPACK host callback for an explicit solve mode."""
    del cache_namespace
    state = _make_umfpack_state()

    def _host_callback(csr_values, csr_columns, csr_offsets, b_values):
        values_np = np.array(csr_values, copy=True)
        columns_np = np.array(np.asarray(csr_columns), copy=True)
        offsets_np = np.array(np.asarray(csr_offsets), copy=True)
        b_np = np.array(b_values, copy=True)

        if not batched:
            return _factorize_and_solve_with_cache(
                state,
                trans=trans,
                values=values_np,
                columns=columns_np,
                offsets=offsets_np,
                rhs=b_np,
            )

        n, nrhs = int(b_np.shape[0]), int(b_np.shape[1])
        b_f = np.asfortranarray(b_np)

        if shared_values:
            return _factorize_and_solve_with_cache(
                state,
                trans=trans,
                values=values_np,
                columns=columns_np,
                offsets=offsets_np,
                rhs=b_f,
            )

        bdm_values, bdm_columns, bdm_offsets = assemble_batched_csr_arrays_np(
            data_blocks=values_np,
            single_indices=columns_np,
            single_indptr=offsets_np,
            block_ncols=n,
        )
        rhs_flat = b_f.reshape(-1, order="F")
        x_flat = _factorize_and_solve_with_cache(
            state,
            trans=trans,
            values=bdm_values,
            columns=bdm_columns,
            offsets=bdm_offsets,
            rhs=rhs_flat,
        )
        return np.asarray(x_flat, dtype=b_f.dtype).reshape(b_f.shape, order="F")

    return _host_callback


@ft.partial(jax.jit, static_argnames=("trans", "cache_namespace"))
def umfpack_solve_single(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    trans: str = "N",
    cache_namespace: str = "global",
):
    """Solve one sparse linear system with UMFPACK."""
    csr_offsets = normalize_static_csr_vector(csr_offsets)
    csr_columns = normalize_static_csr_vector(csr_columns)
    validate_single_csr_inputs(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        solver_name="UMFPACK",
    )

    callback = _get_umfpack_host_callback(
        trans=trans,
        cache_namespace=cache_namespace,
        batched=False,
        shared_values=True,
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


@ft.partial(jax.jit, static_argnames=("trans", "cache_namespace", "vmap_method"))
def umfpack_solve_batched(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    trans: str = "N",
    cache_namespace: str = "global",
    vmap_method: str = _UMFPACK_VMAP_METHOD,
):
    """Solve batched sparse systems with UMFPACK on host.

    For shared matrix values, RHS vectors are packed as columns and solved in a
    single ``umf_solve`` call with Fortran-contiguous storage for better
    host-side throughput.
    """
    csr_offsets = normalize_static_csr_vector(csr_offsets)
    csr_columns = normalize_static_csr_vector(csr_columns)

    batch_shape, total_batch, n, values_payload = prepare_batched_csr_payload(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        broadcast_values=False,
    )
    b_cols = b_values.reshape(total_batch, n).T
    shared_values = csr_values.ndim == 1

    callback = _get_umfpack_host_callback(
        trans=trans,
        cache_namespace=cache_namespace,
        batched=True,
        shared_values=shared_values,
    )
    result_shape = jax.ShapeDtypeStruct(b_cols.shape, b_cols.dtype)
    x_cols = jax.pure_callback(
        callback,
        result_shape,
        values_payload,
        csr_columns,
        csr_offsets,
        b_cols,
        vmap_method=vmap_method,
    )
    return x_cols.T.reshape(batch_shape + (n,))


@ft.lru_cache(maxsize=None)
def _get_umfpack_vmap_dispatch(trans: str, cache_namespace: str, vmap_method: str):
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

        return umfpack_solve_single(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            trans=trans,
            cache_namespace=cache_namespace,
        )

    @_dispatch.def_vmap
    def _umfpack_solve_vmap_rule(
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

        out = umfpack_solve_batched(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            trans=trans,
            cache_namespace=cache_namespace,
            vmap_method=vmap_method,
        )
        return out, True

    return _dispatch


def umfpack_solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    trans: str = "N",
    cache_namespace: str = "global",
    vmap_method: str = _UMFPACK_VMAP_METHOD,
):
    """Public UMFPACK solve API."""
    dispatch = _get_umfpack_vmap_dispatch(trans, cache_namespace, vmap_method)
    return dispatch(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
    )
