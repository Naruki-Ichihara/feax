"""SciPy-based sparse solves via JAX pure callbacks."""

import functools as ft

import jax
from jax import custom_batching
import numpy as np

from ._callback_utils import (
    assemble_batched_csr_arrays_np,
    build_csr_matrix,
    normalize_static_csr_vector,
    prepare_batched_csr_payload,
    prepare_solver_vmap_inputs,
    validate_single_csr_inputs,
)

_SPSOLVE_VMAP_METHOD = "broadcast_all"


def _map_reorder_to_permc_spec(reorder: int) -> str:
    """Map JAX-like reorder flag to SciPy ``permc_spec``."""
    if reorder == 0:
        return "NATURAL"
    if reorder == 1:
        return "COLAMD"
    raise ValueError(f"Unsupported reorder={reorder}; expected 0 (NATURAL) or 1 (COLAMD).")


@ft.lru_cache(maxsize=None)
def _get_spsolve_host_callback(reorder: int, batched: bool):
    """Create one host callback per reorder mode and batching mode."""

    def _host_callback(csr_values, csr_columns, csr_offsets, b_values):
        import scipy.sparse.linalg as spla

        values_np = np.array(csr_values, copy=True)
        columns_np = np.array(csr_columns, copy=True)
        offsets_np = np.array(csr_offsets, copy=True)
        b_np = np.array(b_values, copy=True)

        if not batched:
            A = build_csr_matrix(values_np, columns_np, offsets_np, b_np.shape[0])
            x = spla.spsolve(A, b_np, permc_spec=_map_reorder_to_permc_spec(reorder))
            return np.asarray(x, dtype=b_np.dtype)

        bdm_values, bdm_columns, bdm_offsets = assemble_batched_csr_arrays_np(
            data_blocks=values_np,
            single_indices=columns_np,
            single_indptr=offsets_np,
            block_ncols=b_np.shape[1],
        )
        rhs_flat = b_np.reshape(-1)
        A = build_csr_matrix(
            bdm_values,
            bdm_columns,
            bdm_offsets,
            rhs_flat.shape[0],
        )
        x = spla.spsolve(A, rhs_flat, permc_spec=_map_reorder_to_permc_spec(reorder))
        return np.asarray(x, dtype=b_np.dtype).reshape(b_np.shape)

    return _host_callback


@ft.partial(jax.jit, static_argnames=("reorder",))
def spsolve_single(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    reorder: int = 1,
):
    """Solve one sparse linear system via SciPy in a host callback."""
    csr_offsets = normalize_static_csr_vector(csr_offsets)
    csr_columns = normalize_static_csr_vector(csr_columns)
    validate_single_csr_inputs(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        solver_name="spsolve",
    )

    callback = _get_spsolve_host_callback(reorder, batched=False)
    result_shape = jax.ShapeDtypeStruct(b_values.shape, b_values.dtype)
    return jax.pure_callback(
        callback,
        result_shape,
        csr_values,
        csr_columns,
        csr_offsets,
        b_values,
    )


@ft.partial(jax.jit, static_argnames=("reorder", "vmap_method"))
def spsolve_batched(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    reorder: int = 1,
    vmap_method: str = _SPSOLVE_VMAP_METHOD,
):
    """Solve batched sparse systems via host-side block-diagonal assembly."""
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

    callback = _get_spsolve_host_callback(reorder, batched=True)
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
def _get_spsolve_vmap_dispatch(reorder: int, vmap_method: str):
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

        return spsolve_single(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            reorder=reorder,
        )

    @_dispatch.def_vmap
    def _spsolve_vmap_rule(
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

        out = spsolve_batched(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            reorder=reorder,
            vmap_method=vmap_method,
        )
        return out, True

    return _dispatch


def spsolve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    reorder: int = 1,
    vmap_method: str = _SPSOLVE_VMAP_METHOD,
):
    """Public SciPy direct solve API."""
    dispatch = _get_spsolve_vmap_dispatch(reorder, vmap_method)
    return dispatch(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
    )
