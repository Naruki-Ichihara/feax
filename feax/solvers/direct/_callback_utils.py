"""Shared helpers for direct sparse solver pure callbacks."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def normalize_static_csr_vector(x):
    """Drop spurious leading batch axes (e.g. introduced by transforms)."""
    while getattr(x, "ndim", 0) > 1:
        x = jax.lax.index_in_dim(x, 0, axis=0, keepdims=False)
    return x


def validate_single_csr_inputs(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    solver_name: str,
):
    """Validate the public single-system solve signature."""
    if b_values.ndim != 1:
        raise ValueError(f"b_values must have shape (n,); got {b_values.shape}.")
    if csr_offsets.shape[0] != b_values.shape[0] + 1:
        raise ValueError(
            f"csr_offsets must have length n+1={b_values.shape[0] + 1}; got {csr_offsets.shape[0]}."
        )
    if csr_values.ndim != 1 or csr_columns.ndim != 1:
        raise ValueError(f"csr_values and csr_columns must be rank-1 for non-batched {solver_name}.")
    if csr_values.shape[0] != csr_columns.shape[0]:
        raise ValueError(
            f"csr_values length ({csr_values.shape[0]}) must match csr_columns length ({csr_columns.shape[0]})."
        )


def prepare_batched_csr_payload(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    broadcast_values: bool,
):
    """Validate the public batched solve signature and reshape payloads."""
    if b_values.ndim < 2:
        raise ValueError(f"b_values must have shape (batch, n) or (..., batch, n); got {b_values.shape}.")

    batch_shape = b_values.shape[:-1]
    n = b_values.shape[-1]
    total_batch = int(np.prod(batch_shape))
    nnz = csr_columns.shape[0]

    if csr_offsets.shape[0] != n + 1:
        raise ValueError(
            f"csr_offsets must have length n+1={n + 1} for b_values shape {b_values.shape}; "
            f"got {csr_offsets.shape[0]}."
        )

    if csr_values.ndim == 1:
        if csr_values.shape[0] != nnz:
            raise ValueError(f"csr_values length must be nnz={nnz}; got {csr_values.shape[0]}.")
        values_payload = (
            jnp.broadcast_to(csr_values[None, :], (total_batch, nnz)) if broadcast_values else csr_values
        )
    elif csr_values.ndim >= 2:
        expected = batch_shape + (nnz,)
        if csr_values.shape != expected:
            raise ValueError(f"csr_values must have shape {expected} (or (nnz,)); got {csr_values.shape}.")
        values_payload = csr_values.reshape(total_batch, nnz)
    else:
        raise ValueError("csr_values must be rank-1 or rank-(len(batch_shape)+1).")

    return batch_shape, total_batch, n, values_payload


def assemble_batched_csr_arrays_np(
    data_blocks: np.ndarray,
    single_indices: np.ndarray,
    single_indptr: np.ndarray,
    block_ncols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a block-diagonal CSR matrix by repeating one CSR pattern per block."""
    batch_size = int(data_blocks.shape[0])
    nnz_single = int(single_indptr[-1])
    n_block_rows = int(single_indptr.shape[0] - 1)

    if single_indices.shape[0] < nnz_single:
        raise ValueError(
            f"csr_columns has fewer entries ({single_indices.shape[0]}) than indptr[-1]={nnz_single}."
        )
    if data_blocks.shape[1] < nnz_single:
        raise ValueError(
            f"csr_values has fewer entries ({data_blocks.shape[1]}) than indptr[-1]={nnz_single}."
        )

    single_indices = single_indices[:nnz_single]
    data_blocks = data_blocks[:, :nnz_single]

    col_offsets = (np.arange(batch_size, dtype=single_indices.dtype) * int(block_ncols))[:, None]
    out_indices = (single_indices[None, :] + col_offsets).reshape(-1)

    ptr_offsets = (np.arange(batch_size, dtype=single_indptr.dtype) * nnz_single)[:, None]
    out_indptr = np.empty(batch_size * n_block_rows + 1, dtype=single_indptr.dtype)
    out_indptr[0] = single_indptr[0]
    out_indptr[1:] = (single_indptr[None, 1:] + ptr_offsets).reshape(-1)

    out_data = data_blocks.reshape(-1)
    return out_data, out_indices, out_indptr


def build_csr_matrix(values, columns, offsets, n):
    """Build one CSR matrix from a CSR payload."""
    nnz = int(offsets[-1])
    return sp.csr_matrix((values[:nnz], columns[:nnz], offsets), shape=(n, n))


def prepare_solver_vmap_inputs(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    in_batched,
    axis_size: int,
):
    """Normalize static sparsity inputs and broadcast one mapped payload if needed."""
    b_batched, values_batched, offsets_batched, columns_batched = in_batched

    if offsets_batched:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=0, keepdims=False)
        offsets_batched = False
    if columns_batched:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=0, keepdims=False)
        columns_batched = False
    if offsets_batched or columns_batched:
        raise NotImplementedError("Batches of heterogeneous sparsity patterns are not supported.")

    if b_batched == values_batched:
        return b_values, csr_values, csr_offsets, csr_columns

    if b_batched:
        csr_values = jnp.broadcast_to(csr_values, (axis_size,) + csr_values.shape)
    else:
        b_values = jnp.broadcast_to(b_values, (axis_size,) + b_values.shape)

    return b_values, csr_values, csr_offsets, csr_columns
