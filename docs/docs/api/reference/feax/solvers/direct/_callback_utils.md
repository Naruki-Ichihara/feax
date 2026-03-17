---
sidebar_label: _callback_utils
title: feax.solvers.direct._callback_utils
---

Shared helpers for direct sparse solver pure callbacks.

#### normalize\_static\_csr\_vector

```python
def normalize_static_csr_vector(x)
```

Drop spurious leading batch axes (e.g. introduced by transforms).

#### validate\_single\_csr\_inputs

```python
def validate_single_csr_inputs(b_values, csr_values, csr_offsets, csr_columns,
                               *, solver_name: str)
```

Validate the public single-system solve signature.

#### prepare\_batched\_csr\_payload

```python
def prepare_batched_csr_payload(b_values, csr_values, csr_offsets, csr_columns,
                                *, broadcast_values: bool)
```

Validate the public batched solve signature and reshape payloads.

#### assemble\_batched\_csr\_arrays\_np

```python
def assemble_batched_csr_arrays_np(
        data_blocks: np.ndarray, single_indices: np.ndarray,
        single_indptr: np.ndarray,
        block_ncols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Create a block-diagonal CSR matrix by repeating one CSR pattern per block.

#### build\_csr\_matrix

```python
def build_csr_matrix(values, columns, offsets, n)
```

Build one CSR matrix from a CSR payload.

#### prepare\_solver\_vmap\_inputs

```python
def prepare_solver_vmap_inputs(b_values, csr_values, csr_offsets, csr_columns,
                               in_batched, axis_size: int)
```

Normalize static sparsity inputs and broadcast one mapped payload if needed.

