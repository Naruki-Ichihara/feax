---
sidebar_label: spsolve
title: feax.solvers.direct.spsolve
---

SciPy-based sparse solves via JAX pure callbacks.

#### spsolve\_single

```python
@ft.partial(jax.jit, static_argnames=("reorder", ))
def spsolve_single(b_values,
                   csr_values,
                   csr_offsets,
                   csr_columns,
                   *,
                   reorder: int = 1)
```

Solve one sparse linear system via SciPy in a host callback.

#### spsolve\_batched

```python
@ft.partial(jax.jit, static_argnames=("reorder", "vmap_method"))
def spsolve_batched(b_values,
                    csr_values,
                    csr_offsets,
                    csr_columns,
                    *,
                    reorder: int = 1,
                    vmap_method: str = _SPSOLVE_VMAP_METHOD)
```

Solve batched sparse systems via host-side block-diagonal assembly.

#### spsolve

```python
def spsolve(b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            *,
            reorder: int = 1,
            vmap_method: str = _SPSOLVE_VMAP_METHOD)
```

Public SciPy direct solve API.

