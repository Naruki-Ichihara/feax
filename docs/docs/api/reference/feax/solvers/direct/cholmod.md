---
sidebar_label: cholmod
title: feax.solvers.direct.cholmod
---

CHOLMOD-based sparse solves via JAX pure callbacks.

#### cholmod\_solve\_single

```python
@ft.partial(jax.jit, static_argnames=("lower", "order", "cache_namespace"))
def cholmod_solve_single(b_values,
                         csr_values,
                         csr_offsets,
                         csr_columns,
                         *,
                         lower: bool = False,
                         order: Optional[str] = "default",
                         cache_namespace: str = "global")
```

Solve one SPD sparse linear system with CHOLMOD.

#### cholmod\_solve\_batched

```python
@ft.partial(jax.jit,
            static_argnames=("lower", "order", "cache_namespace",
                             "vmap_method"))
def cholmod_solve_batched(b_values,
                          csr_values,
                          csr_offsets,
                          csr_columns,
                          *,
                          lower: bool = False,
                          order: Optional[str] = "default",
                          cache_namespace: str = "global",
                          vmap_method: str = _CHOLMOD_VMAP_METHOD)
```

Solve batched SPD sparse systems by assembling one host-side block diagonal.

#### cholmod\_solve

```python
def cholmod_solve(b_values,
                  csr_values,
                  csr_offsets,
                  csr_columns,
                  *,
                  lower: bool = False,
                  order: Optional[str] = "default",
                  cache_namespace: str = "global",
                  vmap_method: str = _CHOLMOD_VMAP_METHOD)
```

Public CHOLMOD solve API.

