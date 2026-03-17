---
sidebar_label: umfpack
title: feax.solvers.direct.umfpack
---

UMFPACK-based sparse solves via JAX pure callbacks.

This module mirrors the CHOLMOD batched API for general sparse systems while
using :func:`sksparse.umfpack.umf_solve` as the host-side direct solver.

#### umfpack\_solve\_single

```python
@ft.partial(jax.jit, static_argnames=("trans", "cache_namespace"))
def umfpack_solve_single(b_values,
                         csr_values,
                         csr_offsets,
                         csr_columns,
                         *,
                         trans: str = "N",
                         cache_namespace: str = "global")
```

Solve one sparse linear system with UMFPACK.

#### umfpack\_solve\_batched

```python
@ft.partial(jax.jit,
            static_argnames=("trans", "cache_namespace", "vmap_method"))
def umfpack_solve_batched(b_values,
                          csr_values,
                          csr_offsets,
                          csr_columns,
                          *,
                          trans: str = "N",
                          cache_namespace: str = "global",
                          vmap_method: str = _UMFPACK_VMAP_METHOD)
```

Solve batched sparse systems with UMFPACK on host.

For shared matrix values, RHS vectors are packed as columns and solved in a
single ``umf_solve`` call with Fortran-contiguous storage for better
host-side throughput.

#### umfpack\_solve

```python
def umfpack_solve(b_values,
                  csr_values,
                  csr_offsets,
                  csr_columns,
                  *,
                  trans: str = "N",
                  cache_namespace: str = "global",
                  vmap_method: str = _UMFPACK_VMAP_METHOD)
```

Public UMFPACK solve API.

