---
sidebar_label: common
title: feax.solvers.common
---

Shared solver helpers.

Small stateless utilities used across linear/newton/reduced solver modules.

#### create\_x0

```python
def create_x0(bc_rows=None, bc_vals=None, P_mat=None)
```

Create BC-aware initial guess function for linear increments.

#### create\_x0\_parametric

```python
def create_x0_parametric(P_mat=None)
```

Create BC-aware initial guess function that takes bc as an explicit argument.

Unlike :func:`create_x0` which captures ``bc_rows``/``bc_vals`` in a closure,
this version accepts a :class:`DirichletBC` so it can be traced under ``jax.vmap``.

#### create\_jacobi\_preconditioner

```python
def create_jacobi_preconditioner(A: BCOO, shift: float = 1e-12)
```

Create Jacobi (diagonal) preconditioner from sparse matrix.

#### create\_direct\_solve\_fn

```python
def create_direct_solve_fn(options: DirectSolverOptions,
                           *,
                           cache_namespace: str = "global")
```

Create a direct linear solve function.

#### create\_iterative\_solve\_fn

```python
def create_iterative_solve_fn(options: IterativeSolverOptions)
```

Create an iterative linear solve function.

#### create\_linear\_solve\_fn

```python
def create_linear_solve_fn(solver_options, *, cache_namespace: str = "global")
```

Create a linear solve function based on solver options.

#### prewarm\_cudss\_solvers

```python
def prewarm_cudss_solvers(problem, bc, internal_vars, J_bc_func,
                          forward_options, adjoint_options, forward_solve_fn,
                          adjoint_solve_fn)
```

Pre-warm cuDSS solve closures with concrete CSR structure.

This must run outside JAX tracing so the first-call cuDSS initialization
does not capture tracers in closure state.

#### check\_convergence

```python
def check_convergence(A, x, b, solver_options, matrix_view: MatrixView,
                      solver_label: str)
```

Check relative residual and return NaN solution when convergence fails.

