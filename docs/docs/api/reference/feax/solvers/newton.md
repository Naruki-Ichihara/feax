---
sidebar_label: newton
title: feax.solvers.newton
---

Newton solver utilities for FEAX.

Contains x0 initialization, Armijo line-search implementations,
and Newton solve variants used by ``feax.solver``.

#### create\_newton\_solve\_fn

```python
def create_newton_solve_fn(iter_num, J_bc_func, res_bc_func, bc,
                           newton_options, linear_solver_options,
                           linear_solve_fn, x0_fn, matrix_view)
```

Create a Newton solve callable for adaptive or fixed-iteration solves.

#### create\_newton\_solver

```python
def create_newton_solver(problem,
                         bc,
                         linear_options,
                         adjoint_linear_options,
                         iter_num: Optional[int],
                         newton_options: Optional[NewtonOptions] = None,
                         internal_vars=None)
```

Create a differentiable Newton solver (iter_num is None or &gt;1).

#### create\_armijo\_line\_search\_jax

```python
def create_armijo_line_search_jax(res_bc_applied,
                                  c1=1e-4,
                                  rho=0.5,
                                  max_backtracks=30)
```

Create JAX while_loop Armijo line search.

#### create\_armijo\_line\_search\_scan

```python
def create_armijo_line_search_scan(res_bc_applied,
                                   c1=1e-4,
                                   rho=0.5,
                                   max_backtracks=30)
```

Create JAX scan-based Armijo line search (vmap-friendly).

#### create\_armijo\_line\_search\_python

```python
def create_armijo_line_search_python(res_bc_applied,
                                     c1=1e-4,
                                     rho=0.5,
                                     max_backtracks=30)
```

Create Python-loop Armijo line search (debug path).

#### newton\_solve

```python
def newton_solve(J_bc_applied,
                 res_bc_applied,
                 initial_guess,
                 bc,
                 newton_options,
                 linear_solver_options,
                 internal_vars=None,
                 P_mat=None,
                 linear_solve_fn=None,
                 armijo_search_fn=None,
                 x0_fn=None,
                 matrix_view: MatrixView = MatrixView.FULL)
```

Newton solver using JAX while_loop for JIT compatibility.

#### newton\_solve\_fori

```python
def newton_solve_fori(J_bc_applied,
                      res_bc_applied,
                      initial_guess,
                      bc,
                      newton_options,
                      num_iters,
                      linear_solver_options,
                      internal_vars=None,
                      P_mat=None,
                      linear_solve_fn=None,
                      armijo_search_fn=None,
                      x0_fn=None,
                      matrix_view: MatrixView = MatrixView.FULL)
```

Newton solver using JAX fori_loop for fixed iterations.

#### newton\_solve\_py

```python
def newton_solve_py(J_bc_applied,
                    res_bc_applied,
                    initial_guess,
                    bc,
                    newton_options,
                    linear_solver_options,
                    internal_vars=None,
                    P_mat=None,
                    matrix_view: MatrixView = MatrixView.FULL)
```

Newton solver using Python while loop (non-JIT debug path).

