---
sidebar_label: solver
title: feax.solver
---

Nonlinear and linear solvers for FEAX finite element framework.

This module provides Newton-Raphson solvers and solver configuration
utilities for solving finite element problems. It includes both JAX-based
solvers for performance and Python-based solvers for debugging.

Key Features:
- Newton-Raphson solvers with line search and convergence control
- Multiple solver variants: while loop, fixed iterations, and Python debugging
- Solver configuration via AbstractSolverOptions hierarchy
  (DirectSolverOptions, IterativeSolverOptions, or legacy SolverOptions)
- Support for multipoint constraints via prolongation matrices

Linear solver implementations and preconditioners are provided in
``linear_solver.py``. Solver configuration options are in ``solver_option.py``.

#### create\_x0

```python
def create_x0(bc_rows=None, bc_vals=None, P_mat=None)
```

Create initial guess function for linear solver following JAX-FEM approach.

Parameters
----------
- **bc_rows** (*array-like, optional*): Row indices of boundary condition locations
- **bc_vals** (*array-like, optional*): Boundary condition values
- **P_mat** (*BCOO matrix, optional*): Prolongation matrix for reduced problems (maps reduced to full DOFs)


Returns
-------
- **x0_fn** (*callable*): Function that takes current solution and returns initial guess for increment


Notes
-----
Implements the exact x0 computation from the row elimination solver:
x0_1 = assign_bc(zeros, problem) - sets BC values at BC locations, 0 elsewhere
x0_2 = copy_bc(current_sol, problem) - copies current solution values at BC locations, 0 elsewhere
x0 = x0_1 - x0_2 - the correct initial guess computation

For reduced problems (when P_mat is provided):
x0_2 = copy_bc(P @ current_sol_reduced, problem) - expand reduced sol and copy BC
x0 = P.T @ (x0_1 - x0_2) - transform back to reduced space

Examples
--------
```python
>>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0])
>>> solver_options = IterativeSolverOptions(x0_fn=x0_fn)
```
```python
>>> # Usage with reduced problem
>>> x0_fn = create_x0(bc_rows, bc_vals, P_mat=P)
```

#### create\_armijo\_line\_search\_jax

```python
def create_armijo_line_search_jax(res_bc_applied,
                                  c1=1e-4,
                                  rho=0.5,
                                  max_backtracks=30)
```

Create JAX-compatible Armijo backtracking line search using jax.lax.scan.

This function returns a line search function that can be JIT-compiled.

Parameters
----------
- **res_bc_applied** (*callable*): Residual function with boundary conditions applied. Signature: res_bc_applied(sol, internal_vars=None) -&gt; residual
- **c1** (*float, default 1e-4*): Armijo constant for sufficient decrease condition
- **rho** (*float, default 0.5*): Backtracking factor (alpha *= rho each iteration)
- **max_backtracks** (*int, default 30*): Maximum number of backtracking steps


Returns
-------
- **line_search_fn** (*callable*): Line search function with signature: (sol, delta_sol, initial_res_norm, internal_vars=None) -&gt; (new_sol, new_norm, alpha, success)


#### create\_armijo\_line\_search\_scan

```python
def create_armijo_line_search_scan(res_bc_applied,
                                   c1=1e-4,
                                   rho=0.5,
                                   max_backtracks=30)
```

Create JAX-compatible Armijo backtracking line search using jax.lax.scan.

This version uses scan (fixed iterations) and is vmap-compatible.
Use this for newton_solve_fori where vmap support is required.

Parameters
----------
- **res_bc_applied** (*callable*): Residual function with boundary conditions applied.
- **c1** (*float, default 1e-4*): Armijo constant for sufficient decrease condition
- **rho** (*float, default 0.5*): Backtracking factor
- **max_backtracks** (*int, default 30*): Number of backtracking steps (all are always evaluated)


Returns
-------
- **line_search_fn** (*callable*): Line search function with same signature as while_loop version.


#### create\_armijo\_line\_search\_python

```python
def create_armijo_line_search_python(res_bc_applied,
                                     c1=1e-4,
                                     rho=0.5,
                                     max_backtracks=30)
```

Create Python-based Armijo backtracking line search.

This version uses Python loops and is suitable for debugging or non-JIT contexts.

Parameters
----------
- **res_bc_applied** (*callable*): Residual function with boundary conditions applied
- **c1** (*float, default 1e-4*): Armijo constant
- **rho** (*float, default 0.5*): Backtracking factor
- **max_backtracks** (*int, default 30*): Maximum backtracking steps


Returns
-------
- **line_search_fn** (*callable*): Line search function


#### newton\_solve

```python
def newton_solve(J_bc_applied,
                 res_bc_applied,
                 initial_guess,
                 bc: DirichletBC,
                 solver_options: SolverOptions,
                 internal_vars=None,
                 P_mat=None,
                 linear_solve_fn=None,
                 armijo_search_fn=None,
                 x0_fn=None)
```

Newton solver using JAX while_loop for JIT compatibility.

Parameters
----------
- **J_bc_applied** (*callable*): Jacobian function with BC applied
- **res_bc_applied** (*callable*): Residual function with BC applied
- **initial_guess** (*array*): Initial solution guess
- **bc** (*DirichletBC*): Boundary conditions
- **solver_options** (*SolverOptions*): Solver configuration
- **internal_vars** (*InternalVars, optional*): Material properties and parameters
- **P_mat** (*BCOO matrix, optional*): Prolongation matrix for MPC/periodic BC
- **linear_solve_fn** (*callable, optional*): Pre-created linear solve function. If None, created internally.
- **armijo_search_fn** (*callable, optional*): Pre-created Armijo line search function. If None, created internally.
- **x0_fn** (*callable, optional*): Pre-created initial guess function. If None, created internally.


Returns
-------
sol, res_norm, initial_res_norm, iter_count : tuple
    Solution, residual norm, initial residual norm, and iteration count

#### newton\_solve\_fori

```python
def newton_solve_fori(J_bc_applied,
                      res_bc_applied,
                      initial_guess,
                      bc: DirichletBC,
                      solver_options: SolverOptions,
                      num_iters: int,
                      internal_vars=None,
                      P_mat=None,
                      linear_solve_fn=None,
                      armijo_search_fn=None,
                      x0_fn=None)
```

Newton solver using JAX fori_loop for fixed iterations - optimized for vmap.

Designed for vmap with fixed iterations and consistent computational graph.
Uses scan-based Armijo line search for vmap compatibility.

Parameters
----------
- **J_bc_applied** (*callable*): Jacobian function with BC applied
- **res_bc_applied** (*callable*): Residual function with BC applied
- **initial_guess** (*array*): Initial solution guess
- **bc** (*DirichletBC*): Boundary conditions
- **solver_options** (*SolverOptions*): Solver configuration
- **num_iters** (*int*): Fixed number of iterations
- **internal_vars** (*InternalVars, optional*): Material properties and parameters
- **P_mat** (*BCOO matrix, optional*): Prolongation matrix for MPC/periodic BC
- **linear_solve_fn** (*callable, optional*): Pre-created linear solve function. If None, created internally.
- **armijo_search_fn** (*callable, optional*): Pre-created Armijo line search function. If None, created internally (scan-based).
- **x0_fn** (*callable, optional*): Pre-created initial guess function. If None, created internally.


Returns
-------
sol, final_res_norm, converged : tuple
    Solution, residual norm, and convergence flag

#### newton\_solve\_py

```python
def newton_solve_py(J_bc_applied,
                    res_bc_applied,
                    initial_guess,
                    bc: DirichletBC,
                    solver_options: SolverOptions,
                    internal_vars=None,
                    P_mat=None)
```

Newton solver using Python while loop - non-JIT version for debugging.

Uses Python control flow for easier debugging. Not JIT-compatible.

Parameters
----------
- **J_bc_applied** (*callable*): Jacobian function with BC applied
- **res_bc_applied** (*callable*): Residual function with BC applied
- **initial_guess** (*array*): Initial solution guess
- **bc** (*DirichletBC*): Boundary conditions
- **solver_options** (*SolverOptions*): Solver configuration
- **internal_vars** (*InternalVars, optional*): Material properties and parameters
- **P_mat** (*BCOO matrix, optional*): Prolongation matrix for MPC/periodic BC


Returns
-------
sol, final_res_norm, converged, num_iters : tuple
    Solution, residual norm, convergence flag, and iteration count

#### linear\_solve

```python
def linear_solve(J_bc_applied,
                 res_bc_applied,
                 initial_guess,
                 bc: DirichletBC,
                 solver_options: SolverOptions,
                 matrix_view: MatrixView,
                 internal_vars=None,
                 P_mat=None,
                 linear_solve_fn=None,
                 x0_fn=None)
```

Linear solver for problems that converge in one iteration.

Optimized for linear problems (e.g., linear elasticity). Performs single Newton step.

Parameters
----------
- **J_bc_applied** (*callable*): Jacobian function with BC applied
- **res_bc_applied** (*callable*): Residual function with BC applied
- **initial_guess** (*array*): Initial solution guess
- **bc** (*DirichletBC*): Boundary conditions
- **solver_options** (*SolverOptions*): Solver configuration
- **matrix_view** (*MatrixView*): Matrix storage format from the problem.
- **internal_vars** (*InternalVars, optional*): Material properties and parameters
- **P_mat** (*BCOO matrix, optional*): Prolongation matrix for MPC/periodic BC
- **linear_solve_fn** (*callable, optional*): Pre-created linear solve function. If None, created internally.
- **x0_fn** (*callable, optional*): Pre-created initial guess function. If None, created internally.


Returns
-------
sol, None : tuple
    Solution and None (for compatibility)

For linear problems, this single iteration achieves the exact solution.

#### create\_solver

```python
def create_solver(
        problem: Problem,
        bc: DirichletBC,
        solver_options: Optional[AbstractSolverOptions] = None,
        adjoint_solver_options: Optional[AbstractSolverOptions] = None,
        iter_num: Optional[int] = None,
        P: Optional[BCOO] = None,
        internal_vars=None,
        internal_jit: bool = False) -> Callable
```

Create a differentiable solver with custom VJP for gradient computation.

Parameters
----------
- **problem** (*Problem*): The feax Problem instance.
- **bc** (*DirichletBC*): Boundary conditions.
- **solver_options** (*AbstractSolverOptions, optional*): Options for the forward linear solve. Accepts any subclass of ``AbstractSolverOptions``:
- **adjoint_solver_options** (*AbstractSolverOptions, optional*): Options for the adjoint solve in the backward pass. Defaults to same as ``solver_options``.
- **iter_num** (*int, optional*): Number of Newton iterations:
- **P** (*BCOO matrix, optional*): Prolongation matrix for periodic boundary conditions.
- **internal_vars** (*InternalVars, optional*): Sample internal variables for auto solver selection and cuDSS pre-warming. Required when ``solver=&quot;auto&quot;`` or cuDSS is used.
- **internal_jit** (*bool, optional*): When ``True`` and ``iter_num != 1``, wraps the internal linear solver with ``jax.jit`` so that each Newton iteration&#x27;s linear solve is compiled separately from the outer computation graph.  This is most effective for iterative solvers (CG, BiCGSTAB, GMRES) called only once; for repeated outer calls prefer ``jax.jit(solver)`` instead. Ignored (with a warning) when ``iter_num == 1`` because the linear solver is invoked only once per solve and internal JIT provides no benefit.  Default: ``False``.


Returns
-------
callable
    When ``DirectSolverOptions`` is used:
        ``solver(internal_vars) -&gt; solution``
        (``initial_guess`` is optional and ignored if provided.)
    When ``IterativeSolverOptions`` or ``SolverOptions`` is used:
        ``solver(internal_vars, initial_guess) -&gt; solution``

Examples
--------
```python
>>> # Direct solver (auto-selects cuDSS on GPU, spsolve on CPU)
>>> solver = create_solver(problem, bc, DirectSolverOptions(),
...                        iter_num=1, internal_vars=internal_vars)
>>> solution = solver(internal_vars)
>>>
>>> # Iterative solver with auto selection
>>> solver = create_solver(problem, bc, IterativeSolverOptions(),
...                        iter_num=1, internal_vars=internal_vars)
>>> solution = solver(internal_vars, initial_guess)
>>>
>>> # Explicit solver selection (no internal_vars needed for non-cuDSS)
>>> solver = create_solver(problem, bc, IterativeSolverOptions(solver=&quot;gmres&quot;),
...                        iter_num=1)
>>> solution = solver(internal_vars, initial_guess)
```

