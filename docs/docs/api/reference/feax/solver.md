---
sidebar_label: solver
title: feax.solver
---

Nonlinear and linear solvers for FEAX finite element framework.

This module provides Newton-Raphson solvers, linear solvers, and solver configuration
utilities for solving finite element problems. It includes both JAX-based solvers
for performance and Python-based solvers for debugging.

Key Features:
- Newton-Raphson solvers with line search and convergence control
- Multiple solver variants: while loop, fixed iterations, and Python debugging
- Jacobi preconditioning for improved convergence
- Comprehensive solver configuration through SolverOptions dataclass
- Support for multipoint constraints via prolongation matrices

## SolverOptions Objects

```python
@dataclass(frozen=True)
class SolverOptions()
```

Configuration options for the Newton solver.

Parameters
----------
- **tol** (*float, default 1e-6*): Absolute tolerance for residual vector (l2 norm)
- **rel_tol** (*float, default 1e-8*): Relative tolerance for residual vector (l2 norm)
- **max_iter** (*int, default 100*): Maximum number of Newton iterations
- **linear_solver** (*str, default &quot;cg&quot;*): Linear solver type. Options: &quot;cg&quot;, &quot;bicgstab&quot;, &quot;gmres&quot;, &quot;spsolve&quot;
- **preconditioner** (*callable, optional*): Preconditioner function for linear solver
- **use_jacobi_preconditioner** (*bool, default False*): Whether to use Jacobi (diagonal) preconditioner automatically
- **jacobi_shift** (*float, default 1e-12*): Regularization parameter for Jacobi preconditioner
- **linear_solver_tol** (*float, default 1e-10*): Tolerance for linear solver
- **linear_solver_atol** (*float, default 1e-10*): Absolute tolerance for linear solver
- **linear_solver_maxiter** (*int, default 10000*): Maximum iterations for linear solver
- **linear_solver_x0_fn** (*callable, optional*): Custom function to compute initial guess: f(current_sol) -&gt; x0
- **line_search_max_backtracks** (*int, default 30*): Maximum number of backtracking steps in Armijo line search
- **line_search_c1** (*float, default 1e-4*): Armijo constant for sufficient decrease condition
- **line_search_rho** (*float, default 0.5*): Backtracking factor for line search (alpha *= rho each iteration)
- **verbose** (*bool, default False*): Whether to print convergence information during iterations Uses jax.debug.print() for JIT/vmap compatibility


#### linear\_solver

Options: &quot;cg&quot;, &quot;bicgstab&quot;, &quot;gmres&quot;, &quot;spsolve&quot;

#### linear\_solver\_x0\_fn

Function to compute initial guess: f(current_sol) -&gt; x0

#### create\_jacobi\_preconditioner

```python
def create_jacobi_preconditioner(
        A: jax.experimental.sparse.BCOO,
        shift: float = 1e-12) -> jax.experimental.sparse.BCOO
```

Create Jacobi (diagonal) preconditioner from sparse matrix.

Parameters
----------
- **A** (*BCOO sparse matrix*): The system matrix to precondition
- **shift** (*float, default 1e-12*): Small value added to diagonal for numerical stability


Returns
-------
- **M** (*LinearOperator*): Jacobi preconditioner as diagonal inverse matrix


Notes
-----
This creates a diagonal preconditioner M = diag(A)^{`-1`} with regularization.
The preconditioner is JAX-compatible and avoids dynamic indexing.
For elasticity problems with extreme material contrasts, this helps
condition number significantly.

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
>>> # Usage with BC information
>>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0])
>>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
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
                 P_mat=None)
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


Returns
-------
sol, res_norm, rel_res_norm, iter_count : tuple
    Solution, residual norms, and iteration count

#### newton\_solve\_fori

```python
def newton_solve_fori(J_bc_applied,
                      res_bc_applied,
                      initial_guess,
                      bc: DirichletBC,
                      solver_options: SolverOptions,
                      num_iters: int,
                      internal_vars=None,
                      P_mat=None)
```

Newton solver using JAX fori_loop for fixed iterations - optimized for vmap.

Designed for vmap with fixed iterations and consistent computational graph.

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
                 internal_vars=None,
                 P_mat=None)
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
- **internal_vars** (*InternalVars, optional*): Material properties and parameters
- **P_mat** (*BCOO matrix, optional*): Prolongation matrix for MPC/periodic BC


Returns
-------
sol, None : tuple
    Solution and None (for compatibility)

For linear problems, this single iteration achieves the exact solution.

#### create\_solver

```python
def create_solver(problem,
                  bc,
                  solver_options=None,
                  adjoint_solver_options=None,
                  iter_num=None,
                  P=None)
```

Create a differentiable solver that returns gradients w.r.t. internal_vars using custom VJP.

This solver uses the self-adjoint approach for efficient gradient computation:
- Forward mode: standard Newton solve
- Backward mode: solve adjoint system to compute gradients

Parameters
----------
- **problem** (*Problem*)
- **bc** (*DirichletBC*)
- **solver_options** (*SolverOptions, optional*)
- **adjoint_solver_options** (*dict, optional*)
- **iter_num** (*int, optional*)
- **Note** (*When iter_num is not None, the solver is vmappable since it uses fixed iterations.*)
- **Recommended** (*Use iter_num=1 for linear problems for optimal performance.*)
- **P** (*BCOO matrix, optional*)


Returns
-------
- **differentiable_solve** (*callable*)


Notes
-----
The returned function has signature: differentiable_solve(internal_vars, initial_guess) -&gt; solution
where gradients flow through internal_vars (material properties, loadings, etc.)

The initial_guess parameter is required to avoid conditionals that slow down JAX compilation.
For the first solve, you can pass zeros with BC values set:
initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)

When iter_num is specified (not None), the solver becomes vmappable as it uses fixed
iterations without dynamic control flow. This is essential for parallel solving of
multiple parameter sets using jax.vmap.

Examples
--------
```python
>>> # Create differentiable solver
>>> diff_solve = create_solver(problem, bc)
>>>
>>> # Prepare initial guess
>>> initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
>>> initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
>>>
>>> # First solve
>>> solution = diff_solve(internal_vars, initial_guess)
>>>
>>> # For time-dependent problems, update initial guess each timestep
>>> for t in timesteps:
>>>     solution = diff_solve(internal_vars_at_t, solution)  # Use previous solution
>>>
>>> # For linear problems (e.g., linear elasticity), use single iteration for best performance
>>> # This is both faster and vmappable
>>> diff_solve_linear = create_solver(problem, bc, iter_num=1)
>>>
>>> # For fixed iteration count (e.g., for vmap)
>>> diff_solve_fixed = create_solver(problem, bc, iter_num=10)
>>>
>>> # Define loss function
>>> def loss_fn(internal_vars):
...     initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
...     initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
...     sol = diff_solve(internal_vars, initial_guess)
...     return jnp.sum(sol**2)  # Example loss
>>>
>>> # Compute gradients w.r.t. internal_vars
>>> grad_fn = jax.grad(loss_fn)
>>> gradients = grad_fn(internal_vars)
```

