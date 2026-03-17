---
sidebar_label: linear
title: feax.solvers.linear
---

Linear solver implementations for FEAX finite element framework.

This module provides low-level linear algebra utilities and solver
selection logic for solving systems of the form A x = b arising
in finite element analysis.

Key Features:
- Jacobi (diagonal) preconditioner
- Solver selection: cg, bicgstab, gmres, spsolve, umfpack, cholmod, cudss
- Convergence checking for ill-conditioned problems
- Adjoint linear solve for gradient computation
- create_linear_solver: high-level differentiable solver for linear problems

#### linear\_solve\_adjoint

```python
def linear_solve_adjoint(A,
                         b,
                         solver_options: AbstractSolverOptions,
                         matrix_view: MatrixView,
                         bc=None,
                         linear_solve_fn: Optional[Callable] = None)
```

Solve linear system for adjoint problem.

Parameters
----------
- **A** (*BCOO sparse matrix*): The transposed Jacobian matrix (J^T)
- **b** (*jax.numpy.ndarray*): Right-hand side vector (cotangent vector from VJP)
- **solver_options** (*SolverOptions or DirectSolverOptions or IterativeSolverOptions*): Solver configuration for adjoint solve
- **matrix_view** (*MatrixView*): Matrix storage format from the problem
- **bc** (*DirichletBC, optional*): Boundary conditions for computing initial guess
- **linear_solve_fn** (*callable, optional*): Pre-created linear solve function. If None, one is created from solver_options internally.


Returns
-------
- **sol** (*jax.numpy.ndarray*): Solution to the adjoint system: A @ sol = b


Notes
-----
For the adjoint problem, boundary conditions are already incorporated into
the transposed Jacobian matrix. The initial guess uses BC-aware computation
when bc is provided, which can improve convergence for problems with
Dirichlet boundary conditions.

#### linear\_solve

```python
def linear_solve(J_bc_applied,
                 res_bc_applied,
                 initial_guess,
                 bc: DirichletBC,
                 solver_options: AbstractSolverOptions,
                 matrix_view: MatrixView,
                 internal_vars=None,
                 P_mat=None,
                 linear_solve_fn: Optional[Callable] = None,
                 x0_fn: Optional[Callable] = None)
```

Single-step linear solve used by create_solver(iter_num=1).

#### create\_linear\_solver

```python
def create_linear_solver(
        problem: Problem,
        bc: DirichletBC,
        solver_options: Optional[AbstractSolverOptions] = None,
        adjoint_solver_options: Optional[AbstractSolverOptions] = None,
        internal_vars=None) -> Callable[[Any, np.ndarray], np.ndarray]
```

Create a differentiable solver for linear FE problems.

Simpler and more focused alternative to ``create_solver(iter_num=1)``
when the problem is known to be linear (e.g. linear elasticity).
The returned function supports ``jax.grad`` via a custom VJP based on
the adjoint method.

Parameters
----------
- **problem** (*Problem*): The feax Problem instance.
- **bc** (*DirichletBC*): Boundary conditions.
- **solver_options** (*DirectSolverOptions or IterativeSolverOptions, optional*): Options for the forward linear solve (defaults to IterativeSolverOptions()).
- **adjoint_solver_options** (*DirectSolverOptions or IterativeSolverOptions, optional*): Options for the adjoint solve used in the backward pass. Defaults to the same options as the forward solve.
- **internal_vars** (*InternalVars, optional*): Sample internal variables used to pre-warm cuDSS with concrete CSR structure before any JAX tracing. Recommended when using cuDSS and composing ``jax.jit`` with ``jax.grad``.


Returns
-------
- **differentiable_solve** (*callable*): A function with signature ``(internal_vars, initial_guess) -&gt; solution`` that is differentiable w.r.t. ``internal_vars`` via ``jax.grad``.


Notes
-----
Forward pass performs a single linear solve:

```python
J * delta_sol = -res
sol = initial_guess + delta_sol
```

Backward pass solves the adjoint system:

```python
J^T * adjoint = v
```

and returns the VJP of the residual w.r.t. ``internal_vars``.

Examples
--------
```python
>>> solver = create_linear_solver(problem, bc)
>>> initial = fe.zero_like_initial_guess(problem, bc)
>>> sol = solver(internal_vars, initial)
>>>
>>> # Gradient w.r.t. internal_vars
>>> def loss(internal_vars):
...     sol = solver(internal_vars, initial)
...     return np.sum(sol ** 2)
>>> grad = jax.grad(loss)(internal_vars)
```

