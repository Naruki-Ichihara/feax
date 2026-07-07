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
- **solver_options** (*SolverOptions or DirectSolverOptions or KrylovSolverOptions*): Solver configuration for adjoint solve
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
                 traced_params=None,
                 P_mat=None,
                 linear_solve_fn: Optional[Callable] = None,
                 x0_fn: Optional[Callable] = None)
```

Single-step linear solve used by create_solver(linear=True).

#### create\_linear\_solver

```python
def create_linear_solver(
        problem: Problem,
        bc: DirichletBC,
        solver_options: Optional[AbstractSolverOptions] = None,
        adjoint_solver_options: Optional[AbstractSolverOptions] = None,
        traced_params=None,
        symmetric_elimination: bool = True,
        traced_structure=None) -> Callable[[Any, np.ndarray], np.ndarray]
```

Create a differentiable solver for linear FE problems.

Simpler and more focused alternative to ``create_solver(linear=True)``
when the problem is known to be linear (e.g. linear elasticity).
The returned function supports ``jax.grad`` via a custom VJP based on
the adjoint method.

Parameters
----------
- **problem** (*Problem*): The feax Problem instance.
- **bc** (*DirichletBC*): Boundary conditions.
- **solver_options** (*DirectSolverOptions or KrylovSolverOptions, optional*): Options for the forward linear solve (defaults to KrylovSolverOptions()).
- **adjoint_solver_options** (*DirectSolverOptions or KrylovSolverOptions, optional*): Options for the adjoint solve used in the backward pass. Defaults to the same options as the forward solve.
- **traced_params** (*TracedParams, optional*): Sample internal variables used to pre-warm cuDSS with concrete CSR structure before any JAX tracing. Recommended when using cuDSS and composing ``jax.jit`` with ``jax.grad``.
- **symmetric_elimination** (*bool, default True*): Use symmetric Dirichlet elimination (zero BC rows *and* columns). Linear FE operators are symmetric after symmetric elimination, so the Krylov adjoint reuses the forward matvec (``Jᵀ = J``).


Returns
-------
- **differentiable_solve** (*callable*): A function with signature ``(traced_params, initial_guess) -&gt; solution`` that is differentiable w.r.t. ``traced_params`` via ``jax.grad``.


Notes
-----
The operator representation follows the solve method: ``DirectSolverOptions``
assembles the Jacobian straight into CSR (for factorization), while
``KrylovSolverOptions`` (cg/bicgstab/gmres) is fully **matrix-free** — the
forward and adjoint matvecs are residual JVPs, so the element Jacobian is
never materialized.

Forward pass performs a single linear solve:

```python
J * delta_sol = -res
sol = initial_guess + delta_sol
```

Backward pass solves the adjoint system:

```python
J^T * adjoint = v
```

and returns the VJP of the residual w.r.t. ``traced_params``.

Examples
--------
```python
>>> solver = create_linear_solver(problem, bc)
>>> initial = fe.zero_like_initial_guess(problem, bc)
>>> sol = solver(traced_params, initial)
>>>
>>> # Gradient w.r.t. traced_params
>>> def loss(traced_params):
...     sol = solver(traced_params, initial)
...     return np.sum(sol ** 2)
>>> grad = jax.grad(loss)(traced_params)
```

