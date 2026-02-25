---
sidebar_label: linear_solver
title: feax.linear_solver
---

Linear solver implementations for FEAX finite element framework.

This module provides low-level linear algebra utilities and solver
selection logic for solving systems of the form A x = b arising
in finite element analysis.

Key Features:
- Jacobi (diagonal) preconditioner
- Solver selection: cg, bicgstab, gmres, spsolve, cudss, lineax
- Convergence checking for ill-conditioned problems
- Adjoint linear solve for gradient computation
- create_linear_solver: high-level differentiable solver for linear problems

#### create\_jacobi\_preconditioner

```python
def create_jacobi_preconditioner(A: BCOO, shift: float = 1e-12)
```

Create Jacobi (diagonal) preconditioner from sparse matrix.

Parameters
----------
- **A** (*BCOO sparse matrix*): The system matrix to precondition
- **shift** (*float, default 1e-12*): Small value added to diagonal for numerical stability


Returns
-------
- **M_matvec** (*callable*): Jacobi preconditioner as a matvec function (diag(A)^{`-1`} @ x)


Notes
-----
This creates a diagonal preconditioner M = diag(A)^{`-1`} with regularization.
The preconditioner is JAX-compatible and avoids dynamic indexing.
For elasticity problems with extreme material contrasts, this helps
condition number significantly.

#### create\_direct\_solve\_fn

```python
def create_direct_solve_fn(options: DirectSolverOptions)
```

Create a direct linear solve function.

Parameters
----------
- **options** (*DirectSolverOptions*): Direct solver configuration. The ``solver`` field must be resolved (not &quot;auto&quot;) before calling this function.


Returns
-------
callable
    Function with signature (A, b, x0) -&gt; x.
    Direct solvers ignore x0 but accept it for interface uniformity.

Raises
------
ValueError
    If solver is &quot;auto&quot; (unresolved) or unknown.
RuntimeError
    If backend requirements are not met (e.g. spsolve on GPU).

#### create\_iterative\_solve\_fn

```python
def create_iterative_solve_fn(options: IterativeSolverOptions)
```

Create an iterative linear solve function.

Parameters
----------
- **options** (*IterativeSolverOptions*): Iterative solver configuration. The ``solver`` field must be resolved (not &quot;auto&quot;) before calling this function.


Returns
-------
callable
    Function with signature (A, b, x0) -&gt; x.

Raises
------
ValueError
    If solver is &quot;auto&quot; (unresolved) or unknown.

#### create\_linear\_solve\_fn

```python
def create_linear_solve_fn(solver_options)
```

Create a linear solve function based on solver options.

Accepts SolverOptions (legacy), DirectSolverOptions, or IterativeSolverOptions.
For legacy SolverOptions, internally converts and delegates to the appropriate
specialized function.

Parameters
----------
- **solver_options** (*SolverOptions or DirectSolverOptions or IterativeSolverOptions*): Solver configuration.


Returns
-------
callable
    Function with signature (A, b, x0) -&gt; x.

#### check\_linear\_convergence

```python
def check_linear_convergence(A, x, b, solver_options: SolverOptions,
                             matrix_view: MatrixView, solver_label: str)
```

Check relative residual and return NaN solution when convergence fails.

Parameters
----------
- **A** (*BCOO sparse matrix*): The system matrix
- **x** (*array*): The computed solution
- **b** (*array*): The right-hand side vector
- **solver_options** (*SolverOptions*): Solver configuration (uses verbose and convergence_threshold)
- **matrix_view** (*MatrixView*): Matrix storage format
- **solver_label** (*str*): Label for verbose output (e.g. &quot;Linear solver&quot;, &quot;Adjoint solver&quot;)


Returns
-------
array
    x if converged, NaN array otherwise

#### linear\_solve\_adjoint

```python
def linear_solve_adjoint(A,
                         b,
                         solver_options,
                         matrix_view: MatrixView,
                         bc=None,
                         linear_solve_fn=None)
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

#### create\_linear\_solver

```python
def create_linear_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[SolverOptions] = None,
    adjoint_solver_options: Optional[SolverOptions] = None
) -> Callable[[Any, jnp.ndarray], jnp.ndarray]
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
- **solver_options** (*SolverOptions, optional*): Options for the forward linear solve (defaults to SolverOptions()).
- **adjoint_solver_options** (*SolverOptions, optional*): Options for the adjoint solve used in the backward pass. Defaults to the same options as the forward solve.


Returns
-------
- **differentiable_solve** (*callable*): A function with signature ``(internal_vars, initial_guess) -&gt; solution`` that is differentiable w.r.t. ``internal_vars`` via ``jax.grad``.


Notes
-----
Forward pass performs a single linear solve::

    J * delta_sol = -res
    sol = initial_guess + delta_sol

Backward pass solves the adjoint system::

    J^T * adjoint = v

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
...     return jnp.sum(sol ** 2)
>>> grad = jax.grad(loss)(internal_vars)
```

