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
  (DirectSolverOptions, KrylovSolverOptions, or legacy SolverOptions)
- Support for multipoint constraints via prolongation matrices

Linear/backward/newton/reduced implementations live under ``solvers/``.

#### create\_solver

```python
def create_solver(
        problem: Problem,
        bc: DirichletBC,
        solver_options: Optional[AbstractSolverOptions] = None,
        adjoint_solver_options: Optional[AbstractSolverOptions] = None,
        newton_options: Optional[NewtonOptions] = None,
        linear: bool = False,
        P: Optional[BCOO] = None,
        internal_vars=None,
        extra_residual_fn: Optional[Callable] = None,
        symmetric_bc: bool = True) -> Callable
```

Create a differentiable solver with custom VJP for gradient computation.

Parameters
----------
- **problem** (*Problem*): The feax Problem instance.
- **bc** (*DirichletBC*): Boundary conditions.
- **solver_options** (*AbstractSolverOptions, optional*): Options for the forward linear solve. Accepts any subclass of ``AbstractSolverOptions``:
- **adjoint_solver_options** (*AbstractSolverOptions, optional*): Options for the adjoint solve in the backward pass. Defaults to same as ``solver_options``.
- **newton_options** (*NewtonOptions, optional*): Newton-specific nonlinear controls (tolerances and line search). Only used for the nonlinear path (``linear=False``).
- **linear** (*bool, default False*): Selects the solve path:
- **P** (*BCOO matrix, optional*): Prolongation matrix for periodic boundary conditions.
- **internal_vars** (*InternalVars, optional*): Sample internal variables for auto solver selection and cuDSS pre-warming. Required when ``solver=&quot;auto&quot;`` or cuDSS is used.
- **extra_residual_fn** (*callable, optional*): Additional residual contribution: ``extra_residual_fn(sol_flat) -&gt; residual_flat``. Combined with feax&#x27;s bulk residual via hybrid matrix-free Newton-Krylov: the bulk Jacobian is assembled (sparse), while the extra contribution&#x27;s Jacobian-vector product is computed via ``jax.jvp`` (forward-mode AD). Requires ``KrylovSolverOptions`` and the nonlinear path (``linear=False``).
- **symmetric_bc** (*bool, default True*): Controls how Dirichlet BCs are applied to the Jacobian matrix.


Returns
-------
callable
    When ``DirectSolverOptions`` is used:
        ``solver(internal_vars) -&gt; solution``
        (``initial_guess`` is optional and ignored if provided.)
    When ``KrylovSolverOptions`` is used:
        ``solver(internal_vars, initial_guess, bc=None) -&gt; solution``

    The optional ``bc`` parameter accepts a
    :class:``8 whose ``bc_rows`` match the
    original BC but ``bc_vals`` may differ.  This avoids rebuilding
    the solver when only prescribed values change (e.g. incremental
    loading).  Use :meth:``3 for convenience.

Examples
--------
```python
>>> # Direct solver (auto-selects cuDSS on GPU, spsolve on CPU)
>>> solver = create_solver(problem, bc, solver_options=DirectSolverOptions(),
...                        linear=True, internal_vars=internal_vars)
>>> solution = solver(internal_vars)
>>>
>>> # Iterative solver with auto selection
>>> solver = create_solver(problem, bc, solver_options=KrylovSolverOptions(),
...                        linear=True, internal_vars=internal_vars)
>>> solution = solver(internal_vars, initial_guess)
>>>
>>> # Explicit solver selection (no internal_vars needed for non-cuDSS)
>>> solver = create_solver(problem, bc, solver_options=KrylovSolverOptions(solver=&quot;gmres&quot;),
...                        linear=True)
>>> solution = solver(internal_vars, initial_guess)
>>>
>>> # Incremental loading with non-symmetric BC elimination
>>> solver = create_solver(problem, bc,
...                        solver_options=DirectSolverOptions(solver=&quot;spsolve&quot;),
...                        newton_options=NewtonOptions(tol=1e-6, max_iter=20),
...                        symmetric_bc=False,
...                        internal_vars=internal_vars)
>>> sol = zero_like_initial_guess(problem, bc)
>>> for step in range(1, num_steps + 1):
...     bc_step = bc.replace_vals(new_vals)  # update prescribed values
...     sol = solver(internal_vars, sol, bc=bc_step)
```

