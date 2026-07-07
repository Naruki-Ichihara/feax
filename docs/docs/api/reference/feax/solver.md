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
        traced_params=None,
        extra_residual_fn: Optional[Callable] = None,
        symmetric_elimination: bool = True,
        traced_structure=None,
        return_solution: bool = True) -> Callable
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
- **traced_params** (*TracedParams, optional*): Sample internal variables for auto solver selection and cuDSS pre-warming. Required when ``solver=&quot;auto&quot;`` or cuDSS is used.
- **extra_residual_fn** (*callable, optional*): Additional residual contribution: ``extra_residual_fn(sol_flat) -&gt; residual_flat``. Requires the nonlinear path (``linear=False``). Two treatments:
- **symmetric_elimination** (*bool, default True*): Controls how Dirichlet BCs are applied to the Jacobian matrix.
- **traced_structure** (*TracedStructure, optional*): A :class:``0 for this problem. When given, every sample assembly done at solver-construction time is routed through the TracedStructure path instead of the no-TracedStructure path, namely:
- **return_solution** (*bool, default True*): The solver returns a :class:``7 (the default): ``sol.field(i)`` replaces ``problem.unflatten_fn_sol_list(sol)[i]``, ``sol.node_var(component=)`` bridges into the next solve&#x27;s ``TracedParams`` (or pass the Solution directly as a volume var), and array/arithmetic protocols keep it usable wherever a flat vector is expected (``jnp.dot``, ``sol - ref``, ``band.scatter_sol``, or as the next call&#x27;s ``initial_guess``). Pass ``False`` for the raw flat DOF vector.


Returns
-------
callable
    When ``DirectSolverOptions`` is used:
        ``solver(traced_params, initial_guess=None, bc=None, traced_structure=None) -&gt; solution``
        (``initial_guess`` is optional and ignored if provided.)
    When ``KrylovSolverOptions`` is used:
        ``solver(traced_params, initial_guess, bc=None, traced_structure=None) -&gt; solution``

    The optional ``bc`` parameter accepts a
    :class:``18 whose ``bc_rows`` match the
    original BC but ``bc_vals`` may differ.  This avoids rebuilding
    the solver when only prescribed values change (e.g. incremental
    loading).  Use :meth:``23 for convenience.

    Pass ``traced_structure=ts`` to run the solve on the TracedStructure
    path (required if the host slot maps were released — the default of
    ``TracedStructure.from_problem``); omit it to use the
    no-TracedStructure path.

Examples
--------
```python
>>> # Direct solver (auto-selects cuDSS on GPU, spsolve on CPU)
>>> solver = create_solver(problem, bc, solver_options=DirectSolverOptions(),
...                        linear=True, traced_params=traced_params)
>>> solution = solver(traced_params)
>>>
>>> # Iterative solver with auto selection
>>> solver = create_solver(problem, bc, solver_options=KrylovSolverOptions(),
...                        linear=True, traced_params=traced_params)
>>> solution = solver(traced_params, initial_guess)
>>>
>>> # Explicit solver selection (no traced_params needed for non-cuDSS)
>>> solver = create_solver(problem, bc, solver_options=KrylovSolverOptions(solver=&quot;gmres&quot;),
...                        linear=True)
>>> solution = solver(traced_params, initial_guess)
>>>
>>> # Incremental loading with non-symmetric BC elimination
>>> solver = create_solver(problem, bc,
...                        solver_options=DirectSolverOptions(solver=&quot;spsolve&quot;),
...                        newton_options=NewtonOptions(tol=1e-6, max_iter=20),
...                        symmetric_elimination=False,
...                        traced_params=traced_params)
>>> sol = zero_like_initial_guess(problem, bc)
>>> for step in range(1, num_steps + 1):
...     bc_step = bc.replace_vals(new_vals)  # update prescribed values
...     sol = solver(traced_params, sol, bc=bc_step)
```

