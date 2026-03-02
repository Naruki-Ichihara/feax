"""
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

Linear/backward/newton/reduced implementations live under ``solvers/``.
"""

import logging
from typing import Callable, Optional

from jax.experimental.sparse import BCOO

from .assembler import create_J_bc_function
from .DCboundary import DirichletBC
from .problem import Problem
from .solvers.linear import (
    create_linear_solver,
)
from .solvers.newton import (
    create_newton_solver,
)
from .solvers.options import (
    AbstractSolverOptions,
    DirectSolverOptions,
    IterativeSolverOptions,
    NewtonOptions,
    SolverOptions,
    detect_matrix_property,
    resolve_direct_solver,
    resolve_iterative_solver,
)
from .solvers.reduced import create_reduced_solver

logger = logging.getLogger(__name__)


def create_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[AbstractSolverOptions] = None,
    adjoint_solver_options: Optional[AbstractSolverOptions] = None,
    newton_options: Optional[NewtonOptions] = None,
    iter_num: Optional[int] = None,
    P: Optional[BCOO] = None,
    internal_vars=None,
) -> Callable:
    """Create a differentiable solver with custom VJP for gradient computation.

    Parameters
    ----------
    problem : Problem
        The feax Problem instance.
    bc : DirichletBC
        Boundary conditions.
    solver_options : AbstractSolverOptions, optional
        Options for the forward linear solve. Accepts any subclass of
        ``AbstractSolverOptions``:

        - ``DirectSolverOptions``: Direct solvers (cudss, spsolve, cholesky, lu, qr).
        - ``IterativeSolverOptions``: Iterative solvers (cg, bicgstab, gmres).
        When ``solver="auto"`` (or when ``solver_options`` is ``None``), the
        algorithm is selected automatically by assembling the initial Jacobian
        and calling ``detect_matrix_property``.
        If not specified, forward options default to iterative auto mode.
    adjoint_solver_options : AbstractSolverOptions, optional
        Options for the adjoint solve in the backward pass.
        Defaults to same as ``solver_options``.
    newton_options : NewtonOptions, optional
        Newton-specific nonlinear controls (tolerances and line search).
        Only used when ``iter_num != 1``.
    iter_num : int, optional
        Number of Newton iterations:

        - ``None``: Adaptive Newton solve with while loop (NOT vmappable).
        - ``1``: Single linear solve (vmappable). Recommended for linear problems.
        - ``>1``: Fixed-iteration Newton solve (vmappable).
    P : BCOO matrix, optional
        Prolongation matrix for periodic boundary conditions.
    internal_vars : InternalVars, optional
        Sample internal variables for auto solver selection and cuDSS
        pre-warming. Required when ``solver="auto"`` or cuDSS is used.

    Returns
    -------
    callable
        When ``DirectSolverOptions`` is used:
            ``solver(internal_vars) -> solution``
            (``initial_guess`` is optional and ignored if provided.)
        When ``IterativeSolverOptions`` is used:
            ``solver(internal_vars, initial_guess) -> solution``

    Examples
    --------
    >>> # Direct solver (auto-selects cuDSS on GPU, spsolve on CPU)
    >>> solver = create_solver(problem, bc, solver_options=DirectSolverOptions(),
    ...                        iter_num=1, internal_vars=internal_vars)
    >>> solution = solver(internal_vars)
    >>>
    >>> # Iterative solver with auto selection
    >>> solver = create_solver(problem, bc, solver_options=IterativeSolverOptions(),
    ...                        iter_num=1, internal_vars=internal_vars)
    >>> solution = solver(internal_vars, initial_guess)
    >>>
    >>> # Explicit solver selection (no internal_vars needed for non-cuDSS)
    >>> solver = create_solver(problem, bc, solver_options=IterativeSolverOptions(solver="gmres"),
    ...                        iter_num=1)
    >>> solution = solver(internal_vars, initial_guess)
    """
    linear_options = solver_options
    adjoint_linear_options = adjoint_solver_options

    # SolverOptions is now deprecated
    if isinstance(linear_options, SolverOptions) or isinstance(adjoint_linear_options, SolverOptions):
        raise RuntimeError(
            "SolverOptions has been removed. "
            "Use DirectSolverOptions or IterativeSolverOptions."
        )

    # 1) Reduced path (matrix-free, requires iterative options)
    if P is not None:
        if linear_options is None:
            linear_options = IterativeSolverOptions()
        if adjoint_linear_options is None:
            adjoint_linear_options = linear_options

        if not isinstance(linear_options, IterativeSolverOptions):
            raise ValueError(
                "solver_options must be IterativeSolverOptions when P (prolongation matrix) "
                f"is provided, got {type(linear_options).__name__}. "
                "The reduced problem is matrix-free and only supports iterative solvers "
                "(cg, bicgstab, gmres)."
            )
        if not isinstance(adjoint_linear_options, IterativeSolverOptions):
            raise ValueError(
                "adjoint_solver_options must be IterativeSolverOptions when P is provided, "
                f"got {type(adjoint_linear_options).__name__}."
            )
        return create_reduced_solver(problem, bc, P, linear_options, adjoint_linear_options)

    # Non-reduced paths (Newton / linear): normalize missing options.
    _linear_options_missing = linear_options is None
    if linear_options is None:
        linear_options = IterativeSolverOptions()
    if linear_options is not None and adjoint_linear_options is None:
        adjoint_linear_options = linear_options

    # Resolve "auto" solvers using detect_matrix_property
    _shared_opts = adjoint_linear_options is linear_options
    _needs_auto = (
        _linear_options_missing or
        linear_options.solver == "auto" or
        adjoint_linear_options.solver == "auto"
    )

    if _needs_auto:
        if internal_vars is None:
            raise ValueError(
                "internal_vars is required when solver_options is None or has solver='auto'. "
                "Pass a sample InternalVars to enable automatic matrix property detection, "
                "or specify the solver explicitly (e.g. IterativeSolverOptions(solver='cg'))."
            )

    if not isinstance(linear_options, (DirectSolverOptions, IterativeSolverOptions)):
        raise TypeError(
            "Unsupported solver_options type. "
            f"Expected DirectSolverOptions or IterativeSolverOptions, got {type(linear_options).__name__}."
        )
    if not isinstance(adjoint_linear_options, (DirectSolverOptions, IterativeSolverOptions)):
        raise TypeError(
            "Unsupported adjoint_solver_options type. "
            f"Expected DirectSolverOptions or IterativeSolverOptions, got {type(adjoint_linear_options).__name__}."
        )

    J_bc_func = create_J_bc_function(problem, bc)

    # Assemble sample Jacobian and resolve "auto" only when needed.
    if _needs_auto:
        from .utils import zero_like_initial_guess
        _initial_tmp = zero_like_initial_guess(problem, bc)
        _sample_J = J_bc_func(_initial_tmp, internal_vars)
        _mp = detect_matrix_property(_sample_J, matrix_view=problem.matrix_view)
        from .solvers.options import detect_backend
        _backend = detect_backend()

        if linear_options.solver == "auto":
            _category = "direct" if isinstance(linear_options, DirectSolverOptions) else "iterative"
            if isinstance(linear_options, DirectSolverOptions):
                linear_options = resolve_direct_solver(linear_options, _mp, matrix_view=problem.matrix_view)
            else:
                linear_options = resolve_iterative_solver(linear_options, _mp)
            print(f"[feax] Auto solver ({_category}): backend={_backend.name}, matrix_property={_mp.name} -> {linear_options.solver}")

        if _shared_opts:
            adjoint_linear_options = linear_options
        elif adjoint_linear_options.solver == "auto":
            _category = "direct" if isinstance(adjoint_linear_options, DirectSolverOptions) else "iterative"
            if isinstance(adjoint_linear_options, DirectSolverOptions):
                adjoint_linear_options = resolve_direct_solver(adjoint_linear_options, _mp, matrix_view=problem.matrix_view)
            else:
                adjoint_linear_options = resolve_iterative_solver(adjoint_linear_options, _mp)
            print(f"[feax] Auto adjoint solver ({_category}): backend={_backend.name}, matrix_property={_mp.name} -> {adjoint_linear_options.solver}")

    # 2) Newton path
    if iter_num != 1:
        return create_newton_solver(
            problem=problem,
            bc=bc,
            linear_options=linear_options,
            adjoint_linear_options=adjoint_linear_options,
            iter_num=iter_num,
            newton_options=newton_options,
            internal_vars=internal_vars,
        )

    # 3) Linear path (iter_num == 1)
    if iter_num == 1:
        if newton_options is not None and newton_options.internal_jit:
            import warnings as _warnings
            _warnings.warn(
                "[feax] newton_options.internal_jit=True is ignored when iter_num=1: "
                "the linear solver is called only once per solve and "
                "internal JIT provides no benefit. "
                "Use jax.jit(solver) to JIT the entire solve instead.",
                UserWarning,
                stacklevel=2,
            )
        return create_linear_solver(
            problem=problem,
            bc=bc,
            solver_options=linear_options,
            adjoint_solver_options=adjoint_linear_options,
            internal_vars=internal_vars,
        )
