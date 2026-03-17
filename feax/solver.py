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
from .solvers.matrix_free import (
    MatrixFreeOptions,
    create_matrix_free_solver,
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
    extra_residual_fn: Optional[Callable] = None,
    energy_fn: Optional[Callable] = None,
    symmetric_bc: bool = True,
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

        - ``DirectSolverOptions``: Direct solvers (cudss, cholmod, umfpack, spsolve).
        - ``IterativeSolverOptions``: Iterative solvers (cg, bicgstab, gmres).
        When ``solver="auto"`` (or when ``solver_options`` is ``None``), the
        algorithm is selected automatically by assembling the initial Jacobian
        and calling ``detect_matrix_property``.
        If not specified, defaults to ``DirectSolverOptions(solver="auto")``,
        which selects the best available direct solver (cuDSS on GPU,
        cholmod/umfpack/spsolve on CPU). Use ``IterativeSolverOptions``
        explicitly when a direct solver is too memory-intensive.
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
    extra_residual_fn : callable, optional
        Additional residual contribution: ``extra_residual_fn(sol_flat) -> residual_flat``.
        Combined with feax's bulk residual via hybrid matrix-free Newton-Krylov:
        the bulk Jacobian is assembled (sparse), while the extra contribution's
        Jacobian-vector product is computed via ``jax.jvp`` (forward-mode AD).
        Requires ``IterativeSolverOptions`` and ``iter_num != 1`` (Newton path).
    symmetric_bc : bool, default True
        Controls how Dirichlet BCs are applied to the Jacobian matrix.

        - ``True`` (symmetric elimination): Zeros both BC rows **and** columns
          in the Jacobian, then sets BC diagonal entries to 1. This preserves
          matrix symmetry (enabling symmetric solvers like CG) but removes the
          K₁₀ coupling between BC DOFs and interior DOFs. Suitable when BC
          values are pre-applied to the initial guess and BC DOF increments
          are zero during Newton iterations (e.g. fixed BCs, linear problems).

        - ``False`` (non-symmetric elimination): Zeros only BC **rows** in the
          Jacobian, keeping BC columns (K₁₀ coupling) intact. The Newton solver
          drives BC DOFs to their prescribed values through the modified
          residual: ``res[bc_dof] = sol[bc_dof] - bc_val``. This is required
          when the K₁₀ coupling is important for Newton convergence.

        Use ``symmetric_bc=False`` when:

        1. **Incremental loading**: BC values change per load step and the
           previous solution is reused as initial guess. The K₁₀ coupling
           ensures that prescribed displacement changes propagate correctly
           to interior DOFs in the Newton linearization.
        2. **Soft regions with large stiffness contrast**: e.g. third-medium
           contact where the medium stiffness is scaled by γ₀ ≈ 1e-6. Without
           K₁₀ coupling, the first Newton increment overshoots in soft regions,
           causing divergence.
        3. **Large-deformation nonlinear problems** where BC DOF displacements
           are large and strongly coupled to interior DOFs.

        Note that ``symmetric_bc=False`` produces a non-symmetric Jacobian,
        so symmetric solvers (CG) cannot be used. Use ``spsolve``, ``umfpack``,
        ``bicgstab``, or ``gmres`` instead.

    Returns
    -------
    callable
        When ``DirectSolverOptions`` is used:
            ``solver(internal_vars) -> solution``
            (``initial_guess`` is optional and ignored if provided.)
        When ``IterativeSolverOptions`` is used:
            ``solver(internal_vars, initial_guess, bc=None) -> solution``

        The optional ``bc`` parameter accepts a
        :class:`~feax.DCboundary.DirichletBC` whose ``bc_rows`` match the
        original BC but ``bc_vals`` may differ.  This avoids rebuilding
        the solver when only prescribed values change (e.g. incremental
        loading).  Use :meth:`DirichletBC.replace_vals` for convenience.

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
    >>>
    >>> # Incremental loading with non-symmetric BC elimination
    >>> solver = create_solver(problem, bc,
    ...                        solver_options=DirectSolverOptions(solver="spsolve"),
    ...                        newton_options=NewtonOptions(tol=1e-6, max_iter=20),
    ...                        iter_num=None, symmetric_bc=False,
    ...                        internal_vars=internal_vars)
    >>> sol = zero_like_initial_guess(problem, bc)
    >>> for step in range(1, num_steps + 1):
    ...     bc_step = bc.replace_vals(new_vals)  # update prescribed values
    ...     sol = solver(internal_vars, sol, bc=bc_step)
    """
    linear_options = solver_options
    adjoint_linear_options = adjoint_solver_options

    # SolverOptions is now deprecated
    if isinstance(linear_options, SolverOptions) or isinstance(adjoint_linear_options, SolverOptions):
        raise RuntimeError(
            "SolverOptions has been removed. "
            "Use DirectSolverOptions or IterativeSolverOptions."
        )

    # Validate energy_fn constraints
    if energy_fn is not None and not isinstance(linear_options, MatrixFreeOptions):
        raise ValueError(
            "energy_fn is only supported with MatrixFreeOptions. "
            "Pass solver_options=MatrixFreeOptions(...) to use a custom energy function."
        )

    # Validate extra_residual_fn constraints
    if extra_residual_fn is not None:
        if iter_num == 1:
            raise ValueError(
                "extra_residual_fn requires Newton solver (iter_num != 1). "
                "The hybrid matrix-free approach needs iterative Newton updates."
            )
        if isinstance(linear_options, DirectSolverOptions):
            raise ValueError(
                "extra_residual_fn requires IterativeSolverOptions. "
                "The hybrid Jacobian is a callable matvec, not a sparse matrix."
            )
        if P is not None:
            raise ValueError(
                "extra_residual_fn cannot be combined with P (prolongation matrix). "
                "Use one or the other."
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

    # 2) Matrix-free path (energy-based, JVP tangent)
    if isinstance(linear_options, MatrixFreeOptions):
        if iter_num == 1:
            raise ValueError(
                "MatrixFreeOptions requires nonlinear Newton solve (iter_num != 1). "
                "The matrix-free solver always performs a full Newton iteration."
            )
        if newton_options is not None:
            import warnings as _warnings
            _warnings.warn(
                "[feax] newton_options is ignored when MatrixFreeOptions is used. "
                "Configure Newton tolerances via MatrixFreeOptions(newton_tol=..., newton_max_iter=...).",
                UserWarning,
                stacklevel=2,
            )
        return create_matrix_free_solver(
            problem=problem,
            bc=bc,
            options=linear_options,
            energy_fn=energy_fn,
        )

    # Non-reduced paths (Newton / linear): normalize missing options.
    # Default to DirectSolverOptions (direct solvers are preferred for
    # small-to-medium problems; COMSOL-style: try direct first, fall back
    # to iterative only when memory/performance is an issue).
    _linear_options_missing = linear_options is None
    if linear_options is None:
        linear_options = DirectSolverOptions()
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

    if not isinstance(linear_options, (DirectSolverOptions, IterativeSolverOptions, MatrixFreeOptions)):
        raise TypeError(
            "Unsupported solver_options type. "
            f"Expected DirectSolverOptions, IterativeSolverOptions, or MatrixFreeOptions, got {type(linear_options).__name__}."
        )
    if not isinstance(adjoint_linear_options, (DirectSolverOptions, IterativeSolverOptions, MatrixFreeOptions)):
        raise TypeError(
            "Unsupported adjoint_solver_options type. "
            f"Expected DirectSolverOptions, IterativeSolverOptions, or MatrixFreeOptions, got {type(adjoint_linear_options).__name__}."
        )

    J_bc_func = create_J_bc_function(problem, bc, symmetric=symmetric_bc)

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
            extra_residual_fn=extra_residual_fn,
            symmetric_bc=symmetric_bc,
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