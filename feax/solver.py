"""
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
"""

import logging
from typing import Callable, Optional

from jax.experimental.sparse import BCOO

from .assembler import create_J_bc_csr_function
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
    KrylovSolverOptions,
    NewtonOptions,
    SolverOptions,
    detect_matrix_property,
    resolve_direct_solver,
    resolve_iterative_solver,
)
from .solvers.reduced import create_reduced_solver

logger = logging.getLogger(__name__)


class Solver:
    """The callable :func:`create_solver` returns.

    Calling it runs the solve, ``solver(traced_params, initial_guess=None,
    bc=None, traced_structure=None, preconditioner=None)``, and (by default)
    wraps the result in a :class:`feax.Solution`. It also carries what it was
    built from — ``problem``, ``bc``, ``options``, ``adjoint_options``,
    ``sample_params``, ``traced_structure``, ``preconditioner``, ``linear``,
    ``symmetric_elimination``, ``projection`` — and can assemble its own
    operator (:meth:`assemble`) and residual (:meth:`residual`); this is what
    :func:`feax.vmap` uses to batch it.
    """

    def __init__(self, fn, *, problem, bc, options, adjoint_options, linear,
                 symmetric_elimination, sample_params=None, traced_structure=None,
                 preconditioner=None, adjoint_preconditioner=None, projection=None,
                 return_solution=True):
        self._fn = fn
        self.problem = problem
        self.bc = bc
        self.options = options
        self.adjoint_options = adjoint_options
        self.linear = linear
        self.symmetric_elimination = symmetric_elimination
        self.sample_params = sample_params
        self.traced_structure = traced_structure
        self.preconditioner = preconditioner
        self.adjoint_preconditioner = adjoint_preconditioner
        self.projection = projection
        self.return_solution = return_solution
        self.layout = tuple((fe_.num_total_nodes, int(v_))
                            for fe_, v_ in zip(problem.fes, problem.vec))
        self._res_fn = None
        self._J_fn = None

    def __call__(self, *args, **kwargs):
        from .solution import Solution
        args = tuple(a.dofs if isinstance(a, Solution) else a for a in args)
        kwargs = {k: (v.dofs if isinstance(v, Solution) else v) for k, v in kwargs.items()}
        out = self._fn(*args, **kwargs)
        return Solution(out, self.layout) if self.return_solution else out

    def residual(self, sol, traced_params, bc=None, traced_structure=None):
        """The BC-applied residual ``r(sol; traced_params)`` (flat)."""
        if self._res_fn is None:
            from .assembler import create_res_bc_parametric
            self._res_fn = create_res_bc_parametric(self.problem)
        return self._res_fn(sol, traced_params, bc if bc is not None else self.bc,
                            traced_structure if traced_structure is not None else self.traced_structure)

    def assemble(self, sol, traced_params, bc=None, traced_structure=None):
        """The BC-applied tangent at ``sol`` as a :class:`feax.csr.CSRMatrix`."""
        if self._J_fn is None:
            from .assembler import create_J_bc_csr_parametric
            self._J_fn = create_J_bc_csr_parametric(self.problem, symmetric=self.symmetric_elimination)
        return self._J_fn(sol, traced_params, bc if bc is not None else self.bc,
                          traced_structure if traced_structure is not None else self.traced_structure)

    def __repr__(self):
        kind = "linear" if self.linear else "newton"
        return (f"Solver({kind}, options={type(self.options).__name__}"
                f"({getattr(self.options, 'solver', '?')}), "
                f"preconditioner={getattr(self.preconditioner, '__name__', self.preconditioner)!r})")


def create_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[AbstractSolverOptions] = None,
    adjoint_solver_options: Optional[AbstractSolverOptions] = None,
    newton_options: Optional[NewtonOptions] = None,
    linear: bool = False,
    projection: Optional[BCOO] = None,
    traced_params=None,
    extra_residual_fn: Optional[Callable] = None,
    symmetric_elimination: bool = True,
    traced_structure=None,
    return_solution: bool = True,
    preconditioner=None,
    adjoint_preconditioner=None,
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
        - ``KrylovSolverOptions``: Iterative solvers (cg, bicgstab, gmres).
        When ``solver="auto"`` (or when ``solver_options`` is ``None``), the
        algorithm is selected automatically by assembling the initial Jacobian
        and calling ``detect_matrix_property``.
        If not specified, defaults to ``DirectSolverOptions(solver="auto")``,
        which selects the best available direct solver (cuDSS on GPU,
        cholmod/umfpack/spsolve on CPU). Use ``KrylovSolverOptions``
        explicitly when a direct solver is too memory-intensive.
    adjoint_solver_options : AbstractSolverOptions, optional
        Options for the adjoint solve in the backward pass.
        Defaults to same as ``solver_options``.
    newton_options : NewtonOptions, optional
        Newton-specific nonlinear controls (tolerances and line search).
        Only used for the nonlinear path (``linear=False``).
    linear : bool, default False
        Selects the solve path:

        - ``False`` (default): Adaptive nonlinear Newton solve (iterates to
          ``newton_options.tol`` / ``rel_tol``, capped at ``max_iter``).
        - ``True``: Single linear solve. Recommended for linear problems
          (e.g. linear elasticity, steady diffusion).

        Both paths are differentiable and compose with ``jax.jit`` / ``jax.vmap``
        (the Newton forward is a traced ``jax.lax.while_loop`` — one Newton step
        per loop body, no ``pure_callback`` node — so it jits and vmaps natively;
        batched solves use a block-diagonal direct solve).
    projection : BCOO matrix, optional
        Prolongation (projection) matrix of a reduced solve, e.g. periodic
        boundary conditions: the system is solved for the reduced DOFs of
        ``PᵀJP``.
    traced_params : TracedParams, optional
        Sample internal variables for auto solver selection and cuDSS
        pre-warming. Required when ``solver="auto"`` or cuDSS is used.
    extra_residual_fn : callable, optional
        Additional residual contribution: ``extra_residual_fn(sol_flat) -> residual_flat``.
        Requires the nonlinear path (``linear=False``). Two treatments:

        - With ``KrylovSolverOptions``: hybrid matrix-free Newton-Krylov — the
          bulk Jacobian is assembled (sparse), the extra contribution's
          Jacobian-vector product comes from ``jax.jvp``.
        - With ``DirectSolverOptions``: the extra Jacobian's sparsity is
          detected once via automatic sparse differentiation (asdex), colored,
          and assembled onto the merged (bulk ∪ extra) CSR pattern — the
          direct solver then factorizes the exact combined tangent.
    symmetric_elimination : bool, default True
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

        Use ``symmetric_elimination=False`` when:

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

        Note that ``symmetric_elimination=False`` produces a non-symmetric Jacobian,
        so symmetric solvers (CG) cannot be used. Use ``spsolve``, ``umfpack``,
        ``bicgstab``, or ``gmres`` instead.
    traced_structure : TracedStructure, optional
        A :class:`feax.TracedStructure` for this problem. When given, every
        sample assembly done at solver-construction time is routed through the
        TracedStructure path instead of the no-TracedStructure path, namely:

        1. the ``"auto"`` matrix-property probe (when ``solver="auto"`` /
           ``solver_options`` is ``None``), and
        2. pre-warming of direct solvers (cuDSS / cholmod / umfpack / spsolve),
           which assemble a sample Jacobian to fix the CSR structure.

        This matters because ``TracedStructure.from_problem`` releases the
        host-side slot maps by default (``free_scratch=True``); the
        no-TracedStructure path reads those maps and would raise. The
        **recommended flow** is therefore to build the TracedStructure first,
        pass it here, and also pass it to every ``solver(...)`` call::

            ts = feax.TracedStructure.from_problem(problem)
            solver = feax.create_solver(problem, bc, opts,
                                        traced_params=tp, traced_structure=ts)
            sol = solver(tp, x0, traced_structure=ts)

        Omit it only if you keep the host maps alive
        (``TracedStructure.from_problem(problem, free_scratch=False)`` or no
        TracedStructure at all), in which case the legacy no-TracedStructure
        path is used.
    return_solution : bool, default True
        The solver returns a :class:`feax.Solution` (the default):
        ``sol.field(i)`` replaces ``problem.unflatten_fn_sol_list(sol)[i]``,
        ``sol.node_var(component=)`` bridges into the next solve's
        ``TracedParams`` (or pass the Solution directly as a volume var), and
        array/arithmetic protocols keep it usable wherever a flat vector is
        expected (``jnp.dot``, ``sol - ref``, ``band.scatter_sol``, or as the
        next call's ``initial_guess``). Pass ``False`` for the raw flat DOF
        vector.

    preconditioner : callable, optional
        The preconditioner of a Krylov forward solve, as one callable
        ``preconditioner(op) -> M`` that is called inside each solve with that
        solve's operator (a :class:`feax.LinearOperator`: matvec, and the
        assembled values / diagonal on demand) and returns ``M(r) -> z``.
        Built-ins: :func:`feax.jacobi` (diagonal of the current operator) and
        :func:`feax.amg` (a smoothed-aggregation hierarchy built from a sample
        operator); a fixed ``M`` is ``lambda op: M``. An error with a direct
        solver. It can be overridden per call,
        ``solver(tp, u0, preconditioner=P)``, which is how a pytree
        preconditioner (:class:`feax.AMGPreconditioner`) becomes a traced
        argument of a jitted solve instead of a closure constant. With the
        Newton solver, an AMG preconditioner whose ``rebuild_every`` is not 0
        runs a host loop that rebuilds the hierarchy from the current tangent.
    adjoint_preconditioner : callable, optional
        The same for the adjoint (backward-pass) solve. Defaults to
        ``preconditioner`` when the adjoint solve is Krylov.

    Returns
    -------
    callable
        When ``DirectSolverOptions`` is used:
            ``solver(traced_params, initial_guess=None, bc=None, traced_structure=None) -> solution``
            (``initial_guess`` is optional and ignored if provided.)
        When ``KrylovSolverOptions`` is used:
            ``solver(traced_params, initial_guess, bc=None, traced_structure=None) -> solution``

        The optional ``bc`` parameter accepts a
        :class:`~feax.DCboundary.DirichletBC` whose ``bc_rows`` match the
        original BC but ``bc_vals`` may differ.  This avoids rebuilding
        the solver when only prescribed values change (e.g. incremental
        loading).  Use :meth:`DirichletBC.replace_vals` for convenience.

        Pass ``traced_structure=ts`` to run the solve on the TracedStructure
        path (required if the host slot maps were released — the default of
        ``TracedStructure.from_problem``); omit it to use the
        no-TracedStructure path.

    Examples
    --------
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
    >>> solver = create_solver(problem, bc, solver_options=KrylovSolverOptions(solver="gmres"),
    ...                        linear=True)
    >>> solution = solver(traced_params, initial_guess)
    >>>
    >>> # Incremental loading with non-symmetric BC elimination
    >>> solver = create_solver(problem, bc,
    ...                        solver_options=DirectSolverOptions(solver="spsolve"),
    ...                        newton_options=NewtonOptions(tol=1e-6, max_iter=20),
    ...                        symmetric_elimination=False,
    ...                        traced_params=traced_params)
    >>> sol = zero_like_initial_guess(problem, bc)
    >>> for step in range(1, num_steps + 1):
    ...     bc_step = bc.replace_vals(new_vals)  # update prescribed values
    ...     sol = solver(traced_params, sol, bc=bc_step)
    """
    linear_options = solver_options
    adjoint_linear_options = adjoint_solver_options

    def _wrap(solver_fn):
        """Wrap the solve function in a :class:`Solver` carrying its metadata
        (the resolved options are read when this is called)."""
        return Solver(
            solver_fn, problem=problem, bc=bc, options=linear_options,
            adjoint_options=adjoint_linear_options, linear=bool(linear or projection is not None),
            symmetric_elimination=symmetric_elimination, sample_params=traced_params,
            traced_structure=traced_structure, preconditioner=preconditioner,
            adjoint_preconditioner=adjoint_preconditioner, projection=projection,
            return_solution=return_solution)

    # SolverOptions is now deprecated
    if isinstance(linear_options, SolverOptions) or isinstance(adjoint_linear_options, SolverOptions):
        raise RuntimeError(
            "SolverOptions has been removed. "
            "Use DirectSolverOptions or KrylovSolverOptions."
        )

    # Validate extra_residual_fn constraints
    if extra_residual_fn is not None:
        if linear:
            raise ValueError(
                "extra_residual_fn requires the nonlinear Newton solver (linear=False)."
            )
        if projection is not None:
            raise ValueError(
                "extra_residual_fn cannot be combined with projection (prolongation matrix). "
                "Use one or the other."
            )

    # 1) Reduced path. KrylovSolverOptions runs matrix-free (Pᵀ(J(Pv))
    # matvecs); DirectSolverOptions (and a rebuilt AMG preconditioner) assemble
    # the reduced operator PᵀJP via ASD (pattern = boolean triple product,
    # values = colored probes of the matrix-free action) — see
    # feax.solvers.reduced / feax.asd.
    if projection is not None:
        if linear_options is None:
            linear_options = KrylovSolverOptions()
        if adjoint_linear_options is None:
            adjoint_linear_options = linear_options

        _reduced_ok = (DirectSolverOptions, KrylovSolverOptions)
        if not isinstance(linear_options, _reduced_ok):
            raise ValueError(
                "solver_options must be Krylov/Direct solver options when projection "
                f"(prolongation matrix) is provided, got {type(linear_options).__name__}."
            )
        if not isinstance(adjoint_linear_options, _reduced_ok):
            raise ValueError(
                "adjoint_solver_options must be Krylov/Direct solver options when "
                f"projection is provided, got {type(adjoint_linear_options).__name__}."
            )
        return _wrap(create_reduced_solver(
            problem, bc, projection, linear_options, adjoint_linear_options,
            preconditioner=preconditioner, adjoint_preconditioner=adjoint_preconditioner))

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
    def _is_resolvable_auto(o):
        return (isinstance(o, (DirectSolverOptions, KrylovSolverOptions))
                and o.solver == "auto")
    _needs_auto = (
        _linear_options_missing or
        _is_resolvable_auto(linear_options) or
        _is_resolvable_auto(adjoint_linear_options)
    )

    if _needs_auto:
        if traced_params is None:
            raise ValueError(
                "traced_params is required when solver_options is None or has solver='auto'. "
                "Pass a sample TracedParams to enable automatic matrix property detection, "
                "or specify the solver explicitly (e.g. KrylovSolverOptions(solver='cg'))."
            )

    from .solvers.preconditioners import AMGOptions
    for _role, _opts in (("solver_options", linear_options),
                         ("adjoint_solver_options", adjoint_linear_options)):
        if isinstance(_opts, AMGOptions):
            raise TypeError(
                f"{_role} got AMGOptions. AMG is a preconditioner: pass "
                "KrylovSolverOptions as the options and preconditioner=fe.amg(problem, "
                "bc, tp, AMGOptions(...)).")
        if not isinstance(_opts, (DirectSolverOptions, KrylovSolverOptions)):
            raise TypeError(
                f"Unsupported {_role} type. Expected DirectSolverOptions or "
                f"KrylovSolverOptions, got {type(_opts).__name__}."
            )
    for _role, _pre, _opts in (("preconditioner", preconditioner, linear_options),
                               ("adjoint_preconditioner", adjoint_preconditioner, adjoint_linear_options)):
        if _pre is not None and isinstance(_opts, DirectSolverOptions):
            raise ValueError(
                f"{_role} has no effect with a direct solver ({_opts.solver}); "
                "use KrylovSolverOptions with it, or drop it.")
        if _pre is not None and not callable(_pre):
            raise TypeError(
                f"{_role} must be callable, preconditioner(op) -> M (e.g. fe.jacobi(), "
                f"fe.amg(...)), got {type(_pre).__name__}.")

    # Assemble sample Jacobian (CSR-direct) and resolve "auto" only when needed.
    if _needs_auto:
        from .utils import zero_like_initial_guess
        _initial_tmp = zero_like_initial_guess(problem, bc)
        # Detect the matrix property on the TracedStructure (recommended) path when an
        # ``ts`` is supplied, so the probe never touches the no-TracedStructure slot
        # maps (``csr_volume_slots`` / ``res_volume_dofs``). This lets the caller
        # build TracedStructure first (which frees those host arrays) before
        # create_solver, with no ordering constraint. Falls back to the
        # closure/no-ts assembly when ``traced_structure`` is omitted.
        if traced_structure is not None:
            from .assembler import create_J_bc_csr_parametric
            J_bc_csr_func = create_J_bc_csr_parametric(problem, symmetric=symmetric_elimination)
            _sample_J = J_bc_csr_func(_initial_tmp, traced_params, bc, traced_structure)
        else:
            J_bc_csr_func = create_J_bc_csr_function(problem, bc, symmetric=symmetric_elimination)
            _sample_J = J_bc_csr_func(_initial_tmp, traced_params)
        _mp = detect_matrix_property(_sample_J, matrix_view=problem.matrix_view)
        from .solvers.options import detect_backend
        _backend = detect_backend()

        if _is_resolvable_auto(linear_options):
            _category = "direct" if isinstance(linear_options, DirectSolverOptions) else "iterative"
            if isinstance(linear_options, DirectSolverOptions):
                linear_options = resolve_direct_solver(linear_options, _mp, matrix_view=problem.matrix_view)
            else:
                linear_options = resolve_iterative_solver(linear_options, _mp)
            print(f"[feax] Auto solver ({_category}): backend={_backend.name}, matrix_property={_mp.name} -> {linear_options.solver}")

        if _shared_opts:
            adjoint_linear_options = linear_options
        elif _is_resolvable_auto(adjoint_linear_options):
            _category = "direct" if isinstance(adjoint_linear_options, DirectSolverOptions) else "iterative"
            if isinstance(adjoint_linear_options, DirectSolverOptions):
                adjoint_linear_options = resolve_direct_solver(adjoint_linear_options, _mp, matrix_view=problem.matrix_view)
            else:
                adjoint_linear_options = resolve_iterative_solver(adjoint_linear_options, _mp)
            print(f"[feax] Auto adjoint solver ({_category}): backend={_backend.name}, matrix_property={_mp.name} -> {adjoint_linear_options.solver}")

    # Foot-gun guard: symmetric_elimination=False produces a NON-symmetric
    # tangent, but cuDSS defaults to matrix_type=SYMMETRIC (LDLT) — that
    # combination silently factorizes a symmetrized matrix and Newton stalls
    # in the line search. Warn loudly instead of failing mysteriously.
    if not symmetric_elimination:
        import warnings
        for _role, _opts in (("solver_options", linear_options),
                             ("adjoint_solver_options", adjoint_linear_options)):
            if (isinstance(_opts, DirectSolverOptions) and _opts.solver == "cudss"
                    and getattr(_opts.cudss_options.matrix_type, "value",
                                _opts.cudss_options.matrix_type) != 0):
                warnings.warn(
                    f"symmetric_elimination=False makes the tangent non-symmetric, "
                    f"but {_role} uses cuDSS with matrix_type="
                    f"{_opts.cudss_options.matrix_type} (a symmetric factorization). "
                    "Pass cudss_options=CUDSSOptions(matrix_type=CUDSSMatrixType."
                    "GENERAL) — otherwise Newton directions are wrong and the line "
                    "search will fail.", stacklevel=2)

    # 2) Linear path (single linear solve)
    if linear:
        return _wrap(create_linear_solver(
            problem=problem,
            bc=bc,
            solver_options=linear_options,
            adjoint_solver_options=adjoint_linear_options,
            traced_params=traced_params,
            traced_structure=traced_structure,
            preconditioner=preconditioner,
            adjoint_preconditioner=adjoint_preconditioner,
        ))

    # 3) Nonlinear Newton path
    return _wrap(create_newton_solver(
        problem=problem,
        bc=bc,
        linear_options=linear_options,
        adjoint_linear_options=adjoint_linear_options,
        newton_options=newton_options,
        traced_params=traced_params,
        extra_residual_fn=extra_residual_fn,
        symmetric_elimination=symmetric_elimination,
        traced_structure=traced_structure,
        preconditioner=preconditioner,
        adjoint_preconditioner=adjoint_preconditioner,
    ))