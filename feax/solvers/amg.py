"""Algebraic multigrid preconditioner backend (PyAMG hierarchy -> AMJax/JAX).

This builds a smoothed-aggregation AMG hierarchy *once* on the host (PyAMG, from
a sample assembled CSR Jacobian), converts it to a JAX-native
``amjax.MultilevelSolver``, and returns one multigrid V-cycle as a preconditioner
callable ``M(x) -> y`` for a matrix-free outer Krylov solve.

Why this shape:
- AMG for vector elasticity/structural problems needs the **near-null-space**
  (rigid body modes) to build a good coarse space. PyAMG's
  ``smoothed_aggregation_solver(A, B=...)`` takes it as a first-class argument;
  we generate the rigid body modes from the mesh node coordinates.
- AMJax's JAX-side V-cycle implements only a Jacobi smoother, and *undamped*
  Jacobi diverges on elasticity, so we use a damped Jacobi smoother by default.
- The hierarchy is fixed (built from a sample matrix); the outer Krylov applies
  the current operator matrix-free and uses this AMG cycle as the preconditioner.
  An outer Krylov (gmres/cg/bicgstab) makes the fixed-hierarchy preconditioner
  robust to the operator changing between solves (Newton / parameter sweeps).

Requires the optional ``feax[amg]`` dependency (``amjax`` + ``pyamg``).
"""

import jax.numpy as jnp
import numpy as onp

from ..csr import CSRMatrix


def rigid_body_modes(problem, bc=None):
    """Rigid-body near-null-space modes from mesh node coordinates.

    Returns an ``(n_dof, k)`` array (k=6 in 3D: 3 translations + 3 rotations;
    k=3 in 2D: 2 translations + 1 rotation) for a single vector field with
    ``vec == dim`` (node-major DOF layout ``dof = node*vec + component``). Returns
    ``None`` when the problem is not a single ``vec == dim`` field (e.g. scalar
    Poisson, or a multi-field mixed problem), in which case AMG falls back to the
    default constant near-null-space.

    Dirichlet DOFs (``bc.bc_rows``) are zeroed: constrained DOFs are not part of
    the reduced operator's near-null-space.
    """
    vec = problem.vec
    if not (isinstance(vec, (list, tuple)) and len(vec) == 1):
        return None
    nv = int(vec[0])
    dim = int(problem.dim)
    if nv != dim:
        return None

    pts = onp.asarray(problem.mesh[0].points)          # (n_nodes, dim)
    n_nodes = pts.shape[0]
    ndof = n_nodes * nv
    coords = pts - pts.mean(axis=0)                     # center for well-scaled rotations

    if dim == 3:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        B = onp.zeros((ndof, 6))
        B[0::3, 0] = 1.0; B[1::3, 1] = 1.0; B[2::3, 2] = 1.0    # translations
        B[1::3, 3] = -z;  B[2::3, 3] = y                        # rot about x
        B[0::3, 4] = z;   B[2::3, 4] = -x                       # rot about y
        B[0::3, 5] = -y;  B[1::3, 5] = x                        # rot about z
    elif dim == 2:
        x, y = coords[:, 0], coords[:, 1]
        B = onp.zeros((ndof, 3))
        B[0::2, 0] = 1.0; B[1::2, 1] = 1.0                      # translations
        B[0::2, 2] = -y;  B[1::2, 2] = x                        # rotation
    else:
        return None

    if bc is not None and getattr(bc, "bc_rows", None) is not None:
        B[onp.asarray(bc.bc_rows), :] = 0.0
    return B


def _to_scipy_csr(A):
    """Convert a feax CSRMatrix (or its transpose) to a host scipy CSR matrix."""
    import scipy.sparse as sp
    from ..csr import CSRTranspose
    if isinstance(A, CSRTranspose):
        return _to_scipy_csr(A.parent).T.tocsr()
    if isinstance(A, CSRMatrix):
        return sp.csr_matrix(
            (onp.asarray(A.data), onp.asarray(A.indices), onp.asarray(A.indptr)),
            shape=A.shape,
        )
    raise TypeError(
        f"AMG preconditioner needs a feax CSRMatrix sample, got {type(A).__name__}."
    )


def _default_num_candidates(problem):
    """Rigid-body mode count for a single ``vec == dim`` field, else 1."""
    vec = problem.vec
    if isinstance(vec, (list, tuple)) and len(vec) == 1 and int(vec[0]) == int(problem.dim):
        d = int(problem.dim)
        return d * (d + 1) // 2          # 6 in 3D, 3 in 2D
    return 1


def _resolve_near_nullspace(near_nullspace, problem, bc):
    """Resolve the polymorphic ``near_nullspace`` input.

    Returns ``(B, use_adaptive)`` where ``B`` is an ``(n_dof, k)`` array or
    ``None`` (constant), and ``use_adaptive`` requests PyAMG's numerical
    near-null-space search.
    """
    ns = near_nullspace
    if ns is None:
        # Smart default: rigid body modes for vector elasticity, else constant.
        return rigid_body_modes(problem, bc), False
    if isinstance(ns, str):
        key = ns.lower()
        if key == "rigid_body":
            B = rigid_body_modes(problem, bc)
            if B is None:
                raise ValueError(
                    f"near_nullspace={ns!r} needs a single vec==dim field "
                    f"(got vec={problem.vec}, dim={problem.dim}). Pass an explicit "
                    "(n_dof, k) array, or use 'adaptive_sa' / 'constant'."
                )
            return B, False
        if key == "constant":
            return None, False
        if key == "adaptive_sa":
            return None, True
        raise ValueError(f"Unknown near_nullspace preset: {ns!r}.")
    # array-like, user-defined
    return onp.asarray(ns), False


def build_amg_preconditioner(problem, bc, sample_csr, options):
    """Build an AMG V-cycle preconditioner ``M(x) -> y`` from a sample Jacobian.

    The near-null-space ``B`` is resolved from ``options.near_nullspace`` and may
    be (a) a user-defined array, (b) a known-physics preset (rigid body modes for
    elasticity / constant for scalar), or (c) ``"auto"`` — estimated numerically
    by adaptive smoothed aggregation (relaxing ``A x = 0`` from random starts).

    Parameters
    ----------
    problem, bc :
        Provide the mesh coordinates / DOF layout for the rigid-body presets.
    sample_csr : feax.csr.CSRMatrix
        A representative assembled (BC-applied) Jacobian; the SA-AMG hierarchy is
        built from its values + pattern on the host.
    options : AMGSolverOptions
        Hierarchy / smoother / near-null-space configuration.

    Returns
    -------
    callable
        ``M(b) -> x`` applying one multigrid cycle (an approximate inverse),
        suitable as the ``M=`` preconditioner for a JAX Krylov solver.
    """
    try:
        import pyamg
        from amjax import MultilevelSolver
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "AMGSolverOptions requires the optional 'feax[amg]' extra "
            "(amjax + pyamg). Install with:  pip install 'feax[amg]'"
        ) from e

    A_sp = _to_scipy_csr(sample_csr)

    B, use_adaptive = _resolve_near_nullspace(options.near_nullspace, problem, bc)
    if B is not None:
        B = onp.asarray(B)

    sa_kwargs = {}
    if options.strength is not None:
        sa_kwargs["strength"] = ("symmetric", {"theta": options.strength})

    if use_adaptive:
        # Numerically estimate the near-null-space (relax A x = 0 from random
        # starts) and build the hierarchy from the discovered candidates.
        k = options.num_nullspace or _default_num_candidates(problem)
        result = pyamg.aggregation.adaptive_sa_solver(
            A_sp, num_candidates=int(k), **sa_kwargs)
        ml = result[0] if isinstance(result, (tuple, list)) else result
    else:
        ml = pyamg.smoothed_aggregation_solver(A_sp, B=B, **sa_kwargs)

    smoother = ("jacobi", {"iterations": int(options.smoother_sweeps),
                           "omega": float(options.smoother_omega)})
    mlj = MultilevelSolver.from_pyamg(
        ml,
        presmoother=smoother,
        postsmoother=smoother,
        coarse_solver=options.coarse_solver,
        dtype=jnp.float64,
    )
    return mlj.aspreconditioner(cycle=options.cycle)


def make_self_preconditioned_amg_solve(problem, bc, options):
    """Return a ``solve(A, b, x0) -> x`` that rebuilds the AMG V-cycle from ``A``.

    Each call builds a fresh AMG preconditioner from the *given* operator ``A``
    (a feax ``CSRMatrix`` or its transpose) and runs the outer Krylov method on
    it. This is what makes the Newton ``rebuild_every`` path and the adjoint solve
    use a preconditioner matched to the current/converged tangent rather than a
    stale one built from the initial state. Runs eagerly (a host PyAMG build per
    call), so it is not meant for a jitted while-loop.
    """
    import jax.numpy as _jnp
    import jax.scipy.sparse.linalg as _jsla

    solver = options.solver if options.solver != "auto" else "gmres"
    krylov = getattr(_jsla, solver)

    def solve(A, b, x0=None):
        M = build_amg_preconditioner(problem, bc, A, options)
        kw = dict(M=M, tol=options.tol, atol=options.atol, maxiter=options.maxiter)
        if solver == "gmres":
            kw["restart"] = options.restart or 50
        x0 = _jnp.zeros_like(b) if x0 is None else x0
        x, _ = krylov(lambda v: A @ v, b, x0=x0, **kw)
        return x

    return solve


def amg_to_krylov_options(
    amg_options,
    problem,
    bc,
    traced_params,
    traced_structure=None,
    symmetric_elimination: bool = True,
):
    """Lower an :class:`AMGSolverOptions` to a matrix-free Krylov solve + AMG ``M``.

    Assembles one sample Jacobian (so the AMG hierarchy + rigid-body near-null-space
    can be built on the host), then returns a :class:`KrylovSolverOptions` whose
    ``preconditioner`` is the AMG V-cycle. The outer solver is the requested
    Krylov method (``"auto"`` -> cg for an SPD sample, else gmres). The rest of the
    solver stack then treats it as an ordinary matrix-free Krylov solve.
    """
    from ..assembler import create_J_bc_csr_parametric
    from ..utils import zero_like_initial_guess
    from .options import (
        KrylovSolverOptions,
        MatrixProperty,
        detect_matrix_property,
    )

    if traced_params is None:
        raise ValueError(
            "AMGSolverOptions requires traced_params: a sample Jacobian is "
            "assembled once to build the AMG hierarchy."
        )

    initial = zero_like_initial_guess(problem, bc)
    J_csr = create_J_bc_csr_parametric(problem, symmetric=symmetric_elimination)
    sample = J_csr(initial, traced_params, bc, traced_structure)

    M = build_amg_preconditioner(problem, bc, sample, amg_options)

    solver = amg_options.solver
    if solver == "auto":
        mp = detect_matrix_property(sample, matrix_view=problem.matrix_view)
        solver = "cg" if mp == MatrixProperty.SPD else "gmres"

    return KrylovSolverOptions(
        solver=solver,
        tol=amg_options.tol,
        atol=amg_options.atol,
        maxiter=amg_options.maxiter,
        preconditioner=M,
        restart=amg_options.restart,
        check_convergence=amg_options.check_convergence,
        convergence_threshold=amg_options.convergence_threshold,
        verbose=amg_options.verbose,
    )
