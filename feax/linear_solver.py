"""
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
"""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import logging
from typing import Optional, Callable, Any

from .solver_option import SolverOptions, DirectSolverOptions, IterativeSolverOptions
from .problem import MatrixView, Problem
from .DCboundary import DirichletBC
from .assembler import create_J_bc_function, create_res_bc_function

logger = logging.getLogger(__name__)


def _safe_negate(x):
    """Negate array, handling JAX's float0 type for zero gradients."""
    if hasattr(x, 'dtype'):
        dtype_str = str(x.dtype)
        if 'float0' in dtype_str or 'V' in dtype_str:
            return x
    return -x


def create_jacobi_preconditioner(A: BCOO, shift: float = 1e-12):
    """Create Jacobi (diagonal) preconditioner from sparse matrix.

    Parameters
    ----------
    A : BCOO sparse matrix
        The system matrix to precondition
    shift : float, default 1e-12
        Small value added to diagonal for numerical stability

    Returns
    -------
    M_matvec : callable
        Jacobi preconditioner as a matvec function (diag(A)^{-1} @ x)

    Notes
    -----
    This creates a diagonal preconditioner M = diag(A)^{-1} with regularization.
    The preconditioner is JAX-compatible and avoids dynamic indexing.
    For elasticity problems with extreme material contrasts, this helps
    condition number significantly.
    """

    def extract_diagonal(A):
        """Extract diagonal from BCOO sparse matrix avoiding dynamic indexing."""
        n = A.shape[0]
        diagonal_mask = A.indices[:, 0] == A.indices[:, 1]
        diagonal_indices = jnp.where(diagonal_mask, A.indices[:, 0], n)
        diagonal_values = jnp.where(diagonal_mask, A.data, 0.0)
        diag = jnp.zeros(n)
        diag = diag.at[diagonal_indices].add(diagonal_values)
        return diag

    diagonal = extract_diagonal(A)
    diagonal_inv = 1.0 / (diagonal + shift)

    def M_matvec(x):
        return diagonal_inv * x

    return M_matvec


# ============================================================================
# Direct Solvers
# ============================================================================

def create_direct_solve_fn(options: DirectSolverOptions):
    """Create a direct linear solve function.

    Parameters
    ----------
    options : DirectSolverOptions
        Direct solver configuration. The ``solver`` field must be resolved
        (not "auto") before calling this function.

    Returns
    -------
    callable
        Function with signature (A, b, x0) -> x.
        Direct solvers ignore x0 but accept it for interface uniformity.

    Raises
    ------
    ValueError
        If solver is "auto" (unresolved) or unknown.
    RuntimeError
        If backend requirements are not met (e.g. spsolve on GPU).
    """
    if options.solver == "auto":
        raise ValueError(
            "DirectSolverOptions.solver is 'auto' (unresolved). "
            "Call resolve_direct_solver() before create_direct_solve_fn()."
        )

    _lineax_solvers = {"cholesky", "lu", "qr"}
    if options.solver in _lineax_solvers:
        try:
            import lineax as _lx
        except ImportError:
            raise RuntimeError(
                f"lineax is required to use the '{options.solver}' solver. "
                f"Install it with: pip install lineax"
            )

        _solver_map = {
            "cholesky": (_lx.Cholesky(), (_lx.symmetric_tag, _lx.positive_semidefinite_tag)),
            "lu": (_lx.LU(), ()),
            "qr": (_lx.QR(), ()),
        }
        _lx_solver, _lx_tags = _solver_map[options.solver]

        def solve(A, b, x0):
            input_structure = jax.ShapeDtypeStruct((A.shape[1],), b.dtype)

            def matvec(v):
                return A @ v

            operator = _lx.FunctionLinearOperator(
                matvec,
                input_structure=input_structure,
                tags=_lx_tags,
            )

            sol = _lx.linear_solve(operator, b, solver=_lx_solver)
            return sol.value

        return solve

    if options.solver == "spsolve":
        if jax.default_backend() != "cpu":
            raise RuntimeError(
                "jax.experimental.sparse.linalg.spsolve is only enabled on the CPU. "
                "Run on CPU or use an iterative solver on GPU."
            )

        def solve(A, b, x0=None):
            A_bcsr = jax.experimental.sparse.BCSR.from_bcoo(A.sum_duplicates(nse=A.nse))
            x = jax.experimental.sparse.linalg.spsolve(
                A_bcsr.data, A_bcsr.indices, A_bcsr.indptr, b,
                tol=0.0,
                reorder=1,
            )
            return x
        return solve

    if options.solver == "cudss":
        if jax.default_backend() != "gpu":
            raise RuntimeError(
                "spineax.cudss.solver.CuDSSSolver is only enabled on the GPU"
            )
        try:
            from spineax.cudss.solver import CuDSSSolver
        except Exception as e:
            raise RuntimeError(
                "Failed to import `spineax.cudss.solver.CuDSSSolver`. "
                "The optional dependency `spineax` is required to use the `cudss` solver."
            ) from e
        cudss_solver = None

        def solve(A, b, x0=None):
            nonlocal cudss_solver
            A_bcsr = jax.experimental.sparse.BCSR.from_bcoo(A.sum_duplicates(nse=A.nse))

            csr_values = A_bcsr.data
            if cudss_solver is None:
                csr_offsets = A_bcsr.indptr.astype(jnp.int32)
                csr_columns = A_bcsr.indices.astype(jnp.int32)
                # Suppress "JAX array set as static" warning — the sparsity
                # pattern is fixed for a given FEM problem, so static is safe.
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cudss_solver = CuDSSSolver(
                        csr_offsets=csr_offsets,
                        csr_columns=csr_columns,
                        device_id=options.cudss_options.device_id,
                        mtype_id=options.cudss_options.mtype_id,
                        mview_id=options.cudss_options.mview_id,
                    )
            x, _ = cudss_solver(b, csr_values)
            return x
        return solve

    raise ValueError(
        f"Unknown direct solver: {options.solver}. "
        f"Choose from ('cudss', 'spsolve', 'cholesky', 'lu', 'qr')"
    )


# ============================================================================
# Iterative Solvers
# ============================================================================

def create_iterative_solve_fn(options: IterativeSolverOptions):
    """Create an iterative linear solve function.

    Parameters
    ----------
    options : IterativeSolverOptions
        Iterative solver configuration. The ``solver`` field must be resolved
        (not "auto") before calling this function.

    Returns
    -------
    callable
        Function with signature (A, b, x0) -> x.

    Raises
    ------
    ValueError
        If solver is "auto" (unresolved) or unknown.
    """
    if options.solver == "auto":
        raise ValueError(
            "IterativeSolverOptions.solver is 'auto' (unresolved). "
            "Call resolve_iterative_solver() before create_iterative_solve_fn()."
        )

    def choose_preconditioner(A):
        if options.use_jacobi_preconditioner and options.preconditioner is None:
            return create_jacobi_preconditioner(A, options.jacobi_shift)
        return options.preconditioner

    if options.solver == "cg":
        def solve(A, b, x0):
            M = choose_preconditioner(A)
            x, _ = jax.scipy.sparse.linalg.cg(
                A, b, x0=x0, M=M,
                tol=options.tol,
                atol=options.atol,
                maxiter=options.maxiter
            )
            return x
        return solve

    if options.solver == "bicgstab":
        def solve(A, b, x0):
            M = choose_preconditioner(A)
            x, _ = jax.scipy.sparse.linalg.bicgstab(
                A, b, x0=x0, M=M,
                tol=options.tol,
                atol=options.atol,
                maxiter=options.maxiter
            )
            return x
        return solve

    if options.solver == "gmres":
        def solve(A, b, x0):
            M = choose_preconditioner(A)
            x, _ = jax.scipy.sparse.linalg.gmres(
                A, b, x0=x0, M=M,
                tol=options.tol,
                atol=options.atol,
                maxiter=options.maxiter
            )
            return x
        return solve

    raise ValueError(
        f"Unknown iterative solver: {options.solver}. "
        f"Choose from ('cg', 'bicgstab', 'gmres')"
    )


# ============================================================================
# Unified Interface (backward compatible)
# ============================================================================

def create_linear_solve_fn(solver_options):
    """Create a linear solve function based on solver options.

    Accepts SolverOptions (legacy), DirectSolverOptions, or IterativeSolverOptions.
    For legacy SolverOptions, internally converts and delegates to the appropriate
    specialized function.

    Parameters
    ----------
    solver_options : SolverOptions or DirectSolverOptions or IterativeSolverOptions
        Solver configuration.

    Returns
    -------
    callable
        Function with signature (A, b, x0) -> x.
    """
    # New option types: delegate directly
    if isinstance(solver_options, DirectSolverOptions):
        return create_direct_solve_fn(solver_options)
    if isinstance(solver_options, IterativeSolverOptions):
        return create_iterative_solve_fn(solver_options)

    # Legacy SolverOptions: convert and delegate
    direct_solvers = {"cudss", "spsolve", "cholesky", "lu", "qr"}
    iterative_solvers = {"cg", "bicgstab", "gmres"}
    solver_name = solver_options.linear_solver

    if solver_name in direct_solvers:
        direct_opts = DirectSolverOptions(
            solver=solver_name,
            cudss_options=solver_options.cudss_options,
            check_convergence=solver_options.check_convergence,
            convergence_threshold=solver_options.convergence_threshold,
            verbose=solver_options.verbose,
        )
        return create_direct_solve_fn(direct_opts)

    if solver_name in iterative_solvers:
        iterative_opts = IterativeSolverOptions(
            solver=solver_name,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter,
            preconditioner=solver_options.preconditioner,
            use_jacobi_preconditioner=solver_options.use_jacobi_preconditioner,
            jacobi_shift=solver_options.jacobi_shift,
            x0_fn=solver_options.linear_solver_x0_fn,
            check_convergence=solver_options.check_convergence,
            convergence_threshold=solver_options.convergence_threshold,
            verbose=solver_options.verbose,
        )
        return create_iterative_solve_fn(iterative_opts)

    valid_solvers = ("cg", "bicgstab", "gmres", "spsolve", "cudss", "cholesky", "lu", "qr")
    raise ValueError(
        f"Unknown linear solver: {solver_name}. "
        f"Choose from {valid_solvers}"
    )


def _extract_sparse_diagonal(A: BCOO):
    """Extract sparse matrix diagonal as dense vector."""
    n = A.shape[0]
    diagonal_mask = A.indices[:, 0] == A.indices[:, 1]
    diagonal_indices = jnp.where(diagonal_mask, A.indices[:, 0], n)
    diagonal_values = jnp.where(diagonal_mask, A.data, 0.0)
    diag = jnp.zeros(n, dtype=A.data.dtype)
    return diag.at[diagonal_indices].add(diagonal_values)


def _matvec_with_full_view(A, x, matrix_view: MatrixView):
    """Apply full matrix-vector product for FULL/UPPER/LOWER storage."""
    if matrix_view in (MatrixView.UPPER, MatrixView.LOWER):
        diag = _extract_sparse_diagonal(A)
        return A @ x + A.transpose() @ x - diag * x
    return A @ x


def check_linear_convergence(A, x, b, solver_options: SolverOptions, matrix_view: MatrixView, solver_label: str):
    """Check relative residual and return NaN solution when convergence fails.

    Parameters
    ----------
    A : BCOO sparse matrix
        The system matrix
    x : array
        The computed solution
    b : array
        The right-hand side vector
    solver_options : SolverOptions
        Solver configuration (uses verbose and convergence_threshold)
    matrix_view : MatrixView
        Matrix storage format
    solver_label : str
        Label for verbose output (e.g. "Linear solver", "Adjoint solver")

    Returns
    -------
    array
        x if converged, NaN array otherwise
    """
    residual = _matvec_with_full_view(A, x, matrix_view) - b
    residual_norm = jnp.linalg.norm(residual)
    b_norm = jnp.linalg.norm(b)
    rel_residual = residual_norm / (b_norm + 1e-12)

    if solver_options.verbose:
        jax.debug.print(
            "{label} residual: abs={r:.6e}, rel={rr:.6e}",
            label=solver_label,
            r=residual_norm,
            rr=rel_residual,
        )

    return jnp.where(
        rel_residual < solver_options.convergence_threshold,
        x,
        jnp.full_like(x, jnp.nan),
    )


def linear_solve_adjoint(A, b, solver_options, matrix_view: MatrixView, bc=None,
                         linear_solve_fn=None):
    """Solve linear system for adjoint problem.

    Parameters
    ----------
    A : BCOO sparse matrix
        The transposed Jacobian matrix (J^T)
    b : jax.numpy.ndarray
        Right-hand side vector (cotangent vector from VJP)
    solver_options : SolverOptions or DirectSolverOptions or IterativeSolverOptions
        Solver configuration for adjoint solve
    matrix_view : MatrixView
        Matrix storage format from the problem
    bc : DirichletBC, optional
        Boundary conditions for computing initial guess
    linear_solve_fn : callable, optional
        Pre-created linear solve function. If None, one is created from
        solver_options internally.

    Returns
    -------
    sol : jax.numpy.ndarray
        Solution to the adjoint system: A @ sol = b

    Notes
    -----
    For the adjoint problem, boundary conditions are already incorporated into
    the transposed Jacobian matrix. The initial guess uses BC-aware computation
    when bc is provided, which can improve convergence for problems with
    Dirichlet boundary conditions.
    """
    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(solver_options)

    if bc is not None and hasattr(bc, 'bc_rows') and hasattr(bc, 'bc_vals'):
        x0 = jnp.zeros_like(b)
        bc_rows_array = jnp.array(bc.bc_rows) if not isinstance(bc.bc_rows, jnp.ndarray) else bc.bc_rows
        x0 = x0.at[bc_rows_array].set(b[bc_rows_array])
    else:
        x0 = jnp.zeros_like(b)

    sol = linear_solve_fn(A, b, x0)

    check_convergence = getattr(solver_options, 'check_convergence', False)
    if check_convergence:
        sol = check_linear_convergence(
            A=A,
            x=sol,
            b=b,
            solver_options=solver_options,
            matrix_view=matrix_view,
            solver_label="Adjoint solver",
        )

    return sol


def create_linear_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[SolverOptions] = None,
    adjoint_solver_options: Optional[SolverOptions] = None,
) -> Callable[[Any, jnp.ndarray], jnp.ndarray]:
    """Create a differentiable solver for linear FE problems.

    Simpler and more focused alternative to ``create_solver(iter_num=1)``
    when the problem is known to be linear (e.g. linear elasticity).
    The returned function supports ``jax.grad`` via a custom VJP based on
    the adjoint method.

    Parameters
    ----------
    problem : Problem
        The feax Problem instance.
    bc : DirichletBC
        Boundary conditions.
    solver_options : SolverOptions, optional
        Options for the forward linear solve (defaults to SolverOptions()).
    adjoint_solver_options : SolverOptions, optional
        Options for the adjoint solve used in the backward pass.
        Defaults to the same options as the forward solve.

    Returns
    -------
    differentiable_solve : callable
        A function with signature ``(internal_vars, initial_guess) -> solution``
        that is differentiable w.r.t. ``internal_vars`` via ``jax.grad``.

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
    >>> solver = create_linear_solver(problem, bc)
    >>> initial = fe.zero_like_initial_guess(problem, bc)
    >>> sol = solver(internal_vars, initial)
    >>>
    >>> # Gradient w.r.t. internal_vars
    >>> def loss(internal_vars):
    ...     sol = solver(internal_vars, initial)
    ...     return jnp.sum(sol ** 2)
    >>> grad = jax.grad(loss)(internal_vars)
    """
    if solver_options is None:
        solver_options = SolverOptions()
    if adjoint_solver_options is None:
        adjoint_solver_options = solver_options

    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)
    linear_solve_fn = create_linear_solve_fn(solver_options)
    if adjoint_solver_options is solver_options:
        adjoint_linear_solve_fn = linear_solve_fn
    else:
        adjoint_linear_solve_fn = create_linear_solve_fn(adjoint_solver_options)

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        J = J_bc_func(initial_guess, internal_vars)
        res = res_bc_func(initial_guess, internal_vars)
        x0 = jnp.zeros_like(initial_guess)
        if solver_options.linear_solver_x0_fn is not None:
            x0 = solver_options.linear_solver_x0_fn(initial_guess)
        delta_sol = linear_solve_fn(J, -res, x0)
        if solver_options.check_convergence:
            delta_sol = check_linear_convergence(
                A=J,
                x=delta_sol,
                b=-res,
                solver_options=solver_options,
                matrix_view=problem.matrix_view,
                solver_label="Linear solver",
            )
        return initial_guess + delta_sol

    def f_fwd(internal_vars, initial_guess):
        sol = differentiable_solve(internal_vars, initial_guess)
        return sol, (internal_vars, sol)

    def f_bwd(res, v):
        internal_vars, sol = res

        # Adjoint solve: J^T @ adjoint = v
        J = J_bc_func(sol, internal_vars)
        use_transpose = problem.matrix_view not in (MatrixView.UPPER, MatrixView.LOWER)
        J_adjoint = J.transpose() if use_transpose else J
        adjoint_vec = linear_solve_adjoint(
            J_adjoint, v, adjoint_solver_options, problem.matrix_view, bc,
            linear_solve_fn=adjoint_linear_solve_fn
        )

        # VJP of residual w.r.t. internal_vars
        def res_fn(iv):
            return problem.unflatten_fn_sol_list(res_bc_func(sol, iv))

        adjoint_list = problem.unflatten_fn_sol_list(adjoint_vec)
        _, f_vjp = jax.vjp(res_fn, internal_vars)
        vjp_result, = f_vjp(adjoint_list)
        vjp_result = jax.tree_util.tree_map(_safe_negate, vjp_result)

        return (vjp_result, None)  # No gradient w.r.t. initial_guess

    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve
