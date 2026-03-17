"""
Solver configuration options for FEAX finite element framework.

This module provides configuration dataclasses and enums for controlling
solver behavior, including linear solver selection, tolerances, and
CUDA-specific options.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union
from ..problem import MatrixView

import jax

logger = logging.getLogger(__name__)


# ============================================================================
# Backend Detection
# ============================================================================

class Backend(Enum):
    """JAX compute backend."""
    CPU = "cpu"
    CUDA = "gpu"  # JAX reports GPU as "gpu"; we expose it as CUDA


def detect_backend() -> Backend:
    """Detect the active JAX backend.

    Returns
    -------
    Backend
        Backend.CUDA if a CUDA device is active, Backend.CPU otherwise.
    """
    return Backend(jax.default_backend())


def is_cuda() -> bool:
    """Check if the active backend is CUDA."""
    return detect_backend() == Backend.CUDA


def is_cpu() -> bool:
    """Check if the active backend is CPU."""
    return detect_backend() == Backend.CPU


def has_cudss() -> bool:
    """Check if the cuDSS direct solver is available.

    Requires CUDA backend and the ``spineax`` package.
    """
    if not is_cuda():
        return False
    try:
        from spineax.cudss.solver import CuDSSSolver  # noqa: F401
        return True
    except ImportError:
        return False


def has_spsolve() -> bool:
    """Check if JAX's experimental spsolve is available.

    Only supported on the CPU backend.
    """
    return is_cpu()


def detect_available_solver_backends(solver: str = "auto") -> tuple[str, ...]:
    """Detect available direct solvers for the current runtime.

    Parameters
    ----------
    solver : str, default "auto"
        - "auto": return all available direct solvers.
        - specific solver: return that solver if available, otherwise raise.
    """
    backend = jax.default_backend()

    def _has_cudss() -> bool:
        if backend != "gpu":
            return False
        try:
            from spineax.cudss.solver import CuDSSSolver  # noqa: F401
            return True
        except Exception:
            return False

    def _has_cholmod() -> bool:
        if backend != "cpu":
            return False
        try:
            from sksparse import cholmod as _cholmod  # noqa: F401
            return True
        except Exception:
            return False

    def _has_umfpack() -> bool:
        if backend != "cpu":
            return False
        try:
            from sksparse import umfpack as _umfpack  # noqa: F401
            return True
        except Exception:
            return False

    if solver != "auto":
        if solver == "spsolve":
            if backend != "cpu":
                raise RuntimeError("spsolve is only enabled on the CPU")
            return ("spsolve",)
        if solver == "cudss":
            if backend != "gpu":
                raise RuntimeError(
                    "spineax.cudss.solver.CuDSSSolver is only enabled on the GPU"
                )
            if not _has_cudss():
                raise RuntimeError("spineax is not installed")
            return ("cudss",)
        if solver == "cholmod":
            if backend != "cpu":
                raise RuntimeError("cholmod is only enabled on the CPU")
            if not _has_cholmod():
                raise RuntimeError("sksparse is not installed")
            return ("cholmod",)
        if solver == "umfpack":
            if backend != "cpu":
                raise RuntimeError("umfpack is only enabled on the CPU")
            if not _has_umfpack():
                raise RuntimeError("sksparse is not installed")
            return ("umfpack",)
        raise ValueError(
            f"Unknown direct solver '{solver}'. "
            "Choose from ('auto', 'cudss', 'spsolve', 'umfpack', 'cholmod')"
        )

    available_solvers = {"spsolve"} if backend == "cpu" else set()
    if _has_cudss():
        available_solvers.add("cudss")
    if _has_cholmod():
        available_solvers.add("cholmod")
    if _has_umfpack():
        available_solvers.add("umfpack")

    return tuple(sorted(available_solvers))


# ============================================================================
# Matrix Property Detection
# ============================================================================

class MatrixProperty(Enum):
    """Algebraic properties of the system matrix.

    Used by auto solver selection to choose the best algorithm.

    - GENERAL: No special structure (LU / GMRES)
    - SYMMETRIC: Symmetric A = A^T (LDLT / BICGSTAB)
    - SPD: Symmetric Positive Definite (Cholesky / CG)
    """
    GENERAL = "general"
    SYMMETRIC = "symmetric"
    SPD = "spd"

    def view(self):
        """Return the recommended MatrixView for this property.

        Returns
        -------
        MatrixView
            GENERAL → FULL, SYMMETRIC/SPD → UPPER.
        """
        from ..problem import MatrixView

        if self in (MatrixProperty.SYMMETRIC, MatrixProperty.SPD):
            return MatrixView.UPPER
        return MatrixView.FULL


def detect_matrix_property(A, sym_tol: float = 1e-8, matrix_view=None) -> MatrixProperty:
    """Detect matrix property from an assembled sparse matrix.

    Performs numerical checks on the matrix:
    1. Symmetry: compares ``A @ x`` vs ``A^T @ x`` for a random vector.
       Skipped when ``matrix_view`` is UPPER or LOWER (the matrix is
       symmetric by construction).
    2. Positive definiteness (heuristic): checks that all diagonal entries
       are positive, which is a necessary condition for SPD.

    Parameters
    ----------
    A : BCOO sparse matrix
        The assembled system matrix.
    sym_tol : float, default 1e-8
        Relative tolerance for the symmetry check.
    matrix_view : MatrixView, optional
        Storage format of the matrix.  When UPPER or LOWER, the matrix
        stores only one triangular half and is symmetric by definition,
        so the symmetry check is skipped.

    Returns
    -------
    MatrixProperty
        SPD if symmetric with all-positive diagonal,
        SYMMETRIC if symmetric but diagonal has non-positive entries,
        GENERAL otherwise.

    Notes
    -----
    The diagonal positivity test is a necessary but not sufficient
    condition for positive definiteness.  For FEM stiffness matrices
    from well-posed problems this is a reliable indicator.
    """
    import jax.numpy as np

    from ..problem import MatrixView

    # UPPER/LOWER storage implies symmetry by construction
    if matrix_view in (MatrixView.UPPER, MatrixView.LOWER):
        is_symmetric = True
    else:
        # --- symmetry check via random matvec ---
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (A.shape[1],), dtype=A.dtype)
        Ax = A @ x
        ATx = A.T @ x
        sym_err = np.linalg.norm(Ax - ATx) / (np.linalg.norm(Ax) + 1e-30)
        is_symmetric = bool(sym_err < sym_tol)

    if not is_symmetric:
        return MatrixProperty.GENERAL

    # --- SPD heuristic: diagonal positivity ---
    n = A.shape[0]
    diag_mask = A.indices[:, 0] == A.indices[:, 1]
    diag_idx = np.where(diag_mask, A.indices[:, 0], n)
    diag_val = np.where(diag_mask, A.data, 0.0)
    diag = np.zeros(n, dtype=A.dtype).at[diag_idx].add(diag_val)
    is_diag_positive = bool(np.all(diag > 0))

    if is_diag_positive:
        return MatrixProperty.SPD

    return MatrixProperty.SYMMETRIC


class CUDSSMatrixType(Enum):
    """Matrix type for cuDSS solver.

    Determines which factorization method cuDSS will use internally.
    """
    GENERAL = 0      # Non-symmetric matrix (uses LU factorization)
    SYMMETRIC = 1    # Symmetric matrix (uses LDLT factorization)
    HERMITIAN = 2    # Hermitian matrix (uses LDLT for complex)
    SPD = 3          # Symmetric Positive Definite (uses Cholesky factorization)
    HPD = 4          # Hermitian Positive Definite (uses Cholesky for complex)


class CUDSSMatrixView(Enum):
    """Matrix view (which portion of matrix is provided) for cuDSS solver."""
    FULL = 0   # Full matrix provided
    UPPER = 1  # Upper triangular portion only
    LOWER = 2  # Lower triangular portion only


@dataclass(frozen=True)
class CUDSSOptions:
    """Options specific to the NVIDIA cuDSS direct solver.

    cuDSS matrix configuration is defined by two orthogonal properties:
    1) Matrix Type (algebraic structure determining factorization method)
    2) Matrix View (which triangular portion is provided)

    Parameters
    ----------
    matrix_type : CUDSSMatrixType or str or int, default SYMMETRIC
        Matrix type determining factorization method:
        - GENERAL: Non-symmetric (uses LU factorization)
        - SYMMETRIC: Symmetric indefinite (uses LDLT factorization)
        - HERMITIAN: Hermitian indefinite (uses LDLT for complex)
        - SPD: Symmetric Positive Definite (uses Cholesky factorization)
        - HPD: Hermitian Positive Definite (uses Cholesky for complex)
    matrix_view : CUDSSMatrixView or str or int, default FULL
        Which portion of the matrix is provided:
        - FULL: All entries provided
        - UPPER: Upper triangular portion only
        - LOWER: Lower triangular portion only
    device_id : int, default 0
        CUDA device ID to use

    Examples
    --------
    >>> # Using enums (recommended)
    >>> opts = CUDSSOptions(
    ...     matrix_type=CUDSSMatrixType.SPD,
    ...     matrix_view=CUDSSMatrixView.FULL
    ... )
    >>>
    >>> # Using strings (convenient)
    >>> opts = CUDSSOptions(matrix_type="SPD", matrix_view="FULL")
    >>>
    >>> # Using integers (backward compatible)
    >>> opts = CUDSSOptions(matrix_type=3, matrix_view=0)

    Notes
    -----
    Typical combinations:
    - General sparse: matrix_type=GENERAL, matrix_view=FULL
    - Symmetric: matrix_type=SYMMETRIC, matrix_view=FULL
    - SPD (Cholesky): matrix_type=SPD, matrix_view=FULL

    For GENERAL matrices, cuDSS treats the view as FULL (the view setting is
    effectively ignored).

    With symmetric elimination (default in FEAX), matrix_view should always be FULL
    since both upper and lower triangular portions are modified.
    """

    matrix_type: Union[CUDSSMatrixType, str, int] = CUDSSMatrixType.SYMMETRIC
    matrix_view: Union[CUDSSMatrixView, str, int] = CUDSSMatrixView.FULL
    device_id: int = 0

    def __post_init__(self):
        """Validate and convert matrix_type and matrix_view to enums if needed."""
        # Convert matrix_type to enum if string or int
        if isinstance(self.matrix_type, str):
            try:
                object.__setattr__(self, 'matrix_type', CUDSSMatrixType[self.matrix_type.upper()])
            except KeyError:
                raise ValueError(
                    f"Invalid matrix_type string: {self.matrix_type}. "
                    f"Valid options: {[t.name for t in CUDSSMatrixType]}"
                )
        elif isinstance(self.matrix_type, int):
            try:
                object.__setattr__(self, 'matrix_type', CUDSSMatrixType(self.matrix_type))
            except ValueError:
                raise ValueError(
                    f"Invalid matrix_type integer: {self.matrix_type}. "
                    f"Valid values: 0-4"
                )

        # Convert matrix_view to enum if string or int
        if isinstance(self.matrix_view, str):
            try:
                object.__setattr__(self, 'matrix_view', CUDSSMatrixView[self.matrix_view.upper()])
            except KeyError:
                raise ValueError(
                    f"Invalid matrix_view string: {self.matrix_view}. "
                    f"Valid options: {[v.name for v in CUDSSMatrixView]}"
                )
        elif isinstance(self.matrix_view, int):
            try:
                object.__setattr__(self, 'matrix_view', CUDSSMatrixView(self.matrix_view))
            except ValueError:
                raise ValueError(
                    f"Invalid matrix_view integer: {self.matrix_view}. "
                    f"Valid values: 0-2"
                )

        # Supported FEAX/cuDSS matrix type subset.
        if self.matrix_type not in (CUDSSMatrixType.GENERAL, CUDSSMatrixType.SYMMETRIC, CUDSSMatrixType.SPD):
            raise ValueError(
                "cudss_options.matrix_type must be GENERAL, SYMMETRIC, or SPD. "
                f"Got: {self.matrix_type.name}."
            )

        # Triangular storage is typically paired with symmetric/SPD factorizations.
        if self.matrix_view in (CUDSSMatrixView.UPPER, CUDSSMatrixView.LOWER):
            if self.matrix_type not in (CUDSSMatrixType.SYMMETRIC, CUDSSMatrixType.SPD):
                logger.warning(
                    f"Using matrix_view={self.matrix_view.name} with matrix_type={self.matrix_type.name}. "
                    "For best performance, use matrix_type=SYMMETRIC or SPD with triangular storage."
                )

    @property
    def mtype_id(self) -> int:
        """Get matrix type as integer ID for cuDSS."""
        return self.matrix_type.value

    @property
    def mview_id(self) -> int:
        """Get matrix view as integer ID for cuDSS."""
        return self.matrix_view.value


@dataclass(frozen=True)
class SKSPARSEOptions:
    """Options for host-side sparse direct solvers.

    These options are shared by the CPU direct solver backends. Some fields
    apply to all host-callback solvers, while others are backend-specific.

    Parameters
    ----------
    vmap_method : str, default "broadcast_all"
        JAX ``pure_callback`` vmap behavior. Supported values:
        ``"expand_dims"``, ``"sequential"``, or ``"broadcast_all"``.
        Used by ``spsolve``, ``umfpack``, and ``cholmod``.
    order : str default "amd"
        CHOLMOD ordering strategy. Supported values:
        ``"default"``, ``"best"``, ``"metis"``, ``"nesdis"``, ``"amd"``,
        ``"colamd"``, ``"postordered"``, ``"natural"``, or ``None``.
    lower : bool, default False
        Whether CHOLMOD should use lower-triangular input.
    """

    vmap_method: str = "broadcast_all"
    order: str = "amd"
    lower: bool = False

    def __post_init__(self):
        valid_orders = {"default", "best", "metis", "nesdis", "amd", "colamd", "postordered", "natural"}
        if self.order not in valid_orders:
            raise ValueError(
                f"Invalid sksparse order: {self.order}. "
                f"Choose from {sorted(x for x in valid_orders if x is not None)} or None."
            )
        valid_vmap_methods = {"expand_dims", "sequential", "broadcast_all"}
        if self.vmap_method not in valid_vmap_methods:
            raise ValueError(
                f"Invalid sksparse vmap_method: {self.vmap_method}. "
                f'Choose from {sorted(valid_vmap_methods)}.'
            )
        if self.lower not in (True, False):
            raise ValueError("sksparse_options.lower must be True or False")


@dataclass(frozen=True)
class AbstractSolverOptions:
    """Base class for all solver option types.

    Common parameters shared by DirectSolverOptions,
    IterativeSolverOptions, and SolverOptions.

    Parameters
    ----------
    verbose : bool, default False
        Whether to print solver diagnostics.
    check_convergence : bool, default False
        Whether to verify solution by checking residual norm.
    convergence_threshold : float, default 0.1
        Maximum allowable relative residual for convergence check.
        Only used when check_convergence=True.
    """
    verbose: bool = False
    check_convergence: bool = False
    convergence_threshold: float = 0.1

    def uses_x0(self) -> bool:
        """Whether this solver family consumes an initial iterate ``x0``."""
        raise NotImplementedError


@dataclass(frozen=True)
class NewtonOptions:
    """Configuration for Newton nonlinear iteration and line search.

    Parameters
    ----------
    tol : float, default 1e-6
        Absolute tolerance for residual norm.
    rel_tol : float, default 1e-8
        Relative tolerance for residual norm.
    max_iter : int, default 100
        Maximum Newton iterations.
    line_search_max_backtracks : int, default 30
        Maximum Armijo backtracking steps.
    line_search_c1 : float, default 1e-4
        Armijo sufficient decrease constant.
    line_search_rho : float, default 0.5
        Backtracking shrink factor (alpha *= rho).
    internal_jit : bool, default False
        JIT-compile the internal linear solve used inside Newton iterations.
        Ignored for ``iter_num == 1`` (linear-only path).
    """

    tol: float = 1e-6
    rel_tol: float = 1e-8
    max_iter: int = 100
    line_search_max_backtracks: int = 30
    line_search_c1: float = 1e-4
    line_search_rho: float = 0.5
    internal_jit: bool = False
    make_jittable: bool = False


@dataclass(frozen=True)
class SolverOptions(AbstractSolverOptions):
    """Deprecated legacy solver options.

    This class is intentionally disabled.  It previously mixed linear solver
    configuration and Newton controls in one object, which made solver-path
    behavior difficult to reason about and maintain.

    Use the new option classes instead:

    - ``DirectSolverOptions`` for direct linear solvers
    - ``IterativeSolverOptions`` for iterative linear solvers

    Newton/mode-specific options are being migrated separately.

    Parameters
    ----------
    tol : float, default 1e-6
        Absolute tolerance for residual vector (l2 norm)
    rel_tol : float, default 1e-8
        Relative tolerance for residual vector (l2 norm)
    max_iter : int, default 100
        Maximum number of Newton iterations
    linear_solver : str, optional
        Linear solver type. If not specified, automatically selects based on backend:
        - GPU backend: "cudss" (cuDSS direct solver, requires CUDA)
        - CPU backend: "cg" (Conjugate Gradient, JAX-native)
        Manual options: "cg", "bicgstab", "gmres", "spsolve", "cudss"
    preconditioner : callable, optional
        Preconditioner function for linear solver
    use_jacobi_preconditioner : bool, default False
        Whether to use Jacobi (diagonal) preconditioner automatically
    jacobi_shift : float, default 1e-12
        Regularization parameter for Jacobi preconditioner
    linear_solver_tol : float, default 1e-10
        Tolerance for linear solver
    linear_solver_atol : float, default 1e-10
        Absolute tolerance for linear solver
    linear_solver_maxiter : int, default 10000
        Maximum iterations for linear solver
    linear_solver_x0_fn : callable, optional
        Custom function to compute initial guess: f(current_sol) -> x0
    cudss_options : CUDSSOptions, optional
        CuDSS-specific options (mtype_id, mview_id, device_id)
    line_search_max_backtracks : int, default 30
        Maximum number of backtracking steps in Armijo line search
    line_search_c1 : float, default 1e-4
        Armijo constant for sufficient decrease condition
    line_search_rho : float, default 0.5
        Backtracking factor for line search (alpha *= rho each iteration)
    """

    tol: float = 1e-6
    rel_tol: float = 1e-8
    max_iter: int = 100
    linear_solver: Optional[str] = None  # Auto-detected based on backend if not specified
    preconditioner: Optional[Callable] = None
    use_jacobi_preconditioner: bool = False
    jacobi_shift: float = 1e-12
    linear_solver_tol: float = 1e-10
    linear_solver_atol: float = 1e-10
    linear_solver_maxiter: int = 10000
    linear_solver_x0_fn: Optional[Callable] = None  # Function to compute initial guess: f(current_sol) -> x0
    cudss_options: CUDSSOptions = None  # Will be set to default in __post_init__
    line_search_max_backtracks: int = 30
    line_search_c1: float = 1e-4
    line_search_rho: float = 0.5

    def __post_init__(self):
        raise RuntimeError(
            "SolverOptions has been removed. "
            "Migrate to DirectSolverOptions or IterativeSolverOptions."
        )


# ============================================================================
# Direct Solver Options
# ============================================================================

@dataclass(frozen=True)
class DirectSolverOptions(AbstractSolverOptions):
    """Configuration for direct linear solvers.

    Parameters
    ----------
    solver : str, default "auto"
        Direct solver algorithm:
        - "auto": Automatically selected based on backend and matrix property
          (CUDA -> cudss, CPU SPD -> cholmod, otherwise umfpack).
          Resolved at create_solver time.
        - "cudss": NVIDIA cuDSS direct solver (GPU only)
        - "spsolve": sparse direct solve via SciPy callback (CPU host)
        - "umfpack": sparse direct solve via scikit-sparse UMFPACK (CPU host)
        - "cholmod": CHOLMOD sparse Cholesky via scikit-sparse (SPD)
    cudss_options : CUDSSOptions, optional
        cuDSS-specific configuration. Only used when solver="cudss".
        Auto-configured when solver="auto" and backend is CUDA.
    sksparse_options : SKSPARSEOptions, optional
        Host-side direct solver configuration shared by ``spsolve``,
        ``umfpack``, and ``cholmod``.
    """
    solver: str = "auto"
    cudss_options: CUDSSOptions = None
    sksparse_options: SKSPARSEOptions = None

    def __post_init__(self):
        valid_solvers = ("auto", "cudss", "spsolve", "umfpack", "cholmod")
        if self.solver not in valid_solvers:
            raise ValueError(
                f"Invalid direct solver: {self.solver}. "
                f"Choose from {valid_solvers}"
            )
        if self.cudss_options is None:
            object.__setattr__(self, 'cudss_options', CUDSSOptions())
        if self.sksparse_options is None:
            object.__setattr__(self, 'sksparse_options', SKSPARSEOptions())

    def uses_x0(self) -> bool:
        """Direct solvers do not consume an initial iterate."""
        return False

def resolve_direct_solver(
    options: DirectSolverOptions,
    matrix_property: MatrixProperty,
    matrix_view: MatrixView,
) -> DirectSolverOptions:
    """Resolve "auto" to a concrete direct solver based on backend and matrix property.

    The selection follows a priority order inspired by COMSOL's solver
    guidelines, with cuDSS given highest priority on GPU:

    **GPU (CUDA backend)**::

        cuDSS  (always — matrix type adjusted to SPD/SYMMETRIC/GENERAL)

    **CPU backend, SPD matrix**::

        cholmod  →  umfpack  →  spsolve
        (Cholesky)  (LU)        (SciPy LU)

    **CPU backend, SYMMETRIC or GENERAL matrix**::

        umfpack  →  spsolve
        (LU)        (SciPy LU)

    Parameters
    ----------
    options : DirectSolverOptions
        Options with solver possibly set to "auto".
    matrix_property : MatrixProperty
        Detected matrix property (SPD, SYMMETRIC, GENERAL).
    matrix_view : MatrixView
        Problem's matrix storage format. Used to fill backend defaults when
        corresponding option fields are not explicitly set.

    Returns
    -------
    DirectSolverOptions
        Options with solver resolved to a concrete algorithm.
    """
    if options.solver != "auto":
        detect_available_solver_backends(options.solver)
        return DirectSolverOptions(
            solver=options.solver,
            cudss_options=options.cudss_options,
            sksparse_options=options.sksparse_options,
            check_convergence=options.check_convergence,
            convergence_threshold=options.convergence_threshold,
            verbose=options.verbose,
        )

    available_solvers = set(detect_available_solver_backends("auto"))
    has_spsolve_backend = "spsolve" in available_solvers
    has_cudss_backend = "cudss" in available_solvers
    has_cholmod = "cholmod" in available_solvers
    has_umfpack = "umfpack" in available_solvers

    sksparse_opts = options.sksparse_options

    if has_cudss_backend:
        mp_to_cudss = {
            MatrixProperty.SPD: CUDSSMatrixType.SPD,
            MatrixProperty.SYMMETRIC: CUDSSMatrixType.SYMMETRIC,
            MatrixProperty.GENERAL: CUDSSMatrixType.GENERAL,
        }
        cudss_mtype = mp_to_cudss[matrix_property]
        mv_to_cudss = {
            MatrixView.UPPER: CUDSSMatrixView.UPPER,
            MatrixView.LOWER: CUDSSMatrixView.LOWER,
        }
        cudss_mview = mv_to_cudss.get(matrix_view, options.cudss_options.matrix_view)

        cudss_opts = CUDSSOptions(
            matrix_type=cudss_mtype,
            matrix_view=cudss_mview,
            device_id=options.cudss_options.device_id,
        )
        resolved = DirectSolverOptions(
            solver="cudss",
            cudss_options=cudss_opts,
            sksparse_options=sksparse_opts,
            check_convergence=options.check_convergence,
            convergence_threshold=options.convergence_threshold,
            verbose=options.verbose,
        )
        logger.info(
            f"DirectSolver auto: matrix_property={matrix_property.name} "
            f"-> cudss (matrix_type={cudss_mtype.name})"
        )
        return resolved
    if matrix_property == MatrixProperty.SPD:
        if has_cholmod:
            solver = "cholmod"
        elif has_umfpack:
            solver = "umfpack"
        elif has_spsolve_backend:
            solver = "spsolve"
        else:
            raise RuntimeError(
                "No direct solver backend is available for solver='auto'. "
                f"Available solvers: {tuple(sorted(available_solvers))}"
            )
    else:
        if has_umfpack:
            solver = "umfpack"
        elif has_spsolve_backend:
            solver = "spsolve"
        else:
            raise RuntimeError(
                "No direct solver backend is available for solver='auto'. "
                f"Available solvers: {tuple(sorted(available_solvers))}"
            )

    sksparse_lower = options.sksparse_options.lower
    if matrix_property == MatrixProperty.SPD:
        if matrix_view == MatrixView.UPPER:
            sksparse_lower = False
        elif matrix_view in (MatrixView.LOWER, MatrixView.FULL):
            sksparse_lower = True

    sksparse_opts = SKSPARSEOptions(
        vmap_method=options.sksparse_options.vmap_method,
        order=options.sksparse_options.order,
        lower=sksparse_lower,
    )

    resolved = DirectSolverOptions(
        solver=solver,
        cudss_options=options.cudss_options,
        sksparse_options=sksparse_opts,
        check_convergence=options.check_convergence,
        convergence_threshold=options.convergence_threshold,
        verbose=options.verbose,
    )
    logger.info(
        f"DirectSolver auto: matrix_property={matrix_property.name} -> {solver}"
    )
    return resolved


# ============================================================================
# Iterative Solver Options
# ============================================================================

@dataclass(frozen=True)
class IterativeSolverOptions(AbstractSolverOptions):
    """Configuration for iterative linear solvers (cg, bicgstab, gmres).

    Parameters
    ----------
    solver : str, default "auto"
        Iterative solver algorithm:
        - "auto": Automatically selected based on matrix property.
          SPD -> cg, SYMMETRIC -> bicgstab, GENERAL -> gmres.
          Resolved at create_solver time.
        - "cg": Conjugate Gradient (for SPD matrices)
        - "bicgstab": BiCGSTAB (for symmetric/general matrices)
        - "gmres": GMRES (for general matrices)
    tol : float, default 1e-10
        Relative tolerance for the iterative solver.
    atol : float, default 1e-10
        Absolute tolerance for the iterative solver.
    maxiter : int, default 10000
        Maximum number of iterations.
    preconditioner : callable, optional
        Custom preconditioner function M(x) -> y.
    use_jacobi_preconditioner : bool, default False
        Whether to auto-create Jacobi (diagonal) preconditioner.
    jacobi_shift : float, default 1e-12
        Regularization parameter for Jacobi preconditioner.
    restart : int, optional
        Restart parameter for GMRES. Only used when solver='gmres'.
        If None, defaults to min(200, n) where n is the system size.
    x0_fn : callable, optional
        Custom function to compute initial guess: f(current_sol) -> x0.
    """
    solver: str = "auto"
    tol: float = 1e-10
    atol: float = 1e-10
    maxiter: int = 10000
    preconditioner: Optional[Callable] = None
    use_jacobi_preconditioner: bool = False
    jacobi_shift: float = 1e-12
    restart: Optional[int] = None
    x0_fn: Optional[Callable] = None

    def __post_init__(self):
        valid_solvers = ("auto", "cg", "bicgstab", "gmres")
        if self.solver not in valid_solvers:
            raise ValueError(
                f"Invalid iterative solver: {self.solver}. "
                f"Choose from {valid_solvers}"
            )

    def uses_x0(self) -> bool:
        """Iterative solvers consume an initial iterate."""
        return True

def resolve_iterative_solver(
    options: IterativeSolverOptions,
    matrix_property: MatrixProperty,
) -> IterativeSolverOptions:
    """Resolve "auto" to a concrete iterative solver based on matrix property.

    Selection mapping::

        SPD       →  cg        (Conjugate Gradient — optimal for SPD)
        SYMMETRIC →  bicgstab  (no symmetry exploitation, but robust)
        GENERAL   →  gmres     (general-purpose Krylov method)

    Parameters
    ----------
    options : IterativeSolverOptions
        Options with solver possibly set to "auto".
    matrix_property : MatrixProperty
        Detected matrix property (SPD, SYMMETRIC, GENERAL).

    Returns
    -------
    IterativeSolverOptions
        Options with solver resolved to a concrete algorithm.
        If solver != "auto", returns the input unchanged.
    """
    if options.solver != "auto":
        return options

    mp_to_solver = {
        MatrixProperty.SPD: "cg",
        MatrixProperty.SYMMETRIC: "bicgstab",
        MatrixProperty.GENERAL: "gmres",
    }
    solver_name = mp_to_solver[matrix_property]

    resolved = IterativeSolverOptions(
        solver=solver_name,
        tol=options.tol,
        atol=options.atol,
        maxiter=options.maxiter,
        preconditioner=options.preconditioner,
        use_jacobi_preconditioner=options.use_jacobi_preconditioner,
        jacobi_shift=options.jacobi_shift,
        x0_fn=options.x0_fn,
        check_convergence=options.check_convergence,
        convergence_threshold=options.convergence_threshold,
        verbose=options.verbose,
    )
    logger.info(
        f"IterativeSolver auto: matrix_property={matrix_property.name} "
        f"-> {solver_name}"
    )
    return resolved
