"""
Nonlinear and linear solvers for FEAX finite element framework.

This module provides Newton-Raphson solvers, linear solvers, and solver configuration
utilities for solving finite element problems. It includes both JAX-based solvers
for performance and Python-based solvers for debugging.

Key Features:
- Newton-Raphson solvers with line search and convergence control
- Multiple solver variants: while loop, fixed iterations, and Python debugging
- Jacobi preconditioning for improved convergence
- Comprehensive solver configuration through SolverOptions dataclass
- Support for multipoint constraints via prolongation matrices
"""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Union, Any
from enum import Enum
import logging
from .assembler import create_J_bc_function, create_res_bc_function
from .DCboundary import DirichletBC
from .problem import MatrixView, Problem
import lineax as lx

logger = logging.getLogger(__name__)


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


def _safe_negate(x):
    """Negate array, handling JAX's float0 type for zero gradients.

    When differentiating through computations where some parameters don't
    affect the output, JAX uses a special 'float0' dtype to represent
    zero gradients efficiently. This function handles negation properly
    for both regular arrays and float0 arrays.

    Parameters
    ----------
    x : jax.Array or np.ndarray
        Array to negate. May have float0 dtype.

    Returns
    -------
    jax.Array or np.ndarray
        Negated array, or unchanged if float0 dtype.
    """
    # Check for float0 dtype (JAX's zero gradient type)
    if hasattr(x, 'dtype'):
        dtype_str = str(x.dtype)
        if 'float0' in dtype_str or 'V' in dtype_str:
            # float0 represents zero gradient, negation is still zero
            return x
    return -x


@dataclass(frozen=True)
class CUDSSOptions:
    """Options specific to the NVIDIA cuDSS direct solver.

    cuDSS matrix configuration is defined by two orthogonal properties:
    1) Matrix Type (algebraic structure determining factorization method)
    2) Matrix View (which triangular portion is provided)

    Parameters
    ----------
    matrix_type : CUDSSMatrixType or str or int, default GENERAL
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

    @property
    def mtype_id(self) -> int:
        """Get matrix type as integer ID for cuDSS."""
        return self.matrix_type.value

    @property
    def mview_id(self) -> int:
        """Get matrix view as integer ID for cuDSS."""
        return self.matrix_view.value


@dataclass(frozen=True)
class SolverOptions:
    """Configuration options for the Newton solver.

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
        Manual options: "cg", "bicgstab", "gmres", "spsolve", "cudss", "lineax"
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
    verbose : bool, default False
        Whether to print convergence information during iterations
        Uses jax.debug.print() for JIT/vmap compatibility
    check_convergence : bool, default False
        Whether to verify linear solver convergence by checking residual.
        If True and residual is too large, returns NaN to signal failure.
        Useful for detecting ill-conditioned problems.
    convergence_threshold : float, default 0.1
        Maximum allowable relative residual for convergence check.
        Only used when check_convergence=True.
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
    verbose: bool = False
    check_convergence: bool = False
    convergence_threshold: float = 0.1

    def __post_init__(self):
        """Auto-detect backend and set appropriate defaults."""
        # Auto-detect linear solver based on backend if not specified
        if self.linear_solver is None:
            backend = jax.default_backend()
            if backend == "gpu":
                # Use cuDSS for GPU (faster direct solver)
                object.__setattr__(self, 'linear_solver', 'cudss')
                logger.info(f"JAX backend: {backend.upper()} | Auto-selected linear solver: cudss")
            else:
                # Use CG for CPU (JAX-native iterative solver)
                object.__setattr__(self, 'linear_solver', 'cg')
                logger.info(f"JAX backend: {backend.upper()} | Auto-selected linear solver: cg")
        else:
            # User manually specified solver - just show backend info
            backend = jax.default_backend()
            logger.info(f"JAX backend: {backend.upper()} | Linear solver: {self.linear_solver}")

        # Set default cudss_options if not provided
        if self.cudss_options is None:
            object.__setattr__(self, 'cudss_options', CUDSSOptions())

    @classmethod
    def from_problem(cls, problem, **kwargs):
        """Create SolverOptions with automatic configuration based on Problem.

        This factory method auto-configures solver options to match the Problem's
        matrix storage format. For Problems with UPPER/LOWER matrix_view, it
        automatically sets the cuDSS solver to use matching matrix view.

        Parameters
        ----------
        problem : Problem
            The finite element problem instance
        **kwargs : dict
            Additional SolverOptions parameters to override defaults

        Returns
        -------
        SolverOptions
            Configured solver options matching the problem structure

        Examples
        --------
        >>> problem = Problem(mesh, vec=3, dim=3, matrix_view='UPPER')
        >>> solver_opts = SolverOptions.from_problem(problem, tol=1e-8)
        >>> # Automatically sets cudss_options.matrix_view = UPPER

        Notes
        -----
        - UPPER/LOWER matrix_view requires SYMMETRIC matrix type
        - Automatically validates compatibility
        """
        from feax.problem import MatrixView

        # Determine cudss_options based on problem.matrix_view
        cudss_kwargs = {}
        if hasattr(problem, 'matrix_view'):
            if problem.matrix_view == MatrixView.UPPER:
                cudss_kwargs['matrix_view'] = CUDSSMatrixView.UPPER
                logger.info("Problem uses UPPER triangular storage → Setting cuDSS matrix_view=UPPER")
            elif problem.matrix_view == MatrixView.LOWER:
                cudss_kwargs['matrix_view'] = CUDSSMatrixView.LOWER
                logger.info("Problem uses LOWER triangular storage → Setting cuDSS matrix_view=LOWER")
            elif problem.matrix_view == MatrixView.FULL:
                cudss_kwargs['matrix_view'] = CUDSSMatrixView.FULL

            # For UPPER/LOWER, force SYMMETRIC matrix type
            if problem.matrix_view in (MatrixView.UPPER, MatrixView.LOWER):
                cudss_kwargs['matrix_type'] = CUDSSMatrixType.SYMMETRIC
                logger.info("Triangular storage requires SYMMETRIC matrix type → Setting matrix_type=SYMMETRIC")

        # Allow user to override cudss_options
        if 'cudss_options' not in kwargs:
            kwargs['cudss_options'] = CUDSSOptions(**cudss_kwargs)
        else:
            # User provided cudss_options - validate compatibility
            user_cudss = kwargs['cudss_options']
            if hasattr(problem, 'matrix_view'):
                if problem.matrix_view == MatrixView.UPPER and user_cudss.matrix_view != CUDSSMatrixView.UPPER:
                    logger.warning(
                        f"Problem uses UPPER storage but cudss_options.matrix_view={user_cudss.matrix_view.name}. "
                        f"Consider using matrix_view=CUDSSMatrixView.UPPER for consistency."
                    )
                elif problem.matrix_view == MatrixView.LOWER and user_cudss.matrix_view != CUDSSMatrixView.LOWER:
                    logger.warning(
                        f"Problem uses LOWER storage but cudss_options.matrix_view={user_cudss.matrix_view.name}. "
                        f"Consider using matrix_view=CUDSSMatrixView.LOWER for consistency."
                    )

        return cls(**kwargs)


def create_jacobi_preconditioner(A: jax.experimental.sparse.BCOO, shift: float = 1e-12) -> jax.experimental.sparse.BCOO:
    """Create Jacobi (diagonal) preconditioner from sparse matrix.
    
    Parameters
    ----------
    A : BCOO sparse matrix
        The system matrix to precondition
    shift : float, default 1e-12
        Small value added to diagonal for numerical stability
        
    Returns
    -------
    M : LinearOperator
        Jacobi preconditioner as diagonal inverse matrix
        
    Notes
    -----
    This creates a diagonal preconditioner M = diag(A)^{-1} with regularization.
    The preconditioner is JAX-compatible and avoids dynamic indexing.
    For elasticity problems with extreme material contrasts, this helps
    condition number significantly.
    """
    
    def extract_diagonal(A):
        """Extract diagonal from BCOO sparse matrix avoiding dynamic indexing."""
        # Get matrix dimensions
        n = A.shape[0]
        
        # Find diagonal entries by checking where row == col
        diagonal_mask = A.indices[:, 0] == A.indices[:, 1]
        
        # Extract diagonal values - use scatter_add to handle duplicates
        diag = jnp.zeros(n)
        diagonal_indices = jnp.where(diagonal_mask, A.indices[:, 0], n)  # Use n as dummy index
        diagonal_values = jnp.where(diagonal_mask, A.data, 0.0)
        diag = diag.at[diagonal_indices].add(diagonal_values)  # Handles out-of-bounds gracefully
        
        return diag
    
    def jacobi_matvec(diag_inv, x):
        """Apply Jacobi preconditioner: M @ x = diag_inv * x"""
        return diag_inv * x
    
    # Extract diagonal and compute inverse with regularization
    diagonal = extract_diagonal(A)
    diagonal_regularized = diagonal + shift
    diagonal_inv = 1.0 / diagonal_regularized
    
    # Create LinearOperator-like function
    def M_matvec(x):
        return jacobi_matvec(diagonal_inv, x)
    
    return M_matvec

def create_linear_solve_fn(solver_options: SolverOptions):
    """Create a linear solve function based on solver options.

    Parameters
    ----------
    solver_options : SolverOptions
        Solver configuration, including linear solver selection and tolerances.

    Returns
    -------
    callable
        Function with signature (A, b, x0) -> x.

    Notes
    -----
    The returned function selects between "cg", "bicgstab", "gmres", "spsolve",
    "cudss", "lineax". The "spsolve" option is only available on CPU, while
    "cudss" is only available on CUDA GPUs (tested with CUDA 12). To
    install the CUDSS solver, use `pip install feax[cuda12]`.
    """

    def choose_preconditioner(A):
        if solver_options.use_jacobi_preconditioner and solver_options.preconditioner is None:
            M = create_jacobi_preconditioner(A, solver_options.jacobi_shift)
        return solver_options.preconditioner

    # Define solver functions for JAX compatibility (no conditionals inside JAX-traced code)
    if solver_options.linear_solver == "cg":
        def solve(A, b, x0):
            M = choose_preconditioner(A)
            x, _ = jax.scipy.sparse.linalg.cg(
                A, b, x0=x0, M=M,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )
            return x
        return solve

    if solver_options.linear_solver == "bicgstab":
        def solve(A, b, x0):
            M = choose_preconditioner(A)
            x, _ = jax.scipy.sparse.linalg.bicgstab(
                A, b, x0=x0, M=M,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )
            return x
        return solve

    if solver_options.linear_solver == "gmres":
        def solve(A, b, x0):
            M = choose_preconditioner(A)
            x, _ = jax.scipy.sparse.linalg.gmres(
                A, b, x0=x0, M=M,
                tol=solver_options.linear_solver_tol,
                atol=solver_options.linear_solver_atol,
                maxiter=solver_options.linear_solver_maxiter
            )
            return x
        return solve

    if solver_options.linear_solver == "lineax":
        def solve(A, b, x0):
            input_structure = jax.ShapeDtypeStruct((A.shape[1],), b.dtype)

            def matvec(v):
                return A @ v

            operator = lx.FunctionLinearOperator(
                matvec,
                input_structure=input_structure,
                tags=(
                    lx.symmetric_tag,
                    lx.positive_semidefinite_tag,
                ),
            )

            sol = lx.linear_solve(operator, b, solver=lx.Cholesky())
            return sol.value

        return solve

    if solver_options.linear_solver == "spsolve":
        if jax.default_backend() != "cpu":
            raise RuntimeError(
                "jax.experimental.sparse.linalg.spsolve is only enabled on the CPU. "
                "Run on CPU or use an iterative solver on GPU."
            )

        def solve(A, b, _):
            # spsolve currently does not support batching via vmap
            # CPU -> scipy implementation, UMPFACK solver
            # GPU -> jax implementation, requires 32-bit types + unstable
            # https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
            A_bcsr = jax.experimental.sparse.BCSR.from_bcoo(A.sum_duplicates(nse=A.nse))
            x = jax.experimental.sparse.linalg.spsolve(
                A_bcsr.data, A_bcsr.indices, A_bcsr.indptr, b,
                tol=solver_options.linear_solver_tol,
                reorder=1,
            )
            return x
        return solve
    
    if solver_options.linear_solver == "cudss":
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

        def solve(A, b, _):
            nonlocal cudss_solver
            A_bcsr = jax.experimental.sparse.BCSR.from_bcoo(A.sum_duplicates(nse=A.nse))

            csr_values  = A_bcsr.data
            if cudss_solver is None:
                csr_offsets = A_bcsr.indptr.astype(jnp.int32)
                csr_columns = A_bcsr.indices.astype(jnp.int32)
                cudss_solver = CuDSSSolver(
                    csr_offsets=csr_offsets,
                    csr_columns=csr_columns,
                    device_id=solver_options.cudss_options.device_id,
                    mtype_id=solver_options.cudss_options.mtype_id,
                    mview_id=solver_options.cudss_options.mview_id,
                )
            x, _ = cudss_solver(b, csr_values)
            return x
        return solve
        

    valid_solvers = ("cg", "bicgstab", "gmres", "spsolve", "cudss", "lineax")
    raise ValueError(
        f"Unknown linear solver: {solver_options.linear_solver}. "
        f"Choose from {valid_solvers}"
    )


def create_x0(bc_rows=None, bc_vals=None, P_mat=None):
    """Create initial guess function for linear solver following JAX-FEM approach.
    
    Parameters
    ----------
    bc_rows : array-like, optional
        Row indices of boundary condition locations
    bc_vals : array-like, optional  
        Boundary condition values
    P_mat : BCOO matrix, optional
        Prolongation matrix for reduced problems (maps reduced to full DOFs)
        
    Returns
    -------
    x0_fn : callable
        Function that takes current solution and returns initial guess for increment
        
    Notes
    -----
    Implements the exact x0 computation from the row elimination solver:
    x0_1 = assign_bc(zeros, problem) - sets BC values at BC locations, 0 elsewhere
    x0_2 = copy_bc(current_sol, problem) - copies current solution values at BC locations, 0 elsewhere  
    x0 = x0_1 - x0_2 - the correct initial guess computation
    
    For reduced problems (when P_mat is provided):
    x0_2 = copy_bc(P @ current_sol_reduced, problem) - expand reduced sol and copy BC
    x0 = P.T @ (x0_1 - x0_2) - transform back to reduced space
    
    Examples
    --------
    >>> # Usage with BC information
    >>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0]) 
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    
    >>> # Usage with reduced problem
    >>> x0_fn = create_x0(bc_rows, bc_vals, P_mat=P)
    """
    
    def x0_fn(current_sol):
        """BC-aware strategy: correct x0 method from row elimination solver."""
        if bc_rows is None or bc_vals is None:
            # Fallback to zeros if BC info not provided
            return jnp.zeros_like(current_sol)
            
        # Convert to JAX arrays if needed (for JIT compatibility)
        bc_rows_array = jnp.array(bc_rows) if isinstance(bc_rows, (tuple, list)) else bc_rows
        bc_vals_array = jnp.array(bc_vals) if isinstance(bc_vals, (tuple, list)) else bc_vals
        
        if P_mat is not None:
            # Reduced problem case - following ref.py logic
            # x0_1 = assign_bc(zeros_full, problem)
            x0_1 = jnp.zeros(P_mat.shape[0])  # Full size
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)
            
            # x0_2 = copy_bc(P @ current_sol_reduced, problem)
            current_sol_full = P_mat @ current_sol  # Expand reduced to full
            x0_2 = jnp.zeros(P_mat.shape[0])
            x0_2 = x0_2.at[bc_rows_array].set(current_sol_full[bc_rows_array])
            
            # x0 = P.T @ (x0_1 - x0_2) - transform to reduced space
            x0 = P_mat.T @ (x0_1 - x0_2)
        else:
            # Standard (non-reduced) problem case
            # x0_1 = assign_bc(zeros, problem) - sets BC values at BC locations, 0 elsewhere
            x0_1 = jnp.zeros_like(current_sol)
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)
            
            # x0_2 = copy_bc(current_sol, problem) - copies current solution values at BC locations, 0 elsewhere
            x0_2 = jnp.zeros_like(current_sol)
            x0_2 = x0_2.at[bc_rows_array].set(current_sol[bc_rows_array])
            
            # x0 = x0_1 - x0_2 (the original correct implementation)
            x0 = x0_1 - x0_2
        
        return x0
        
    return x0_fn


def create_armijo_line_search_jax(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX-compatible Armijo backtracking line search using jax.lax.scan.

    This function returns a line search function that can be JIT-compiled.

    Parameters
    ----------
    res_bc_applied : callable
        Residual function with boundary conditions applied.
        Signature: res_bc_applied(sol, internal_vars=None) -> residual
    c1 : float, default 1e-4
        Armijo constant for sufficient decrease condition
    rho : float, default 0.5
        Backtracking factor (alpha *= rho each iteration)
    max_backtracks : int, default 30
        Maximum number of backtracking steps

    Returns
    -------
    line_search_fn : callable
        Line search function with signature:
        (sol, delta_sol, initial_res_norm, internal_vars=None) -> (new_sol, new_norm, alpha, success)
    """

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        """Execute Armijo line search.

        Parameters
        ----------
        sol : array
            Current solution
        delta_sol : array
            Search direction (Newton step)
        res : array
            Residual at current solution
        res_norm : float
            Norm of residual
        internal_vars : InternalVars, optional
            Internal variables for residual evaluation

        Returns
        -------
        new_sol : array
            Updated solution
        new_norm : float
            Residual norm at new solution
        alpha : float
            Step size used
        success : bool
            Whether a valid step was found
        """
        # Initialize with full Newton step
        init_carry = (1.0, False, sol + delta_sol, res_norm, internal_vars)

        # Run line search with closure over delta_sol and res
        def body_with_closure(carry, x):
            alpha, found_good, best_sol, best_norm, iv = carry
            trial_sol = sol + alpha * delta_sol  # Use closure variables
            if iv is not None:
                trial_res = res_bc_applied(trial_sol, iv)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)

            is_valid = jnp.logical_not(jnp.any(jnp.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            grad_merit = -jnp.dot(res, res)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            is_acceptable = is_valid & armijo_satisfied

            new_found = found_good | is_acceptable
            new_sol = jnp.where(jnp.logical_not(found_good) & is_acceptable, trial_sol, best_sol)
            new_norm = jnp.where(jnp.logical_not(found_good) & is_acceptable, trial_norm, best_norm)
            new_alpha = jnp.where(is_acceptable, alpha, alpha * rho)

            return (new_alpha, new_found, new_sol, new_norm, iv), None

        final_carry, _ = jax.lax.scan(
            body_with_closure,
            init_carry,
            jnp.arange(max_backtracks)
        )

        final_alpha, found_good, new_sol, new_norm, _ = final_carry

        # If no good step found, use very small step as fallback
        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)

        # Use fallback if nothing worked
        final_sol = jnp.where(found_good, new_sol, fallback_sol)
        final_norm = jnp.where(found_good, new_norm, fallback_norm)
        final_alpha_out = jnp.where(found_good, final_alpha, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search


def create_armijo_line_search_python(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create Python-based Armijo backtracking line search.

    This version uses Python loops and is suitable for debugging or non-JIT contexts.

    Parameters
    ----------
    res_bc_applied : callable
        Residual function with boundary conditions applied
    c1 : float, default 1e-4
        Armijo constant
    rho : float, default 0.5
        Backtracking factor
    max_backtracks : int, default 30
        Maximum backtracking steps

    Returns
    -------
    line_search_fn : callable
        Line search function
    """
    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        """Execute Armijo line search using Python loop."""
        grad_merit = -jnp.dot(res, res)

        alpha = 1.0
        for _ in range(max_backtracks):
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)

            is_valid = not jnp.any(jnp.isnan(trial_res))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit

            if is_valid and armijo_satisfied:
                return trial_sol, trial_norm, alpha, True

            alpha *= rho

        # Fallback
        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        return fallback_sol, fallback_norm, 1e-8, False

    return line_search


def newton_solve(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None, P_mat=None):
    """Newton solver using JAX while_loop for JIT compatibility.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC

    Returns
    -------
    sol, res_norm, rel_res_norm, iter_count : tuple
        Solution, residual norms, and iteration count
    """
    
    # Resolve x0 function based on options (at function definition time, not JAX-traced)
    if solver_options.linear_solver_x0_fn is not None:
        # User provided custom function
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        # Create bc_aware x0 function
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver function
    linear_solve_fn = create_linear_solve_fn(solver_options)
    
    def linear_solve_jit(A, b, x0=None):
        """Solve linear system Ax = b using JAX sparse solvers."""
        # Assume A is already in BCOO format (which it should be from the assembler)
        # Use pre-selected solver function (no conditionals in JAX-traced code)
        return linear_solve_fn(A, b, x0)

    # Create line search function once (outside the loop)
    # Use JAX scan version (JIT-compatible, no early exit)
    armijo_search = create_armijo_line_search_jax(
        res_bc_applied,
        c1=solver_options.line_search_c1,
        rho=solver_options.line_search_rho,
        max_backtracks=solver_options.line_search_max_backtracks
    )

    def cond_fun(state):
        """Condition function for while loop."""
        sol, res_norm, rel_res_norm, iter_count = state
        continue_iter = (res_norm > solver_options.tol) & (rel_res_norm > solver_options.rel_tol) & (iter_count < solver_options.max_iter)
        return continue_iter
    
    def body_fun(state):
        """Body function for while loop - performs one Newton iteration."""
        sol, res_norm, rel_res_norm, iter_count = state

        # Compute residual and Jacobian
        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        # Compute initial guess for increment
        x0 = x0_fn(sol)

        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_jit(J, -res, x0=x0)

        # Armijo backtracking line search (using function created outside loop)
        new_sol, new_norm, alpha, success = armijo_search(sol, delta_sol, res, res_norm, internal_vars)

        # Print iteration info if verbose (uses jax.debug.print for JIT compatibility)
        if solver_options.verbose:
            jax.debug.print(
                "Newton iter {i:3d}: res_norm = {r:.6e}, alpha = {a:.4f}, success = {s}",
                i=iter_count, r=new_norm, a=alpha, s=success
            )

        sol = new_sol
        res_norm = new_norm

        # Update iteration count
        iter_count = iter_count + 1

        return (sol, res_norm, rel_res_norm, iter_count)
    
    # Initial state
    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_guess, initial_res_norm, 1.0, 0)

    # Print initial residual if verbose
    if solver_options.verbose:
        jax.debug.print("Newton solver starting: initial res_norm = {r:.6e}", r=initial_res_norm)

    # Run Newton iterations using while_loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    # Print final convergence info if verbose
    if solver_options.verbose:
        final_sol, final_res_norm, final_rel_res_norm, final_iter = final_state
        jax.debug.print(
            "Newton solver converged: final_iter = {i}, final_res_norm = {r:.6e}",
            i=final_iter, r=final_res_norm
        )

    return final_state


def newton_solve_fori(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, num_iters: int, internal_vars=None, P_mat=None):
    """Newton solver using JAX fori_loop for fixed iterations - optimized for vmap.

    Designed for vmap with fixed iterations and consistent computational graph.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    num_iters : int
        Fixed number of iterations
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC

    Returns
    -------
    sol, final_res_norm, converged : tuple
        Solution, residual norm, and convergence flag
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver function
    linear_solve_fn = create_linear_solve_fn(solver_options)

    # Create line search function once (outside the loop)
    armijo_search = create_armijo_line_search_jax(
        res_bc_applied,
        c1=solver_options.line_search_c1,
        rho=solver_options.line_search_rho,
        max_backtracks=solver_options.line_search_max_backtracks
    )

    def newton_iteration(i, state):
        """Single Newton iteration for fori_loop."""
        sol, res_norm = state

        # Compute residual and Jacobian
        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        # Compute initial guess for increment
        x0 = x0_fn(sol)

        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_fn(J, -res, x0)

        # Armijo backtracking line search using shared function
        new_sol, new_res_norm, alpha, success = armijo_search(sol, delta_sol, res, res_norm, internal_vars)

        # Print iteration info if verbose (uses jax.debug.print for JIT compatibility)
        if solver_options.verbose:
            jax.debug.print(
                "Newton iter {iter:3d}: res_norm = {r:.6e}, alpha = {a:.4f}, success = {s}",
                iter=i, r=new_res_norm, a=alpha, s=success
            )

        return (new_sol, new_res_norm)
    
    # Initial residual norm
    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)

    # Print initial residual if verbose
    if solver_options.verbose:
        jax.debug.print("Newton solver (fori) starting: initial res_norm = {r:.6e}", r=initial_res_norm)

    # Run fixed number of iterations using fori_loop
    final_state = jax.lax.fori_loop(
        0, num_iters,
        newton_iteration,
        (initial_guess, initial_res_norm)
    )

    final_sol, final_res_norm = final_state

    # Check convergence
    converged = final_res_norm < solver_options.tol

    # Print final convergence info if verbose
    if solver_options.verbose:
        jax.debug.print(
            "Newton solver (fori) finished: {n} iterations, final_res_norm = {r:.6e}, converged = {c}",
            n=num_iters, r=final_res_norm, c=converged
        )

    return final_sol, final_res_norm, converged


def newton_solve_py(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None, P_mat=None):
    """Newton solver using Python while loop - non-JIT version for debugging.

    Uses Python control flow for easier debugging. Not JIT-compatible.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC

    Returns
    -------
    sol, final_res_norm, converged, num_iters : tuple
        Solution, residual norm, convergence flag, and iteration count
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver function
    linear_solve_fn = create_linear_solve_fn(solver_options)

    # Create line search function using shared Python version
    armijo_line_search = create_armijo_line_search_python(
        res_bc_applied,
        c1=solver_options.line_search_c1,
        rho=solver_options.line_search_rho,
        max_backtracks=solver_options.line_search_max_backtracks
    )
    
    # Initialize
    sol = initial_guess
    if internal_vars is not None:
        initial_res = res_bc_applied(sol, internal_vars)
    else:
        initial_res = res_bc_applied(sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    res_norm = initial_res_norm
    iter_count = 0

    # Print initial residual if verbose
    if solver_options.verbose:
        logger.info(f"Newton solver (py) starting: initial res_norm = {initial_res_norm:.6e}")

    # Main Newton loop
    while (res_norm > solver_options.tol and
           res_norm / initial_res_norm > solver_options.rel_tol and
           iter_count < solver_options.max_iter):

        # Compute residual and Jacobian
        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        # Compute initial guess for increment
        x0 = x0_fn(sol)

        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_fn(J, -res, x0)

        # Line search using shared function
        new_sol, new_res_norm, alpha, success = armijo_line_search(
            sol, delta_sol, res, res_norm, internal_vars
        )

        # Print iteration info if verbose
        if solver_options.verbose:
            logger.info(f"Newton iter {iter_count:3d}: res_norm = {new_res_norm:.6e}, alpha = {alpha:.4f}, success = {success}")

        # Update solution
        sol = new_sol
        res_norm = new_res_norm
        iter_count += 1

    # Check convergence
    converged = (res_norm <= solver_options.tol or
                res_norm / initial_res_norm <= solver_options.rel_tol)

    # Print final convergence info if verbose
    if solver_options.verbose:
        logger.info(f"Newton solver (py) finished: iter_count = {iter_count}, final_res_norm = {res_norm:.6e}, converged = {converged}")

    return sol, res_norm, converged, iter_count


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


def _check_linear_convergence(A, x, b, solver_options: SolverOptions, matrix_view: MatrixView, solver_label: str):
    """Check relative residual and return NaN solution when convergence fails."""
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


def linear_solve(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, matrix_view: MatrixView, internal_vars=None, P_mat=None):
    """Linear solver for problems that converge in one iteration.

    Optimized for linear problems (e.g., linear elasticity). Performs single Newton step.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    matrix_view : MatrixView, optional
        Matrix storage format from the problem.
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC

    Returns
    -------
    sol, None : tuple
        Solution and None (for compatibility)
    
    For linear problems, this single iteration achieves the exact solution.
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver function
    linear_solve_fn = create_linear_solve_fn(solver_options)
    
    # Single Newton iteration (no while loop)
    # Step 1: Compute residual and Jacobian
    if internal_vars is not None:
        res = res_bc_applied(initial_guess, internal_vars)
        J = J_bc_applied(initial_guess, internal_vars)
    else:
        res = res_bc_applied(initial_guess)
        J = J_bc_applied(initial_guess)
    
    # Step 2: Compute initial guess for increment
    x0 = x0_fn(initial_guess)
    
    # Step 3: Solve linear system: J * delta_sol = -res
    b = -res
    delta_sol = linear_solve_fn(J, b, x0)

    # Step 4: Verify convergence if requested
    if solver_options.check_convergence:
        delta_sol = _check_linear_convergence(
            A=J,
            x=delta_sol,
            b=b,
            solver_options=solver_options,
            matrix_view=matrix_view,
            solver_label="Linear solver",
        )

    # Step 5: Update solution
    sol = initial_guess + delta_sol

    return sol, None


def __linear_solve_adjoint(A, b, solver_options: SolverOptions, matrix_view: MatrixView, bc=None):
    """Solve linear system for adjoint problem.

    Parameters
    ----------
    A : BCOO sparse matrix
        The transposed Jacobian matrix (J^T)
    b : jax.numpy.ndarray
        Right-hand side vector (cotangent vector from VJP)
    solver_options : SolverOptions
        Solver configuration for adjoint solve
    matrix_view : MatrixView
        Matrix storage format from the problem
    bc : DirichletBC, optional
        Boundary conditions for computing initial guess

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

    # Define solver function
    linear_solve_fn = create_linear_solve_fn(solver_options)

    # Compute initial guess for adjoint solve
    # For adjoint problem, BC rows have identity on diagonal of J^T,
    # so the solution at BC rows should be the RHS values at those rows
    if bc is not None and hasattr(bc, 'bc_rows') and hasattr(bc, 'bc_vals'):
        # Start with zeros
        x0 = jnp.zeros_like(b)
        # Set BC row values from RHS (since J^T has identity at BC rows)
        bc_rows_array = jnp.array(bc.bc_rows) if not isinstance(bc.bc_rows, jnp.ndarray) else bc.bc_rows
        x0 = x0.at[bc_rows_array].set(b[bc_rows_array])
    else:
        # Fallback to zero initial guess
        x0 = jnp.zeros_like(b)

    sol = linear_solve_fn(A, b, x0)

    # Verify convergence if requested
    if solver_options.check_convergence:
        sol = _check_linear_convergence(
            A=A,
            x=sol,
            b=b,
            solver_options=solver_options,
            matrix_view=matrix_view,
            solver_label="Adjoint solver",
        )

    return sol


def _create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options, iter_num):
    """Create matrix-free reduced solver for periodic boundary conditions."""
    
    # Create full space functions
    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)
    
    # Matrix-free reduced operations
    def create_reduced_matvec(sol_full, internal_vars):
        """Create matrix-vector product function for reduced Jacobian."""
        J_full = J_bc_func(sol_full, internal_vars)
        
        def reduced_matvec(v_reduced):
            v_full = P @ v_reduced          # Expand to full space
            Jv_full = J_full @ v_full       # Apply full Jacobian
            Jv_reduced = P.T @ Jv_full      # Reduce back
            return Jv_reduced
        return reduced_matvec
    
    def compute_reduced_residual(sol_full, internal_vars):
        """Compute residual in reduced space."""
        res_full = res_bc_func(sol_full, internal_vars)
        return P.T @ res_full
    
    # Matrix-free solver function
    def reduced_solve_fn(internal_vars, initial_guess_full):
        """Solve in reduced space using matrix-free CG."""
        # Compute reduced residual
        res_reduced = compute_reduced_residual(initial_guess_full, internal_vars)
        
        # Create reduced Jacobian matvec
        J_reduced_matvec = create_reduced_matvec(initial_guess_full, internal_vars)
        
        # Solve reduced system: J_reduced @ sol_reduced = -res_reduced
        sol_reduced, _ = jax.scipy.sparse.linalg.cg(J_reduced_matvec, -res_reduced, 
                                                   tol=solver_options.tol,
                                                   maxiter=solver_options.linear_solver_maxiter)
        
        # Map back to full space
        sol_full = P @ sol_reduced
        return sol_full, None
    
    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        """Matrix-free reduced solver with automatic differentiation."""
        return reduced_solve_fn(internal_vars, initial_guess)[0]
    
    def f_fwd(internal_vars, initial_guess):
        """Forward function for custom VJP."""
        sol = differentiable_solve(internal_vars, initial_guess)
        return sol, (internal_vars, sol)
    
    def f_bwd(res, v):
        """Backward function using matrix-free adjoint."""
        internal_vars, sol = res

        # Create adjoint matvec operator: (P.T @ J.T @ P) @ adjoint = P.T @ v
        J_full = J_bc_func(sol, internal_vars)
        rhs_reduced = P.T @ v

        def adjoint_matvec(adjoint_reduced):
            adjoint_full = P @ adjoint_reduced    # Expand to full space
            Jt_adjoint_full = J_full.T @ adjoint_full  # Apply transpose Jacobian
            return P.T @ Jt_adjoint_full          # Reduce back

        # Initialize with zero vector for better convergence
        x0_reduced = jnp.zeros_like(rhs_reduced)

        # Solve adjoint system: J_reduced.T @ adjoint_reduced = rhs_reduced
        adjoint_reduced, _ = jax.scipy.sparse.linalg.cg(
            adjoint_matvec, rhs_reduced,
            x0=x0_reduced,
            tol=adjoint_solver_options.linear_solver_tol,
            atol=adjoint_solver_options.linear_solver_atol,
            maxiter=adjoint_solver_options.linear_solver_maxiter
        )
        
        # Compute VJP for internal variables
        adjoint_full = P @ adjoint_reduced
        
        def constraint_fn(dofs, internal_vars):
            return res_bc_func(dofs, internal_vars)
        
        def constraint_fn_sol_to_sol(sol_list, internal_vars):
            dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
            con_vec = constraint_fn(dofs, internal_vars)
            return problem.unflatten_fn_sol_list(con_vec)
        
        def get_partial_params_c_fn(sol_list):
            def partial_params_c_fn(internal_vars):
                return constraint_fn_sol_to_sol(sol_list, internal_vars)
            return partial_params_c_fn

        def get_vjp_contraint_fn_params(internal_vars, sol_list):
            partial_c_fn = get_partial_params_c_fn(sol_list)
            def vjp_linear_fn(v_list):
                _, f_vjp = jax.vjp(partial_c_fn, internal_vars)
                val, = f_vjp(v_list)
                return val
            return vjp_linear_fn
        
        sol_list = problem.unflatten_fn_sol_list(sol)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_full))
        vjp_result = jax.tree_util.tree_map(_safe_negate, vjp_result)

        return (vjp_result, None)  # No gradient w.r.t. initial_guess
    
    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve


def create_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[SolverOptions] = None,
    adjoint_solver_options: Optional[SolverOptions] = None,
    iter_num: Optional[int] = None,
    P: Optional[BCOO] = None,
) -> Callable[[Any, jax.Array], jax.Array]:
    """Create a differentiable solver that returns gradients w.r.t. internal_vars using custom VJP.
    
    This solver uses the self-adjoint approach for efficient gradient computation:
    - Forward mode: standard Newton solve
    - Backward mode: solve adjoint system to compute gradients
    
    Parameters
    ----------
    problem : Problem
        The feax Problem instance (modular API - no internal_vars in constructor)
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions, optional
        Options for forward solve (defaults to SolverOptions())
    adjoint_solver_options : SolverOptions, optional
        Options for adjoint solve (defaults to same as forward solve)
    iter_num : int, optional
        Number of iterations to perform. Controls which solver is used:
        - None: Use while loop newton_solve (adaptive iterations, NOT vmappable)
        - 1: Use linear_solve (single iteration for linear problems, vmappable)
        - >1: Use newton_solve_fori with fixed number of iterations (vmappable)
        Note: When iter_num is not None, the solver is vmappable since it uses fixed iterations.
        Recommended: Use iter_num=1 for linear problems for optimal performance.
    P : BCOO matrix, optional
        Prolongation matrix for periodic boundary conditions (maps reduced to full DOFs).
        If provided, solver works in reduced space using matrix-free operations for memory efficiency.
        
    Returns
    -------
    differentiable_solve : callable
        Function that takes (internal_vars, initial_guess) and returns solution with gradient support
        
    Notes
    -----
    The returned function has signature: differentiable_solve(internal_vars, initial_guess) -> solution
    where gradients flow through internal_vars (material properties, loadings, etc.)
    
    The initial_guess parameter is required to avoid conditionals that slow down JAX compilation.
    For the first solve, you can pass zeros with BC values set:
        initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
        initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    
    When iter_num is specified (not None), the solver becomes vmappable as it uses fixed
    iterations without dynamic control flow. This is essential for parallel solving of
    multiple parameter sets using jax.vmap.
    
    Examples
    --------
    >>> # Create differentiable solver
    >>> diff_solve = create_solver(problem, bc)
    >>> 
    >>> # Prepare initial guess
    >>> initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
    >>> initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    >>> 
    >>> # First solve
    >>> solution = diff_solve(internal_vars, initial_guess)
    >>> 
    >>> # For time-dependent problems, update initial guess each timestep
    >>> for t in timesteps:
    >>>     solution = diff_solve(internal_vars_at_t, solution)  # Use previous solution
    >>> 
    >>> # For linear problems (e.g., linear elasticity), use single iteration for best performance
    >>> # This is both faster and vmappable
    >>> diff_solve_linear = create_solver(problem, bc, iter_num=1)
    >>> 
    >>> # For fixed iteration count (e.g., for vmap)
    >>> diff_solve_fixed = create_solver(problem, bc, iter_num=10)
    >>> 
    >>> # Define loss function
    >>> def loss_fn(internal_vars):
    ...     initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
    ...     initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    ...     sol = diff_solve(internal_vars, initial_guess)
    ...     return jnp.sum(sol**2)  # Example loss
    >>> 
    >>> # Compute gradients w.r.t. internal_vars
    >>> grad_fn = jax.grad(loss_fn)
    >>> gradients = grad_fn(internal_vars)
    """
    
    # Set default options
    if solver_options is None:
        solver_options = SolverOptions()
    if adjoint_solver_options is None:
        # Default: use same solver options for adjoint as forward
        # This is simpler and works well for symmetric problems
        adjoint_solver_options = solver_options

    def _validate_cudss_options(opts: CUDSSOptions, role: str) -> None:
        # Matrix view validation - allow UPPER/LOWER for triangular storage
        # The Problem class handles the actual storage format (FULL/UPPER/LOWER)
        # The cuDSS solver just needs to know how to interpret that storage
        if opts.matrix_view not in (CUDSSMatrixView.FULL, CUDSSMatrixView.UPPER, CUDSSMatrixView.LOWER):
            raise ValueError(
                f"cudss_options.matrix_view must be FULL, UPPER, or LOWER for {role} solver. "
                f"Got: {opts.matrix_view.name}"
            )

        # For UPPER/LOWER views, SYMMETRIC or SPD matrix type is required
        if opts.matrix_view in (CUDSSMatrixView.UPPER, CUDSSMatrixView.LOWER):
            if opts.matrix_type not in (CUDSSMatrixType.SYMMETRIC, CUDSSMatrixType.SPD):
                logger.warning(
                    f"{role} solver: Using matrix_view={opts.matrix_view.name} with matrix_type={opts.matrix_type.name}. "
                    f"For best performance, use matrix_type=SYMMETRIC or matrix_type=SPD with triangular storage."
                )

        if opts.matrix_type not in (CUDSSMatrixType.GENERAL, CUDSSMatrixType.SYMMETRIC, CUDSSMatrixType.SPD):
            raise ValueError(
                f"cudss_options.matrix_type must be GENERAL, SYMMETRIC, or SPD for {role} solver. "
                f"Got: {opts.matrix_type.name}."
            )

    if solver_options.linear_solver == "cudss":
        _validate_cudss_options(solver_options.cudss_options, "primary")
    if adjoint_solver_options.linear_solver == "cudss":
        _validate_cudss_options(adjoint_solver_options.cudss_options, "adjoint")

    # Branch between standard and reduced solver
    if P is not None:
        return _create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options, iter_num)
    
    # Standard solver (original implementation)
    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)

    if iter_num is None:
        solve_fn = lambda internal_vars, initial_sol: newton_solve(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, internal_vars
        )
    elif iter_num == 1:
        solve_fn = lambda internal_vars, initial_sol: linear_solve(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, problem.matrix_view, internal_vars
        )
    else:
        solve_fn = lambda internal_vars, initial_sol: newton_solve_fori(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, iter_num, internal_vars
        )

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        """Forward solve: standard Newton iteration.
        
        Parameters
        ----------
        internal_vars : InternalVars
            Material properties, loadings, etc.
        initial_guess : jax.numpy.ndarray
            Initial guess for the solution vector. For time-dependent problems,
            this should be the solution from the previous time step.
            
        Returns
        -------
        sol : jax.numpy.ndarray
            Solution vector
        """
        return solve_fn(internal_vars, initial_guess)[0]
    
    def f_fwd(internal_vars, initial_guess):
        """Forward function for custom VJP.
        
        Returns solution and residuals needed for backward pass.
        """
        sol = differentiable_solve(internal_vars, initial_guess)
        return sol, (internal_vars, sol)
    
    def f_bwd(res, v):
        internal_vars, sol = res
        
        def constraint_fn(dofs, internal_vars):
            return res_bc_func(dofs, internal_vars)
        
        def constraint_fn_sol_to_sol(sol_list, internal_vars):
            dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
            con_vec = constraint_fn(dofs, internal_vars)
            return problem.unflatten_fn_sol_list(con_vec)
        
        def get_partial_params_c_fn(sol_list):

            def partial_params_c_fn(internal_vars):
                return constraint_fn_sol_to_sol(sol_list, internal_vars)
            
            return partial_params_c_fn

        def get_vjp_contraint_fn_params(internal_vars, sol_list):
            partial_c_fn = get_partial_params_c_fn(sol_list)
            def vjp_linear_fn(v_list):
                _, f_vjp = jax.vjp(partial_c_fn, internal_vars)
                val, = f_vjp(v_list)
                return val
            return vjp_linear_fn
        
        J = J_bc_func(sol, internal_vars)
        v_vec = jax.flatten_util.ravel_pytree(v)[0]

        # If the problem stores only one triangular part (UPPER/LOWER),
        # transposing swaps storage convention and is not meaningful here.
        # Use J directly for adjoint solve in that case.
        use_transpose = True
        if problem.matrix_view in (MatrixView.UPPER, MatrixView.LOWER):
            use_transpose = False
            logger.debug(
                "Using J directly (no transpose) for adjoint solve with problem.matrix_view=%s",
                problem.matrix_view.name,
            )

        J_adjoint = J.transpose() if use_transpose else J
        adjoint_vec = __linear_solve_adjoint(J_adjoint, v_vec, adjoint_solver_options, problem.matrix_view, bc)
        sol_list = problem.unflatten_fn_sol_list(sol)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
        vjp_result = jax.tree_util.tree_map(_safe_negate, vjp_result)

        return (vjp_result, None)  # No gradient w.r.t. initial_guess

    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve
