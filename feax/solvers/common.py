"""Shared solver helpers.

Small stateless utilities used across linear/newton/reduced solver modules.
"""

import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO

from ..problem import MatrixView
from .options import DirectSolverOptions, IterativeSolverOptions, SolverOptions


def _safe_negate(x):
    """Negate array, handling JAX's float0 type for zero gradients."""
    if hasattr(x, 'dtype'):
        dtype_str = str(x.dtype)
        if 'float0' in dtype_str or 'V' in dtype_str:
            return x
    return -x


def create_x0(bc_rows=None, bc_vals=None, P_mat=None):
    """Create BC-aware initial guess function for linear increments."""

    def x0_fn(current_sol):
        if bc_rows is None or bc_vals is None:
            return np.zeros_like(current_sol)

        bc_rows_array = np.array(bc_rows) if isinstance(bc_rows, (tuple, list)) else bc_rows
        bc_vals_array = np.array(bc_vals) if isinstance(bc_vals, (tuple, list)) else bc_vals

        if P_mat is not None:
            x0_1 = np.zeros(P_mat.shape[0])
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)

            current_sol_full = P_mat @ current_sol
            x0_2 = np.zeros(P_mat.shape[0])
            x0_2 = x0_2.at[bc_rows_array].set(current_sol_full[bc_rows_array])

            x0 = P_mat.T @ (x0_1 - x0_2)
        else:
            x0_1 = np.zeros_like(current_sol)
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)

            x0_2 = np.zeros_like(current_sol)
            x0_2 = x0_2.at[bc_rows_array].set(current_sol[bc_rows_array])

            x0 = x0_1 - x0_2

        return x0

    return x0_fn


def create_jacobi_preconditioner(A: BCOO, shift: float = 1e-12):
    """Create Jacobi (diagonal) preconditioner from sparse matrix."""
    diagonal = _extract_sparse_diagonal(A)
    diagonal_inv = 1.0 / (diagonal + shift)

    def matvec(x):
        return diagonal_inv * x

    return matvec


def create_direct_solve_fn(options: DirectSolverOptions):
    """Create a direct linear solve function."""
    if options.solver == "auto":
        raise ValueError(
            "DirectSolverOptions.solver is 'auto' (unresolved). "
            "Call resolve_direct_solver() before create_direct_solve_fn()."
        )

    lineax_solvers = {"cholesky", "lu", "qr"}
    if options.solver in lineax_solvers:
        try:
            import lineax as _lx
        except ImportError:
            raise RuntimeError(
                f"lineax is required to use the '{options.solver}' solver. "
                f"Install it with: pip install lineax"
            )

        solver_map = {
            "cholesky": (_lx.Cholesky(), (_lx.symmetric_tag, _lx.positive_semidefinite_tag)),
            "lu": (_lx.LU(), ()),
            "qr": (_lx.QR(), ()),
        }
        lx_solver, lx_tags = solver_map[options.solver]

        def solve(A, b, x0=None):
            input_structure = jax.ShapeDtypeStruct((A.shape[1],), b.dtype)

            def matvec(v):
                return A @ v

            operator = _lx.FunctionLinearOperator(
                matvec,
                input_structure=input_structure,
                tags=lx_tags,
            )

            sol = _lx.linear_solve(operator, b, solver=lx_solver)
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
                csr_offsets = A_bcsr.indptr.astype(np.int32)
                csr_columns = A_bcsr.indices.astype(np.int32)
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


def create_iterative_solve_fn(options: IterativeSolverOptions):
    """Create an iterative linear solve function."""
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
        "Choose from ('cg', 'bicgstab', 'gmres')"
    )


def create_linear_solve_fn(solver_options):
    """Create a linear solve function based on solver options."""
    if isinstance(solver_options, SolverOptions):
        raise RuntimeError(
            "SolverOptions has been removed. "
            "Use DirectSolverOptions or IterativeSolverOptions."
        )

    if isinstance(solver_options, DirectSolverOptions):
        return create_direct_solve_fn(solver_options)
    if isinstance(solver_options, IterativeSolverOptions):
        return create_iterative_solve_fn(solver_options)
    raise TypeError(
        "Unsupported solver option type. "
        f"Expected DirectSolverOptions or IterativeSolverOptions, got {type(solver_options).__name__}."
    )


def _extract_sparse_diagonal(A):
    """Extract sparse matrix diagonal as dense vector."""
    n = A.shape[0]
    diagonal_mask = A.indices[:, 0] == A.indices[:, 1]
    diagonal_indices = np.where(diagonal_mask, A.indices[:, 0], n)
    diagonal_values = np.where(diagonal_mask, A.data, 0.0)
    diag = np.zeros(n, dtype=A.data.dtype)
    return diag.at[diagonal_indices].add(diagonal_values)


def _matvec_with_matrix_view(A, x, matrix_view: MatrixView):
    """Apply matrix-vector product respecting stored matrix view."""
    if callable(A):
        return A(x)

    if matrix_view in (MatrixView.UPPER, MatrixView.LOWER):
        diag = _extract_sparse_diagonal(A)
        return A @ x + A.transpose() @ x - diag * x

    return A @ x


def check_convergence(A, x, b, solver_options, matrix_view: MatrixView, solver_label: str):
    """Check relative residual and return NaN solution when convergence fails."""
    residual = _matvec_with_matrix_view(A, x, matrix_view) - b
    residual_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    rel_residual = residual_norm / (b_norm + 1e-12)

    if getattr(solver_options, "verbose", False):
        jax.debug.print(
            "{label} residual: abs={r:.6e}, rel={rr:.6e}",
            label=solver_label,
            r=residual_norm,
            rr=rel_residual,
        )

    return np.where(
        rel_residual < getattr(solver_options, "convergence_threshold", 0.1),
        x,
        np.full_like(x, np.nan),
    )
