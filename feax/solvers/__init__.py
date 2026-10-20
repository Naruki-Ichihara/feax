"""Public solver API for FEAX.

This package-level module re-exports the stable solver-facing symbols from
``options``, ``common``, ``linear``, ``newton``, and ``reduced``.
"""

from .common import (
    check_convergence,
    create_direct_solve_fn,
    create_iterative_solve_fn,
    create_linear_solve_fn,
    create_x0,
)
from .amg import rigid_body_modes
from .batched import BatchedSolveInfo, batched_pcg, batched_spmv
from .preconditioners import AMGOptions, AMGPreconditioner, LinearOperator, amg, jacobi, shared_factor
from .linear import create_linear_solver, linear_solve
from .newton import (
    create_newton_solver,
)
from .options import (
    AbstractSolverOptions,
    AMGSolverOptions,  # removed: a migration stub that raises
    CUDSSMatrixType,
    CUDSSMatrixView,
    CUDSSOptions,
    DirectSolverOptions,
    KrylovSolverOptions,
    MatrixProperty,
    NewtonOptions,
    SKSPARSEOptions,
    SolverOptions,
    detect_matrix_property,
    resolve_direct_solver,
    resolve_iterative_solver,
)
from .reduced import create_reduced_solver
from .time_solver import (
    AdaptiveDtConfig,
    Callback,
    ExplicitPipeline,
    ImplicitPipeline,
    TimeConfig,
    TimePipeline,
    TimeResult,
    run as run_time,
)
from .direct import (
    cholmod_solve,
    spsolve,
    umfpack_solve,
)
from .eigen import (
    BucklingConvergenceError,
    create_linear_buckling_solver,
    generalized_eigh,
)

__all__ = [
    "AbstractSolverOptions",
    "AMGOptions",
    "AMGPreconditioner",
    "LinearOperator",
    "amg",
    "jacobi",
    "shared_factor",
    "BatchedSolveInfo",
    "batched_pcg",
    "batched_spmv",
    "CUDSSMatrixType",
    "CUDSSMatrixView",
    "CUDSSOptions",
    "DirectSolverOptions",
    "KrylovSolverOptions",
    "MatrixProperty",
    "NewtonOptions",
    "SKSPARSEOptions",
    "SolverOptions",
    "detect_matrix_property",
    "resolve_direct_solver",
    "resolve_iterative_solver",
    "check_convergence",
    "rigid_body_modes",
    "create_direct_solve_fn",
    "create_iterative_solve_fn",
    "create_linear_solve_fn",
    "create_x0",
    "create_linear_solver",
    "linear_solve",
    "create_newton_solver",
    "create_reduced_solver",
    "cholmod_solve",
    "spsolve",
    "umfpack_solve",
    "AdaptiveDtConfig",
    "Callback",
    "ExplicitPipeline",
    "ImplicitPipeline",
    "TimeConfig",
    "TimePipeline",
    "TimeResult",
    "run_time",
    "create_linear_buckling_solver",
    "generalized_eigh",
    "BucklingConvergenceError",
]
