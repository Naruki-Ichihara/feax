"""Public solver API for FEAX.

This package-level module re-exports the stable solver-facing symbols from
``options``, ``common``, ``linear``, ``newton``, and ``reduced``.
"""

from .options import (
    AbstractSolverOptions,
    CUDSSMatrixType,
    CUDSSMatrixView,
    CUDSSOptions,
    DirectSolverOptions,
    IterativeSolverOptions,
    MatrixProperty,
    NewtonOptions,
    SolverOptions,
    detect_matrix_property,
    resolve_direct_solver,
    resolve_iterative_solver,
)
from .common import (
    check_convergence,
    create_direct_solve_fn,
    create_iterative_solve_fn,
    create_jacobi_preconditioner,
    create_linear_solve_fn,
    create_x0,
)
from .linear import create_linear_solver, linear_solve
from .newton import (
    create_newton_solver,
    newton_solve,
    newton_solve_fori,
    newton_solve_py,
)
from .reduced import create_reduced_solver

__all__ = [
    "AbstractSolverOptions",
    "CUDSSMatrixType",
    "CUDSSMatrixView",
    "CUDSSOptions",
    "DirectSolverOptions",
    "IterativeSolverOptions",
    "MatrixProperty",
    "NewtonOptions",
    "SolverOptions",
    "detect_matrix_property",
    "resolve_direct_solver",
    "resolve_iterative_solver",
    "check_convergence",
    "create_direct_solve_fn",
    "create_iterative_solve_fn",
    "create_jacobi_preconditioner",
    "create_linear_solve_fn",
    "create_x0",
    "create_linear_solver",
    "linear_solve",
    "create_newton_solver",
    "newton_solve",
    "newton_solve_fori",
    "newton_solve_py",
    "create_reduced_solver",
]
