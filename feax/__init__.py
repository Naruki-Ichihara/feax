import jax
jax.config.update("jax_enable_x64", True)

# Version info
try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("feax")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0"

# Main API
from .problem import Problem, MatrixView
from .internal_vars import InternalVars
from .assembler import get_jacobian_info, get_res, create_J_bc_function, create_res_bc_function
from .mesh import Mesh
from .DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res, DirichletBCSpec, DirichletBCConfig, dirichlet_bc_config
from .solvers.options import (
    AbstractSolverOptions,
    SolverOptions, CUDSSOptions, CUDSSMatrixType, CUDSSMatrixView,
    Backend, detect_backend, is_cuda, is_cpu, has_cudss, has_spsolve,
    MatrixProperty, detect_matrix_property,
    DirectSolverOptions, IterativeSolverOptions, NewtonOptions,
    resolve_direct_solver, resolve_iterative_solver,
)
from .solvers.linear import create_linear_solver, linear_solve
from .solvers.newton import newton_solve, newton_solve_fori, newton_solve_py
from .solver import (
    create_solver,
    create_x0,
    create_jacobi_preconditioner,
    create_direct_solve_fn,
    create_iterative_solve_fn,
    create_linear_solve_fn,
    check_convergence,
)
from .utils import zero_like_initial_guess

# Note: Experimental features available in feax.experimental
# from feax.experimental import SymbolicProblem

# Generative design toolkit (gene module)
# Gene = Generative design in FEAX
# Available: feax.gene.create_compliance_fn, feax.gene.create_volume_fn,
#            feax.gene.create_helmholtz_filter, feax.gene.mdmm, etc.
