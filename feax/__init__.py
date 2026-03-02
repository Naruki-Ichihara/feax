import jax

jax.config.update("jax_enable_x64", True)

# Version info
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("feax")
except (ImportError, PackageNotFoundError):
    __version__ = "0.1.0"

# Main API
from .assembler import create_J_bc_function, create_res_bc_function, get_jacobian_info, get_res
from .DCboundary import (
    DirichletBC,
    DirichletBCConfig,
    DirichletBCSpec,
    apply_boundary_to_J,
    apply_boundary_to_res,
    dirichlet_bc_config,
)
from .internal_vars import InternalVars
from .mesh import Mesh
from .problem import MatrixView, Problem
from .solver import (
    check_convergence,
    create_direct_solve_fn,
    create_iterative_solve_fn,
    create_jacobi_preconditioner,
    create_linear_solve_fn,
    create_solver,
    create_x0,
)
from .solvers.linear import create_linear_solver, linear_solve
from .solvers.newton import newton_solve, newton_solve_fori, newton_solve_py
from .solvers.options import (
    AbstractSolverOptions,
    Backend,
    CUDSSMatrixType,
    CUDSSMatrixView,
    CUDSSOptions,
    DirectSolverOptions,
    IterativeSolverOptions,
    MatrixProperty,
    NewtonOptions,
    SolverOptions,
    detect_backend,
    detect_matrix_property,
    has_cudss,
    has_spsolve,
    is_cpu,
    is_cuda,
    resolve_direct_solver,
    resolve_iterative_solver,
)
from .utils import zero_like_initial_guess

# Note: Experimental features available in feax.experimental
# from feax.experimental import SymbolicProblem

# Generative design toolkit (gene module)
# Gene = Generative design in FEAX
# Available: feax.gene.create_compliance_fn, feax.gene.create_volume_fn,
#            feax.gene.create_helmholtz_filter, feax.gene.mdmm, etc.
