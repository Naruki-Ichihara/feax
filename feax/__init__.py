import jax

jax.config.update("jax_enable_x64", True)

# Version info
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("feax")
except (ImportError, PackageNotFoundError):
    __version__ = "0.4.0"

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
    create_solver,
)
from .solvers.common import (
    check_convergence,
    create_direct_solve_fn,
    create_iterative_solve_fn,
    create_jacobi_preconditioner,
    create_linear_solve_fn,
    create_x0,
)
from .solvers.linear import create_linear_solver, linear_solve
from .solvers.matrix_free import (
    LinearSolverOptions,
    MatrixFreeOptions,
    NewtonInfo,
    create_energy_fn,
    create_matrix_free_solver,
    newton_solve as matrix_free_newton_solve,
)
from .solvers.newton import newton_solve, newton_solve_fori, newton_solve_py
from .solvers.time_solver import (
    AdaptiveDtConfig,
    Callback,
    ExplicitPipeline,
    ImplicitPipeline,
    TimeConfig,
    TimePipeline,
    TimeResult,
    run as run_time,
)
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
    SKSPARSEOptions,
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
from .profiler import (
    JaxprInfo,
    MemorySnapshot,
    SolverProfile,
    TimingResult,
    format_cost_analysis,
    get_cost_analysis,
    get_hlo,
    memory_snapshot,
    profile,
    profile_solver,
    profile_solver_py,
    time_fn,
    trace_jaxpr,
)