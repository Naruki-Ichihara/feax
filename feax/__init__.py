import os as _os

import jax

# ── Floating-point precision ────────────────────────────────────────────────
# feax defaults to **float64** (double precision): FE assembly and direct
# linear solves on the resulting stiffness matrices are ill-conditioned in
# float32 and routinely produce garbage Newton steps / spurious adjoints.
#
# Override with the ``FEAX_X64`` environment variable, read here at import
# time (before any JAX array is created):
#
#     FEAX_X64=1   (default, or unset)  →  float64
#     FEAX_X64=0                        →  float32
#
# or programmatically right after ``import feax`` via :func:`feax.enable_x64`.


def _resolve_x64_env() -> bool:
    raw = _os.environ.get("FEAX_X64")
    if raw is None:
        return True                       # default: double precision
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


jax.config.update("jax_enable_x64", _resolve_x64_env())


def enable_x64(flag: bool = True) -> None:
    """Switch JAX between float64 (``flag=True``) and float32 (``flag=False``).

    .. warning::
        JAX's x64 setting is global and only affects arrays created
        *after* it is set.  Call this immediately after ``import feax``
        and before constructing any meshes / problems / arrays — arrays
        created earlier keep their original dtype.  For a guaranteed-clean
        run prefer the ``FEAX_X64`` environment variable instead.

    Examples
    --------
    >>> import feax
    >>> feax.enable_x64(False)        # run the rest of the script in float32
    >>> import feax as fe             # (re-import is a no-op; flag persists)
    """
    jax.config.update("jax_enable_x64", bool(flag))


def x64_enabled() -> bool:
    """Return ``True`` if JAX is currently in float64 (double-precision) mode."""
    return bool(jax.config.read("jax_enable_x64"))


# Version info
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("feax")
except (ImportError, PackageNotFoundError):
    __version__ = "0.5.2"

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
from .solvers.eigen import (
    BucklingConvergenceError,
    create_linear_buckling_solver,
    generalized_eigh,
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
from .utils import zero_like_initial_guess, XDMFWriter
from . import distributed
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