import os as _os

# ── GPU memory preallocation ────────────────────────────────────────────────
# JAX/XLA preallocates ~75% of GPU memory on the first device op. For feax this
# is usually harmful: the sparse direct solvers (cuDSS) allocate their large
# factorization buffers *outside* XLA's pool via cudaMalloc, and on unified-
# memory devices (e.g. GB10) the 75% grab competes with host RAM. feax therefore
# **disables preallocation by default**.
#
# This MUST be set before JAX initializes the GPU backend (the first device
# array / ``jax.devices()`` call), so it is done here at import — before
# ``import jax`` — making the per-script ``os.environ[...]`` line unnecessary.
#
# Override with the ``FEAX_PREALLOCATE`` environment variable:
#
#     FEAX_PREALLOCATE=0  (default, or unset)  →  XLA_PYTHON_CLIENT_PREALLOCATE=false
#     FEAX_PREALLOCATE=1                        →  XLA_PYTHON_CLIENT_PREALLOCATE=true
#
# or programmatically right after ``import feax`` via :func:`feax.enable_preallocate`.
# An explicitly pre-set ``XLA_PYTHON_CLIENT_PREALLOCATE`` always takes precedence.


def _resolve_preallocate_env() -> bool:
    raw = _os.environ.get("FEAX_PREALLOCATE")
    if raw is None:
        return False                      # default: preallocation OFF
    return raw.strip().lower() in ("1", "true", "yes", "on")


if "XLA_PYTHON_CLIENT_PREALLOCATE" not in _os.environ:
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "true" if _resolve_preallocate_env() else "false")

# ── XLA constant folding ────────────────────────────────────────────────────
# XLA's constant-folding pass evaluates ops whose inputs are all constants at
# COMPILE time. feax assembly bakes in very large constant index arrays (BC
# masks, CSR slot maps — tens of millions of int32/bool), and folding a big
# ``gather`` over them materializes the result on the HOST during compilation.
# On unified-memory devices (GB10) that host spike can crash the whole machine,
# and everywhere it blows up compile time ("Constant folding an instruction is
# taking > Ns"). The runtime win is negligible for these index gathers, so feax
# **disables the constant_folding pass by default** (leaves those gathers as
# cheap runtime ops; numerical results are unchanged).
#
# Set before ``import jax`` so it is present when XLA initializes its backend.
# Any pre-existing ``XLA_FLAGS`` is preserved: ``constant_folding`` is merged into
# an existing ``--xla_disable_hlo_passes=...`` list, or a new flag is appended.
#
# Override with ``FEAX_CONSTANT_FOLDING`` (keeps XLA's default folding ON):
#
#     FEAX_CONSTANT_FOLDING=0  (default, or unset)  →  constant folding disabled
#     FEAX_CONSTANT_FOLDING=1                        →  constant folding kept ON


def _keep_constant_folding_env() -> bool:
    raw = _os.environ.get("FEAX_CONSTANT_FOLDING")
    if raw is None:
        return False                      # default: disable constant folding
    return raw.strip().lower() in ("1", "true", "yes", "on")


if not _keep_constant_folding_env():
    import re as _re
    _flags = _os.environ.get("XLA_FLAGS", "")
    if "constant_folding" not in _flags:
        _m = _re.search(r"--xla_disable_hlo_passes=(\S+)", _flags)
        if _m:                            # merge into the existing disable list
            _passes = _m.group(1).split(",") + ["constant_folding"]
            _flags = (_flags[:_m.start()]
                      + "--xla_disable_hlo_passes=" + ",".join(_passes)
                      + _flags[_m.end():])
        else:                             # append a fresh flag
            _flags = ((_flags + " " if _flags else "")
                      + "--xla_disable_hlo_passes=constant_folding")
        _os.environ["XLA_FLAGS"] = _flags
    del _re

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


def enable_preallocate(flag: bool = True) -> None:
    """Enable (``flag=True``) or disable (``flag=False``) XLA's GPU memory
    preallocation by setting ``XLA_PYTHON_CLIENT_PREALLOCATE``.

    feax disables preallocation by default; call this to turn the ~75% upfront
    grab back on (e.g. to reduce fragmentation for a fixed-shape workload).

    .. warning::
        XLA reads ``XLA_PYTHON_CLIENT_PREALLOCATE`` only when the GPU backend
        initializes (the first device op). Call this — or set
        ``FEAX_PREALLOCATE`` / ``XLA_PYTHON_CLIENT_PREALLOCATE`` — *before* any
        JAX array is created; afterwards it has no effect. For a guaranteed-clean
        run prefer the ``FEAX_PREALLOCATE`` environment variable.

    Examples
    --------
    >>> import feax
    >>> feax.enable_preallocate(True)   # re-enable XLA's 75% preallocation
    """
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if flag else "false"


def preallocate_enabled() -> bool:
    """Return ``True`` if XLA GPU memory preallocation is currently enabled."""
    return _os.environ.get(
        "XLA_PYTHON_CLIENT_PREALLOCATE", "false").strip().lower() in (
        "1", "true", "yes", "on")


# Version info
try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("feax")
except (ImportError, PackageNotFoundError):
    __version__ = "0.7.0"

# Main API
from .assembler import create_energy_fn, create_J_bc_csr_function, create_res_bc_function, get_jacobian, get_jacobian_info, get_res
from .DCboundary import (
    DirichletBC,
    DirichletBCConfig,
    DirichletBCSpec,
    apply_boundary_to_res,
    dirichlet_bc_config,
)
from .traced_params import TracedParams
from .traced_structure import TracedStructure
from .solution import Solution
from .mesh import Mesh
from .spgrid import SparseDesign, StructuredGrid, voxelize_mesh
from .narrowband import NarrowBand, SupersetBand
from .solvers.cmg import NarrowBandCMG
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
from .solvers.amg import build_amg_preconditioner, rigid_body_modes
from .solvers.eigen import (
    BucklingConvergenceError,
    create_linear_buckling_solver,
    generalized_eigh,
)
from .solvers.linear import create_linear_solver, linear_solve
from .solvers.nullspace import NullspaceConstraint
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
    AMGSolverOptions,
    Backend,
    CUDSSMatrixType,
    CUDSSMatrixView,
    CUDSSOptions,
    DirectSolverOptions,
    KrylovSolverOptions,
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


def __getattr__(name):
    # Lazy submodule: ``feax.asd`` pulls in asdex (+ numba) — deferred so plain
    # ``import feax`` doesn't pay for it.
    if name == "asd":
        import importlib
        return importlib.import_module(".asd", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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