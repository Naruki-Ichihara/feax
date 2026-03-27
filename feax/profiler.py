"""Profiling and tracing utilities for FEAX.

Provides tools for inspecting JAX computation graphs (jaxpr),
timing function execution, and tracking device memory usage.

Examples
--------
>>> from feax.profiler import trace_jaxpr, profile, time_fn

>>> # Inspect the jaxpr of a function
>>> info = trace_jaxpr(res_fn, sol, internal_vars)
>>> print(info)

>>> # Time a solver call
>>> with profile("newton_solve") as t:
...     sol = solver(internal_vars, u0)
>>> print(t.summary())

>>> # Per-iteration Newton profiling
>>> from feax.profiler import profile_solver_py
>>> sol, prof = profile_solver_py(problem, bc, internal_vars, initial)
>>> print(prof.summary())
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jaxpr inspection
# ---------------------------------------------------------------------------


@dataclass
class JaxprInfo:
    """Container for jaxpr analysis results.

    Attributes
    ----------
    jaxpr : jax.core.ClosedJaxpr
        The closed jaxpr representation.
    num_eqns : int
        Number of equations (primitive operations) in the jaxpr.
    primitives : Dict[str, int]
        Count of each primitive operation type.
    input_avals : list
        Abstract values of dynamic inputs (invars).
    output_avals : list
        Abstract values of outputs.
    static_avals : list
        Abstract values of static constants (constvars — mesh data etc.).
    static_bytes : int
        Total bytes occupied by static constants.
    """

    jaxpr: Any
    num_eqns: int
    primitives: Dict[str, int]
    input_avals: list
    output_avals: list
    static_avals: list
    static_bytes: int

    def __repr__(self) -> str:
        def _mb(b: int) -> str:
            return f"{b / 1e6:.2f} MB" if b >= 1_000 else f"{b} B"

        lines = [
            "JaxprInfo(",
            f"  num_eqns         = {self.num_eqns},",
            f"  dynamic_inputs   = {len(self.input_avals)},",
            f"  static_constants = {len(self.static_avals)}  ({_mb(self.static_bytes)}),",
            f"  outputs          = {len(self.output_avals)},",
            f"  top_primitives   = {dict(_top_n(self.primitives, 10))},",
            ")",
        ]
        return "\n".join(lines)

    def full_jaxpr(self) -> str:
        """Return the full jaxpr string representation."""
        return str(self.jaxpr)

    def save(self, path: str) -> None:
        """Save the full jaxpr text to a file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.full_jaxpr())
        logger.info("Jaxpr saved to %s", path)


def _top_n(d: Dict[str, int], n: int) -> List[Tuple[str, int]]:
    return sorted(d.items(), key=lambda x: -x[1])[:n]


def _count_primitives(jaxpr) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for eqn in jaxpr.jaxpr.eqns:
        name = eqn.primitive.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def trace_jaxpr(fn: Callable, *args, **kwargs) -> JaxprInfo:
    """Trace a function and return its jaxpr analysis.

    Parameters
    ----------
    fn : Callable
        A JAX-compatible function to trace.
    *args
        Example arguments (used for shape/dtype inference only).

    Returns
    -------
    JaxprInfo
        Analysis of the traced jaxpr.

    Examples
    --------
    >>> info = trace_jaxpr(res_fn, sol, internal_vars)
    >>> print(info)
    >>> info.save("jaxpr_residual.txt")
    """
    closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    primitives = _count_primitives(closed_jaxpr)

    static_avals = [v.aval for v in closed_jaxpr.jaxpr.constvars]
    import math
    static_bytes = sum(
        math.prod(a.shape) * a.dtype.itemsize
        for a in static_avals
        if hasattr(a, "shape") and hasattr(a, "dtype")
    )

    return JaxprInfo(
        jaxpr=closed_jaxpr,
        num_eqns=len(closed_jaxpr.jaxpr.eqns),
        primitives=primitives,
        input_avals=[v.aval for v in closed_jaxpr.jaxpr.invars],
        output_avals=[v.aval for v in closed_jaxpr.jaxpr.outvars],
        static_avals=static_avals,
        static_bytes=static_bytes,
    )


# ---------------------------------------------------------------------------
# Timing profiler
# ---------------------------------------------------------------------------


@dataclass
class TimingRecord:
    """A single timing measurement."""

    label: str
    wall_time_s: float


@dataclass
class TimingResult:
    """Accumulated timing results for a profiling session."""

    records: List[TimingRecord] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return sum(r.wall_time_s for r in self.records)

    def summary(self) -> str:
        """Return a formatted summary of all timing records."""
        if not self.records:
            return "No timing records."
        lines = ["Timing Summary", "-" * 50]
        for r in self.records:
            lines.append(f"  {r.label:<30s} {r.wall_time_s:>10.4f} s")
        lines.append("-" * 50)
        lines.append(f"  {'Total':<30s} {self.total_s:>10.4f} s")
        return "\n".join(lines)


@contextmanager
def profile(label: str = "default", block_until_ready: bool = True):
    """Context manager for timing JAX operations.

    Parameters
    ----------
    label : str
        A label for this timing measurement.
    block_until_ready : bool
        If ``True``, flush JAX async dispatch before stopping the clock.

    Examples
    --------
    >>> with profile("assembly") as t:
    ...     J = assemble_jacobian(...)
    >>> print(t.summary())
    """
    result = TimingResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        if block_until_ready:
            jax.effects_barrier()
        elapsed = time.perf_counter() - start
        result.records.append(TimingRecord(label=label, wall_time_s=elapsed))


def time_fn(
    fn: Callable,
    *args,
    label: Optional[str] = None,
    n_runs: int = 1,
    warmup: int = 1,
    **kwargs,
) -> Tuple[Any, TimingResult]:
    """Time a JAX function with warmup runs.

    Parameters
    ----------
    fn : Callable
        Function to time.
    *args
        Arguments to pass to ``fn``.
    label : str, optional
        Label for timing records. Defaults to the function name.
    n_runs : int
        Number of timed runs (after warmup).
    warmup : int
        Number of warmup runs (not timed, triggers JIT compilation).

    Returns
    -------
    result
        The return value of the last call to ``fn``.
    timing : TimingResult
        Timing information.
    """
    if label is None:
        label = getattr(fn, "__name__", "fn")

    result = TimingResult()

    for _ in range(warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)

    for i in range(n_runs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - start
        run_label = f"{label}" if n_runs == 1 else f"{label}[{i}]"
        result.records.append(TimingRecord(label=run_label, wall_time_s=elapsed))

    return out, result


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """Device memory usage snapshot."""

    device: str
    live_buffers: int
    live_bytes: Optional[int] = None

    def __repr__(self) -> str:
        bytes_str = f", {self.live_bytes / 1e6:.1f} MB" if self.live_bytes else ""
        return f"MemorySnapshot({self.device}, {self.live_buffers} buffers{bytes_str})"


def memory_snapshot(backend: Optional[str] = None) -> List[MemorySnapshot]:
    """Take a snapshot of current device memory usage.

    Parameters
    ----------
    backend : str, optional
        JAX backend to query (``"gpu"``, ``"cpu"``). If ``None``,
        queries the default backend.

    Returns
    -------
    list of MemorySnapshot
        One snapshot per device.
    """
    devices = jax.devices(backend)
    snapshots = []
    for dev in devices:
        live_arrays = jax.live_arrays(platform=dev.platform)
        n_buffers = len(live_arrays)
        total_bytes = None
        try:
            total_bytes = sum(a.nbytes for a in live_arrays)
        except (AttributeError, TypeError):
            pass
        snapshots.append(MemorySnapshot(
            device=str(dev),
            live_buffers=n_buffers,
            live_bytes=total_bytes,
        ))
    return snapshots


# ---------------------------------------------------------------------------
# Solver profiler
# ---------------------------------------------------------------------------


@dataclass
class SolverProfile:
    """Profile data collected from a solver run.

    Attributes
    ----------
    label : str
        Solver identifier.
    wall_time_s : float
        Total wall-clock time.
    iteration_times : List[float]
        Per-iteration wall-clock times.
    residual_norms : List[float]
        Residual norms per iteration.
    converged : bool
        Whether the solver converged.
    """

    label: str = "solver"
    wall_time_s: float = 0.0
    iteration_times: List[float] = field(default_factory=list)
    residual_norms: List[float] = field(default_factory=list)
    converged: bool = False

    def summary(self) -> str:
        lines = [
            f"SolverProfile: {self.label}",
            "-" * 50,
            f"  Total time:     {self.wall_time_s:.4f} s",
            f"  Iterations:     {len(self.residual_norms)}",
            f"  Converged:      {self.converged}",
        ]
        if self.residual_norms:
            lines.append(f"  Initial res:    {self.residual_norms[0]:.6e}")
            lines.append(f"  Final res:      {self.residual_norms[-1]:.6e}")
        if self.iteration_times:
            avg = sum(self.iteration_times) / len(self.iteration_times)
            lines.append(f"  Avg iter time:  {avg:.4f} s")
        return "\n".join(lines)

    def convergence_plot(self, ax=None):
        """Plot residual convergence history (requires matplotlib)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.semilogy(range(len(self.residual_norms)), self.residual_norms, "o-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual norm")
        ax.set_title(self.label)
        ax.grid(True, alpha=0.3)
        return ax


def profile_solver(
    solver_fn: Callable,
    *args,
    label: str = "solver",
) -> Tuple[Any, SolverProfile]:
    """Profile a feax solver call (total time only).

    Parameters
    ----------
    solver_fn : Callable
        A feax solver function (e.g. from ``create_solver``).
    *args
        Arguments to pass to the solver.
    label : str
        Label for the profile.

    Returns
    -------
    result
        Solver output.
    prof : SolverProfile
        Profile data.
    """
    prof = SolverProfile(label=label)
    start = time.perf_counter()
    result = solver_fn(*args)
    jax.block_until_ready(result)
    prof.wall_time_s = time.perf_counter() - start
    return result, prof


def profile_solver_py(
    problem,
    bc,
    internal_vars,
    initial_guess,
    *,
    solver_options=None,
    label: str = "newton_py",
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple[Any, SolverProfile]:
    """Profile Newton iterations using a Python loop.

    Runs Newton's method step-by-step (not JIT-compiled as a whole),
    recording per-iteration wall time and residual norms.

    Parameters
    ----------
    problem : Problem
        The feax Problem instance.
    bc : DirichletBC
        Boundary conditions.
    internal_vars : InternalVars
        Material parameters / internal variables.
    initial_guess : array
        Initial solution vector.
    solver_options : DirectSolverOptions, optional
        Linear solver options. Defaults to ``DirectSolverOptions()``.
    label : str
        Label for the profile.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance on residual norm.

    Returns
    -------
    sol : array
        Solution vector.
    prof : SolverProfile
        Per-iteration profile data.
    """
    from .assembler import create_J_bc_function, create_res_bc_function
    from .solvers.common import create_linear_solve_fn
    from .solvers.options import (
        DirectSolverOptions,
        detect_matrix_property,
        resolve_direct_solver,
    )

    if solver_options is None:
        solver_options = DirectSolverOptions()

    res_bc_fn = create_res_bc_function(problem, bc)
    J_bc_fn = create_J_bc_function(problem, bc)

    # Resolve "auto" solver by sampling a Jacobian at the initial guess
    if isinstance(solver_options, DirectSolverOptions) and solver_options.solver == "auto":
        _sample_J = J_bc_fn(initial_guess, internal_vars)
        _mp = detect_matrix_property(_sample_J, matrix_view=problem.matrix_view)
        solver_options = resolve_direct_solver(solver_options, _mp, matrix_view=problem.matrix_view)

    linear_solve_fn = create_linear_solve_fn(solver_options)

    prof = SolverProfile(label=label)
    sol = initial_guess
    start_total = time.perf_counter()

    for _ in range(max_iter):
        iter_start = time.perf_counter()

        res = res_bc_fn(sol, internal_vars)
        jax.block_until_ready(res)
        res_norm = float(np.linalg.norm(res))
        prof.residual_norms.append(res_norm)

        if res_norm < tol:
            prof.converged = True
            prof.iteration_times.append(time.perf_counter() - iter_start)
            break

        J = J_bc_fn(sol, internal_vars)
        jax.block_until_ready(J)
        du = linear_solve_fn(J, -res)
        jax.block_until_ready(du)
        sol = sol + du
        prof.iteration_times.append(time.perf_counter() - iter_start)

    prof.wall_time_s = time.perf_counter() - start_total
    return sol, prof


# ---------------------------------------------------------------------------
# XLA/HLO utilities
# ---------------------------------------------------------------------------


def get_hlo(fn: Callable, *args) -> str:
    """Get the HLO text representation of a compiled function.

    Parameters
    ----------
    fn : Callable
        A JAX function.
    *args
        Example arguments for tracing.

    Returns
    -------
    str
        HLO text.
    """
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    return compiled.as_text()


def get_cost_analysis(fn: Callable, *args) -> Any:
    """Get XLA cost analysis for a compiled function.

    Parameters
    ----------
    fn : Callable
        A JAX function.
    *args
        Example arguments for tracing.

    Returns
    -------
    dict or list
        Cost analysis from XLA (flops, bytes accessed, etc.).
    """
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    return compiled.cost_analysis()


def _si(value: float, unit: str) -> str:
    """Format a non-negative float with SI prefix; return 'N/A' if negative."""
    if value < 0:
        return "N/A"
    for prefix, scale in [("G", 1e9), ("M", 1e6), ("K", 1e3)]:
        if value >= scale:
            return f"{value / scale:.2f} {prefix}{unit}"
    return f"{value:.2f} {unit}"


def _format_cost_dict(raw: dict, label: str = "") -> str:
    title = f"XLA Cost Analysis: {label}" if label else "XLA Cost Analysis"
    lines = [title, "-" * 50]

    flops = raw.get("flops", -1.0)
    bytes_total = raw.get("bytes accessed", -1.0)
    opt = raw.get("optimal_seconds", -1.0)

    lines.append(f"  FLOPs:             {_si(flops, 'FLOP')}")
    lines.append(f"  Memory accessed:   {_si(bytes_total, 'B')}")

    operand_keys = sorted(
        k for k in raw if k.startswith("bytes accessed") and k != "bytes accessed"
    )
    for k in operand_keys:
        tag = k.replace("bytes accessed", "").strip("{} ")
        if tag.startswith("out"):
            idx = tag[3:].strip("{} ")
            operand_label = "out" if not idx else f"out[{idx}]"
        else:
            operand_label = f"in[{tag}]"
        lines.append(f"    {operand_label:<10s}         {_si(raw[k], 'B')}")

    lines.append(f"  Optimal time:      {'N/A' if opt < 0 else f'{opt * 1e3:.3f} ms'}")

    if flops > 0 and bytes_total > 0:
        lines.append(f"  Arith. intensity:  {flops / bytes_total:.2f} FLOP/B")

    return "\n".join(lines)


def format_cost_analysis(fn: Callable, *args, label: str = "") -> str:
    """Return a human-readable summary of XLA cost analysis.

    Extracts FLOPs, memory access, and arithmetic intensity from the
    XLA compiled cost analysis and formats them with SI units.

    Parameters
    ----------
    fn : Callable
        A JAX function.
    *args
        Example arguments for tracing.
    label : str, optional
        Title shown in the header line.

    Returns
    -------
    str
        Formatted cost analysis summary.

    Examples
    --------
    >>> print(format_cost_analysis(solve_forward, internal_vars,
    ...                            label="solve_forward"))
    XLA Cost Analysis: solve_forward
    --------------------------------------------------
      FLOPs:             12.34 GFLOP
      Memory accessed:   1.23 GB
      Arith. intensity:  10.03 FLOP/B
    """
    raw = get_cost_analysis(fn, *args)
    if isinstance(raw, (list, tuple)):
        return "\n\n".join(
            _format_cost_dict(d, label=f"{label}[{i}]")
            for i, d in enumerate(raw)
        )
    return _format_cost_dict(raw, label=label)
