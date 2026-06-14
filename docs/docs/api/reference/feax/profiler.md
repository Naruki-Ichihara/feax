---
sidebar_label: profiler
title: feax.profiler
---

Profiling and tracing utilities for FEAX.

Provides tools for inspecting JAX computation graphs (jaxpr),
timing function execution, and tracking device memory usage.

Examples
--------
```python
>>> from feax.profiler import trace_jaxpr, profile, time_fn
```
```python
>>> # Inspect the jaxpr of a function
>>> info = trace_jaxpr(res_fn, sol, traced_params)
>>> print(info)
```
```python
>>> # Time a solver call
>>> with profile(&quot;newton_solve&quot;) as t:
...     sol = solver(traced_params, u0)
>>> print(t.summary())
```
```python
>>> # Per-iteration Newton profiling
>>> from feax.profiler import profile_solver_py
>>> sol, prof = profile_solver_py(problem, bc, traced_params, initial)
>>> print(prof.summary())
```

## JaxprInfo Objects

```python
@dataclass
class JaxprInfo()
```

Container for jaxpr analysis results.

Attributes
----------
- **jaxpr** (*jax.core.ClosedJaxpr*): The closed jaxpr representation.
- **num_eqns** (*int*): Number of equations (primitive operations) in the jaxpr.
- **primitives** (*Dict[str, int]*): Count of each primitive operation type.
- **input_avals** (*list*): Abstract values of dynamic inputs (invars).
- **output_avals** (*list*): Abstract values of outputs.
- **static_avals** (*list*): Abstract values of static constants (constvars — mesh data etc.).
- **static_bytes** (*int*): Total bytes occupied by static constants.


#### full\_jaxpr

```python
def full_jaxpr() -> str
```

Return the full jaxpr string representation.

#### save

```python
def save(path: str) -> None
```

Save the full jaxpr text to a file.

#### trace\_jaxpr

```python
def trace_jaxpr(fn: Callable, *args, **kwargs) -> JaxprInfo
```

Trace a function and return its jaxpr analysis.

Parameters
----------
- **fn** (*Callable*): A JAX-compatible function to trace.


Returns
-------
JaxprInfo
    Analysis of the traced jaxpr.

Examples
--------
```python
>>> info = trace_jaxpr(res_fn, sol, traced_params)
>>> print(info)
>>> info.save(&quot;jaxpr_residual.txt&quot;)
```

## TimingRecord Objects

```python
@dataclass
class TimingRecord()
```

A single timing measurement.

## TimingResult Objects

```python
@dataclass
class TimingResult()
```

Accumulated timing results for a profiling session.

#### summary

```python
def summary() -> str
```

Return a formatted summary of all timing records.

#### profile

```python
@contextmanager
def profile(label: str = "default", block_until_ready: bool = True)
```

Context manager for timing JAX operations.

Parameters
----------
- **label** (*str*): A label for this timing measurement.
- **block_until_ready** (*bool*): If ``True``, flush JAX async dispatch before stopping the clock.


Examples
--------
```python
>>> with profile(&quot;assembly&quot;) as t:
...     J = assemble_jacobian(...)
>>> print(t.summary())
```

#### time\_fn

```python
def time_fn(fn: Callable,
            *args,
            label: Optional[str] = None,
            n_runs: int = 1,
            warmup: int = 1,
            **kwargs) -> Tuple[Any, TimingResult]
```

Time a JAX function with warmup runs.

Parameters
----------
- **fn** (*Callable*): Function to time.
- **label** (*str, optional*): Label for timing records. Defaults to the function name.
- **n_runs** (*int*): Number of timed runs (after warmup).
- **warmup** (*int*): Number of warmup runs (not timed, triggers JIT compilation).


Returns
-------
- **timing** (*TimingResult*): Timing information.


## MemorySnapshot Objects

```python
@dataclass
class MemorySnapshot()
```

Device memory usage snapshot.

#### memory\_snapshot

```python
def memory_snapshot(backend: Optional[str] = None) -> List[MemorySnapshot]
```

Take a snapshot of current device memory usage.

Parameters
----------
- **backend** (*str, optional*): JAX backend to query (``&quot;gpu&quot;``, ``&quot;cpu&quot;``). If ``None``, queries the default backend.


Returns
-------
list of MemorySnapshot
    One snapshot per device.

## SolverProfile Objects

```python
@dataclass
class SolverProfile()
```

Profile data collected from a solver run.

Attributes
----------
- **label** (*str*): Solver identifier.
- **wall_time_s** (*float*): Total wall-clock time.
- **iteration_times** (*List[float]*): Per-iteration wall-clock times.
- **residual_norms** (*List[float]*): Residual norms per iteration.
- **converged** (*bool*): Whether the solver converged.


#### convergence\_plot

```python
def convergence_plot(ax=None)
```

Plot residual convergence history (requires matplotlib).

#### profile\_solver

```python
def profile_solver(solver_fn: Callable,
                   *args,
                   label: str = "solver") -> Tuple[Any, SolverProfile]
```

Profile a feax solver call (total time only).

Parameters
----------
- **solver_fn** (*Callable*): A feax solver function (e.g. from ``create_solver``).
- **label** (*str*): Label for the profile.


Returns
-------
- **prof** (*SolverProfile*): Profile data.


#### profile\_solver\_py

```python
def profile_solver_py(problem,
                      bc,
                      traced_params,
                      initial_guess,
                      *,
                      solver_options=None,
                      label: str = "newton_py",
                      max_iter: int = 50,
                      tol: float = 1e-8) -> Tuple[Any, SolverProfile]
```

Profile Newton iterations using a Python loop.

Runs Newton&#x27;s method step-by-step (not JIT-compiled as a whole),
recording per-iteration wall time and residual norms.

Parameters
----------
- **problem** (*Problem*): The feax Problem instance.
- **bc** (*DirichletBC*): Boundary conditions.
- **traced_params** (*TracedParams*): Material parameters / internal variables.
- **initial_guess** (*array*): Initial solution vector.
- **solver_options** (*DirectSolverOptions, optional*): Linear solver options. Defaults to ``DirectSolverOptions()``.
- **label** (*str*): Label for the profile.
- **max_iter** (*int*): Maximum Newton iterations.
- **tol** (*float*): Convergence tolerance on residual norm.


Returns
-------
- **sol** (*array*): Solution vector.
- **prof** (*SolverProfile*): Per-iteration profile data.


#### get\_hlo

```python
def get_hlo(fn: Callable, *args) -> str
```

Get the HLO text representation of a compiled function.

Parameters
----------
- **fn** (*Callable*): A JAX function.


Returns
-------
str
    HLO text.

#### get\_cost\_analysis

```python
def get_cost_analysis(fn: Callable, *args) -> Any
```

Get XLA cost analysis for a compiled function.

Parameters
----------
- **fn** (*Callable*): A JAX function.


Returns
-------
dict or list
    Cost analysis from XLA (flops, bytes accessed, etc.).

#### format\_cost\_analysis

```python
def format_cost_analysis(fn: Callable, *args, label: str = "") -> str
```

Return a human-readable summary of XLA cost analysis.

Extracts FLOPs, memory access, and arithmetic intensity from the
XLA compiled cost analysis and formats them with SI units.

Parameters
----------
- **fn** (*Callable*): A JAX function.
- **label** (*str, optional*): Title shown in the header line.


Returns
-------
str
    Formatted cost analysis summary.

Examples
--------
```python
>>> print(format_cost_analysis(solve_forward, traced_params,
...                            label=&quot;solve_forward&quot;))
```
XLA Cost Analysis: solve_forward
--------------------------------------------------
  FLOPs:             12.34 GFLOP
  Memory accessed:   1.23 GB
  Arith. intensity:  10.03 FLOP/B

