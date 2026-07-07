---
sidebar_label: time_solver
title: feax.solvers.time_solver
---

Time-stepping solver with Pipeline interface.

Provides a ``TimePipeline`` abstract class and ``run()`` function that
mirror the topology-optimisation driver (``feax.gene.optimizer``) but
for transient / quasi-static problems.

Three pipeline levels:

* **TimePipeline** — full control: implement ``step()`` directly.
  Use for staggered multi-physics (thermal + phase-field + mechanics).
* **ImplicitPipeline** — one implicit solve per step (backward Euler
  pattern).  Implement ``update_vars()``; ``step()`` calls
  ``self.solver(tp, state)`` automatically.
* **ExplicitPipeline** — explicit ODE integration with lumped mass.
  Implement ``compute_rhs()``; ``step()`` applies Forward Euler / RK2 / RK4.

Usage:

```python
from feax.solvers.time_solver import ImplicitPipeline, TimeConfig, run

class HeatPipeline(ImplicitPipeline):
    def build(self, mesh):
        ...
        self.solver = fe.create_solver(...)

    def initial_state(self):
        return jnp.zeros(...)

    def update_vars(self, state, t, dt):
        T_old = ...
        return fe.TracedParams(volume_vars=(T_old,))

result = run(HeatPipeline(), mesh, TimeConfig(dt=1e-5, t_end=1e-2))
```

## TimePipeline Objects

```python
class TimePipeline(ABC)
```

Abstract base class for time-stepping pipelines.

Subclass and implement :meth:`build`, :meth:`initial_state`, and
:meth:`step`.  Override :meth:`save` and :meth:`monitor` for output.

The interface mirrors ``feax.gene.optimizer.Pipeline``:

============= ==================== ===================================
Optimizer      TimePipeline         Purpose
============= ==================== ===================================
``build``      ``build``            Mesh-dependent setup
``objective``  ``step``             Core computation per iteration
``filter``     —                    (state is user-defined)
``@constraint``  ``monitor``        Scalar logging
============= ==================== ===================================

#### build

```python
@abstractmethod
def build(mesh: Mesh) -> None
```

Create mesh-dependent objects (problems, BCs, solvers).

Called once before the time loop starts.  Store results as
instance attributes for use in :meth:`step`, :meth:`save`,
and :meth:`monitor`.

#### initial\_state

```python
@abstractmethod
def initial_state() -> Any
```

Return the initial state (any JAX-compatible pytree).

Called once after :meth:`build`.  The returned object is passed
as ``state`` to the first call of :meth:`step`.

Examples::

# Single-field problem: flat solution vector
return jnp.zeros(problem.num_total_dofs_all_vars)

# Multi-physics staggered: dict of fields
return {`&#x27;T_flat&#x27;: T0, &#x27;u_flat&#x27;: u0, &#x27;history&#x27;: H0`}

#### step

```python
@abstractmethod
def step(state: Any, t: float, dt: float) -> Any
```

Advance one time step and return the new state.

Parameters
----------
- **t** (*float*): Time at the *beginning* of this step.
- **dt** (*float*): Time step size.


Returns
-------
new_state
    Updated state at time ``t + dt``.

#### save

```python
def save(state: Any, step: int, t: float, output_dir: str) -> None
```

Save a snapshot (VTU, etc.) to *output_dir*.

Override to write problem-specific output.  The default is a no-op.

Parameters
----------
- **step** (*int*): Time step index (0 = initial condition).
- **t** (*float*): Current simulation time.
- **output_dir** (*str*): Directory for output files (already created by ``run``).


#### monitor

```python
def monitor(state: Any, step: int, t: float) -> Dict[str, float]
```

Return scalar quantities to log each step.

Override to report problem-specific diagnostics.

Returns
-------
dict
    ``{`name: value`}`` of scalar quantities.  Printed to stdout
    and written to ``history.csv`` when *output_dir* is given.

## ImplicitPipeline Objects

```python
class ImplicitPipeline(TimePipeline)
```

Implicit time stepping with a single solver call per step.

Covers the common backward-Euler pattern where each time step
solves one (non)linear system:

1. ``update_vars(state, t, dt)`` → ``TracedParams``
2. ``self.solver(tp, state)`` → new state

Set ``self.solver`` in :meth:`build` and implement
:meth:`update_vars`.

For **staggered multi-physics** (multiple solves per step),
subclass :class:``0 directly and override ``step()``.

Pseudo-time mode
----------------
Set the class attribute ``pseudo_time = True`` when the implicit
residual ``R(u_{`n+1`}, t_{`n+1`}) = 0`` does **not** depend on
``u_n`` — i.e. when the previous-step state is used purely as a
Newton warm-start (load stepping, parameter continuation, steady-
state homotopy) and not as a transient history term.  In that
case Newton&#x27;s converged solution at step ``n+1`` is independent
of its initial guess, so we cut the gradient through the state
handed from one step to the next via :func:``1.
Differentiating a ``run()`` chain (or any direct chain of
``step()`` calls) under :func:``6 then needs only
the **final step&#x27;s** adjoint solve instead of one per step — the
intermediate steps still execute their forward Newton (warm-start
chain) but contribute nothing to the gradient.

**Do not enable** ``pseudo_time`` for genuinely transient problems
(mass terms, ``(u_{`n+1`} - u_n)/dt`` in the residual, history
variables) — gradients through time would be silently dropped and
the resulting derivatives would be incorrect.

Example::

class CahnHilliardPipeline(ImplicitPipeline):
def build(self, mesh):
self.problem = CahnHilliard(mesh=[mesh, mesh], ...)
self.bc = ...
self.solver = fe.create_solver(self.problem, self.bc, ...)

def initial_state(self):
return jax.flatten_util.ravel_pytree([c0, mu0])[0]

def update_vars(self, state, t, dt):
c_old = self.problem.unflatten_fn_sol_list(state)[0][:, 0]
return fe.TracedParams(volume_vars=(c_old,))

# Pseudo-time / load-stepping example (gradient-friendly)::

class ThermalRampPipeline(ImplicitPipeline):
pseudo_time = True            # ← enable adjoint short-circuit

def update_vars(self, state, t, dt):
lam = t + dt
return fe.TracedParams(
volume_vars=(jnp.full(self._n_nodes, lam),),
)

#### pseudo\_time

When ``True`` the warm-start state is detached from the gradient
at every step (see class docstring).  Default ``False`` keeps the
full transient adjoint behaviour.

#### update\_vars

```python
@abstractmethod
def update_vars(state: Any, t: float, dt: float)
```

Prepare ``TracedParams`` for the implicit solve.

Parameters
----------
- **t** (*float*): Time at the beginning of this step.
- **dt** (*float*): Time step size.


Returns
-------
feax.TracedParams
    Internal variables encoding the transient term
    (e.g. ``T_old`` for backward Euler).

#### step

```python
def step(state: Any, t: float, dt: float) -> Any
```

Solve the implicit system for this time step.

When :attr:`pseudo_time` is true the input ``state`` is wrapped
in :func:`jax.lax.stop_gradient` before being forwarded to
:meth:`update_vars` and to ``self.solver`` (warm-start) — this
cuts the gradient through the previous-step state so that
differentiating a chain of steps only requires one adjoint
solve at the final step.

## ExplicitPipeline Objects

```python
class ExplicitPipeline(TimePipeline)
```

Explicit time stepping for du/dt = f(u, t).

No linear system solve is needed per step — only matrix-vector
products and element-wise division by a lumped (diagonal) mass
matrix.  This makes each step very cheap, but the time step ``dt``
is restricted by a CFL-like stability condition.

Implement :meth:`compute_rhs` to return f(u, t).  The integration
scheme is selected by setting ``self.scheme`` (default ``&#x27;euler&#x27;``):

* ``&#x27;euler&#x27;`` — Forward Euler  (1st order)
* ``&#x27;rk2&#x27;``   — Midpoint RK2   (2nd order)
* ``&#x27;rk4&#x27;``   — Classic RK4    (4th order)

Set ``self.scheme`` in :meth:``5 or at class level.

For specialised explicit schemes (Velocity Verlet, central
difference, symplectic integrators), override ``step()`` directly.

Example::

class WavePipeline(ExplicitPipeline):
scheme = &#x27;rk4&#x27;

def build(self, mesh):
self.M_inv = ...  # lumped mass inverse
self.K = ...      # stiffness matrix

def initial_state(self):
return {`&#x27;u&#x27;: u0, &#x27;v&#x27;: v0`}

def compute_rhs(self, state, t):
a = self.M_inv @ (-self.K @ state[&#x27;u&#x27;] + f_ext(t))
return {`&#x27;u&#x27;: state[&#x27;v&#x27;], &#x27;v&#x27;: a`}

#### compute\_rhs

```python
@abstractmethod
def compute_rhs(state: Any, t: float) -> Any
```

Compute the right-hand side f(u, t) for du/dt = f(u, t).

Parameters
----------
- **t** (*float*): Current time.


Returns
-------
rhs
    Same pytree structure as *state*.

#### step

```python
def step(state: Any, t: float, dt: float) -> Any
```

Advance one step using the selected explicit scheme.

## TimeConfig Objects

```python
@dataclass
class TimeConfig()
```

Time-stepping configuration.

Parameters
----------
- **dt** (*float*): Time step size.
- **t_end** (*float*): End time of the simulation.
- **t_start** (*float*): Start time (default 0).
- **save_every** (*int*): Call ``pipeline.save()`` every N steps (and at step 0 and final).
- **print_every** (*int*): Print monitor output every N steps.


## AdaptiveDtConfig Objects

```python
@dataclass
class AdaptiveDtConfig()
```

Configuration for adaptive time stepping.

At each step the pipeline&#x27;s :meth:`~TimePipeline.adapt_dt` method
(if overridden) can propose a new ``dt``.  The value is clamped
between ``dt_min`` and ``dt_max``.

Parameters
----------
- **dt_min** (*float*): Minimum allowed time step.
- **dt_max** (*float*): Maximum allowed time step.
- **growth_factor** (*float*): Maximum factor by which dt can grow in one step.


## TimeResult Objects

```python
@dataclass
class TimeResult()
```

Time integration result.

Attributes
----------
- **history** (*dict*): ``{`&#x27;step&#x27;: [...], &#x27;time&#x27;: [...], **monitor_keys`}``.
- **t_final** (*float*): Actual final time reached.
- **n_steps** (*int*): Total number of time steps taken.


## Callback Objects

```python
@dataclass
class Callback()
```

User callback invoked at configurable frequency.

Parameters
----------
- **fn** (*callable*): ``fn(state, step, t)`` — called for its side effects.
- **every** (*int*): Call every N steps (default 1 = every step).


#### run

```python
def run(pipeline: TimePipeline,
        mesh: Mesh,
        time: TimeConfig,
        adaptive_dt: Optional[AdaptiveDtConfig] = None,
        output_dir: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None) -> TimeResult
```

Run a time-stepping simulation.

Parameters
----------
- **pipeline** (*TimePipeline*): Pipeline instance defining the transient problem.
- **mesh** (*Mesh*): Finite-element mesh.
- **time** (*TimeConfig*): Time stepping parameters.
- **adaptive_dt** (*AdaptiveDtConfig, optional*): Enable adaptive time stepping.  The pipeline must override ``adapt_dt(state, step, t, dt_old) -&gt; float`` for this to take effect.
- **output_dir** (*str, optional*): Write VTU snapshots and ``history.csv`` here.
- **callbacks** (*list of Callback, optional*): Extra callbacks invoked during the loop.


Returns
-------
TimeResult

