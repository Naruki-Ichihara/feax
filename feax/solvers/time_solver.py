"""Time-stepping solver with Pipeline interface.

Provides a ``TimePipeline`` abstract class and ``run()`` function that
mirror the topology-optimisation driver (``feax.gene.optimizer``) but
for transient / quasi-static problems.

Three pipeline levels:

* **TimePipeline** — full control: implement ``step()`` directly.
  Use for staggered multi-physics (thermal + phase-field + mechanics).
* **ImplicitPipeline** — one implicit solve per step (backward Euler
  pattern).  Implement ``update_vars()``; ``step()`` calls
  ``self.solver(iv, state)`` automatically.
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
        return fe.InternalVars(volume_vars=(T_old,))

result = run(HeatPipeline(), mesh, TimeConfig(dt=1e-5, t_end=1e-2))
```
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as onp

from feax.mesh import Mesh


# ---------------------------------------------------------------------------
# TimePipeline abstract base class
# ---------------------------------------------------------------------------

class TimePipeline(ABC):
    """Abstract base class for time-stepping pipelines.

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
    """

    @abstractmethod
    def build(self, mesh: Mesh) -> None:
        """Create mesh-dependent objects (problems, BCs, solvers).

        Called once before the time loop starts.  Store results as
        instance attributes for use in :meth:`step`, :meth:`save`,
        and :meth:`monitor`.
        """

    @abstractmethod
    def initial_state(self) -> Any:
        """Return the initial state (any JAX-compatible pytree).

        Called once after :meth:`build`.  The returned object is passed
        as ``state`` to the first call of :meth:`step`.

        Examples::

            # Single-field problem: flat solution vector
            return jnp.zeros(problem.num_total_dofs_all_vars)

            # Multi-physics staggered: dict of fields
            return {'T_flat': T0, 'u_flat': u0, 'history': H0}
        """

    @abstractmethod
    def step(self, state: Any, t: float, dt: float) -> Any:
        """Advance one time step and return the new state.

        Parameters
        ----------
        state
            Current state (same type returned by :meth:`initial_state`).
        t : float
            Time at the *beginning* of this step.
        dt : float
            Time step size.

        Returns
        -------
        new_state
            Updated state at time ``t + dt``.
        """

    def save(self, state: Any, step: int, t: float, output_dir: str) -> None:
        """Save a snapshot (VTU, etc.) to *output_dir*.

        Override to write problem-specific output.  The default is a no-op.

        Parameters
        ----------
        state
            Current state.
        step : int
            Time step index (0 = initial condition).
        t : float
            Current simulation time.
        output_dir : str
            Directory for output files (already created by ``run``).
        """

    def monitor(self, state: Any, step: int, t: float) -> Dict[str, float]:
        """Return scalar quantities to log each step.

        Override to report problem-specific diagnostics.

        Returns
        -------
        dict
            ``{name: value}`` of scalar quantities.  Printed to stdout
            and written to ``history.csv`` when *output_dir* is given.
        """
        return {}


# ---------------------------------------------------------------------------
# ImplicitPipeline — single implicit solve per step
# ---------------------------------------------------------------------------

class ImplicitPipeline(TimePipeline):
    """Implicit time stepping with a single solver call per step.

    Covers the common backward-Euler pattern where each time step
    solves one (non)linear system:

    1. ``update_vars(state, t, dt)`` → ``InternalVars``
    2. ``self.solver(iv, state)`` → new state

    Set ``self.solver`` in :meth:`build` and implement
    :meth:`update_vars`.

    For **staggered multi-physics** (multiple solves per step),
    subclass :class:`TimePipeline` directly and override ``step()``.

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
                return fe.InternalVars(volume_vars=(c_old,))
    """

    @abstractmethod
    def update_vars(self, state: Any, t: float, dt: float):
        """Prepare ``InternalVars`` for the implicit solve.

        Parameters
        ----------
        state
            Current state (solution at time *t*).
        t : float
            Time at the beginning of this step.
        dt : float
            Time step size.

        Returns
        -------
        feax.InternalVars
            Internal variables encoding the transient term
            (e.g. ``T_old`` for backward Euler).
        """

    def step(self, state: Any, t: float, dt: float) -> Any:
        """Solve the implicit system for this time step."""
        iv = self.update_vars(state, t, dt)
        return self.solver(iv, state)


# ---------------------------------------------------------------------------
# ExplicitPipeline — explicit ODE integration
# ---------------------------------------------------------------------------

class ExplicitPipeline(TimePipeline):
    """Explicit time stepping for du/dt = f(u, t).

    No linear system solve is needed per step — only matrix-vector
    products and element-wise division by a lumped (diagonal) mass
    matrix.  This makes each step very cheap, but the time step ``dt``
    is restricted by a CFL-like stability condition.

    Implement :meth:`compute_rhs` to return f(u, t).  The integration
    scheme is selected by setting ``self.scheme`` (default ``'euler'``):

    * ``'euler'`` — Forward Euler  (1st order)
    * ``'rk2'``   — Midpoint RK2   (2nd order)
    * ``'rk4'``   — Classic RK4    (4th order)

    Set ``self.scheme`` in :meth:`build` or at class level.

    For specialised explicit schemes (Velocity Verlet, central
    difference, symplectic integrators), override ``step()`` directly.

    Example::

        class WavePipeline(ExplicitPipeline):
            scheme = 'rk4'

            def build(self, mesh):
                self.M_inv = ...  # lumped mass inverse
                self.K = ...      # stiffness matrix

            def initial_state(self):
                return {'u': u0, 'v': v0}

            def compute_rhs(self, state, t):
                a = self.M_inv @ (-self.K @ state['u'] + f_ext(t))
                return {'u': state['v'], 'v': a}
    """

    scheme: str = 'euler'

    @abstractmethod
    def compute_rhs(self, state: Any, t: float) -> Any:
        """Compute the right-hand side f(u, t) for du/dt = f(u, t).

        Parameters
        ----------
        state
            Current state.
        t : float
            Current time.

        Returns
        -------
        rhs
            Same pytree structure as *state*.
        """

    def step(self, state: Any, t: float, dt: float) -> Any:
        """Advance one step using the selected explicit scheme."""
        import jax.tree_util as jtu

        def _axpy(a, x, y):
            """y + a * x  (pytree-aware)."""
            return jtu.tree_map(lambda xi, yi: yi + a * xi, x, y)

        if self.scheme == 'euler':
            k1 = self.compute_rhs(state, t)
            return _axpy(dt, k1, state)

        elif self.scheme == 'rk2':
            k1 = self.compute_rhs(state, t)
            s_mid = _axpy(0.5 * dt, k1, state)
            k2 = self.compute_rhs(s_mid, t + 0.5 * dt)
            return _axpy(dt, k2, state)

        elif self.scheme == 'rk4':
            k1 = self.compute_rhs(state, t)
            k2 = self.compute_rhs(_axpy(0.5 * dt, k1, state), t + 0.5 * dt)
            k3 = self.compute_rhs(_axpy(0.5 * dt, k2, state), t + 0.5 * dt)
            k4 = self.compute_rhs(_axpy(dt, k3, state), t + dt)
            # state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            rhs_sum = jtu.tree_map(
                lambda a, b, c, d: a + 2 * b + 2 * c + d, k1, k2, k3, k4)
            return _axpy(dt / 6.0, rhs_sum, state)

        else:
            raise ValueError(
                f"Unknown scheme {self.scheme!r}. "
                f"Choose from 'euler', 'rk2', 'rk4'.")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimeConfig:
    """Time-stepping configuration.

    Parameters
    ----------
    dt : float
        Time step size.
    t_end : float
        End time of the simulation.
    t_start : float
        Start time (default 0).
    save_every : int
        Call ``pipeline.save()`` every N steps (and at step 0 and final).
    print_every : int
        Print monitor output every N steps.
    """
    dt: float
    t_end: float
    t_start: float = 0.0
    save_every: int = 10
    print_every: int = 1


@dataclass
class AdaptiveDtConfig:
    """Configuration for adaptive time stepping.

    At each step the pipeline's :meth:`~TimePipeline.adapt_dt` method
    (if overridden) can propose a new ``dt``.  The value is clamped
    between ``dt_min`` and ``dt_max``.

    Parameters
    ----------
    dt_min : float
        Minimum allowed time step.
    dt_max : float
        Maximum allowed time step.
    growth_factor : float
        Maximum factor by which dt can grow in one step.
    """
    dt_min: float = 1e-12
    dt_max: float = 1e30
    growth_factor: float = 2.0


@dataclass
class TimeResult:
    """Time integration result.

    Attributes
    ----------
    final_state
        State at the end of the simulation.
    history : dict
        ``{'step': [...], 'time': [...], **monitor_keys}``.
    t_final : float
        Actual final time reached.
    n_steps : int
        Total number of time steps taken.
    """
    final_state: Any
    history: Dict[str, list]
    t_final: float
    n_steps: int


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------

@dataclass
class Callback:
    """User callback invoked at configurable frequency.

    Parameters
    ----------
    fn : callable
        ``fn(state, step, t)`` — called for its side effects.
    every : int
        Call every N steps (default 1 = every step).
    """
    fn: Callable[[Any, int, float], None]
    every: int = 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    pipeline: TimePipeline,
    mesh: Mesh,
    time: TimeConfig,
    adaptive_dt: Optional[AdaptiveDtConfig] = None,
    output_dir: Optional[str] = None,
    callbacks: Optional[List[Callback]] = None,
) -> TimeResult:
    """Run a time-stepping simulation.

    Parameters
    ----------
    pipeline : TimePipeline
        Pipeline instance defining the transient problem.
    mesh : Mesh
        Finite-element mesh.
    time : TimeConfig
        Time stepping parameters.
    adaptive_dt : AdaptiveDtConfig, optional
        Enable adaptive time stepping.  The pipeline must override
        ``adapt_dt(state, step, t, dt_old) -> float`` for this to
        take effect.
    output_dir : str, optional
        Write VTU snapshots and ``history.csv`` here.
    callbacks : list of Callback, optional
        Extra callbacks invoked during the loop.

    Returns
    -------
    TimeResult
    """
    callbacks = callbacks or []

    # -- Build ----------------------------------------------------------------
    pipeline.build(mesh)
    state = pipeline.initial_state()

    # -- Discover monitor keys from a trial call ------------------------------
    trial_monitor = pipeline.monitor(state, 0, time.t_start)
    monitor_keys = list(trial_monitor.keys())

    history: Dict[str, list] = {
        'step': [], 'time': [],
        **{k: [] for k in monitor_keys},
    }

    # -- File output ----------------------------------------------------------
    csv_file = csv_writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_file = open(
            os.path.join(output_dir, 'history.csv'), 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'time'] + monitor_keys)
        csv_file.flush()

    # -- Record & save initial condition --------------------------------------
    def _record(state, step, t):
        vals = pipeline.monitor(state, step, t)
        history['step'].append(step)
        history['time'].append(t)
        for k in monitor_keys:
            history[k].append(vals.get(k, 0.0))
        return vals

    def _print_step(step, t, dt, vals):
        mon_str = '  '.join(f'{k}={v:.4e}' for k, v in vals.items())
        print(f"  step {step:5d}  t={t:.4e}  dt={dt:.4e}  {mon_str}")

    vals = _record(state, 0, time.t_start)

    # -- Header ---------------------------------------------------------------
    n_steps_est = int((time.t_end - time.t_start) / time.dt)
    print("Starting time integration")
    print(f"  t          : [{time.t_start}, {time.t_end}]")
    print(f"  dt         : {time.dt:.4e}" + (
        f"  (adaptive: [{adaptive_dt.dt_min:.2e}, {adaptive_dt.dt_max:.2e}])"
        if adaptive_dt else ""))
    print(f"  est. steps : {n_steps_est}")
    if monitor_keys:
        print(f"  monitor    : {', '.join(monitor_keys)}")
    if output_dir:
        print(f"  output     : {output_dir}")
    print("-" * 60)

    if output_dir:
        pipeline.save(state, 0, time.t_start, output_dir)

    # -- Time loop ------------------------------------------------------------
    t = time.t_start
    dt = time.dt
    step = 0

    while t < time.t_end - 1e-14 * time.dt:
        # Clamp dt so we land exactly on t_end
        dt_eff = min(dt, time.t_end - t)

        step += 1
        state = pipeline.step(state, t, dt_eff)
        t += dt_eff

        # Monitor & log
        vals = _record(state, step, t)

        if step % time.print_every == 0:
            _print_step(step, t, dt_eff, vals)

        # CSV
        if csv_writer:
            csv_writer.writerow(
                [step, t] + [vals.get(k, 0.0) for k in monitor_keys])
            csv_file.flush()

        # Save snapshot
        if output_dir and (step % time.save_every == 0):
            pipeline.save(state, step, t, output_dir)

        # Callbacks
        for cb in callbacks:
            if step % cb.every == 0:
                cb.fn(state, step, t)

        # Adaptive dt
        if adaptive_dt and hasattr(pipeline, 'adapt_dt'):
            dt_new = pipeline.adapt_dt(state, step, t, dt)
            dt_new = max(adaptive_dt.dt_min, min(adaptive_dt.dt_max, dt_new))
            dt_new = min(dt_new, dt * adaptive_dt.growth_factor)
            dt = dt_new

    # -- Final snapshot -------------------------------------------------------
    if output_dir and (step % time.save_every != 0):
        pipeline.save(state, step, t, output_dir)

    # -- Summary --------------------------------------------------------------
    print("-" * 60)
    print(f"Completed: {step} steps, t_final={t:.6e}")
    if vals:
        for k, v in vals.items():
            print(f"  {k:16s}: {v:.6e}")

    if csv_file:
        csv_file.close()

    return TimeResult(
        final_state=state,
        history=history,
        t_final=t,
        n_steps=step,
    )
