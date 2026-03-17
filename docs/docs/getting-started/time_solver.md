# Time Solver Guide

This page explains how to use the time-stepping infrastructure in `feax.solvers.time_solver` for transient and quasi-static problems.

## Architecture

The time solver provides a **Pipeline** pattern — an abstract base class that separates problem definition from the time loop. You implement a few methods; the `run()` function handles the rest (time loop, output, logging, adaptive dt).

Three pipeline levels are available, from most flexible to most automated:

| Pipeline | What you implement | Time integration | Use case |
|---|---|---|---|
| `TimePipeline` | `step()` | Full control | Staggered multi-physics |
| `ImplicitPipeline` | `update_vars()` | Backward Euler (one `solver()` call per step) | Most transient problems |
| `ExplicitPipeline` | `compute_rhs()` | Forward Euler / RK2 / RK4 | Wave propagation, explicit dynamics |

```python
from feax.solvers.time_solver import ImplicitPipeline, TimeConfig, run
```

## `ImplicitPipeline`

The most common choice. Each time step solves one (non)linear system:

1. `update_vars(state, t, dt)` → `InternalVars` (pack the previous solution)
2. `self.solver(iv, state)` → new state

### Minimal Example: Heat Equation

```python
from feax.solvers.time_solver import ImplicitPipeline, TimeConfig, run

class HeatPipeline(ImplicitPipeline):

    def build(self, mesh):
        self.problem = HeatProblem(mesh, vec=1, dim=2, ele_type='QUAD4')
        bc = fe.DirichletBCConfig([...]).create_bc(self.problem)
        self.solver = fe.create_solver(self.problem, bc,
            solver_options=fe.DirectSolverOptions(),
            iter_num=1, internal_vars=fe.InternalVars(volume_vars=(T0,)))

    def initial_state(self):
        return T0_flat  # flat solution vector

    def update_vars(self, state, t, dt):
        T_old = self.problem.unflatten_fn_sol_list(state)[0][:, 0]
        return fe.InternalVars(volume_vars=(T_old,))

result = run(HeatPipeline(), mesh, TimeConfig(dt=1e-5, t_end=1e-2))
```

### Methods to Implement

| Method | Required | Purpose |
|---|---|---|
| `build(mesh)` | Yes | Create problem, BCs, solver. Called once before the time loop. |
| `initial_state()` | Yes | Return the initial flat solution vector (or any JAX pytree). |
| `update_vars(state, t, dt)` | Yes | Pack the current state into `InternalVars` for the implicit solve. |
| `save(state, step, t, output_dir)` | No | Write VTK or other output files. |
| `monitor(state, step, t)` | No | Return `{name: value}` dict for logging. |

### How Backward Euler Works

The transient term (e.g., $\partial T / \partial t$) is discretized as $(T^{n+1} - T^n) / \Delta t$ inside the weak form. The "old" value $T^n$ is passed to the solver as an `InternalVars` entry. FEAX then solves the nonlinear system for $T^{n+1}$.

In the Problem class, the weak form receives the old value as an extra argument:

```python
class HeatProblem(fe.problem.Problem):
    def get_weak_form(self):
        def weak_form(vals, grads, x, T_old):
            T = vals[0]
            grad_T = grads[0]
            R_mass = (T - T_old) / dt      # transient term
            R_grad = kappa * grad_T         # diffusion
            return ([R_mass], [R_grad])
        return weak_form
```

The `step()` method is provided automatically:

```python
def step(self, state, t, dt):
    iv = self.update_vars(state, t, dt)
    return self.solver(iv, state)
```

## `ExplicitPipeline`

For problems where an explicit time integrator is preferred (wave propagation, explicit dynamics). No linear system solve per step — only matrix-vector products and element-wise operations.

### Schemes

| Scheme | Order | Set via |
|---|---|---|
| `'euler'` | 1st (Forward Euler) | `scheme = 'euler'` (default) |
| `'rk2'` | 2nd (Midpoint RK2) | `scheme = 'rk2'` |
| `'rk4'` | 4th (Classic RK4) | `scheme = 'rk4'` |

### Example

```python
from feax.solvers.time_solver import ExplicitPipeline, TimeConfig, run

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

result = run(WavePipeline(), mesh, TimeConfig(dt=1e-7, t_end=1e-3))
```

The state can be any JAX pytree (dict, list, flat array). The `compute_rhs` return must have the same structure.

## `TimePipeline`

The base class for full control. Implement `step()` directly. Use this for staggered multi-physics where you need multiple solves per step or custom logic:

```python
from feax.solvers.time_solver import TimePipeline

class StaggaredPipeline(TimePipeline):

    def build(self, mesh):
        self.thermal_solver = ...
        self.mech_solver = ...

    def initial_state(self):
        return {'T': T0, 'u': u0}

    def step(self, state, t, dt):
        # Thermal solve
        iv_thermal = fe.InternalVars(volume_vars=(state['T'],))
        T_new = self.thermal_solver(iv_thermal, state['T'])

        # Mechanical solve with updated temperature
        iv_mech = fe.InternalVars(volume_vars=(T_new,))
        u_new = self.mech_solver(iv_mech, state['u'])

        return {'T': T_new, 'u': u_new}
```

## `TimeConfig`

Controls the time loop parameters:

```python
TimeConfig(
    dt=5e-6,          # time step size
    t_end=2.5e-4,     # final simulation time
    t_start=0.0,      # start time (default 0)
    save_every=10,     # VTK output interval (steps)
    print_every=1,     # console log interval (steps)
)
```

## `run()`

The main entry point that drives the time loop:

```python
result = run(
    pipeline,                # TimePipeline instance
    mesh,                    # fe.mesh.Mesh
    TimeConfig(dt=..., t_end=...),
    output_dir='data/vtk',   # optional: write VTK + history.csv
)
```

`run()` handles:

1. Calls `pipeline.build(mesh)` once
2. Gets `pipeline.initial_state()`
3. Saves initial condition (step 0) if `output_dir` is set
4. Loops: `state = pipeline.step(state, t, dt)` until `t >= t_end`
5. Calls `pipeline.monitor()` and `pipeline.save()` at configured intervals
6. Writes `history.csv` with all monitor values
7. Returns `TimeResult`

### `TimeResult`

```python
result.final_state    # state at the end
result.history        # {'step': [...], 'time': [...], 'c_min': [...], ...}
result.t_final        # actual final time
result.n_steps        # total steps taken
```

## Adaptive Time Stepping

Enable adaptive $\Delta t$ by passing `AdaptiveDtConfig` and overriding `adapt_dt` in your pipeline:

```python
from feax.solvers.time_solver import AdaptiveDtConfig

class MyPipeline(ImplicitPipeline):
    ...

    def adapt_dt(self, state, step, t, dt_old):
        # Return desired new dt (will be clamped to [dt_min, dt_max])
        error = compute_error_estimate(state)
        return dt_old * (tol / error) ** 0.5

result = run(
    MyPipeline(), mesh,
    TimeConfig(dt=1e-5, t_end=1e-2),
    adaptive_dt=AdaptiveDtConfig(dt_min=1e-8, dt_max=1e-3, growth_factor=2.0),
)
```

## Callbacks

Add custom logic at configurable intervals:

```python
from feax.solvers.time_solver import Callback

def check_energy(state, step, t):
    print(f"  Energy at step {step}: {compute_energy(state):.6e}")

result = run(
    pipeline, mesh, time_config,
    callbacks=[Callback(fn=check_energy, every=10)],
)
```

## Summary

| Component | Purpose |
|---|---|
| `ImplicitPipeline` | Backward Euler: implement `update_vars()` |
| `ExplicitPipeline` | Forward Euler / RK2 / RK4: implement `compute_rhs()` |
| `TimePipeline` | Full control: implement `step()` |
| `TimeConfig` | Time step size, end time, output intervals |
| `run()` | Drives the time loop, handles I/O and logging |
| `TimeResult` | Final state, history dict, step count |
