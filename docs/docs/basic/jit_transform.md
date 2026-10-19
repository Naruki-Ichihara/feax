# Just-in-Time Compilation Transform

This tutorial demonstrates the performance benefits of JAX's just-in-time (JIT) compilation for finite element solvers in FEAX. JIT compilation dramatically accelerates repeated solver calls by eliminating Python overhead and enabling XLA optimizations.

## Overview

JAX's `jax.jit` compiles functions to optimized machine code. For FEA solvers this provides:

- **Significant speedup** for repeated solver calls
- **One-time compilation overhead** on the first call
- **Cached compiled code** for subsequent calls with the same structure

The recommended pattern in FEAX has three parts:

1. Build a `fe.TracedStructure` from the problem **first**, and pass it both to
   `fe.create_solver` and to every solver call.
2. Keep everything that varies — material parameters (`TracedParams`), boundary
   values (`bc.replace_vals`), the initial guess — as **arguments** of the jitted
   function, not as closure constants.
3. Wrap the **call** to the solver in `jax.jit` (a small function or a lambda),
   rather than `jax.jit`-ing the object returned by `create_solver`.

## Problem Setup

We use the same cantilever beam problem from the [Linear Elasticity](./linear_elasticity.md) tutorial:

```python
import feax as fe
import jax
import jax.numpy as np
import time

E        = 70e3
nu       = 0.3
traction = 1.
tol      = 1e-5

L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)

left  = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L,  tol)

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu      = E / (2. * (1. + nu))
            lmbda   = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
])
bc = bc_config.create_bc(problem)
```

## Traced Structure

`Problem` is a JAX pytree with **no dynamic leaves**: every mesh-sized array it
holds (quadrature geometry, CSR slot maps, Dirichlet masks, scatter maps) is
static structure. If a solver is traced under `jax.jit` straight from the
`Problem`, all of those arrays are captured as closure constants and baked into
the executable. XLA then constant-folds whole subgraphs of them at compile time,
which spikes host memory (on unified-memory devices such as GB10 this can take
the machine down), inflates compile time, and leaves one copy per mesh shape
pinned in the compilation cache.

`fe.TracedStructure` is the runtime-argument counterpart of `TracedParams`: it
carries the same arrays as **pytree leaves**, so they enter the compiled program
as traced inputs. Build it right after the problem and boundary conditions, and
pass it to `create_solver` *and* to every call:

```python
ts = fe.TracedStructure.from_problem(problem)
```

Building it also releases the large host-side slot maps held by the `Problem`,
so do it before creating the solver.

## Creating the Solver

```python
traction_array = fe.TracedParams.create_uniform_surface_var(problem, traction)
traced_params  = fe.TracedParams(volume_vars=(), surface_vars=[(traction_array,)])

solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(),
    linear=True,
    traced_params=traced_params,
    traced_structure=ts,
)

initial = fe.zero_like_initial_guess(problem, bc)
```

`create_solver` returns a differentiable callable (a `custom_vjp`) that can be
called eagerly or under `jax.jit`. There is no "jit solver" to construct; the
transformation is applied where the solver is **called**:

```python
# Eager (op-by-op) call — no compilation
sol_no_jit = solver(traced_params, initial, traced_structure=ts)

# Jitted call — everything that may change is an argument
@jax.jit
def solve(tp, ts):
    return solver(tp, initial, traced_structure=ts)

sol_jit = solve(traced_params, ts)
```

Compilation happens on the first call to `solve` and is cached for subsequent
calls with the same array shapes. Because `tp` and `ts` are arguments, a new
material field or a different problem with the **same shapes** reuses the
compiled executable instead of triggering a recompile.

:::note Direct solvers and pre-warming
With `DirectSolverOptions`, `create_solver` factorizes a sample Jacobian once
outside any trace (`prewarm=True`, the default). Keep it on whenever the solver
is called under `jit`, `vmap`, or `grad`: the direct backend's first-call
initialization must not see tracers. Only a single eager forward solve that is
never differentiated can set `prewarm=False` to skip that extra factorization.
:::

## Benchmarking

```python
# No JIT
start = time.perf_counter()
sol_no_jit = solver(traced_params, initial, traced_structure=ts).block_until_ready()
time_no_jit = time.perf_counter() - start

# JIT — first call (includes compilation)
start = time.perf_counter()
sol_jit_first = solve(traced_params, ts).block_until_ready()
time_jit_with_compile = time.perf_counter() - start

# JIT — second call (compiled, fast)
start = time.perf_counter()
sol_jit_second = solve(traced_params, ts).block_until_ready()
time_jit_compiled = time.perf_counter() - start

compile_overhead = time_jit_with_compile - time_jit_compiled
speedup = time_no_jit / time_jit_compiled
print(f"No JIT: {time_no_jit:.3f}s | JIT (1st): {time_jit_with_compile:.3f}s | JIT (2nd): {time_jit_compiled:.3f}s | Speedup: {speedup:.2f}x")
```

Use `.block_until_ready()` when benchmarking to account for JAX's asynchronous execution.

## Visualization

```python
import matplotlib.pyplot as plt

jax_blue   = '#5E94D4'
jax_orange = '#F26B38'

fig, ax = plt.subplots(figsize=(8, 6))

categories      = ['No JIT', 'JIT\n(1st call)', 'JIT\n(2nd call)']
execution_times = [time_no_jit, time_jit_compiled, time_jit_compiled]
compile_times   = [0, compile_overhead, 0]

x_pos = np.arange(len(categories))
ax.bar(x_pos, execution_times, color=jax_blue,   alpha=0.85, label='Execution',    edgecolor='white', linewidth=1.5)
ax.bar(x_pos, compile_times,   color=jax_orange, alpha=0.85, label='Compilation',  edgecolor='white', linewidth=1.5,
       bottom=execution_times)

for i, (exec_t, comp_t) in enumerate(zip(execution_times, compile_times)):
    total = exec_t + comp_t
    ax.text(i, total + 0.15, f'{total:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if exec_t > 0.5:
        ax.text(i, exec_t / 2, f'{exec_t:.2f}s', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    if comp_t > 0.5:
        ax.text(i, exec_t + comp_t / 2, f'+{comp_t:.2f}s', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax.text(0.98, 0.97, f'{speedup:.1f}× faster\nafter compilation',
        transform=ax.transAxes, ha='right', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=jax_blue, alpha=0.2, edgecolor=jax_blue, linewidth=2))

ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title(f'JIT Compilation Performance (Speedup: {speedup:.1f}×)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('jit_comparison.png', dpi=150, bbox_inches='tight')
```

![JIT Performance Comparison](/img/tutorials/jit_comparison.png)

## Save Solution

```python
import os

sol_unflat   = problem.unflatten_fn_sol_list(sol_jit_second)
displacement = sol_unflat[0]

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
fe.utils.save_sol(mesh=mesh, sol_file=os.path.join(data_dir, 'vtk/u_jit.vtu'),
                  point_infos=[("displacement", displacement)])
```

## Composing with `jax.grad` and `jax.vmap`

The same rule — jit the call, keep the varying data as arguments — extends to
the other transformations. In optimization loops, jit the **whole** objective
(filters, solve, response) rather than the solver alone:

```python
def compliance(tp, ts):
    u = solver(tp, initial, traced_structure=ts)   # fe.Solution
    return np.dot(u.dofs, u.dofs)

obj_and_grad = jax.jit(jax.value_and_grad(compliance))
c, dc = obj_and_grad(traced_params, ts)            # dc is a TracedParams

# Batch over boundary values: BC locations stay fixed, only the values vary
solve_batch = jax.jit(jax.vmap(
    lambda v: solver(traced_params, initial, bc=bc.replace_vals(v), traced_structure=ts)))
```

## Key Takeaways

1. **Build `fe.TracedStructure` first** and pass it to `create_solver` and to every call — it keeps mesh-sized arrays out of the compiled executable
2. **Jit the call, not `create_solver`**: `jax.jit(lambda tp, ts: solver(tp, initial, traced_structure=ts))`
3. **Keep `prewarm=True`** on `DirectSolverOptions` whenever the solver runs under `jit` / `vmap` / `grad`
4. **One-time compilation overhead** on the first call — quickly amortized, and reused across problems with the same shapes
5. **Combine with `jax.vmap` and `jax.grad`** by jitting the whole objective or batch function
6. **Use `.block_until_ready()`** when benchmarking asynchronous JAX computations

## Further Reading

- [Vectorization Transform](./vmap_transform.md) — combine JIT with `jax.vmap`
- [Nonlinear Hyperelasticity](./hyperelasticity.md) — JIT with Newton iterations
