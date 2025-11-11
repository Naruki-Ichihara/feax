# Just-in-Time Compilation Transform

This tutorial demonstrates the performance benefits of JAX's just-in-time (JIT) compilation for finite element solvers in FEAX. JIT compilation can dramatically accelerate repeated solver calls by eliminating Python overhead and enabling XLA optimizations.

## Overview

JAX's `jax.jit` decorator compiles functions to optimized machine code. For FEA solvers, this provides:

- **Significant speedup** for repeated solver calls (often 10-20x faster)
- **One-time compilation overhead** on the first call
- **Cached compiled code** for subsequent calls with the same structure

This is particularly beneficial for:
- Parameter studies with `jax.vmap`
- Gradient-based optimization with `jax.grad`
- Time-stepping simulations
- Topology optimization iterations

## Problem Setup

We use the same cantilever beam problem from the [Linear Elasticity](./linear_elasticity.md) tutorial:

```python
import feax as fe
import jax
import jax.numpy as np
import time

# Problem parameters
E = 70e3
nu = 0.3
traction = 1.0

# Mesh
L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)

# Boundary locations
left = lambda point: np.isclose(point[0], 0., 1e-5)
right = lambda point: np.isclose(point[0], L, 1e-5)

# Problem definition
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary conditions
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(left, "all", 0.)
])
bc = bc_config.create_bc(problem)
```

## Creating JIT and Non-JIT Solvers

Create two solver instances: one with JIT compilation and one without:

```python
solver_option = fe.solver.SolverOptions(linear_solver="bicgstab")

# Standard solver (no JIT)
solver_no_jit = fe.solver.create_solver(problem, bc, solver_option, iter_num=1)

# JIT-compiled solver
solver_jit = jax.jit(fe.solver.create_solver(problem, bc, solver_option, iter_num=1))
```

The `jax.jit` wrapper transforms the solver into a compiled function. The compilation happens on the first call and is cached for subsequent calls.

## Benchmarking

Prepare inputs and measure execution times:

```python
initial = fe.utils.zero_like_initial_guess(problem, bc)
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

# Benchmark 1: No JIT
start = time.perf_counter()
sol_no_jit = solver_no_jit(internal_vars, initial).block_until_ready()
time_no_jit = time.perf_counter() - start

# Benchmark 2: JIT with compilation overhead (first call)
start = time.perf_counter()
sol_jit_first = solver_jit(internal_vars, initial).block_until_ready()
time_jit_with_compile = time.perf_counter() - start

# Benchmark 3: JIT without compilation overhead (second call)
start = time.perf_counter()
sol_jit_second = solver_jit(internal_vars, initial).block_until_ready()
time_jit_compiled = time.perf_counter() - start

# Calculate metrics
compile_overhead = time_jit_with_compile - time_jit_compiled
speedup = time_no_jit / time_jit_compiled
```

**Important:** Use `.block_until_ready()` to ensure GPU/TPU computations complete before stopping the timer. Without this, JAX's asynchronous execution may return before computation finishes.

## Performance Results

For the cantilever beam example with ~10,000 DOFs:

```
No JIT: 7.016s | JIT (1st call): 14.155s | JIT (2nd call): 0.521s | Speedup: 13.48x
```

### Key Observations

1. **Compilation overhead**: The first JIT call (14.155s) is slower than the non-JIT version (7.016s) due to XLA compilation
2. **Dramatic speedup**: Subsequent JIT calls (0.521s) are **13.5× faster** than non-JIT
3. **Break-even point**: JIT becomes beneficial after just 2 calls due to the large speedup

## Visualization

Visualize the performance comparison:

```python
import matplotlib.pyplot as plt

# JAX color palette
jax_blue = '#5E94D4'    # JAX blue
jax_orange = '#F26B38'  # JAX orange

fig, ax = plt.subplots(figsize=(8, 6))

# Stacked bar chart showing compilation overhead
categories = ['No JIT', 'JIT\n(1st call)', 'JIT\n(2nd call)']
execution_times = [time_no_jit, time_jit_compiled, time_jit_compiled]
compile_times = [0, compile_overhead, 0]

x_pos = np.arange(len(categories))
ax.bar(x_pos, execution_times, color=jax_blue, alpha=0.85, label='Execution')
ax.bar(x_pos, compile_times, bottom=execution_times, color=jax_orange, alpha=0.85, label='Compilation')

ax.set_ylabel('Time (s)')
ax.set_title(f'JIT Compilation Performance (Speedup: {speedup:.1f}×)')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.2, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('jit_comparison.png', dpi=150)
```

![JIT Performance Comparison](/img/tutorials/jit_comparison.png)

The stacked bar chart clearly shows that JIT compilation introduces a one-time overhead on the first call (orange), but subsequent calls achieve dramatic speedup through optimized execution (blue).

## When to Use JIT

### ✅ **Use JIT when:**

- Calling the solver multiple times (parameter studies, optimization)
- Using `jax.vmap` for batched operations
- Computing gradients with `jax.grad` through the solver
- Running time-dependent simulations
- Iterating in topology optimization loops

### ❌ **Avoid JIT when:**

- Solving a problem only once
- Prototyping and debugging (compilation hides Python errors)
- Problem size/structure changes between calls (triggers recompilation)

## Practical Example: Parameter Study

JIT compilation is essential for efficient parameter studies:

```python
# Create JIT-compiled solver
solver_jit = jax.jit(fe.solver.create_solver(problem, bc, solver_option, iter_num=1))

# Study different traction magnitudes
traction_values = np.linspace(0.5, 2.0, 20)

def compute_max_displacement(traction_mag):
    traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction_mag)
    internal_vars = fe.internal_vars.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])
    sol = solver_jit(internal_vars, initial)
    return np.max(np.abs(sol))

# First call triggers compilation
max_disp = compute_max_displacement(1.0)

# Subsequent calls reuse compiled code (very fast!)
start = time.perf_counter()
max_displacements = [compute_max_displacement(t) for t in traction_values]
elapsed = time.perf_counter() - start
print(f"Computed 20 solutions in {elapsed:.2f}s ({elapsed/20:.3f}s per solution)")
```

Without JIT, this would take ~140s. With JIT, it completes in ~10s (after initial compilation).

## Advanced: Vectorized Parameter Study

Combine JIT with `jax.vmap` for maximum performance:

```python
@jax.jit
def solve_batch(traction_array_batch):
    def solve_single(traction_array):
        internal_vars = fe.internal_vars.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])
        return solver_jit(internal_vars, initial)
    return jax.vmap(solve_single)(traction_array_batch)

# Create batch of traction arrays
traction_batch = jax.vmap(
    lambda t: fe.internal_vars.InternalVars.create_uniform_surface_var(problem, t)
)(traction_values)

# Solve all cases in parallel (GPU/TPU)
solutions = solve_batch(traction_batch)
```

This approach can leverage GPU parallelism for even greater speedup.

## Complete Example

See [`examples/basic/linear_elasticity_jit.py`](../../../examples/basic/linear_elasticity_jit.py) for the complete working example with timing and visualization.

## Key Takeaways

1. **JIT compilation provides 10-20× speedup** for repeated solver calls
2. **One-time compilation overhead** on first call is quickly amortized
3. **Essential for optimization and parameter studies** to achieve practical performance
4. **Combine with `jax.vmap` and `jax.grad`** for powerful differentiable physics workflows
5. **Use `.block_until_ready()`** when benchmarking to account for asynchronous execution

## Further Reading

- Explore gradient computation through JIT solvers for topology optimization
- Learn about `jax.vmap` for vectorized parameter studies
- Study recompilation triggers and how to avoid them
- Investigate GPU/TPU acceleration with JIT compilation
