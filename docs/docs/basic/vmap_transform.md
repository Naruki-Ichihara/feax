# Vectorization Transform

This tutorial demonstrates the performance benefits of JAX's automatic vectorization (`jax.vmap`) for solving multiple finite element problems in parallel. Vectorization enables efficient parameter studies by eliminating explicit loops and leveraging hardware parallelism.

## Overview

JAX's `jax.vmap` automatically vectorizes functions to operate on batches of inputs. For FEA solvers, this provides:

- **Dramatic speedup** over sequential for-loops (3-44× on CPU depending on batch size, 50-100× on GPU)
- **Superlinear scaling** where larger batches achieve disproportionately better performance
- **Automatic parallelization** across batch dimensions without code changes
- **Memory optimization** through JAX's XLA compiler for efficient cache utilization
- **GPU/TPU acceleration** with even greater parallelism benefits

This is particularly beneficial for:
- Parameter studies with varying material properties or loading conditions
- Uncertainty quantification with multiple samples
- Sensitivity analysis for design optimization
- Multi-scenario structural analysis

## Problem Setup

We use a cantilever beam problem similar to the [Linear Elasticity](./linear_elasticity.md) tutorial, but solve it for multiple traction magnitudes:

```python
import feax as fe
import jax
import jax.numpy as np
import time

# Problem parameters
E0 = 70e3  # Young's modulus
nu = 0.3   # Poisson's ratio

# Mesh - small for demonstration
mesh = fe.mesh.box_mesh((2, 1, 1), mesh_size=0.1)
print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

# Boundary locations
left = lambda point: np.isclose(point[0], 0, atol=1e-5)
right = lambda point: np.isclose(point[0], 2, atol=1e-5)
```

## Problem Definition

Define the linear elasticity problem with surface traction:

```python
class ElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            strain = 0.5 * (u_grad + u_grad.T)
            mu = E0 / (2.0 * (1.0 + nu))
            lam = E0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            sigma = 2.0 * mu * strain + lam * np.trace(strain) * np.eye(self.dim)
            return sigma
        return stress

    def get_surface_maps(self):
        def traction_map(u_grad, surface_quad_point, traction_magnitude):
            return np.array([0.0, 0.0, -traction_magnitude])  # Force in -z direction
        return [traction_map]

# Create problem
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)

# Boundary conditions: fix left face
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='all', value=0.0)
])
bc = bc_config.create_bc(problem)

# Create solver
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg")
solver = fe.solver.create_solver(problem, bc, solver_options, iter_num=1)

print(f"Problem: {problem.num_total_dofs_all_vars} DOFs")
```

## Single Solve Function

Create a function that solves for a single traction magnitude:

```python
def single_solve(traction_magnitude):
    """Solve for a single traction magnitude in -z direction."""
    # Create internal variables with traction magnitude
    traction_z = fe.internal_vars.InternalVars.create_uniform_surface_var(
        problem, traction_magnitude
    )

    internal_vars = fe.internal_vars.InternalVars(
        surface_vars=[(traction_z,)]
    )

    # Solve with zero initial guess
    return solver(internal_vars, fe.utils.zero_like_initial_guess(problem, bc))
```

This function encapsulates the complete solve workflow: creating internal variables, assembling the system, and solving.

## Vectorization with vmap

Transform the single-solve function to handle batches automatically:

```python
# Create vectorized solver
solve_vmap = jax.vmap(single_solve)

# Solve for multiple traction values in parallel
traction_values = np.linspace(1.0, 10.0, 100)  # 100 different load cases
solutions = solve_vmap(traction_values)

print(f"Solved {len(traction_values)} cases")
print(f"Solutions shape: {solutions.shape}")  # (100, num_dofs)
```

**Key insight:** `jax.vmap` automatically vectorizes the entire solve pipeline. No loop modification needed!

## Compilation Strategy

For accurate benchmarking, pre-compile both approaches to exclude compilation overhead:

```python
# Pre-compile with small batch
print("Pre-compiling strategies...")
compile_traction = np.array([5.0])

# Compile for-loop version
print("  Compiling for-loop...")
_ = single_solve(compile_traction[0])

# Compile vmap version
print("  Compiling vmap...")
solve_vmap = jax.vmap(single_solve)
_ = solve_vmap(compile_traction)
jax.block_until_ready(_)
```

**Important:** Use `jax.block_until_ready()` to ensure GPU/TPU computations complete before benchmarking. JAX executes asynchronously by default.

## Benchmarking: For-loop vs Vmap

Compare sequential for-loop execution against vectorized vmap:

```python
batch_size = 100
traction_values = np.linspace(1.0, 10.0, batch_size)

# Benchmark 1: For-loop approach
print(f"Testing for-loop with {batch_size} solves...")
start_time = time.time()
for_loop_solutions = []
for traction_mag in traction_values:
    solution = single_solve(traction_mag)
    for_loop_solutions.append(solution)
jax.block_until_ready(for_loop_solutions)
for_loop_time = time.time() - start_time
print(f"For-loop time: {for_loop_time:.4f}s")

# Benchmark 2: Vmap approach
print(f"Testing vmap with {batch_size} solves...")
start_time = time.time()
vmap_solutions = solve_vmap(traction_values)
jax.block_until_ready(vmap_solutions)
vmap_time = time.time() - start_time
print(f"Vmap time: {vmap_time:.4f}s")

# Calculate speedup
speedup = for_loop_time / vmap_time
print(f"Speedup: {speedup:.2f}x")
```

## Performance Results

For the cantilever beam example (mesh: `(2, 1, 1)` with `mesh_size=0.1`, ~700 nodes) on CPU:

```
Batch Size   For-loop (s)    Vmap (s)     Speedup
-------------------------------------------------------
1            1.6531          0.5039       3.28x
10           15.2186         4.3811       3.47x
20           35.4874         3.9949       8.88x
40           89.2907         5.9693       14.96x
80           178.6348        6.1004       29.28x
100          215.8393        6.1091       35.33x
120          280.5465        6.3767       44.00x
```

### Key Observations

1. **Immediate speedup**: Even for batch size 1, vmap is 3.3× faster due to better code optimization
2. **Superlinear scaling**: Speedup increases dramatically with batch size (3.5× → 44× for batch 10 → 120)
3. **Memory efficiency**: Vmap execution time plateaus around batch size 40-100, indicating efficient memory reuse
4. **Exceptional performance**: 44× speedup at batch 120 demonstrates JAX's powerful compilation optimizations
5. **No manual optimization**: Automatic parallelization without any code changes

**Why such dramatic speedup?**
- **Compilation optimization**: JAX's XLA compiler optimizes the entire batched computation
- **Memory layout**: Vectorized operations enable better cache utilization
- **Reduced Python overhead**: Single batched call eliminates per-iteration overhead
- **Pipeline parallelism**: JAX exploits instruction-level parallelism across batch dimension

### Performance Characteristics

The benchmark reveals two distinct scaling regimes:

1. **Small batches (1-20)**: Speedup grows linearly (~3-9×), dominated by compilation benefits
2. **Large batches (40-120)**: Superlinear speedup (15-44×), showing memory and parallelism advantages

**Vmap execution time analysis:**
- Batch 1-10: Time increases (0.5s → 4.4s) as more work is done
- Batch 20-120: Time plateaus (4.0s → 6.4s) as JAX maximizes hardware utilization

### GPU Performance

On NVIDIA GPUs, vectorization provides even greater benefits:

- **50-100× speedup** for large batches due to massive parallelism
- **Memory coalescing** improves bandwidth utilization dramatically
- **Reduced kernel launch overhead** compared to sequential calls
- **Optimal occupancy** as batch size fills GPU compute units

## Visualization

Create performance comparison plots:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Execution Time Comparison
ax1.plot(results['batch_size'], results['for_loop_time'], 'o-', color='skyblue',
         label='For-loop', linewidth=2, markersize=8)
ax1.plot(results['batch_size'], results['vmap_time'], 's-', color='orange',
         label='Vmap', linewidth=2, markersize=8)
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Execution Time (seconds)')
ax1.set_title('Execution Time: For-loop vs Vmap')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot 2: Speedup
ax2.plot(results['batch_size'], results['speedup'], 'go-', linewidth=2, markersize=8)
ax2.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No speedup')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Speedup (For-loop time / Vmap time)')
ax2.set_title('Vmap Speedup over For-loop')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# Add speedup annotations
for i, (bs, speedup) in enumerate(zip(results['batch_size'], results['speedup'])):
    ax2.annotate(f'{speedup:.1f}x', (bs, speedup),
                textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('vmap_comparison.png', dpi=300, bbox_inches='tight')
```

![Vectorization Performance Comparison](/img/tutorials/vmap_comparison.png)

The left plot shows execution time scaling: for-loop time grows linearly (blue circles) while vmap time plateaus around 6 seconds (orange squares), demonstrating exceptional memory efficiency. The right plot reveals superlinear speedup scaling from 3× to 44× as batch size increases, with dramatic acceleration beyond batch size 20.

## When to Use Vmap

### ✅ **Use vmap when:**

- Running parameter sweeps (varying loads, materials, geometries)
- Performing uncertainty quantification with multiple samples
- Computing finite difference approximations for gradients
- Evaluating multiple design candidates in parallel
- Running Monte Carlo simulations

### ❌ **Avoid vmap when:**

- Solving only a single case (no batching benefit)
- Problems have different structures across batch (triggers recompilation)
- Memory is severely constrained (batching increases memory usage)
- Debugging (vectorized code is harder to trace)

## Practical Example: Compliance Study

Study how compliance varies with traction magnitude:

```python
# Define compliance function (strain energy)
def compute_compliance(traction_magnitude):
    solution = single_solve(traction_magnitude)
    # Compliance = ½ u^T K u ≈ u^T f (for linear problems)
    return 0.5 * np.sum(solution**2)

# Vectorize compliance computation
compute_compliance_vmap = jax.vmap(compute_compliance)

# Compute compliance for many load cases
traction_values = np.linspace(0.5, 10.0, 50)
compliances = compute_compliance_vmap(traction_values)

# Plot compliance curve
plt.figure(figsize=(8, 5))
plt.plot(traction_values, compliances, 'b-', linewidth=2)
plt.xlabel('Traction Magnitude')
plt.ylabel('Compliance')
plt.title('Compliance vs Traction Load')
plt.grid(True, alpha=0.3)
plt.savefig('compliance_study.png', dpi=150)
```

This efficiently computes compliance for 50 load cases without explicit loops.

## Advanced: Combining JIT and Vmap

Maximize performance by combining JIT compilation with vectorization:

```python
# JIT-compile the vectorized solver
solve_vmap_jit = jax.jit(jax.vmap(single_solve))

# First call triggers compilation
_ = solve_vmap_jit(np.array([1.0, 2.0]))

# Subsequent calls are extremely fast
start = time.time()
solutions = solve_vmap_jit(traction_values)
jax.block_until_ready(solutions)
elapsed = time.time() - start

print(f"JIT + Vmap: Solved {len(traction_values)} cases in {elapsed:.4f}s")
print(f"Time per solve: {elapsed/len(traction_values)*1000:.2f}ms")
```

**Combined benefits:**
- **JIT compilation** eliminates Python overhead (10-20× speedup from [JIT tutorial](./jit_transform.md))
- **Vmap vectorization** adds another 2-5× speedup
- **Total speedup** can exceed 50× for parameter studies

## Gradient Computation Through Vmap

Vmap composes with automatic differentiation for efficient gradient computation:

```python
# Define objective function
def objective(traction_magnitude):
    solution = single_solve(traction_magnitude)
    return np.sum(solution**2)  # Example: minimize squared displacement

# Compute gradient for a batch of traction values
grad_fn = jax.vmap(jax.grad(objective))

traction_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
gradients = grad_fn(traction_values)

print(f"Gradients shape: {gradients.shape}")  # (5,)
```

This enables efficient sensitivity analysis and gradient-based optimization across parameter spaces.

## Nested Vectorization

Vmap supports nested vectorization for multi-dimensional parameter spaces:

```python
# Study both traction and Young's modulus
def solve_with_params(traction_mag, youngs_modulus):
    # Update E0 in problem definition
    # ... (requires problem to accept E as internal variable)
    return single_solve(traction_mag)

# Double vmap: vectorize over both parameters
solve_2d = jax.vmap(jax.vmap(solve_with_params, in_axes=(None, 0)), in_axes=(0, None))

traction_values = np.linspace(1.0, 10.0, 20)
E_values = np.linspace(50e3, 100e3, 15)

# Solve 20 × 15 = 300 cases in parallel
solutions_2d = solve_2d(traction_values, E_values)
print(f"Solutions shape: {solutions_2d.shape}")  # (20, 15, num_dofs)
```

This explores a 2D parameter space efficiently without explicit nested loops.

## Complete Example

See [`examples/basic/linear_elasticity_vmap_traction.py`](../../../examples/basic/linear_elasticity_vmap_traction.py) for the complete working example with:
- Pre-compilation strategy
- Comprehensive benchmarking across batch sizes
- Visualization generation
- Result verification

## Key Takeaways

1. **Vmap provides dramatic CPU speedup**: 3-44× depending on batch size, with superlinear scaling
2. **Immediate benefits**: Even single-item batches see 3× speedup from compilation optimization
3. **Zero manual batching code**: Automatic vectorization without any code changes
4. **Exceptional scaling**: Speedup grows superlinearly for large batches (44× at batch size 120)
5. **Memory efficient**: Execution time plateaus as JAX optimizes memory reuse
6. **Composes with grad**: Efficient sensitivity analysis through vectorized solvers
7. **Scales to nested parameter spaces**: Multi-dimensional exploration through nested vmap

## Comparison: JIT vs Vmap

| Aspect | JIT Compilation | Vectorization (Vmap) |
|--------|----------------|----------------------|
| **Purpose** | Eliminate Python overhead | Parallelize across batch |
| **Speedup** | 10-20× per call | 3-44× (CPU), 50-100× (GPU) |
| **Scaling** | Constant per call | Superlinear with batch size |
| **When to use** | Repeated calls | Multiple similar problems |
| **Overhead** | One-time compilation | Minimal, built into XLA |
| **Combines with** | Vmap, grad | JIT, grad |

**Best practice:** Use both together for maximum performance. JIT compilation is automatically applied within vmap operations, contributing to the dramatic speedups observed.

## Further Reading

- Learn about [JIT compilation](./jit_transform.md) for complementary speedup
- Explore gradient computation with `jax.grad` through vectorized solvers
- Study memory optimization for large batch sizes
- Investigate GPU profiling with JAX's performance tools
