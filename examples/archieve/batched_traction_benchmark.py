"""
Performance benchmark for batched traction forces using create_solver.
Compares different approaches to solving multiple problems with varying traction loads in parallel.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, DirichletBC, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh_gmsh
import time
import matplotlib.pyplot as plt

# Problem setup
E = 70e3  # Young's modulus
nu = 0.3  # Poisson's ratio

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_x, traction_y, traction_z):
            return np.array([traction_x, traction_y, traction_z])
        return [surface_map]

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(30, 30, 30, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

def zero_disp(point):
    return 0.0

# Create boundary conditions using dataclass approach
bc_config = DirichletBCConfig([
    # Fixed left face - all components to zero
    DirichletBCSpec(location=left, component='all', value=0.0)
])

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)

# Create solver
bc = bc_config.create_bc(problem)
solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_options, iter_num=1)

# Method 1: Solve with different traction values one by one
def solve_single_traction(traction_values):
    """Solve for each traction value individually"""
    solutions = []
    for traction in traction_values:
        # Create traction arrays for x, y, z components
        traction_x = InternalVars.create_uniform_surface_var(problem, traction[0])
        traction_y = InternalVars.create_uniform_surface_var(problem, traction[1])
        traction_z = InternalVars.create_uniform_surface_var(problem, traction[2])
        
        internal_vars = InternalVars(
            surface_vars=[(traction_x, traction_y, traction_z)]
        )
        sol = solver(internal_vars, zero_like_initial_guess(problem, bc))
        solutions.append(sol)
    return solutions

# Method 2: JIT compiled single solve
# Prepare initial guess outside
initial_guess_default = zero_like_initial_guess(problem, bc)

solve_single_jit = jax.jit(lambda tx, ty, tz: solver(
    InternalVars(surface_vars=[(
        InternalVars.create_uniform_surface_var(problem, tx),
        InternalVars.create_uniform_surface_var(problem, ty),
        InternalVars.create_uniform_surface_var(problem, tz)
    )]),
    initial_guess_default
))

def solve_single_traction_jit(traction_values):
    """Solve for each traction value with JIT"""
    solutions = []
    for traction in traction_values:
        sol = solve_single_jit(traction[0], traction[1], traction[2])
        solutions.append(sol)
    return solutions

# Method 3: Vectorized solve using vmap
def solve_batched_traction_vmap(traction_values):
    """Solve all traction values at once using vmap"""
    # Extract components
    tx_values = traction_values[:, 0]
    ty_values = traction_values[:, 1]
    tz_values = traction_values[:, 2]
    
    # Create vmapped solver
    def single_solve(tx, ty, tz):
        internal_vars = InternalVars(
            surface_vars=[(
                InternalVars.create_uniform_surface_var(problem, tx),
                InternalVars.create_uniform_surface_var(problem, ty),
                InternalVars.create_uniform_surface_var(problem, tz)
            )]
        )
        # Create initial guess: zeros with BC values
        initial_guess = np.zeros(problem.num_total_dofs_all_vars)
        initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
        return solver(internal_vars, initial_guess)
    
    solve_vmap = jax.jit(jax.vmap(single_solve))
    return solve_vmap(tx_values, ty_values, tz_values)

def run_benchmark(batch_sizes):
    results = {
        'for_loop': [],
        'for_loop_jit': [],
        'vmap': []
    }
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Create batch of traction values
        # Varying traction in different directions
        traction_values = np.zeros((batch_size, 3))
        traction_values = traction_values.at[:, 0].set(np.linspace(0.05, 0.15, batch_size))  # x-direction
        traction_values = traction_values.at[:, 1].set(np.linspace(0.0, 0.1, batch_size))    # y-direction
        traction_values = traction_values.at[:, 2].set(np.linspace(0.0, 0.05, batch_size))   # z-direction
        
        # 1. For loop (no JIT) - only for small batch sizes
        if batch_size <= 10:
            print("  Running for loop (no JIT)...")
            start_time = time.time()
            solutions_loop = solve_single_traction(traction_values)
            jax.block_until_ready(solutions_loop[-1])
            loop_time = time.time() - start_time
            results['for_loop'].append(loop_time)
            print(f"    Time: {loop_time:.4f}s")
        else:
            results['for_loop'].append(None)
        
        # 2. For loop (with JIT) - no additional warmup needed
        print("  Running for loop (with JIT)...")
        start_time = time.time()
        solutions_loop_jit = solve_single_traction_jit(traction_values)
        jax.block_until_ready(solutions_loop_jit[-1])
        loop_jit_time = time.time() - start_time
        results['for_loop_jit'].append(loop_jit_time)
        print(f"    Time: {loop_jit_time:.4f}s")
        
        # 3. Vmap - no additional warmup needed
        print("  Running vmap...")
        start_time = time.time()
        solutions_vmap = solve_batched_traction_vmap(traction_values)
        jax.block_until_ready(solutions_vmap)
        vmap_time = time.time() - start_time
        results['vmap'].append(vmap_time)
        print(f"    Time: {vmap_time:.4f}s")
        
        # Verify solutions are consistent
        if batch_size <= 10:
            # Check that all methods give similar results for the first solution
            sol_loop = solutions_loop[0]
            sol_jit = solutions_loop_jit[0]
            sol_vmap = solutions_vmap[0]
            
            diff_jit = np.max(np.abs(sol_loop - sol_jit))
            diff_vmap = np.max(np.abs(sol_loop - sol_vmap))
            print(f"  Solution consistency check:")
            print(f"    Max diff (loop vs jit): {diff_jit:.2e}")
            print(f"    Max diff (loop vs vmap): {diff_vmap:.2e}")
    
    return results

# Warmup phase - compile all functions with different batch sizes
print("Starting warmup phase...")
print("  Warming up for loop (no JIT)...")
warmup_traction = np.zeros((2, 3))
warmup_traction = warmup_traction.at[:, 0].set([0.1, 0.15])
warmup_solutions = solve_single_traction(warmup_traction[:1])
if warmup_solutions:
    jax.block_until_ready(warmup_solutions[0])

print("  Warming up for loop (JIT)...")
for size in [1, 2, 5]:
    _ = solve_single_jit(0.1, 0.0, 0.0)
    jax.block_until_ready(_)

print("  Warming up vmap...")
for size in [1, 2, 5, 10]:
    warmup_vals = np.zeros((size, 3))
    warmup_vals = warmup_vals.at[:, 0].set(0.1)
    _ = solve_batched_traction_vmap(warmup_vals)
    jax.block_until_ready(_)

print("Warmup complete!\n")

# Run benchmarks
print("Starting batched traction force benchmark using create_solver...")
batch_sizes = [1, 10, 50, 100, 150, 200]
results = run_benchmark(batch_sizes)

# Plot results
plt.figure(figsize=(10, 6))

# Plot for loop (no JIT) only where we have data
valid_indices = [i for i, val in enumerate(results['for_loop']) if val is not None]
if valid_indices:
    valid_batch_sizes = [batch_sizes[i] for i in valid_indices]
    valid_times = [results['for_loop'][i] for i in valid_indices]
    plt.plot(valid_batch_sizes, valid_times, 'o-', label='For loop (no JIT)', linewidth=2, markersize=8)

plt.plot(batch_sizes, results['for_loop_jit'], 's-', label='For loop (JIT)', linewidth=2, markersize=8)
plt.plot(batch_sizes, results['vmap'], '^-', label='Vmap', linewidth=2, markersize=8)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Batched Traction Forces - Performance Comparison (30x30x30 mesh)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('/workspace/batched_traction_benchmark.png', dpi=300, bbox_inches='tight')
print(f"\nBenchmark complete! Results saved to '/workspace/batched_traction_benchmark.png'")

# Print speedup factors
print("\nSpeedup factors:")
for i, batch_size in enumerate(batch_sizes):
    print(f"  Batch size {batch_size}:")
    if results['for_loop'][i] is not None:
        speedup_jit = results['for_loop'][i] / results['for_loop_jit'][i]
        speedup_vmap = results['for_loop'][i] / results['vmap'][i]
        print(f"    JIT speedup: {speedup_jit:.2f}x")
        print(f"    Vmap speedup: {speedup_vmap:.2f}x")
    
    speedup_vmap_vs_jit = results['for_loop_jit'][i] / results['vmap'][i]
    print(f"    Vmap vs JIT loop: {speedup_vmap_vs_jit:.2f}x")

# Additional analysis: scaling behavior
print("\nScaling analysis:")
print("  Batch size increase | JIT loop time ratio | Vmap time ratio")
for i in range(1, len(batch_sizes)):
    batch_ratio = batch_sizes[i] / batch_sizes[i-1]
    jit_ratio = results['for_loop_jit'][i] / results['for_loop_jit'][i-1]
    vmap_ratio = results['vmap'][i] / results['vmap'][i-1]
    print(f"  {batch_sizes[i-1]}->{batch_sizes[i]} ({batch_ratio:.1f}x) | {jit_ratio:.2f}x | {vmap_ratio:.2f}x")