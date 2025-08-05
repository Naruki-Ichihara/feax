"""
Simple batched linear elasticity example using vmap for parallel traction solving.

Demonstrates:
1. Single solve function for one traction value
2. Using jax.vmap to solve multiple traction values in parallel
3. Compatibility with the new solver design
"""

import jax
import jax.numpy as np
import time
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh_gmsh
from feax.utils import save_sol
import os
import matplotlib.pyplot as plt

# Problem setup
E0 = 70e3
nu = 0.3
batch_sizes = [1, 10, 30, 50, 70, 100, 120, 150, 200]

class ElasticityProblem(Problem):
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
            return np.array([0.0, 0.0, -traction_magnitude])  # Only -z direction
        return [traction_map]

# Create mesh (small for demo)
print("Creating mesh...")
meshio_mesh = box_mesh_gmsh(40, 20, 20, 2., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

# Boundary locations
def left(point):
    return np.isclose(point[0], 0, atol=1e-5)

def right(point):
    return np.isclose(point[0], 2, atol=1e-5)

# Boundary conditions: fix left face completely
bc_config = DirichletBCConfig([
    DirichletBCSpec(location=left, component='all', value=0.0)
])

# Create problem and solver
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)

bc = bc_config.create_bc(problem)
solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_options, iter_num=1)

print(f"Problem: {problem.num_total_dofs_all_vars} DOFs")

# Single solve function for one traction magnitude (always -z direction)
def single_solve(traction_magnitude):
    """Solve for a single traction magnitude in -z direction."""
    # Only apply traction in -z direction (traction_magnitude is positive, but force is -z)
    traction_z = InternalVars.create_uniform_surface_var(problem, traction_magnitude)
    
    internal_vars = InternalVars(
        surface_vars=[(traction_z, )]
    )
    
    # Solve with zero initial guess
    return solver(internal_vars, zero_like_initial_guess(problem, bc))

# Benchmark Setup
print("\n=== Batched Benchmark Test ===")

results = {'batch_size': [], 'for_loop_time': [], 'vmap_time': [], 'speedup': []}

# Pre-compile both strategies with small batch to avoid compilation overhead in benchmark
print("Pre-compiling strategies...")
compile_traction = np.array([5.0])
solve_vmap = jax.vmap(single_solve)

# Compile for-loop version
print("  Compiling for-loop...")
_ = single_solve(compile_traction[0])

# Compile vmap version  
print("  Compiling vmap...")
_ = solve_vmap(compile_traction)
jax.block_until_ready(_)

print("Compilation completed!\n")

# Run benchmarks for each batch size
for batch_size in batch_sizes:
    print(f"=== Batch Size: {batch_size} ===")
    
    # Create traction values for this batch size
    traction_values = np.linspace(1.0, 10.0, batch_size)
    
    # Benchmark 1: For-loop approach
    print(f"  Testing for-loop with {batch_size} solves...")
    start_time = time.time()
    for_loop_solutions = []
    for traction_mag in traction_values:
        solution = single_solve(traction_mag)
        for_loop_solutions.append(solution)
    jax.block_until_ready(for_loop_solutions)
    for_loop_time = time.time() - start_time
    print(f"  For-loop time: {for_loop_time:.4f}s")
    
    # Benchmark 2: Vmap approach
    print(f"  Testing vmap with {batch_size} solves...")
    start_time = time.time()
    vmap_solutions = solve_vmap(traction_values)
    jax.block_until_ready(vmap_solutions)
    vmap_time = time.time() - start_time
    print(f"  Vmap time: {vmap_time:.4f}s")
    
    # Calculate speedup
    speedup = for_loop_time / vmap_time
    print(f"  Speedup: {speedup:.2f}x")
    
    # Verify results match
    if batch_size <= 10:  # Only verify for smaller batches to save time
        diffs = [np.max(np.abs(s - v)) for s, v in zip(for_loop_solutions, vmap_solutions)]
        max_diff = max(diffs)
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  ✅ Results match!" if max_diff < 1e-10 else "  ❌ Results differ!")
    
    # Store results
    results['batch_size'].append(batch_size)
    results['for_loop_time'].append(for_loop_time)
    results['vmap_time'].append(vmap_time)
    results['speedup'].append(speedup)
    
    print()

# Create benchmark plots
print("=== Creating Benchmark Plots ===")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Execution Time Comparison
ax1.plot(results['batch_size'], results['for_loop_time'], 'o-', color='skyblue', label='For-loop', linewidth=2, markersize=8)
ax1.plot(results['batch_size'], results['vmap_time'], 's-', color='orange', label='Vmap', linewidth=2, markersize=8)
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

# Add text annotations with results
for i, (bs, speedup) in enumerate(zip(results['batch_size'], results['speedup'])):
    ax2.annotate(f'{speedup:.1f}x', (bs, speedup), 
                textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()

# Save plot
output_dir = "/workspace/examples/data/vmap_traction_results"
os.makedirs(output_dir, exist_ok=True)
plot_filename = f"{output_dir}/vmap_benchmark.svg"
plt.savefig(plot_filename, format='svg', dpi=300, bbox_inches='tight')
print(f"Benchmark plot saved to: {plot_filename}")

# Print summary
print("\n=== Benchmark Summary ===")
print(f"{'Batch Size':<12} {'For-loop (s)':<15} {'Vmap (s)':<12} {'Speedup':<10}")
print("-" * 55)
for i in range(len(results['batch_size'])):
    print(f"{results['batch_size'][i]:<12} {results['for_loop_time'][i]:<15.4f} "
          f"{results['vmap_time'][i]:<12.4f} {results['speedup'][i]:<10.2f}x")

print(f"\n✅ Benchmark completed!")
print(f"✅ Results show vmap speedup ranges from {min(results['speedup']):.1f}x to {max(results['speedup']):.1f}x")
print(f"✅ Plot saved as SVG: {plot_filename}")