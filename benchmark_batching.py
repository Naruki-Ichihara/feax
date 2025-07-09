import jax
import jax.numpy as np
import jax.scipy as scipy
import numpy as onp
import time
import matplotlib.pyplot as plt

from feax.problem import Problem
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, create_boundary_functions

nu = 0.3

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad, internal_vars):
            E_val = internal_vars[0]
            mu = E_val / (2. * (1. + nu))
            lmbda = E_val * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, internal_vars):
            return np.array([0., 0., 100.])
        return [surface_map]

# Setup with mesh size (50, 10, 10)
print("Setting up mesh (50x10x10)...")
meshio_mesh = box_mesh_gmsh(Nx=50, Ny=10, Nz=10, Lx=10., Ly=2., Lz=2., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
boundary_fns = create_boundary_functions(10., 2., 2.)
bc_data = prepare_bc_info(mesh, FixedBC(boundary_fns['left'], components=[0, 1, 2]), vec_dim=3)
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[boundary_fns['right']])
sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))

print(f"Mesh info: {problem.num_cells} cells, {problem.fes[0].num_total_nodes} nodes, {problem.fes[0].num_total_nodes * 3} DOFs")

@jax.jit
def solve(state):
    from feax.problem import get_sparse_system
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    x_sol, _ = scipy.sparse.linalg.cg(A_bc, b_bc, maxiter=1000)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return np.max(np.linalg.norm(sol_final[0], axis=1))  # Return just max displacement for speed

def single_solve(E_values):
    state = problem.get_functional_state(internal_vars=[E_values])
    return solve(state)

# Warm up JAX compilation
print("Warming up JAX compilation...")
E_test = np.full((problem.num_cells * 8,), 50e3)
_ = single_solve(E_test)
print("Compilation complete.")

# Benchmark different batch sizes
batch_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
times_per_case = []
total_times = []

print("\nRunning benchmark...")
print("Batch Size | Total Time (s) | Time per Case (s) | Throughput (cases/s)")
print("-" * 70)

for batch_size in batch_sizes:
    # Create E values for this batch
    E_range = np.linspace(10e3, 100e3, batch_size)
    E_arrays = np.array([np.full((problem.num_cells * 8,), E) for E in E_range])
    
    # Time the batched solve
    start_time = time.time()
    vmapped_solve = jax.vmap(single_solve)
    max_displacements = vmapped_solve(E_arrays)
    
    # Ensure computation is complete
    max_displacements.block_until_ready()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_case = total_time / batch_size
    throughput = batch_size / total_time
    
    total_times.append(total_time)
    times_per_case.append(time_per_case)
    
    print(f"{batch_size:10d} | {total_time:13.3f} | {time_per_case:16.4f} | {throughput:18.2f}")

print("\nBenchmark complete!")

# Calculate additional metrics
throughputs = [batch_sizes[i] / total_times[i] for i in range(len(batch_sizes))]
efficiency_ratio = [times_per_case[0] / t for t in times_per_case]

print("\n=== Detailed Results ===")
print("Batch | Time/Case | Throughput | Efficiency")
print("Size  |   (sec)   | (cases/s)  |   Ratio   ")
print("------|-----------|------------|----------")
for i, batch_size in enumerate(batch_sizes):
    print(f"{batch_size:5d} | {times_per_case[i]:8.4f}  | {throughputs[i]:9.2f}  | {efficiency_ratio[i]:8.2f}x")

# Summary statistics
best_batch_idx = onp.argmin(times_per_case)
best_batch_size = batch_sizes[best_batch_idx]
best_time_per_case = times_per_case[best_batch_idx]
best_throughput = throughputs[best_batch_idx]

print(f"\n=== Benchmark Summary ===")
print(f"Mesh: {problem.num_cells} cells, {problem.fes[0].num_total_nodes * 3} DOFs")
print(f"Best batch size: {best_batch_size} cases")
print(f"Best time per case: {best_time_per_case:.4f} seconds")
print(f"Best throughput: {best_throughput:.2f} cases/second")
print(f"Efficiency improvement: {efficiency_ratio[best_batch_idx]:.2f}x vs batch=10")

# Calculate speedup over sequential solving
sequential_time_estimate = times_per_case[0] * batch_sizes[0]  # Estimate based on batch=10
parallel_time = total_times[best_batch_idx]
speedup = (sequential_time_estimate * best_batch_size / batch_sizes[0]) / parallel_time

print(f"Estimated speedup vs sequential: {speedup:.1f}x")

# Non-JIT for loop comparison
print("\n=== For Loop Comparison (No Batching) ===")

# Create non-JIT solve function
def solve_no_jit(state):
    from feax.problem import get_sparse_system
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    x_sol, _ = scipy.sparse.linalg.cg(A_bc, b_bc, maxiter=1000)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return np.max(np.linalg.norm(sol_final[0], axis=1))

# Test for loop on batch sizes 10 and 50
for_loop_sizes = [1, 50, 100]
for_loop_times = []
for_loop_times_per_case = []

for batch_size in for_loop_sizes:
    E_range = np.linspace(10e3, 100e3, batch_size)
    
    # Time the for loop version
    start_time = time.time()
    max_disps_loop = []
    
    for E_val in E_range:
        E_array = np.full((problem.num_cells * 8,), E_val)
        state = problem.get_functional_state(internal_vars=[E_array])
        max_disp = solve_no_jit(state)
        max_disps_loop.append(max_disp)
    
    end_time = time.time()
    
    total_time_loop = end_time - start_time
    time_per_case_loop = total_time_loop / batch_size
    
    for_loop_times.append(total_time_loop)
    for_loop_times_per_case.append(time_per_case_loop)
    
    # Find corresponding vmap time
    vmap_idx = batch_sizes.index(batch_size)
    vmap_time = total_times[vmap_idx]
    vmap_time_per_case = times_per_case[vmap_idx]
    
    speedup_vs_loop = total_time_loop / vmap_time
    
    print(f"\nBatch size: {batch_size}")
    print(f"For loop time: {total_time_loop:.3f}s ({time_per_case_loop:.4f}s per case)")
    print(f"Vmap time: {vmap_time:.3f}s ({vmap_time_per_case:.4f}s per case)")
    print(f"Speedup: {speedup_vs_loop:.2f}x faster with vmap")

# Create performance plots
plt.figure(figsize=(14, 10))

# Plot 1: Time per case comparison
plt.subplot(2, 2, 1)
plt.plot(batch_sizes, times_per_case, 'b-o', linewidth=2, markersize=8, label='JAX vmap')
plt.scatter(for_loop_sizes, for_loop_times_per_case, color='red', s=100, marker='s', label='For loop (no JIT)', zorder=5)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Time per Case (seconds)', fontsize=12)
plt.title('Computational Efficiency: vmap vs For Loop', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 2: Total time comparison
plt.subplot(2, 2, 2)
plt.plot(batch_sizes, total_times, 'b-o', linewidth=2, markersize=8, label='JAX vmap')
plt.scatter(for_loop_sizes, for_loop_times, color='red', s=100, marker='s', label='For loop (no JIT)', zorder=5)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Total Time (seconds)', fontsize=12)
plt.title('Total Computation Time Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot 3: Speedup vs batch size
speedups = []
for i, bs in enumerate(batch_sizes):
    if bs in for_loop_sizes:
        idx = for_loop_sizes.index(bs)
        speedup = for_loop_times_per_case[idx] / times_per_case[i]
        speedups.append(speedup)
    else:
        # Interpolate/extrapolate for other batch sizes
        speedup = for_loop_times_per_case[0] / times_per_case[i]
        speedups.append(speedup)

plt.subplot(2, 2, 3)
plt.plot(batch_sizes, speedups, 'g-^', linewidth=2, markersize=8)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Speedup Factor', fontsize=12)
plt.title('vmap Speedup vs For Loop', fontsize=14)
plt.grid(True, alpha=0.3)

# Plot 4: Throughput comparison
plt.subplot(2, 2, 4)
plt.plot(batch_sizes, throughputs, 'b-o', linewidth=2, markersize=8, label='JAX vmap')
for_loop_throughputs = [for_loop_sizes[i]/for_loop_times[i] for i in range(len(for_loop_sizes))]
plt.scatter(for_loop_sizes, for_loop_throughputs, color='red', s=100, marker='s', label='For loop (no JIT)', zorder=5)
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Throughput (cases/second)', fontsize=12)
plt.title('Throughput Comparison', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.suptitle(f'Batching Performance Benchmark\nMesh: {problem.num_cells} cells, {problem.fes[0].num_total_nodes * 3} DOFs', fontsize=16)
plt.tight_layout()
plt.savefig('batching_benchmark_with_forloop.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as: batching_benchmark_with_forloop.png")

print("\n=== Performance Summary ===")
print(f"JAX vmap provides up to {max(speedups):.1f}x speedup over for loops")
print(f"Best throughput: {best_throughput:.1f} cases/second with batch={best_batch_size}")
print(f"For loop throughput: {for_loop_throughputs[0]:.2f} cases/second")
print(f"This enables parameter studies that would be impractical with sequential solving!")