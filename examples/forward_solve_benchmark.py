"""
Performance comparison for forward solve using create_solver
Compares for loop, for loop with JIT, and vmap for different batch sizes
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, DirichletBC, SolverOptions
from feax.mesh import box_mesh_gmsh
import time
import matplotlib.pyplot as plt

# Problem setup
E0 = 70e3
E_eps = 1e-3
nu = 0.3
p = 3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = (E0 - E_eps) * rho**p + E_eps
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
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

dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_disp, zero_disp, zero_disp]]

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

# Create solver
bc = DirichletBC.from_problem(problem)
solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_options, iter_num=1)

# Create traction
traction_array = InternalVars.create_uniform_surface_var(problem, 1.0)

def solve_forward(rho_value):
    # Create rho array on-demand from scalar value
    rho_array = InternalVars.create_uniform_volume_var(problem, rho_value)
    internal_vars = InternalVars(
        volume_vars=(rho_array,),
        surface_vars=[(traction_array,)]
    )
    return solver(internal_vars)

# JIT compiled version
solve_forward_jit = jax.jit(solve_forward)

# Vmap version
solve_forward_vmap = jax.jit(jax.vmap(solve_forward))

def run_benchmark(batch_sizes):
    results = {
        'for_loop': [],
        'for_loop_jit': [],
        'vmap': []
    }
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Create batch of density values (just scalars, not full arrays)
        rho_values = np.linspace(0.2, 0.8, batch_size)
        
        # 2. For loop (with JIT) - warmup first
        print("  Running for loop (with JIT)...")
        _ = solve_forward_jit(rho_values[0])  # Warmup
        start_time = time.time()
        solutions_loop_jit = []
        for rho in rho_values:
            sol = solve_forward_jit(rho)
            solutions_loop_jit.append(sol)
        _ = jax.block_until_ready(solutions_loop_jit[-1])
        loop_jit_time = time.time() - start_time
        results['for_loop_jit'].append(loop_jit_time)
        print(f"    Time: {loop_jit_time:.4f}s")
        
        # 3. Vmap - warmup first
        print("  Running vmap...")
        _ = solve_forward_vmap(rho_values[:1])  # Warmup
        start_time = time.time()
        solutions_vmap = solve_forward_vmap(rho_values)
        _ = jax.block_until_ready(solutions_vmap)
        vmap_time = time.time() - start_time
        results['vmap'].append(vmap_time)
        print(f"    Time: {vmap_time:.4f}s")
    
    return results

# Run benchmarks
print("Starting forward solve performance benchmark...")
batch_sizes = [1, 10, 20, 50]
results = run_benchmark(batch_sizes)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, results['for_loop_jit'], 's-', label='For loop (JIT)', linewidth=2, markersize=8)
plt.plot(batch_sizes, results['vmap'], '^-', label='Vmap', linewidth=2, markersize=8)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Forward Solve Performance Comparison (10x10x10 mesh)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(batch_sizes)
plt.tight_layout()
plt.savefig('/workspace/forward_solve_benchmark.png', dpi=300, bbox_inches='tight')
print(f"\nBenchmark complete! Results saved to '/workspace/forward_solve_benchmark.png'")