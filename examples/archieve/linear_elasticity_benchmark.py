"""
Performance benchmark for linear elasticity solver using clean FEAX API.
Compares different implementation approaches: for loop (no jit), for loop (jit), and vmap implementations.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, get_J, get_res, create_J_bc_function, create_res_bc_function
from feax import Mesh, DirichletBC, linear_solve, SolverOptions, apply_boundary_to_J, apply_boundary_to_res
from feax.mesh import box_mesh_gmsh
import time
import matplotlib.pyplot as plt
import numpy as onp

# Problem setup
E = 70e3
nu = 0.3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, E_quad):
            mu = E_quad / (2. * (1. + nu))
            lmbda = E_quad * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(40, 40, 40, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

def zero_disp(point):
    return 0.0

def tension_disp(point):
    return 0.1

# Create problem - simpler BC without surface forces
dirichlet_bc_info = [[left] * 3 + [right], [0, 1, 2, 0], 
                     [zero_disp, zero_disp, zero_disp, tension_disp]]

# Create clean Problem (NO internal_vars!)
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info
)

# Create InternalVars separately
E_array = InternalVars.create_uniform_volume_var(problem, E)

internal_vars = InternalVars(
    volume_vars=(E_array,)
)

# Single solve function
def solve_single(initial_sol, static_J, bc):
    def J_bc_func(sol_flat, internal_vars_param):
        return static_J
    
    def res_bc_func(sol_flat, internal_vars_param):
        sol_unflat = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_unflat, internal_vars_param)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)
    
    solver_options = SolverOptions(
        tol=1e-8,
        linear_solver="cg"
    )
    
    sol = linear_solve(J_bc_func, res_bc_func, initial_sol, bc, solver_options, internal_vars)
    return sol

# JIT compiled version of single solve
solve_single_jit = jax.jit(solve_single)

# Vmap version
solve_vmap = jax.jit(jax.vmap(solve_single, in_axes=(0, None, 0)))

# Benchmark function
def run_benchmark(batch_sizes):
    results = {
        'for_loop': [],
        'for_loop_jit': [],
        'vmap': []
    }
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Create batch of boundary conditions
        tension_values = np.linspace(0.05, 0.15, batch_size)
        bc_list = []
        initial_sol_batch = []
        
        for tension in tension_values:
            def make_tension_func(t):
                def tension_func(point):
                    return t
                return tension_func
            
            tension_func = make_tension_func(float(tension))
            bc_info = [[left] * 3 + [right], [0, 1, 2, 0], 
                       [zero_disp, zero_disp, zero_disp, tension_func]]
            
            bc = DirichletBC.from_bc_info(problem, bc_info)
            bc_list.append(bc)
            
            initial_sol = np.zeros(problem.num_total_dofs_all_vars)
            initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)
            initial_sol_batch.append(initial_sol)
        
        # Create batched BC for vmap
        bc_batch = DirichletBC(
            bc_rows=np.tile(bc_list[0].bc_rows, (batch_size, 1)),
            bc_mask=np.tile(bc_list[0].bc_mask, (batch_size, 1)),
            bc_vals=np.stack([bc.bc_vals for bc in bc_list], axis=0),
            total_dofs=bc_list[0].total_dofs
        )
        
        initial_sol_batch_array = np.stack(initial_sol_batch, axis=0)
        
        # Static Jacobian
        initial_sol_unflat = problem.unflatten_fn_sol_list(initial_sol_batch[0])
        J = get_J(problem, initial_sol_unflat, internal_vars)
        static_J = apply_boundary_to_J(bc_list[0], J)
        
        # 1. For loop (no JIT)
        print("  Running for loop (no JIT)...")
        start_time = time.time()
        solutions_loop = []
        for i in range(batch_size):
            sol = solve_single(initial_sol_batch[i], static_J, bc_list[i])
            solutions_loop.append(sol)
        _ = jax.block_until_ready(solutions_loop[-1])
        loop_time = time.time() - start_time
        results['for_loop'].append(loop_time)
        print(f"    Time: {loop_time:.3f}s")
        
        # 2. For loop (with JIT) - compile first
        print("  Running for loop (with JIT)...")
        _ = solve_single_jit(initial_sol_batch[0], static_J, bc_list[0])  # Warmup
        start_time = time.time()
        solutions_loop_jit = []
        for i in range(batch_size):
            sol = solve_single_jit(initial_sol_batch[i], static_J, bc_list[i])
            solutions_loop_jit.append(sol)
        _ = jax.block_until_ready(solutions_loop_jit[-1])
        loop_jit_time = time.time() - start_time
        results['for_loop_jit'].append(loop_jit_time)
        print(f"    Time: {loop_jit_time:.3f}s")
        
        # 3. Vmap
        print("  Running vmap...")
        start_time = time.time()
        solutions_vmap = solve_vmap(initial_sol_batch_array, static_J, bc_batch)
        _ = jax.block_until_ready(solutions_vmap)
        vmap_time = time.time() - start_time
        results['vmap'].append(vmap_time)
        print(f"    Time: {vmap_time:.3f}s")
    
    return results

# Run benchmarks
print("Starting performance benchmark...")
batch_sizes = [1, 10, 50, 100]
results = run_benchmark(batch_sizes)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, results['for_loop'], 'o-', label='For loop (no JIT)', linewidth=2, markersize=8)
plt.plot(batch_sizes, results['for_loop_jit'], 's-', label='For loop (JIT)', linewidth=2, markersize=8)
plt.plot(batch_sizes, results['vmap'], '^-', label='Vmap', linewidth=2, markersize=8)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Linear Elasticity Solver Performance Benchmark', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.xscale('log')
plt.xticks(batch_sizes, batch_sizes)

# Add speedup annotations
for i, batch_size in enumerate(batch_sizes):
    if batch_size > 1:
        speedup_vmap = results['for_loop'][i] / results['vmap'][i]
        speedup_jit = results['for_loop'][i] / results['for_loop_jit'][i]
        plt.text(batch_size, results['vmap'][i] * 0.8, f'{speedup_vmap:.1f}x', 
                ha='center', va='top', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('linear_elasticity_benchmark.png', dpi=300, bbox_inches='tight')
print("\nBenchmark complete! Results saved to 'linear_elasticity_benchmark.png'")

# Print summary
print("\nPerformance Summary:")
print(f"{'Batch Size':<12} {'For Loop':<15} {'For Loop (JIT)':<15} {'Vmap':<15}")
print("-" * 60)
for i, batch_size in enumerate(batch_sizes):
    print(f"{batch_size:<12} {results['for_loop'][i]:<15.3f} {results['for_loop_jit'][i]:<15.3f} {results['vmap'][i]:<15.3f}")