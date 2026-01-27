"""
Batched linear elasticity benchmark using vmap for parallel density-based solving.

Demonstrates:
1. Single solve function for one density value using SIMP material interpolation
2. Using jax.vmap to solve multiple density values in parallel
3. Benchmark comparison between for-loop and vmap approaches
4. Density range: 0.1-1.0 with batch sizes 1 and 10
"""
import feax as fe
import jax
import jax.numpy as np
import time
import os
import gc
import matplotlib.pyplot as plt

# Problem setup
E0 = 70e3
E_eps = 1e-3
nu = 0.3
p = 3  # SIMP penalization parameter
T = 1e2  # Traction magnitude (fixed)

# Batch sizes for benchmarking
batch_sizes = [1, 10, 20, 30, 40]

class DensityElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            # SIMP material interpolation: E(rho) = (E0 - E_eps) * rho^p + E_eps
            E = (E0 - E_eps) * rho**p + E_eps
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            strain = 0.5 * (u_grad + u_grad.T)
            sigma = lam * np.trace(strain) * np.eye(self.dim) + 2.0 * mu * strain
            return sigma
        return stress
    
    def get_surface_maps(self):
        def traction_map(u_grad, surface_quad_point, traction_magnitude):
            return np.array([0.0, 0.0, -traction_magnitude])  # Fixed traction in -z direction
        return [traction_map]

# Create mesh
print("Creating mesh...")
mesh = fe.mesh.box_mesh((2, 1, 1), mesh_size=0.1)
print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

# Boundary locations
def left(point):
    return np.isclose(point[0], 0, atol=1e-5)

def right(point):
    return np.isclose(point[0], 2, atol=1e-5)

# Boundary conditions: fix left face completely
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='all', value=0.0)
])

# Create problem and solver
problem = DensityElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)

bc = bc_config.create_bc(problem)
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="cudss")
solver = fe.solver.create_solver(problem, bc, solver_options, iter_num=1)

print(f"Problem: {problem.num_total_dofs_all_vars} DOFs")

# Single solve function for one density value
def single_solve(density):
    """Solve for a single density value using SIMP material interpolation."""
    # Create uniform density field
    rho = fe.internal_vars.InternalVars.create_uniform_volume_var(problem, density)
    
    # Create fixed traction
    traction_z = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, T)
    
    internal_vars = fe.internal_vars.InternalVars(
        volume_vars=[rho],
        surface_vars=[(traction_z,)]
    )
    
    # Solve with zero initial guess
    return solver(internal_vars, fe.utils.zero_like_initial_guess(problem, bc))

# Run benchmark for single density range
density_ranges = [
    {"name": "Density", "range": (0.1, 1.0), "filename": "density"}
]

all_results = {}

for density_config in density_ranges:
    print(f"{density_config['name']} Range: {density_config['range'][0]}-{density_config['range'][1]}")

    results = {'batch_size': [], 'for_loop_time': [], 'vmap_time': [], 'speedup': []}
    
    # Pre-compile both strategies with small batch to avoid compilation overhead
    print("Pre-compiling strategies...")
    compile_density = np.array([0.5])
    solve_vmap = jax.vmap(single_solve)
    
    # Compile vmap version  
    print("  Compiling vmap...")
    _ = solve_vmap(compile_density)
    jax.block_until_ready(_)
    
    # Run benchmarks for each batch size
    for batch_size in batch_sizes:
        print(f"=== Batch Size: {batch_size} ===")
        
        # Create density values for this batch size
        density_values = np.linspace(density_config['range'][0], density_config['range'][1], batch_size)
        
        # Benchmark 2: Vmap approach
        print(f"  Testing vmap with {batch_size} solves...")
        start_time = time.time()
        vmap_solutions = solve_vmap(density_values)
        jax.block_until_ready(vmap_solutions)
        vmap_time = time.time() - start_time
        print(f"  Vmap time: {vmap_time:.4f}s")
        
        gc.collect()