"""
Periodic linear elasticity with macroscopic strain.
Demonstrates periodic homogenization with zero mean fluctuation constraint to remove rigid body motion.
Based on: https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCConfig
from feax.mesh import box_mesh
from feax.utils import save_sol
import os
jax.config.update("jax_enable_x64", True)  # Use 64-bit precision for better accuracy

# Problem setup
E0 = 70e3
E_eps = 1e-3
rho_0 = 1
T = 1e2
nu = 0.3
p = 3

epsilon_macro = np.array([[0.1, 0.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]])

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = (E0 - E_eps) * rho + E_eps
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

# Create mesh and problem
from feax.lattice_toolkit.unitcell import UnitCell
class Cube(UnitCell):
    def mesh_build(self, **kwargs):
        return box_mesh(20, 20, 20, 1, 1, 1)
unitcell = Cube()
mesh = unitcell.mesh

from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
from feax.lattice_toolkit.solver import create_affine_displacement_solver

pbc = periodic_bc_3D(unitcell, 3, 3)
P = prolongation_matrix(pbc, mesh, 3)

# For periodic homogenization, typically no Dirichlet BCs are needed
# The periodic constraints provide the necessary constraints
bc_config = DirichletBCConfig([])

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[]
)

# Create FCC lattice density distribution using built-in graph toolkit
from feax.lattice_toolkit.graph import create_fcc_density

# Create FCC lattice density field (element-based)
print("Creating FCC lattice density field...")
rho_element = create_fcc_density(
    problem, 
    radius=0.08,           # Lattice strut radius
    scale=1.0,             # Unit cell scale
    density_solid=1.0,     # Density in solid regions (struts)
    density_void=0.0       # Density in void regions  
)

print(f"FCC density shape: {rho_element.shape} (element-based)")
print(f"Problem num_cells: {problem.num_cells}")
print(f"Density values range: [{np.min(rho_element):.1f}, {np.max(rho_element):.1f}]")
print(f"Volume fraction: {np.mean(rho_element):.3f}")

# Convert to FEAX volume variable format (broadcast to quad points)
template = InternalVars.create_uniform_volume_var(problem, 1.0)
num_elements, num_quad_points = template.shape
rho_array_0 = np.broadcast_to(rho_element[:, None], (num_elements, num_quad_points))
print(f"FEAX volume variable shape: {rho_array_0.shape}")

# Create boundary conditions from config
bc = bc_config.create_bc(problem)

initial_guess = zero_like_initial_guess(problem, bc)

# Use the new affine displacement solver with matrix-free CG for large problems
affine_solver_options = SolverOptions(
    linear_solver="cg",
    linear_solver_tol=1e-10,
    linear_solver_maxiter=10000
)

solver_with_macro = create_affine_displacement_solver(
    problem, bc, P, epsilon_macro, mesh, affine_solver_options
)

@jax.jit  
def solve_forward(rho_array):
    internal_vars = InternalVars(
        volume_vars=(rho_array,),
        surface_vars=[]
    )
    return solver_with_macro(internal_vars, initial_guess)

sol = solve_forward(rho_array_0)

# Solve
import time
print("solve..")
start = time.time()
sol = solve_forward(rho_array_0)
end = time.time()
sol_time = end - start
numdofs = problem.num_total_dofs_all_vars
print(f"sol time {sol_time}, dofs {numdofs}")

# Analyze displacement components
sol_reshaped = sol.reshape(-1, 3)
mean_disp = np.mean(sol_reshaped, axis=0)
print(f"Mean displacement: [{mean_disp[0]:.6e}, {mean_disp[1]:.6e}, {mean_disp[2]:.6e}]")
print(f"Max displacement magnitude: {np.max(np.linalg.norm(sol_reshaped, axis=1)):.6f}")

# Create macro displacement for comparison
from feax.lattice_toolkit.solver import create_macro_displacement_field
u_macro = create_macro_displacement_field(mesh, epsilon_macro).reshape(-1, 3)
u_fluctuation = sol_reshaped - u_macro

print(f"Macro displacement mean: [{np.mean(u_macro, axis=0)[0]:.6e}, {np.mean(u_macro, axis=0)[1]:.6e}, {np.mean(u_macro, axis=0)[2]:.6e}]")
print(f"Macro displacement max magnitude: {np.max(np.linalg.norm(u_macro, axis=1)):.6f}")

print(f"Fluctuation mean: [{np.mean(u_fluctuation, axis=0)[0]:.6e}, {np.mean(u_fluctuation, axis=0)[1]:.6e}, {np.mean(u_fluctuation, axis=0)[2]:.6e}]")  
print(f"Fluctuation max magnitude: {np.max(np.linalg.norm(u_fluctuation, axis=1)):.6f}")
print(f"Fluctuation RMS: {np.sqrt(np.mean(u_fluctuation**2)):.6f}")

sol_unflat = problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]

# Save solution with displacement (point data) and density (cell data)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_with_density.vtu')

save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)],
    cell_infos=[("density", rho_element)]  # Save element-based density directly as cell data
)

print(f"Saved solution with FCC density field to: {vtk_path}")
print(f"Element-based density memory: {rho_element.nbytes / 1024:.1f} KB")
print(f"Old quad-point density would be: {rho_array_0.nbytes / 1024:.1f} KB") 
print(f"Memory savings: {(1 - rho_element.nbytes / rho_array_0.nbytes) * 100:.1f}%")