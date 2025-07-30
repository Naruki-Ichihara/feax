"""
Simple example: newton_solve inside @jax.jit
"""

import jax
import jax.numpy as np
from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.assembler import get_J, get_res
from feax.DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res
from feax.solver import linear_solve, SolverOptions
from feax.utils import save_sol
import os

# Problem setup
E = 70e3
nu = 0.3

class ElasticityProblem(FeaxProblem):
    def get_tensor_map(self):
        def stress(u_grad, internal_vars=None):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, internal_vars=None):
            return np.array([0., 0., 1.])
        return [surface_map]

# Create mesh and problem
meshio_mesh = box_mesh_gmsh(30, 30, 30, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

# Fix boundary conditions for a proper elasticity problem
# Left boundary: fix all displacements to 0 (full constraint)  
# Right boundary: apply x-displacement of 0.1 (tension)
def zero_disp(point):
    return 0.0

def tension_disp(point):
    return 0.1

# Constrain left boundary completely, apply tension on right boundary x-direction
dirichlet_bc_info = [[left] * 3 + [right], [0, 1, 2, 0], 
                     [zero_disp, zero_disp, zero_disp, tension_disp]]

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

bc = DirichletBC.from_problem(problem)

# Create 10 different boundary conditions with varying tension values using from_bc_info
tension_values = np.linspace(0.05, 0.15, 100)
bc_list = []
initial_sol_batch = []

for tension in tension_values:
    # Create tension function for this specific value
    def make_tension_func(t):
        def tension_func(point):
            return t
        return tension_func
    
    tension_func = make_tension_func(float(tension))
    
    # Create BC info for this tension value
    bc_info = [[left] * 3 + [right], [0, 1, 2, 0], 
               [zero_disp, zero_disp, zero_disp, tension_func]]
    
    # Use the new from_bc_info method
    bc = DirichletBC.from_bc_info(problem, bc_info)
    bc_list.append(bc)
    
    # Create initial solution with BC values set
    initial_sol = np.zeros(problem.num_total_dofs_all_vars)
    initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)
    initial_sol_batch.append(initial_sol)

# Create batched BC
bc_batch = DirichletBC(
    bc_rows=np.tile(bc_list[0].bc_rows, (100, 1)),
    bc_mask=np.tile(bc_list[0].bc_mask, (100, 1)),
    bc_vals=np.stack([bc.bc_vals for bc in bc_list], axis=0),
    total_dofs=bc_list[0].total_dofs
)

initial_sol_batch = np.stack(initial_sol_batch, axis=0)

# Static Jacobian computation
initial_sol_unflat = problem.unflatten_fn_sol_list(initial_sol_batch[0])
J = get_J(problem, initial_sol_unflat)
static_J = apply_boundary_to_J(bc, J)

def solve(initial_sol, static_J, bc):
    def J_bc_func(sol_flat):
        return static_J
    
    def res_bc_func(sol_flat):
        sol_unflat = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_unflat)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)
    
    solver_options = SolverOptions(
        tol=1e-8,
        linear_solver="cg",
        x0_strategy="zeros"
    )
    
    sol = linear_solve(J_bc_func, res_bc_func, initial_sol, solver_options)
    return sol

# Batch solve
solve_batch = jax.vmap(solve, in_axes=(0, None, 0))
print("Solving batch...")
solutions = solve_batch(initial_sol_batch, static_J, bc_batch)

# Save all solutions as VTK files
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving solutions to {output_dir}/...")
for i, (sol, tension) in enumerate(zip(solutions, tension_values)):
    vtk_path = os.path.join(output_dir, f"solution_tension_{tension:.3f}.vtk")
    # Unflatten the solution to get displacement field
    sol_unflat = problem.unflatten_fn_sol_list(sol)
    displacement = sol_unflat[0]  # First (and only) field is displacement
    
    # Save using save_sol with displacement as point data
    save_sol(
        mesh=mesh,
        sol_file=vtk_path,
        point_infos=[("displacement", displacement)]
    )
    print(f"  Saved: {vtk_path}")