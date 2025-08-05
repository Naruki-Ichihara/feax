"""
Linear elasticity example with SIMP-based material interpolation.
Demonstrates density-dependent material properties using the SIMP (Solid Isotropic Material with Penalization) method.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh_gmsh
from feax.utils import save_sol
import os

# Problem setup
E0 = 70e3
E_eps = 1e-3
rho_0 = 0.5
T = 1e2
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
meshio_mesh = box_mesh_gmsh(40, 20, 20, 2., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 2, atol=1e-5)

# Create boundary conditions using the new dataclass approach
bc_config = DirichletBCConfig([
    # Fix left boundary completely (all components to zero)
    DirichletBCSpec(
        location=left,
        component='all',  # Fix x, y, z components
        value=0.0
    )
])

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[right]
)

# Create InternalVars separately
rho_array_0 = InternalVars.create_uniform_volume_var(problem, rho_0)
traction_array = InternalVars.create_uniform_surface_var(problem, T)

# Create boundary conditions from config
bc = bc_config.create_bc(problem)
solver_option = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_option, iter_num=1)

initial_guess = zero_like_initial_guess(problem, bc)

@jax.jit
def solve_forward(rho_array):
    internal_vars = InternalVars(
    volume_vars=(rho_array,),
    surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial_guess)

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
sol_unflat = problem.unflatten_fn_sol_list(sol)

displacement = sol_unflat[0]
# Save solution

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')

save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])