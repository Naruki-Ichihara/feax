"""
Linear elasticity example with SIMP-based material interpolation.
Demonstrates density-dependent material properties using the SIMP (Solid Isotropic Material with Penalization) method.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, DirichletBC, SolverOptions
from feax.mesh import box_mesh_gmsh

# Problem setup
E0 = 70e3
E_eps = 1e-3
rho_0 = 0.5
T = 1.0
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

# Fix boundary conditions for a proper elasticity problem
# Left boundary: fix all displacements to 0 (full constraint)  
# Right boundary: apply x-displacement of 0.1 (tension)
def zero_disp(point):
    return 0.0

# Constrain left boundary completely, apply tension on right boundary x-direction
dirichlet_bc_info = [[left] * 3, [0, 1, 2], 
                     [zero_disp, zero_disp, zero_disp]]

# Create clean Problem (NO internal_vars!)
problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
)

# Create InternalVars separately
rho_array_0 = InternalVars.create_uniform_volume_var(problem, rho_0)
traction_array = InternalVars.create_uniform_surface_var(problem, T)

# Create boundary conditions
bc = DirichletBC.from_problem(problem)
solver_option = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_option, iter_num=1)

def solve_forward(rho_array):
    internal_vars = InternalVars(
    volume_vars=(rho_array,),
    surface_vars=[(traction_array,)]
    )
    return solver(internal_vars)

def test_func(sol_vec):
    return np.sum(sol_vec**2)

def compose_func(rho_array):
    return test_func(solve_forward(rho_array))

# Solve
print("solve..")
sol = solve_forward(rho_array_0)
sol_unflat = problem.unflatten_fn_sol_list(sol)

value = test_func(sol)
print(value)

displacement = sol_unflat[0]
# Save solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution.vtk",
    point_infos=[("displacement", displacement)])

# Transformation
solve_backward = jax.grad(compose_func)
print("solve backward..")
vals = solve_backward(rho_array_0)
print(np.linalg.norm(vals))