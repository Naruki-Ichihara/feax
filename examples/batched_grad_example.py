"""
Simple example: batched gradients with create_solver for linear elasticity
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, DirichletBC, SolverOptions
from feax.mesh import box_mesh_gmsh

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
meshio_mesh = box_mesh_gmsh(5, 5, 5, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1, atol=1e-5)

def zero_disp(point):
    return 0.0

# BC: fix left, apply traction on right
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

def solve_forward(rho_array):
    internal_vars = InternalVars(
        volume_vars=(rho_array,),
        surface_vars=[(traction_array,)]
    )
    return solver(internal_vars)

def objective(rho_array):
    sol = solve_forward(rho_array)
    return np.sum(sol**2)

# Batch of density values
rho_batch = np.linspace(0.1, 1.0, 50)

# Batched gradients
grad_fn = jax.grad(objective)
batched_grad_fn = jax.vmap(grad_fn)

print("Computing batched gradients...")
rho_arrays = [InternalVars.create_uniform_volume_var(problem, rho) for rho in rho_batch]
gradients = batched_grad_fn(np.stack(rho_arrays))

print(f"Batch size: {len(rho_batch)}")
print(f"Gradient norms: {[np.linalg.norm(g) for g in gradients]}")