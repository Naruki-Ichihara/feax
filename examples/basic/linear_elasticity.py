import feax as fe
import jax
import jax.numpy as np
import os

elastic_moduli = 70e3
poisson_ratio = 0.3
traction = 1e-3
tol = 1e-5

# Define mesh
L = 100
W = 10
H = 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=1)
    
# Locations
left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol)

# Define problem
E = elastic_moduli
nu = poisson_ratio
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]
    
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary
left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)

# Solver
solver = fe.solver.create_solver(problem, bc, iter_num=1)
initial = fe.utils.zero_like_initial_guess(problem, bc)

def solve_forward(traction_array):
    internal_vars = fe.internal_vars.InternalVars(
    volume_vars=(),
    surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial)

traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction)
sol = solve_forward(traction_array)
sol_unflat = problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]

# Save solution
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')

fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])