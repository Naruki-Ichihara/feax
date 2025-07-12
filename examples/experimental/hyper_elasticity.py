import jax
import jax.numpy as np

from feax.problem import Problem, get_sparse_system
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, DirichletBC
from feax.solvers import SolverOptions, solve
from feax.utils import save_as_vtk

tol = 1e-5
resolution = 60
E = 10.
nu = 0.3

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        # No surface tractions - using displacement BCs only
        return []
    

# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):

        def psi(F):
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
# Mesh
meshio_mesh = box_mesh_gmsh(Nx=resolution, Ny=resolution, Nz=resolution, Lx=1., Ly=1., Lz=1., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Boundary condition
left = lambda x: np.isclose(x[0], 0., atol=tol)
right = lambda x: np.isclose(x[0], 1., atol=tol)
left_bc = FixedBC(left, [0, 1, 2])
right_bc = DirichletBC(right, 0.1, components=[0])
bc_info = prepare_bc_info(mesh, [left_bc, right_bc], 3)

# Create problem without surface tractions (using only displacement BC)
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[])

solver_options = SolverOptions(max_iter=1000, tol=1e-8)
state = problem.get_functional_state()

sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))
sol_flat = jax.flatten_util.ravel_pytree(sol_list)[0]

@jax.jit
def linear_solve_jit(state, sol_flat, bc_info):
    A, b = get_sparse_system(state, sol_flat)
    A_bc, b_bc = apply_bc(A, b, bc_info)
    sol = solve(A_bc, b_bc, "cg", solver_options)
    return sol

sol = linear_solve_jit(state, sol_flat, bc_info)

# Reshape solution from flat array to (num_points, 3) for VTK
sol_reshaped = sol.reshape(-1, 3)

vtk_filename = f'outputs/test.vtu'
save_as_vtk(
        mesh=mesh,
        sol_file=vtk_filename,
        point_infos=[
            ('displacement', sol_reshaped),
        ]
    )