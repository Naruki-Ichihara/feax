import jax
import jax.numpy as np
from feax import Problem, InternalVars, get_J, get_res, create_J_bc_function, create_res_bc_function
from feax import Mesh, DirichletBC, newton_solve, SolverOptions, create_differentiable_solver
from feax.mesh import box_mesh_gmsh
import time

class HyperElasticityFeax(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):
        def psi(F):
            E = 100.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad, E_quad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
meshio_mesh = box_mesh_gmsh(15, 15, 15, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], 1, atol=1e-5)


def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                     [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                     [zero_dirichlet_val] * 3]

# Create clean Problem (NO internal_vars!)
feax_problem = HyperElasticityFeax(mesh,
                          vec=3,
                          dim=3,
                          dirichlet_bc_info=dirichlet_bc_info)

# Create InternalVars separately (even though not used in this hyperelastic formulation)
E_array = InternalVars.create_uniform_volume_var(feax_problem, 100.0)
internal_vars = InternalVars(volume_vars=(E_array,))

bc = DirichletBC.from_bc_info(feax_problem, dirichlet_bc_info)


solver_options = SolverOptions(tol=1e-8, linear_solver="bicgstab", x0_strategy="bc_aware", bc_rows=bc.bc_rows, bc_vals=bc.bc_vals)
solver = create_differentiable_solver(feax_problem, bc, solver_options)

@jax.jit
def solve_fn(internal_vars):
    sol = solver(internal_vars)
    return sol

start = time.time()
sol = solve_fn(internal_vars)
end = time.time()

print(f"Solve took: {end-start} s.")

# Recall
start = time.time()
sol = solve_fn(internal_vars)
end = time.time()

sol_unflat = feax_problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]
# Save solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution.vtk",
    point_infos=[("displacement", displacement)])

print(f"Solve took: {end-start} s.")