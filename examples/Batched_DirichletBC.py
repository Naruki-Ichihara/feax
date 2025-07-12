import jax
import jax.numpy as np

from feax.problem import Problem, get_sparse_system
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, DirichletBC
from feax.solvers import SolverOptions, solve
from feax.utils import save_as_vtk

tol = 1e-5
resolution = 50
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
        return []

# Mesh
meshio_mesh = box_mesh_gmsh(Nx=resolution, Ny=resolution, Nz=resolution, Lx=1., Ly=1., Lz=1., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Boundary condition functions
left = lambda x: np.isclose(x[0], 0., atol=tol)
right = lambda x: np.isclose(x[0], 1., atol=tol)

# Prepare batched boundary conditions
displacement_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

def prepare_batched_bc_info(mesh, displacement_val):
    left_bc = FixedBC(left, [0, 1, 2])
    right_bc = DirichletBC(right, displacement_val, components=[0])
    return prepare_bc_info(mesh, [left_bc, right_bc], 3)

# Pre-compute all boundary condition info
bc_infos = [prepare_batched_bc_info(mesh, disp_val) for disp_val in displacement_values]

# Create problem
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[])
solver_options = SolverOptions(max_iter=1000, tol=1e-8)
state = problem.get_functional_state()

sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))
sol_flat = jax.flatten_util.ravel_pytree(sol_list)[0]

@jax.jit
def solve_single_case(bc_info, state, sol_flat):
    A, b = get_sparse_system(state, sol_flat)
    A_bc, b_bc = apply_bc(A, b, bc_info)
    sol = solve(A_bc, b_bc, "cg", solver_options)
    return sol

# Use vmap to vectorize over bc_infos
solve_batched = jax.vmap(solve_single_case, in_axes=(0, None, None))

# Convert bc_infos to arrays for vmap
bc_infos_array = jax.tree.map(lambda *x: np.stack(x), *bc_infos)

# Solve all cases in batch
solutions = solve_batched(bc_infos_array, state, sol_flat)

# Save results for each case
for i, (disp_val, sol) in enumerate(zip(displacement_values, solutions)):
    sol_reshaped = sol.reshape(-1, 3)
    vtk_filename = f'outputs/batched_test_disp_{disp_val:.1f}.vtu'
    save_as_vtk(
        mesh=mesh,
        sol_file=vtk_filename,
        point_infos=[
            ('displacement', sol_reshaped),
        ]
    )
    print(f"Solved case {i+1}/3: displacement = {disp_val}, max displacement = {np.max(np.abs(sol_reshaped)):.6f}")

print("Batched linear elasticity simulation completed!")