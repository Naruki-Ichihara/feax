import jax
import jax.numpy as np
import jax.scipy as scipy

from feax.problem import Problem
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, create_boundary_functions
from feax.utils import save_as_vtk

nu = 0.3

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad, internal_vars):
            E_val = internal_vars[0]
            mu = E_val / (2. * (1. + nu))
            lmbda = E_val * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, internal_vars):
            return np.array([0., 0., 100.])
        return [surface_map]

# Setup
meshio_mesh = box_mesh_gmsh(Nx=50, Ny=10, Nz=10, Lx=10., Ly=2., Lz=2., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
boundary_fns = create_boundary_functions(10., 2., 2.)
bc_data = prepare_bc_info(mesh, FixedBC(boundary_fns['left'], components=[0, 1, 2]), vec_dim=3)
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[boundary_fns['right']])
sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))

@jax.jit
def solve(state):
    from feax.problem import get_sparse_system
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    x_sol, _ = scipy.sparse.linalg.cg(A_bc, b_bc, maxiter=1000)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return sol_final[0]  # Return full displacement field

def single_solve(E_values):
    state = problem.get_functional_state(internal_vars=[E_values])
    return solve(state)

# Solve 30 cases with different E values
E_range = np.linspace(10e3, 100e3, 60)
E_arrays = np.array([np.full((problem.num_cells * 8,), E) for E in E_range])
print(f"Solving {len(E_range)} cases...")
all_solutions = jax.vmap(single_solve)(E_arrays)

# Calculate max displacements for summary
max_displacements = np.array([np.max(np.linalg.norm(sol, axis=1)) for sol in all_solutions])

print(f"Solved {len(E_range)} cases")
print(f"E: {E_range[0]:.0f} to {E_range[-1]:.0f}")
print(f"Max displacement: {np.max(max_displacements):.3f} to {np.min(max_displacements):.3f}")
print(f"Stiffness ratio: {max_displacements[-1]/max_displacements[0]:.2f}")

# Save all solutions as VTK files
print(f"Saving {len(E_range)} VTK files...")
for i, (E_val, solution) in enumerate(zip(E_range, all_solutions)):
    displacement_magnitude = np.linalg.norm(solution, axis=1)
    vtk_filename = f'outputs/batched_E_{E_val:.0f}_{i:02d}.vtu'
    
    save_as_vtk(
        mesh=mesh,
        sol_file=vtk_filename,
        point_infos=[
            ('displacement', solution),
            ('displacement_magnitude', displacement_magnitude),
            ('E_modulus', np.full(len(solution), E_val))
        ]
    )

print(f"VTK files saved: batched_E_*.vtu")