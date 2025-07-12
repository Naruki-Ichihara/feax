import jax
import jax.numpy as np

from feax.problem import Problem, get_sparse_system
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, DirichletBC
from feax.solvers import SolverOptions, solve
from feax.utils import save_as_vtk

tol = 1e-5
resolution = 10
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

# Pre-compute boundary condition info (static part)
left_bc = FixedBC(left, [0, 1, 2])
right_bc = DirichletBC(right, 0.0, components=[0])  # dummy value
static_bc_info = prepare_bc_info(mesh, [left_bc, right_bc], 3)

# Count the number of right boundary DOFs for x-component
right_nodes = np.where(jax.vmap(right)(mesh.points))[0]
num_right_x_dofs = len(right_nodes)
static_bc_info['num_right_x_dofs'] = num_right_x_dofs

# Create problem
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[])
solver_options = SolverOptions(max_iter=1000, tol=1e-8)
state = problem.get_functional_state()

sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))
sol_flat = jax.flatten_util.ravel_pytree(sol_list)[0]

@jax.jit
def solve_with_displacement(displacement_val):
    """Solve the problem for a given displacement value"""
    # Create dynamic bc_info by modifying the prescribed values
    # We need to update the displacement values for the right boundary (x-component)
    bc_info = {
        'dof_indices': static_bc_info['dof_indices'],
        'prescribed_values': static_bc_info['prescribed_values'].at[-static_bc_info['num_right_x_dofs']:].set(displacement_val),
        'vec_dim': static_bc_info['vec_dim'],
        'has_bc': static_bc_info['has_bc'],
        'num_right_x_dofs': static_bc_info['num_right_x_dofs']
    }
    
    A, b = get_sparse_system(state, sol_flat)
    A_bc, b_bc = apply_bc(A, b, bc_info)
    sol = solve(A_bc, b_bc, "cg", solver_options)
    return sol

def compute_compliance(displacement_val):
    """Compute compliance (objective function) for given displacement"""
    sol = solve_with_displacement(displacement_val)
    # Compliance = u^T * K * u (work done by external forces)
    # For linear elasticity, this represents the strain energy
    A, b = get_sparse_system(state, sol_flat)
    compliance = 0.5 * np.dot(sol, A @ sol)
    return compliance

def compute_max_displacement(displacement_val):
    """Compute maximum displacement magnitude for given BC displacement"""
    sol = solve_with_displacement(displacement_val)
    sol_reshaped = sol.reshape(-1, 3)
    # Use soft maximum to ensure differentiability
    disp_magnitudes = np.linalg.norm(sol_reshaped, axis=1)
    # Soft maximum with temperature parameter
    temperature = 10.0
    soft_max = temperature * jax.nn.logsumexp(disp_magnitudes / temperature)
    return soft_max

# Compute gradients using jax.grad
compliance_grad_fn = jax.grad(compute_compliance)
max_disp_grad_fn = jax.grad(compute_max_displacement)

# Test displacement values
test_displacement = 0.1

print("=== JAX Gradient Analysis w.r.t. Dirichlet BC ===")
print(f"Test displacement value: {test_displacement}")

# Compute function values
compliance_val = compute_compliance(test_displacement)

print(f"Compliance: {compliance_val:.6e}")

# Compute gradients
compliance_grad = compliance_grad_fn(test_displacement)

print(f"∂(compliance)/∂(displacement_BC): {compliance_grad:.6e}")

# Verify gradients with finite differences
eps = 1e-5
compliance_fd = (compute_compliance(test_displacement + eps) - 
                compute_compliance(test_displacement - eps)) / (2 * eps)

print("\n=== Finite Difference Verification ===")
print(f"Compliance gradient (FD): {compliance_fd:.6e}")
print(f"Compliance gradient error: {abs(compliance_grad - compliance_fd):.2e}")

# Demonstrate gradient-based sensitivity analysis
print("\n=== Sensitivity Analysis ===")
displacement_range = np.linspace(0.05, 0.2, 5)
for disp_val in displacement_range:
    compliance_grad_val = compliance_grad_fn(disp_val)

# Save solution for visualization
sol_final = solve_with_displacement(test_displacement)
sol_reshaped = sol_final.reshape(-1, 3)
vtk_filename = 'outputs/gradient_dirichlet_example.vtu'
save_as_vtk(
    mesh=mesh,
    sol_file=vtk_filename,
    point_infos=[
        ('displacement', sol_reshaped),
    ]
)

print(f"\nSolution saved to: {vtk_filename}")
print("Gradient-based Dirichlet BC analysis completed!")