"""Test differentiability of homogenization solver."""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, SolverOptions, DirichletBCConfig
from feax.mesh import box_mesh
from feax.lattice_toolkit.unitcell import UnitCell
from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
from feax.lattice_toolkit.solver import create_homogenization_solver
from feax.lattice_toolkit.graph import create_fcc_density

jax.config.update("jax_enable_x64", True)

# Material properties
E0 = 70e3
nu = 0.3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = E0 * rho
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

# Create coarse unit cell mesh
class Cube(UnitCell):
    def mesh_build(self, **kwargs):
        return box_mesh(size=(1.0, 1.0, 1.0), mesh_size=0.2, element_type='HEX8')

unitcell = Cube()
mesh = unitcell.mesh

# Setup periodic boundary conditions
pbc = periodic_bc_3D(unitcell, 3, 3)
P = prolongation_matrix(pbc, mesh, 3)

bc_config = DirichletBCConfig([])

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[]
)

bc = bc_config.create_bc(problem)

# Create homogenization solver
solver_options = SolverOptions(
    linear_solver="cg",
    linear_solver_tol=1e-8,
    linear_solver_maxiter=5000
)

print("Creating homogenization solver...")
compute_C_hom = create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3
)

# Test with simple density field
print("\nTesting differentiability...")
rho_test = np.ones(problem.num_cells) * 0.5

@jax.jit
def objective(rho_cell):
    """Compute trace of stiffness matrix as scalar objective."""
    internal_vars = InternalVars(volume_vars=(rho_cell,), surface_vars=[])
    C_hom = compute_C_hom(internal_vars)
    # Return trace (sum of diagonal elements)
    return np.trace(C_hom)

# Compute objective
print("Computing objective...")
obj_value = objective(rho_test)
print(f"Objective value (trace(C_hom)): {obj_value:.2f}")

# Compute gradient
print("\nComputing gradient via JAX autodiff...")
grad_fn = jax.grad(objective)
grad = grad_fn(rho_test)

print(f"Gradient shape: {grad.shape}")
print(f"Gradient range: [{np.min(grad):.2e}, {np.max(grad):.2e}]")
print(f"Gradient mean: {np.mean(grad):.2e}")
print(f"Gradient std: {np.std(grad):.2e}")

print("\n✓ Homogenization solver is differentiable!")
print("  Can now be used for topology optimization with homogenized stiffness objective.")
