"""
Periodic homogenization to compute effective stiffness matrix.
Uses vmap to analyze all strain cases and compute homogenized stiffness via asymptotic homogenization.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, SolverOptions, DirichletBCConfig
from feax.mesh import box_mesh
import os
import time

from feax.lattice_toolkit.unitcell import UnitCell
from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
from feax.lattice_toolkit.solver import create_homogenization_solver
from feax.lattice_toolkit.graph import create_fcc_density
from feax.lattice_toolkit.utils import visualize_stiffness_sphere

# Material properties
E0 = 70e3
E_eps = 1e-3
nu = 0.3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = (E0 - E_eps) * rho + E_eps
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

# Create unit cell mesh


class Cube(UnitCell):
    def mesh_build(self, **kwargs):
        return box_mesh(size=(1.0, 1.0, 1.0), mesh_size=0.1, element_type='HEX8')

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

# Create FCC lattice density
print("Creating FCC lattice...")
rho_element = create_fcc_density(
    problem,
    radius=0.08,
    scale=1.0,
    density_solid=1.0,
    density_void=0.0
)

print(f"Density shape: {rho_element.shape}, volume fraction: {np.mean(rho_element):.3f}")

bc = bc_config.create_bc(problem)

# Create homogenization solver
solver_options = SolverOptions(
    linear_solver="cg",
    linear_solver_tol=1e-10,
    linear_solver_maxiter=10000
)

print("\nComputing homogenized stiffness matrix...")
compute_C_hom = create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3
)

@jax.jit
def compute_stiffness(rho_cell):
    # Cell-based variables (num_cells,) are automatically broadcast to quad points
    internal_vars = InternalVars(volume_vars=(rho_cell,), surface_vars=[])
    return compute_C_hom(internal_vars)

# Compute homogenized stiffness using cell-based density
start = time.time()
C_hom = compute_stiffness(rho_element)
end = time.time()

print(f"Computation time: {end - start:.2f}s")
print(f"\nHomogenized stiffness matrix (6×6 Voigt notation):")
print(C_hom)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Visualize FCC lattice stiffness
vtk_fcc = os.path.join(data_dir, 'vtk/stiffness_sphere_fcc.vtu')
os.makedirs(os.path.dirname(vtk_fcc), exist_ok=True)
stats_fcc = visualize_stiffness_sphere(C_hom, vtk_fcc, n_theta=30, n_phi=60)
