"""BCC lattice homogenization example.

This example computes the homogenized stiffness tensor of a BCC lattice structure
using periodic boundary conditions and visualizes the stiffness distribution.
"""

import os
import feax as fe
import feax.flat as flat
import jax.numpy as np

# Material properties
E_base = 210e9  # Pa (steel)
nu = 0.3

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu_val):
            mu = E / (2.0 * (1.0 + nu_val))
            lmbda = E * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

class BCCUnitCell(flat.unitcell.UnitCell):
    """BCC lattice unit cell."""

    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

# Create unit cell and BCC graph structure
print("Creating BCC lattice unit cell...")
unitcell = BCCUnitCell(mesh_size=0.05)
mesh = unitcell.mesh
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} elements")

# Define BCC lattice: 8 corners + 1 center node
corners = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=np.float32)
center = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
nodes = np.vstack([corners, center])

# BCC edges: all corners connect to center
edges = np.array([[i, 8] for i in range(8)])

# Create problem first
problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])

# Create lattice density field using graph
print("Creating BCC strut structure...")
lattice_func = flat.graph.create_lattice_function(nodes, edges, radius=0.05)
rho = flat.graph.create_lattice_density_field(problem, lattice_func, density_solid=1.0, density_void=0.01)

# Periodic boundary conditions
print("Setting up periodic boundary conditions...")
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=3)
print(f"Prolongation matrix: {P.shape}")

# Boundary conditions (empty for fully periodic)
bc_config = fe.DCboundary.DirichletBCConfig([])
bc = bc_config.create_bc(problem)

# Internal variables with density-based Young's modulus
print("Setting up material properties...")
E_field = fe.internal_vars.InternalVars.create_cell_var(problem, E_base * rho)
nu_field = fe.internal_vars.InternalVars.create_cell_var(problem, nu)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_field, nu_field), surface_vars=())

# Homogenization solver
print("Creating homogenization solver...")
solver_options = fe.solver.SolverOptions(
    tol=1e-8,
    linear_solver="cudss",
    verbose=False
)

compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3
)

# Compute homogenized stiffness
print("Computing homogenized stiffness tensor...")
C_hom = compute_C_hom(internal_vars)
print(f"Homogenized stiffness matrix shape: {C_hom.shape}")

# Extract engineering constants
# For isotropic/cubic material: C11, C12, C44
C11 = C_hom[0, 0]
C12 = C_hom[0, 1]
C44 = C_hom[3, 3]

# Effective Young's modulus (assuming cubic symmetry)
E_eff = (C11 - C12) * (C11 + 2*C12) / (C11 + C12)
nu_eff = C12 / (C11 + C12)
G_eff = C44

print(f"\nHomogenized properties:")
print(f"  Effective Young's modulus: {E_eff/1e9:.2f} GPa")
print(f"  Effective Poisson's ratio: {nu_eff:.3f}")
print(f"  Effective shear modulus: {G_eff/1e9:.2f} GPa")
print(f"  Relative stiffness (E_eff/E_base): {E_eff/E_base:.3f}")

# Save lattice structure and results
print("\nSaving results...")

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "data", "vtk")
os.makedirs(output_dir, exist_ok=True)

# Save lattice mesh with density field
lattice_file = os.path.join(output_dir, "bcc_lattice_structure.vtu")
fe.utils.save_sol(
    mesh=mesh,
    sol_file=lattice_file,
    cell_infos=[("density", rho)]
)
print(f"  Saved: {lattice_file}")

# Visualize stiffness sphere
sphere_file = os.path.join(output_dir, "bcc_stiffness_sphere.vtk")
flat.utils.visualize_stiffness_sphere(
    C_hom,
    output_file=sphere_file,
)
print(f"  Saved: {sphere_file}")

print("\nHomogenization complete!")