"""BCC lattice homogenization example.

This example computes the homogenized stiffness tensor of a BCC lattice structure
using periodic boundary conditions and visualizes the stiffness distribution.
"""

import os
import time

import jax
import jax.numpy as np

import feax as fe
import feax.flat as flat

# Material properties
E_base = 9026 #MPa
nu = 0.3
mesh_size = 0.15#0.05
unit_cell_length = 3

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
        return fe.mesh.box_mesh(size=unit_cell_length, mesh_size=mesh_size, element_type='HEX8')

# Create unit cell and BCC graph structure
print("Creating BCC lattice unit cell...")
unitcell = BCCUnitCell(mesh_size=mesh_size)
mesh = unitcell.mesh
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} elements")

# Define BCC lattice: 8 corners + 1 center node
corners = np.array([[i, j, k] for i in [0, unit_cell_length] for j in [0, unit_cell_length] for k in [0, unit_cell_length]], dtype=np.float32)
center = np.array([[unit_cell_length/2, unit_cell_length/2, unit_cell_length/2]], dtype=np.float32)
nodes = np.vstack([corners, center])

# BCC edges: all corners connect to center
edges = np.array([[i, 8] for i in range(8)])

# Create problem first
problem = LinearElasticity(mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])

# Create lattice density field using graph (node-based)
print("Creating BCC strut structure...")
lattice_func = flat.graph.create_lattice_function(nodes, edges, radius=0.6)
rho = flat.graph.create_lattice_density_field_nodal(problem, lattice_func, density_solid=1.0, density_void=1e-6)

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
E_field = E_base * rho  # rho is nodal, from create_lattice_density_field_nodal
nu_field = fe.traced_params.TracedParams.create_cell_var(problem, nu)
traced_params = fe.traced_params.TracedParams(volume_vars=(E_field, nu_field), surface_vars=())

# Homogenization solver
print("Creating homogenization solver...")
solver_options = fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-10, verbose=True)

compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P, mesh, solver_options=solver_options, dim=3
)

print("\n--- Without JIT ---")
t0 = time.time()
result = compute_C_hom(traced_params)
jax.block_until_ready(result)
t_no_jit = time.time() - t0
print(f"  Time: {t_no_jit:.4f}s")


C_hom = result.C_hom
print(f"\nHomogenized stiffness matrix shape: {C_hom.shape}")

# Extract engineering constants
# For isotropic/cubic material: C11, C12, C44
C11 = C_hom[0, 0]
C12 = C_hom[0, 1]
C44 = C_hom[3, 3]

# Effective Young's modulus (assuming cubic symmetry)
E_eff = (C11 - C12) * (C11 + 2*C12) / (C11 + C12)
nu_eff = C12 / (C11 + C12)
G_eff = C44

print("\nHomogenized properties:")
print(f"  Effective Young's modulus: {C11/1e3:.2f} GPa")
print(f"  Effective Young's modulus: {C12/1e3:.2f} GPa")
print(f"  Effective Young's modulus: {E_eff/1e3:.2f} GPa")
print(f"  Effective Poisson's ratio: {nu_eff:.3f}")
print(f"  Effective shear modulus: {G_eff/1e3:.2f} GPa")
print(f"  Relative stiffness (E_eff/E_base): {E_eff/E_base:.3f}")
print(f"  Relative density: {rho.mean():.3f}")

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
    point_infos=[("density", rho)]
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
