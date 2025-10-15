"""
Complex lattice periodic homogenization example.
Demonstrates advanced lattice structures with multiple connectivity patterns:
1. BCC-like lattice with face and edge center nodes
2. Octet truss lattice
3. Kelvin foam lattice

Each lattice has significantly more nodes and complex adjacency patterns.
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
from feax.lattice_toolkit.graph import create_lattice_function_from_adjmat, create_lattice_density_field
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

bc = bc_config.create_bc(problem)

# Create homogenization solver
solver_options = SolverOptions(
    linear_solver="cg",
    linear_solver_tol=1e-10,
    linear_solver_maxiter=10000
)

compute_C_hom = create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3
)


# ============================================================================
# LATTICE DEFINITION 1: BCC with Face Centers (19 nodes)
# ============================================================================
def create_bcc_face_lattice():
    """
    BCC lattice with additional face center nodes.
    - 8 corner nodes (cube vertices)
    - 1 body center node
    - 6 face center nodes
    - 12 edge center nodes
    Total: 27 nodes
    """
    # Corner nodes (8)
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=np.float32)

    # Body center (1)
    body_center = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    # Face centers (6)
    face_centers = np.array([
        [0.5, 0.5, 0.0],  # z=0 face
        [0.5, 0.5, 1.0],  # z=1 face
        [0.5, 0.0, 0.5],  # y=0 face
        [0.5, 1.0, 0.5],  # y=1 face
        [0.0, 0.5, 0.5],  # x=0 face
        [1.0, 0.5, 0.5],  # x=1 face
    ], dtype=np.float32)

    # Edge centers (12)
    edge_centers = np.array([
        # Bottom face edges (z=0)
        [0.5, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 0.0],
        # Top face edges (z=1)
        [0.5, 0.0, 1.0], [1.0, 0.5, 1.0], [0.5, 1.0, 1.0], [0.0, 0.5, 1.0],
        # Vertical edges
        [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5],
    ], dtype=np.float32)

    nodes = np.vstack([corners, body_center, face_centers, edge_centers])

    # Build adjacency matrix (27x27)
    n = 27
    adj_mat = np.zeros((n, n), dtype=np.int32)

    # BCC connectivity: body center to all corners
    for i in range(8):
        adj_mat = adj_mat.at[8, i].set(1)
        adj_mat = adj_mat.at[i, 8].set(1)

    # Face centers to corners of their face
    face_corner_connections = [
        (9, [0, 1, 2, 3]),    # z=0 face
        (10, [4, 5, 6, 7]),   # z=1 face
        (11, [0, 1, 4, 5]),   # y=0 face
        (12, [2, 3, 6, 7]),   # y=1 face
        (13, [0, 2, 4, 6]),   # x=0 face
        (14, [1, 3, 5, 7]),   # x=1 face
    ]

    for face_idx, corner_indices in face_corner_connections:
        for corner in corner_indices:
            adj_mat = adj_mat.at[face_idx, corner].set(1)
            adj_mat = adj_mat.at[corner, face_idx].set(1)

    # Edge centers to their endpoints
    edge_connections = [
        # Bottom face (z=0)
        (15, [0, 1]), (16, [1, 3]), (17, [2, 3]), (18, [0, 2]),
        # Top face (z=1)
        (19, [4, 5]), (20, [5, 7]), (21, [6, 7]), (22, [4, 6]),
        # Vertical edges
        (23, [0, 4]), (24, [1, 5]), (25, [3, 7]), (26, [2, 6]),
    ]

    for edge_idx, endpoints in edge_connections:
        for endpoint in endpoints:
            adj_mat = adj_mat.at[edge_idx, endpoint].set(1)
            adj_mat = adj_mat.at[endpoint, edge_idx].set(1)

    # Connect body center to face centers
    for face_idx in range(9, 15):
        adj_mat = adj_mat.at[8, face_idx].set(1)
        adj_mat = adj_mat.at[face_idx, 8].set(1)

    return nodes, adj_mat


# ============================================================================
# LATTICE DEFINITION 2: Octet Truss (14 nodes)
# ============================================================================
def create_octet_truss_lattice():
    """
    Octet truss lattice structure.
    - 8 corner nodes
    - 6 face center nodes
    Total: 14 nodes

    Combines face-diagonal connections creating tetrahedral-octahedral pattern.
    """
    # Corner nodes (8)
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=np.float32)

    # Face centers (6)
    face_centers = np.array([
        [0.5, 0.5, 0.0],  # z=0 face (index 8)
        [0.5, 0.5, 1.0],  # z=1 face (index 9)
        [0.5, 0.0, 0.5],  # y=0 face (index 10)
        [0.5, 1.0, 0.5],  # y=1 face (index 11)
        [0.0, 0.5, 0.5],  # x=0 face (index 12)
        [1.0, 0.5, 0.5],  # x=1 face (index 13)
    ], dtype=np.float32)

    nodes = np.vstack([corners, face_centers])

    # Build adjacency matrix (14x14)
    n = 14
    adj_mat = np.zeros((n, n), dtype=np.int32)

    # Connect each face center to its 4 corners
    face_corner_map = [
        (8, [0, 1, 2, 3]),    # z=0 face
        (9, [4, 5, 6, 7]),    # z=1 face
        (10, [0, 1, 4, 5]),   # y=0 face
        (11, [2, 3, 6, 7]),   # y=1 face
        (12, [0, 2, 4, 6]),   # x=0 face
        (13, [1, 3, 5, 7]),   # x=1 face
    ]

    for face_idx, corner_list in face_corner_map:
        for corner in corner_list:
            adj_mat = adj_mat.at[face_idx, corner].set(1)
            adj_mat = adj_mat.at[corner, face_idx].set(1)

    # Connect face centers to adjacent face centers (12 connections)
    face_adjacency = [
        (8, 10), (8, 11), (8, 12), (8, 13),   # z=0 to others
        (9, 10), (9, 11), (9, 12), (9, 13),   # z=1 to others
        (10, 12), (10, 13),                   # y=0 to x faces
        (11, 12), (11, 13),                   # y=1 to x faces
    ]

    for i, j in face_adjacency:
        adj_mat = adj_mat.at[i, j].set(1)
        adj_mat = adj_mat.at[j, i].set(1)

    return nodes, adj_mat


# ============================================================================
# LATTICE DEFINITION 3: Kelvin Foam (Tetrakaidecahedron) (24 nodes)
# ============================================================================
def create_kelvin_foam_lattice():
    """
    Kelvin foam lattice (tetrakaidecahedron).
    - 8 corner nodes at cube vertices
    - 6 face center nodes
    - 8 nodes at positions creating truncated octahedron
    - 2 additional internal nodes
    Total: 24 nodes

    Creates a space-filling foam structure.
    """
    # Corner nodes (8)
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=np.float32)

    # Face centers (6)
    face_centers = np.array([
        [0.5, 0.5, 0.0],  # z=0 face (8)
        [0.5, 0.5, 1.0],  # z=1 face (9)
        [0.5, 0.0, 0.5],  # y=0 face (10)
        [0.5, 1.0, 0.5],  # y=1 face (11)
        [0.0, 0.5, 0.5],  # x=0 face (12)
        [1.0, 0.5, 0.5],  # x=1 face (13)
    ], dtype=np.float32)

    # Truncation nodes (8) - slightly offset from corners
    offset = 0.25
    truncation_nodes = np.array([
        [offset, offset, offset],
        [1-offset, offset, offset],
        [offset, 1-offset, offset],
        [1-offset, 1-offset, offset],
        [offset, offset, 1-offset],
        [1-offset, offset, 1-offset],
        [offset, 1-offset, 1-offset],
        [1-offset, 1-offset, 1-offset],
    ], dtype=np.float32)

    # Internal nodes (2)
    internal_nodes = np.array([
        [0.33, 0.33, 0.5],
        [0.67, 0.67, 0.5],
    ], dtype=np.float32)

    nodes = np.vstack([corners, face_centers, truncation_nodes, internal_nodes])

    # Build adjacency matrix (24x24)
    n = 24
    adj_mat = np.zeros((n, n), dtype=np.int32)

    # Connect corners to their truncation nodes
    for i in range(8):
        adj_mat = adj_mat.at[i, 14+i].set(1)
        adj_mat = adj_mat.at[14+i, i].set(1)

    # Connect truncation nodes to nearby face centers
    truncation_face_connections = [
        (14, [8, 10, 12]),  # corner (0,0,0) truncation
        (15, [8, 10, 13]),  # corner (1,0,0) truncation
        (16, [8, 11, 12]),  # corner (0,1,0) truncation
        (17, [8, 11, 13]),  # corner (1,1,0) truncation
        (18, [9, 10, 12]),  # corner (0,0,1) truncation
        (19, [9, 10, 13]),  # corner (1,0,1) truncation
        (20, [9, 11, 12]),  # corner (0,1,1) truncation
        (21, [9, 11, 13]),  # corner (1,1,1) truncation
    ]

    for trunc_idx, faces in truncation_face_connections:
        for face in faces:
            adj_mat = adj_mat.at[trunc_idx, face].set(1)
            adj_mat = adj_mat.at[face, trunc_idx].set(1)

    # Connect truncation nodes to form hexagonal patterns
    truncation_ring_connections = [
        (14, 15), (15, 17), (17, 16), (16, 14),  # bottom ring
        (18, 19), (19, 21), (21, 20), (20, 18),  # top ring
        (14, 18), (15, 19), (16, 20), (17, 21),  # vertical connections
    ]

    for i, j in truncation_ring_connections:
        adj_mat = adj_mat.at[i, j].set(1)
        adj_mat = adj_mat.at[j, i].set(1)

    # Connect internal nodes to truncation nodes
    for trunc_idx in range(14, 22):
        for internal_idx in [22, 23]:
            adj_mat = adj_mat.at[trunc_idx, internal_idx].set(1)
            adj_mat = adj_mat.at[internal_idx, trunc_idx].set(1)

    # Connect internal nodes to each other
    adj_mat = adj_mat.at[22, 23].set(1)
    adj_mat = adj_mat.at[23, 22].set(1)

    return nodes, adj_mat


# ============================================================================
# VTK EXPORT UTILITIES
# ============================================================================

def save_lattice_vtk(nodes, adj_mat, filename):
    """
    Save lattice structure as VTK file for visualization.

    Args:
        nodes: Node coordinates (num_nodes, 3)
        adj_mat: Adjacency matrix (num_nodes, num_nodes)
        filename: Output VTK file path
    """
    import meshio

    # Extract edges from adjacency matrix (upper triangle only)
    n = adj_mat.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_mat[i, j] != 0:
                edges.append([i, j])

    edges = np.array(edges, dtype=np.int32)

    # Create meshio mesh with line elements
    cells = [("line", edges)]

    # Create mesh object
    mesh = meshio.Mesh(
        points=np.array(nodes),
        cells=cells,
    )

    # Write to VTK file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    meshio.write(filename, mesh)
    print(f"  Saved lattice structure to: {filename}")


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

radius = 0.08

def create_lattice(nodes, adj_mat):
    lattice_fn = create_lattice_function_from_adjmat(nodes, adj_mat, radius)
    return create_lattice_density_field(problem, lattice_fn, density_void=1e-5)

@jax.jit
def compute_stiffness(nodes, adj_mat):
    rho = create_lattice(nodes, adj_mat)
    internal_vars = InternalVars(volume_vars=(rho,), surface_vars=[])
    return compute_C_hom(internal_vars)


# Test all three lattice structures
lattice_configs = [
    ("BCC with Face Centers (27 nodes)", create_bcc_face_lattice()),
    ("Octet Truss (14 nodes)", create_octet_truss_lattice()),
    ("Kelvin Foam (24 nodes)", create_kelvin_foam_lattice()),
]

print("=" * 80)
print("Complex Lattice Homogenization Analysis")
print("=" * 80)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# Create output directories
vtk_dir = os.path.join(data_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)

for idx, (lattice_name, (nodes, adj_mat)) in enumerate(lattice_configs):
    print(f"\n{lattice_name}")
    print("-" * 80)
    print(f"Number of nodes: {nodes.shape[0]}")
    print(f"Number of edges: {np.sum(adj_mat) // 2}")

    start = time.time()
    C_hom = compute_stiffness(nodes, adj_mat)
    end = time.time()

    print(f"Computation time: {end - start:.2f}s")
    print(f"\nHomogenized stiffness matrix (6×6 Voigt notation):")
    print(C_hom)

    # Compute effective properties
    # Young's modulus in x-direction (approximate)
    E_eff = C_hom[0, 0] - 2 * C_hom[0, 1]**2 / (C_hom[0, 1] + C_hom[2, 2])
    # Shear modulus
    G_eff = C_hom[3, 3]
    # Bulk modulus
    K_eff = (C_hom[0, 0] + C_hom[1, 1] + C_hom[2, 2] +
             2 * (C_hom[0, 1] + C_hom[0, 2] + C_hom[1, 2])) / 9

    print(f"\nEffective properties:")
    print(f"  Young's modulus (approx): {E_eff:.2f} MPa")
    print(f"  Shear modulus: {G_eff:.2f} MPa")
    print(f"  Bulk modulus: {K_eff:.2f} MPa")
    print(f"  Relative density (approx): {E_eff/E0:.4f}")

    # Save lattice structure as VTK
    lattice_filename = ['bcc_face', 'octet_truss', 'kelvin_foam'][idx]
    lattice_vtk = os.path.join(vtk_dir, f'lattice_{lattice_filename}.vtu')
    save_lattice_vtk(nodes, adj_mat, lattice_vtk)

    # Visualize stiffness sphere
    stiffness_vtk = os.path.join(vtk_dir, f'stiffness_sphere_{lattice_filename}.vtu')
    print(f"  Generating stiffness sphere visualization...")
    stats = visualize_stiffness_sphere(C_hom, stiffness_vtk, n_theta=90, n_phi=180)
    print(f"  Saved stiffness sphere to: {stiffness_vtk}")

    # Print stiffness statistics
    print(f"\n  Stiffness sphere statistics:")
    print(f"    Min Young's modulus: {stats['E_min']:.2f} MPa")
    print(f"    Max Young's modulus: {stats['E_max']:.2f} MPa")
    print(f"    Anisotropy ratio: {stats['anisotropy_ratio']:.3f}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print(f"\nOutput files saved to: {data_dir}/")
print(f"  - Lattice structures: {vtk_dir}/lattice_*.vtu")
print(f"  - Stiffness spheres: {vtk_dir}/stiffness_sphere_*.vtu")
