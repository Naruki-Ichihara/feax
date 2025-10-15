"""
Vectorized homogenization analysis using vmap to analyze 30 lattice cases in parallel.
Demonstrates efficient parameter studies with JAX's automatic vectorization.

This example creates 30 different lattice configurations by:
1. Varying connectivity patterns (different adjacency matrices)
2. Using vmap to solve all cases in parallel
3. Comparing effective properties across designs
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
        return box_mesh(size=(1.0, 1.0, 1.0), mesh_size=0.12, element_type='HEX8')

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
# LATTICE GENERATION: Create 30 different lattice configurations
# ============================================================================

# Base node set: 14 nodes (8 corners + 6 face centers)
nodes = np.array([
    # Corner nodes (0-7)
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    # Face centers (8-13)
    [0.5, 0.5, 0.0],  # z=0 face
    [0.5, 0.5, 1.0],  # z=1 face
    [0.5, 0.0, 0.5],  # y=0 face
    [0.5, 1.0, 0.5],  # y=1 face
    [0.0, 0.5, 0.5],  # x=0 face
    [1.0, 0.5, 0.5],  # x=1 face
], dtype=np.float32)

n_nodes = 14
n_cases = 30

def create_lattice_case(case_id):
    """
    Create a unique lattice configuration based on case_id.
    Uses different connectivity patterns for each case.
    """
    adj_mat = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    # Base connectivity: Each face center to its 4 corners
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

    # Additional connectivity varies by case
    # We'll add different patterns of face-to-face connections

    # All possible face-to-face connections (12 total)
    face_pairs = [
        (8, 10), (8, 11), (8, 12), (8, 13),   # z=0 to others
        (9, 10), (9, 11), (9, 12), (9, 13),   # z=1 to others
        (10, 12), (10, 13),                   # y=0 to x faces
        (11, 12), (11, 13),                   # y=1 to x faces
    ]

    # Use case_id to select which face connections to add
    # Each case gets a different subset of connections
    for pair_idx, (i, j) in enumerate(face_pairs):
        # Use binary representation of case_id to determine connectivity
        if (case_id >> pair_idx) & 1:
            adj_mat = adj_mat.at[i, j].set(1)
            adj_mat = adj_mat.at[j, i].set(1)

    return adj_mat

# Generate all 30 adjacency matrices
print("Generating 30 lattice configurations...")
adj_matrices = []
for case_id in range(n_cases):
    adj_mat = create_lattice_case(case_id)
    adj_matrices.append(adj_mat)
    n_edges = np.sum(adj_mat) // 2
    print(f"  Case {case_id:2d}: {n_edges:2d} edges")

# Stack into a batch array (n_cases, n_nodes, n_nodes)
adj_matrices_batch = np.stack(adj_matrices, axis=0)

print(f"\nBatch shape: {adj_matrices_batch.shape}")

# ============================================================================
# VECTORIZED COMPUTATION
# ============================================================================

radius = 0.08

def create_lattice(adj_mat):
    """Create lattice density field from adjacency matrix."""
    lattice_fn = create_lattice_function_from_adjmat(nodes, adj_mat, radius)
    return create_lattice_density_field(problem, lattice_fn, density_void=1e-5)

def compute_stiffness_single(adj_mat):
    """Compute stiffness for a single lattice configuration."""
    rho = create_lattice(adj_mat)
    internal_vars = InternalVars(volume_vars=(rho,), surface_vars=[])
    return compute_C_hom(internal_vars)

# Vectorize the computation over batch dimension
print("\nVectorizing solver with vmap...")
compute_stiffness_batch = jax.vmap(compute_stiffness_single)

# JIT compile the vectorized computation
print("JIT compiling vectorized solver...")
compute_stiffness_batch_jit = jax.jit(compute_stiffness_batch)

# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("Running vectorized homogenization for 30 cases...")
print("=" * 80)

# First call includes JIT compilation time
print("\nFirst run (includes JIT compilation)...")
start = time.time()
C_hom_batch = compute_stiffness_batch_jit(adj_matrices_batch)
C_hom_batch.block_until_ready()  # Wait for computation to complete
end = time.time()
first_run_time = end - start

print(f"Time: {first_run_time:.2f}s")
print(f"Time per case: {first_run_time/n_cases:.3f}s")

# Second call uses cached JIT
print("\nSecond run (cached JIT)...")
start = time.time()
C_hom_batch = compute_stiffness_batch_jit(adj_matrices_batch)
C_hom_batch.block_until_ready()
end = time.time()
second_run_time = end - start

print(f"Time: {second_run_time:.2f}s")
print(f"Time per case: {second_run_time/n_cases:.3f}s")
print(f"Speedup: {first_run_time/second_run_time:.2f}x")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("Analyzing results...")
print("=" * 80)

def compute_effective_properties(C):
    """Compute effective material properties from stiffness matrix."""
    # Young's modulus (approximate)
    E_eff = C[0, 0] - 2 * C[0, 1]**2 / (C[0, 1] + C[2, 2])
    # Shear modulus
    G_eff = C[3, 3]
    # Bulk modulus
    K_eff = (C[0, 0] + C[1, 1] + C[2, 2] +
             2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    return E_eff, G_eff, K_eff

# Vectorized property computation
compute_props_vec = jax.vmap(compute_effective_properties)
E_batch, G_batch, K_batch = compute_props_vec(C_hom_batch)

# Find best and worst designs
best_stiffness_idx = np.argmax(E_batch)
worst_stiffness_idx = np.argmin(E_batch)

print("\nSummary statistics:")
print(f"  Young's modulus range: {np.min(E_batch):.2f} - {np.max(E_batch):.2f} MPa")
print(f"  Shear modulus range:   {np.min(G_batch):.2f} - {np.max(G_batch):.2f} MPa")
print(f"  Bulk modulus range:    {np.min(K_batch):.2f} - {np.max(K_batch):.2f} MPa")

print(f"\nBest design (Case {best_stiffness_idx}):")
print(f"  Young's modulus: {E_batch[best_stiffness_idx]:.2f} MPa")
print(f"  Shear modulus:   {G_batch[best_stiffness_idx]:.2f} MPa")
print(f"  Bulk modulus:    {K_batch[best_stiffness_idx]:.2f} MPa")
print(f"  Number of edges: {np.sum(adj_matrices_batch[best_stiffness_idx]) // 2}")

print(f"\nWorst design (Case {worst_stiffness_idx}):")
print(f"  Young's modulus: {E_batch[worst_stiffness_idx]:.2f} MPa")
print(f"  Shear modulus:   {G_batch[worst_stiffness_idx]:.2f} MPa")
print(f"  Bulk modulus:    {K_batch[worst_stiffness_idx]:.2f} MPa")
print(f"  Number of edges: {np.sum(adj_matrices_batch[worst_stiffness_idx]) // 2}")

# ============================================================================
# DETAILED OUTPUT FOR ALL CASES
# ============================================================================

print("\n" + "=" * 80)
print("Detailed results for all 30 cases:")
print("=" * 80)
print(f"{'Case':>4s} | {'Edges':>5s} | {'E (MPa)':>10s} | {'G (MPa)':>10s} | {'K (MPa)':>10s} | {'E/E0':>8s}")
print("-" * 80)

for i in range(n_cases):
    n_edges = np.sum(adj_matrices_batch[i]) // 2
    E_val = E_batch[i]
    G_val = G_batch[i]
    K_val = K_batch[i]
    relative_E = E_val / E0

    marker = ""
    if i == best_stiffness_idx:
        marker = " <- BEST"
    elif i == worst_stiffness_idx:
        marker = " <- WORST"

    print(f"{i:4d} | {n_edges:5d} | {E_val:10.2f} | {G_val:10.2f} | {K_val:10.2f} | {relative_E:8.4f}{marker}")

# ============================================================================
# SAVE VISUALIZATIONS FOR TOP 5 DESIGNS
# ============================================================================

print("\n" + "=" * 80)
print("Saving visualizations for top 5 designs...")
print("=" * 80)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_dir = os.path.join(data_dir, 'vtk_vmap')
os.makedirs(vtk_dir, exist_ok=True)

# Get top 5 designs by stiffness
top_5_indices = np.argsort(E_batch)[-5:][::-1]

for rank, idx in enumerate(top_5_indices):
    print(f"\nRank {rank+1}: Case {idx}")
    print(f"  Young's modulus: {E_batch[idx]:.2f} MPa")

    # Save lattice structure
    import meshio
    adj_mat = adj_matrices_batch[idx]
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if adj_mat[i, j] != 0:
                edges.append([i, j])

    if len(edges) > 0:
        edges_array = np.array(edges, dtype=np.int32)
        lattice_mesh = meshio.Mesh(
            points=np.array(nodes),
            cells=[("line", edges_array)],
        )
        lattice_file = os.path.join(vtk_dir, f'lattice_rank{rank+1}_case{idx}.vtu')
        meshio.write(lattice_file, lattice_mesh)
        print(f"  Saved lattice: {lattice_file}")

    # Save stiffness sphere
    stiffness_file = os.path.join(vtk_dir, f'stiffness_rank{rank+1}_case{idx}.vtu')
    stats = visualize_stiffness_sphere(C_hom_batch[idx], stiffness_file, n_theta=60, n_phi=120)
    print(f"  Saved stiffness sphere: {stiffness_file}")
    print(f"  Anisotropy ratio: {stats['anisotropy_ratio']:.3f}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("Correlation analysis:")
print("=" * 80)

# Number of edges vs stiffness
edge_counts = np.array([np.sum(adj_matrices_batch[i]) // 2 for i in range(n_cases)])

# Compute correlation coefficient
corr_E_edges = np.corrcoef(edge_counts, np.array(E_batch))[0, 1]
corr_G_edges = np.corrcoef(edge_counts, np.array(G_batch))[0, 1]

print(f"Correlation between edge count and Young's modulus: {corr_E_edges:.3f}")
print(f"Correlation between edge count and Shear modulus:   {corr_G_edges:.3f}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print(f"\nOutput files saved to: {vtk_dir}/")
print(f"Total computation time: {first_run_time:.2f}s for {n_cases} cases")
print(f"Average time per case: {first_run_time/n_cases:.3f}s")
