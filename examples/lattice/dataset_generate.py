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

# Import compression utilities
from lattice_generation import (
    create_random_adjacency_matrix,
    create_cube_nodes,
    check_connectivity,
    verify_periodic_connections
)
from dataset_utils import (
    compress_symmetric_matrix,
    decompress_symmetric_matrix,
    compress_stiffness_voigt,
    decompress_stiffness_voigt
)

# Material properties
E0 = 70e3
E_eps = 1e-3
nu = 0.3
radius = 0.1

# Dataset generation parameters
N = 100 # Total number of samples to generate
M = 2   # Batch size for processing (memory management)

# Problem setup
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
    def mesh_build(self):
        return box_mesh(size=(1.0, 1.0, 1.0), mesh_size=0.05, element_type='HEX8')

unitcell = Cube()
mesh = unitcell.mesh

problem = ElasticityProblem(
    mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
    location_fns=[]
)

# Setup periodic boundary conditions
pbc = periodic_bc_3D(unitcell, 3, 3)
P = prolongation_matrix(pbc, mesh, 3)
bc_config = DirichletBCConfig([])
bc = bc_config.create_bc(problem)

# Create homogenization solver
solver_options = SolverOptions(
    linear_solver="cg",
    linear_solver_tol=1e-10,
    linear_solver_maxiter=10000
)

solver = create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3)

# Generate nodes for lattice (shared across all random variations)
nodes = create_cube_nodes(3, 3, 3)
num_nodes = nodes.shape[0]

print(f"=== Generating {N} lattice samples in batches of {M} ===")
print("Constraints enabled:")
print("  ✓ No island struts (fully connected)")
print("  ✓ Periodic boundary conditions")

# Setup solver functions
def create_lattice(adj_mat):
    """Create lattice density field from adjacency matrix."""
    lattice_fn = create_lattice_function_from_adjmat(nodes, adj_mat, radius)
    return create_lattice_density_field(problem, lattice_fn, density_void=1e-5)

def compute_C_mat(adj_mat):
    """Compute stiffness for a single lattice configuration."""
    rho = create_lattice(adj_mat)
    internal_vars = InternalVars(volume_vars=(rho,), surface_vars=[])
    return solver(internal_vars)

compute_C_mats = jax.jit(jax.vmap(compute_C_mat))

# Initialize storage for all results
all_adj_compressed = []
all_stiffness_compressed = []
all_num_connections = []

# Progress bar using tqdm
from tqdm import tqdm

# Process in batches
num_batches = (N + M - 1) // M  # Ceiling division
total_time = 0.0

for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
    batch_start_idx = batch_idx * M
    batch_end_idx = min(batch_start_idx + M, N)
    current_batch_size = batch_end_idx - batch_start_idx

    # Generate adjacency matrices for this batch with constraints
    adj_matrices_list = []
    for i in range(batch_start_idx, batch_end_idx):
        adj_mat = create_random_adjacency_matrix(
            nodes,
            connection_prob=0.3,
            max_distance=0.6,
            seed=i,
            enforce_connectivity=True,   # No island struts
            enforce_periodic=True,        # Periodic boundary conditions
            max_retries=100               # Allow retries to find valid lattice
        )
        adj_matrices_list.append(adj_mat)

    adj_matrices = np.stack(adj_matrices_list, axis=0)

    # Compute stiffness matrices
    start = time.time()
    C_hom_batch = compute_C_mats(adj_matrices)
    C_hom_batch.block_until_ready()
    end = time.time()
    batch_time = end - start
    total_time += batch_time

    # Compress data
    adj_compressed = compress_symmetric_matrix(adj_matrices)
    stiffness_compressed = compress_stiffness_voigt(C_hom_batch)
    num_connections = np.array([int(np.sum(adj) / 2) for adj in adj_matrices])

    # Store results
    all_adj_compressed.append(adj_compressed)
    all_stiffness_compressed.append(stiffness_compressed)
    all_num_connections.append(num_connections)

# Concatenate all batches
all_adj_compressed = np.concatenate(all_adj_compressed, axis=0)
all_stiffness_compressed = np.concatenate(all_stiffness_compressed, axis=0)
all_num_connections = np.concatenate(all_num_connections, axis=0)

print(f"\n=== Generation Complete ===")
print(f"Total samples: {N}")
print(f"Total time: {total_time:.2f}s ({total_time/N:.3f}s per sample)")
print(f"Adjacency compressed shape: {all_adj_compressed.shape}")
print(f"Stiffness compressed shape: {all_stiffness_compressed.shape}")

# Save as CSV format
import pandas as pd

print("\n=== Saving dataset as CSV ===")

# Create column names for adjacency (upper triangle indices)
adj_cols = [f"adj_{i}_{j}" for i in range(num_nodes) for j in range(i+1, num_nodes)]

# Create column names for stiffness (Voigt notation)
stiff_cols = [
    'C11', 'C22', 'C33', 'C44', 'C55', 'C66',  # Diagonal
    'C12', 'C13', 'C23',                         # Upper 3x3 off-diagonal
    'C14', 'C15', 'C16',                         # Row 0 off-diagonal
    'C24', 'C25', 'C26',                         # Row 1 off-diagonal
    'C34', 'C35', 'C36',                         # Row 2 off-diagonal
    'C45', 'C46', 'C56'                          # Rows 3-4 off-diagonal
]

# Create DataFrame
df = pd.DataFrame({
    'sample_id': np.arange(N),
    'num_connections': all_num_connections,
})

# Add adjacency columns
for i, col in enumerate(adj_cols):
    df[col] = all_adj_compressed[:, i]

# Add stiffness columns
for i, col in enumerate(stiff_cols):
    df[col] = all_stiffness_compressed[:, i]

# Save to CSV
csv_file = 'lattice_dataset.csv'
df.to_csv(csv_file, index=False)
print(f"Dataset saved to {csv_file}")
print(f"CSV shape: {df.shape}")
print(f"CSV size: {os.path.getsize(csv_file) / 1024:.2f} KB")

# Also save node positions separately
nodes_df = pd.DataFrame(nodes, columns=['x', 'y', 'z'])
nodes_df.to_csv('lattice_nodes.csv', index=False)
print(f"Node positions saved to lattice_nodes.csv")

# Save metadata
metadata_df = pd.DataFrame([{
    'num_nodes': num_nodes,
    'num_samples': N,
    'connection_prob': 0.3,
    'max_distance': 0.6,
    'radius': radius,
    'E0': E0,
    'nu': nu,
}])
metadata_df.to_csv('lattice_metadata.csv', index=False)
print(f"Metadata saved to lattice_metadata.csv")

# Print sample results
print("\n=== Sample Results ===")
for i in range(min(3, N)):
    stiff_sample = all_stiffness_compressed[i]
    print(f"\nLattice {i+1} ({all_num_connections[i]} connections):")
    print(f"  C11 = {stiff_sample[0]:.2f} MPa")
    print(f"  C12 = {stiff_sample[6]:.2f} MPa")
    print(f"  C44 = {stiff_sample[3]:.2f} MPa")

# Verification: Test decompression and constraints
print("\n=== Verification ===")
adj_test = decompress_symmetric_matrix(all_adj_compressed[0:1], num_nodes)
stiffness_test = decompress_stiffness_voigt(all_stiffness_compressed[0:1])
print(f"Decompressed adjacency shape: {adj_test.shape}")
print(f"Decompressed stiffness shape: {stiffness_test.shape}")
print("✓ Compression/decompression verified")

# Verify constraints on first sample
bbox_min = np.min(nodes, axis=0)
bbox_max = np.max(nodes, axis=0)
is_connected = check_connectivity(adj_test[0])
is_periodic = verify_periodic_connections(adj_test[0], nodes, bbox_min, bbox_max)
print(f"\nConstraint validation (sample 0):")
print(f"  Connectivity: {'PASS ✓' if is_connected else 'FAIL ✗'}")
print(f"  Periodicity: {'PASS ✓' if is_periodic else 'FAIL ✗'}")