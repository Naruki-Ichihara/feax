"""
Direct comparison: For-loop vs vmap for lattice homogenization.
Demonstrates the performance difference between sequential and vectorized computation.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, SolverOptions, DirichletBCConfig
from feax.mesh import box_mesh
import time

from feax.lattice_toolkit.unitcell import UnitCell
from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
from feax.lattice_toolkit.solver import create_homogenization_solver
from feax.lattice_toolkit.graph import create_lattice_function_from_adjmat, create_lattice_density_field

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
        return box_mesh(size=(1.0, 1.0, 1.0), mesh_size=0.05, element_type='HEX8')

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
# GENERATE TEST CASES
# ============================================================================

# Base node set: 14 nodes (8 corners + 6 face centers)
nodes = np.array([
    # Corner nodes (0-7)
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    # Face centers (8-13)
    [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
    [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
    [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],
], dtype=np.float32)

n_nodes = 14
n_cases = 10
radius = 0.08

def create_lattice_case(case_id):
    """Create a unique lattice configuration based on case_id."""
    adj_mat = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    # Base connectivity: Each face center to its 4 corners
    face_corner_map = [
        (8, [0, 1, 2, 3]), (9, [4, 5, 6, 7]),
        (10, [0, 1, 4, 5]), (11, [2, 3, 6, 7]),
        (12, [0, 2, 4, 6]), (13, [1, 3, 5, 7]),
    ]

    for face_idx, corner_list in face_corner_map:
        for corner in corner_list:
            adj_mat = adj_mat.at[face_idx, corner].set(1)
            adj_mat = adj_mat.at[corner, face_idx].set(1)

    # Additional connectivity varies by case
    face_pairs = [
        (8, 10), (8, 11), (8, 12), (8, 13),
        (9, 10), (9, 11), (9, 12), (9, 13),
        (10, 12), (10, 13), (11, 12), (11, 13),
    ]

    for pair_idx, (i, j) in enumerate(face_pairs):
        if (case_id >> pair_idx) & 1:
            adj_mat = adj_mat.at[i, j].set(1)
            adj_mat = adj_mat.at[j, i].set(1)

    return adj_mat

# Generate all adjacency matrices
print(f"Generating {n_cases} lattice configurations...")
adj_matrices = [create_lattice_case(i) for i in range(n_cases)]
adj_matrices_batch = np.stack(adj_matrices, axis=0)

print(f"Batch shape: {adj_matrices_batch.shape}")
print(f"Mesh info: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements\n")

# ============================================================================
# DEFINE SOLVER FUNCTIONS
# ============================================================================

def create_lattice(adj_mat):
    """Create lattice density field from adjacency matrix."""
    lattice_fn = create_lattice_function_from_adjmat(nodes, adj_mat, radius)
    return create_lattice_density_field(problem, lattice_fn, density_void=1e-5)

def compute_stiffness_single(adj_mat):
    """Compute stiffness for a single lattice configuration."""
    rho = create_lattice(adj_mat)
    internal_vars = InternalVars(volume_vars=(rho,), surface_vars=[])
    return compute_C_hom(internal_vars)

# JIT compile single case solver
compute_stiffness_single_jit = jax.jit(compute_stiffness_single)

# Create vectorized version
compute_stiffness_batch = jax.vmap(compute_stiffness_single)
compute_stiffness_batch_jit = jax.jit(compute_stiffness_batch)

# ============================================================================
# METHOD 1: FOR LOOP (Sequential)
# ============================================================================

print("=" * 80)
print("METHOD 1: FOR LOOP (Sequential Processing)")
print("=" * 80)

print("\nFirst run (includes JIT compilation)...")
results_loop = []
start = time.time()
for i, adj_mat in enumerate(adj_matrices):
    C_hom = compute_stiffness_single_jit(adj_mat)
    C_hom.block_until_ready()  # Wait for computation
    results_loop.append(C_hom)
    if i == 0:
        first_case_time = time.time() - start
        print(f"  First case time: {first_case_time:.3f}s (includes JIT compilation)")
end = time.time()
loop_time_first = end - start

print(f"\nTotal time: {loop_time_first:.2f}s")
print(f"Average per case: {loop_time_first/n_cases:.3f}s")

# Stack results
C_hom_loop = np.stack(results_loop, axis=0)

print("\nSecond run (cached JIT)...")
results_loop2 = []
start = time.time()
for adj_mat in adj_matrices:
    C_hom = compute_stiffness_single_jit(adj_mat)
    C_hom.block_until_ready()
    results_loop2.append(C_hom)
end = time.time()
loop_time_cached = end - start

print(f"Total time: {loop_time_cached:.2f}s")
print(f"Average per case: {loop_time_cached/n_cases:.3f}s")

# ============================================================================
# METHOD 2: VMAP (Vectorized)
# ============================================================================

print("\n" + "=" * 80)
print("METHOD 2: VMAP (Vectorized Processing)")
print("=" * 80)

print("\nFirst run (includes JIT compilation)...")
start = time.time()
C_hom_vmap = compute_stiffness_batch_jit(adj_matrices_batch)
C_hom_vmap.block_until_ready()
end = time.time()
vmap_time_first = end - start

print(f"Total time: {vmap_time_first:.2f}s")
print(f"Average per case: {vmap_time_first/n_cases:.3f}s")

print("\nSecond run (cached JIT)...")
start = time.time()
C_hom_vmap = compute_stiffness_batch_jit(adj_matrices_batch)
C_hom_vmap.block_until_ready()
end = time.time()
vmap_time_cached = end - start

print(f"Total time: {vmap_time_cached:.2f}s")
print(f"Average per case: {vmap_time_cached/n_cases:.3f}s")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

print("\nFirst run (with JIT compilation):")
print(f"  For loop: {loop_time_first:.2f}s")
print(f"  Vmap:     {vmap_time_first:.2f}s")
print(f"  Speedup:  {loop_time_first/vmap_time_first:.2f}x")

print("\nCached run (JIT already compiled):")
print(f"  For loop: {loop_time_cached:.2f}s")
print(f"  Vmap:     {vmap_time_cached:.2f}s")
print(f"  Speedup:  {loop_time_cached/vmap_time_cached:.2f}x")

# Verify results match
max_diff = np.max(np.abs(C_hom_loop - C_hom_vmap))
print(f"\nResult verification:")
print(f"  Max difference between methods: {max_diff:.2e}")
print(f"  Results match: {max_diff < 1e-6}")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

def compute_effective_youngs(C):
    return C[0, 0] - 2 * C[0, 1]**2 / (C[0, 1] + C[2, 2])

# Compute effective properties
E_values = np.array([compute_effective_youngs(C) for C in C_hom_vmap])

print(f"\nYoung's modulus statistics:")
print(f"  Min: {np.min(E_values):.2f} MPa")
print(f"  Max: {np.max(E_values):.2f} MPa")
print(f"  Mean: {np.mean(E_values):.2f} MPa")
print(f"  Std: {np.std(E_values):.2f} MPa")

best_idx = np.argmax(E_values)
print(f"\nBest design: Case {best_idx} with E = {E_values[best_idx]:.2f} MPa")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{n_cases} lattice cases analyzed:")
print(f"  For loop (sequential):  {loop_time_cached:.2f}s")
print(f"  Vmap (vectorized):      {vmap_time_cached:.2f}s")
print(f"  Speedup:                {loop_time_cached/vmap_time_cached:.2f}x")

print(f"\nKey advantages of vmap:")
print(f"  1. Faster execution ({loop_time_cached/vmap_time_cached:.1f}x speedup)")
print(f"  2. Cleaner code (no explicit loops)")
print(f"  3. Automatic parallelization by JAX")
print(f"  4. GPU-ready (same code)")
print(f"  5. Better memory efficiency")

print(f"\nWhen to use each method:")
print(f"  For loop: Single/few cases, debugging, varying solver params")
print(f"  Vmap:     Many cases, parameter studies, optimization")

print("\n" + "=" * 80)
