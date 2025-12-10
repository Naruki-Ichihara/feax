"""Generate multiple periodic spinodoid structures with different anisotropy using vmap."""
import os
import jax
import jax.numpy as np
import feax as fe
import feax.flat as flat
import time
import pandas as pd
from tqdm import tqdm

class SpinodoidUnitCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

# Setup mesh and periodic boundary conditions
unitcell = SpinodoidUnitCell(mesh_size=0.04)
mesh = unitcell.mesh
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=1, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
cell_centers = np.mean(mesh.points[mesh.cells], axis=1)

# Dataset generation parameters
total_samples = 100  # Total number of samples to generate
batch_size = 10  # Number of samples per batch
num_batches = total_samples // batch_size

# Spinodoid generation parameters
radius = 0.1
target_vf_min = 0.3  # Minimum volume fraction
target_vf_max = 1.0  # Maximum volume fraction
beta_heaviside = 10.0
beta_grf = 15.0
N = 100
theta_min = 0.0  # Minimum theta value (radians)
theta_max = np.pi / 4  # Maximum theta value (radians)
solver_opts = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg", verbose=False)

# Material properties for homogenization
E0 = 21e3  # Young's modulus of solid phase (Pa)
nu_val = 0.3  # Poisson's ratio
p = 3.0  # SIMP penalty parameter

# Setup homogenization problem (once, outside loop)
class ElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(3) + 2 * mu * eps
        return stress

problem = ElasticityProblem(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])
bc = fe.DCboundary.DirichletBCConfig([]).create_bc(problem)
pairings_3d = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P_3d = flat.pbc.prolongation_matrix(pairings_3d, mesh, vec=3)

# Create homogenization solver
compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P_3d, solver_opts, mesh, dim=3
)

# Pipeline: GRF → Helmholtz → Heaviside
def generate_spinodoid(n_vectors, gamma, target_vf):
    rho = flat.spinodoid.evaluate_grf_field(cell_centers, n_vectors, gamma, beta_grf)
    rho = flat.filters.helmholtz_filter(rho, mesh, radius, P, solver_opts)
    rho = (rho - rho.min()) / (rho.max() - rho.min())
    thresh = flat.filters.compute_volume_fraction_threshold(rho, target_vf)
    return flat.filters.heaviside_projection(rho, beta_heaviside, thresh)

# JIT compile the generation pipeline and vectorize
generate_spinodoid_batch = jax.jit(jax.vmap(generate_spinodoid))

# Vmap over density fields for homogenization
def compute_single_C_hom(rho_nodal):
    # Convert nodal values to cell values by averaging nodes of each cell
    rho_cell = np.mean(rho_nodal[mesh.cells], axis=1)
    E_field = E0 * rho_cell**p
    E_array = fe.internal_vars.InternalVars.create_cell_var(problem, E_field)
    nu_array = fe.internal_vars.InternalVars.create_cell_var(problem, nu_val)
    internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_array, nu_array), surface_vars=())
    return compute_C_hom(internal_vars)

# JIT compile the homogenization and vectorize
compute_C_hom_batch = jax.jit(jax.vmap(compute_single_C_hom))

# Output setup
output_dir = os.path.join(os.path.dirname(__file__), "data", "csv")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "spinodoid_results.csv")

# Check if CSV exists and count existing samples
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    initial_count = len(df_existing)
    print(f"Loaded existing CSV with {initial_count} samples")
else:
    initial_count = 0
    print("Starting new dataset")

# Generate dataset in batches
initial_seed = int(time.time() * 1000) % (2**32)

for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
    # Create unique seed for this batch
    seed = (initial_seed + batch_idx) % (2**32)
    rng_key = jax.random.PRNGKey(seed)
    rng_key, theta_key, vf_key = jax.random.split(rng_key, 3)

    # Generate random parameters for this batch
    theta_configs = jax.random.uniform(
        theta_key,
        shape=(batch_size, 3),
        minval=theta_min,
        maxval=theta_max
    )
    target_vf_batch = jax.random.uniform(
        vf_key,
        shape=(batch_size,),
        minval=target_vf_min,
        maxval=target_vf_max
    )

    # Pre-generate random parameters for GRF
    keys = jax.random.split(rng_key, batch_size)
    n_vecs, gammas = [], []

    for key, thetas in zip(keys, theta_configs):
        theta1, theta2, theta3 = thetas
        key_n, key_g = jax.random.split(key)
        n_vecs.append(flat.spinodoid.generate_direction_vectors(theta1, theta2, theta3, N, key_n))
        gammas.append(jax.random.uniform(key_g, shape=(N,), minval=0.0, maxval=2*np.pi))

    n_vecs = np.stack(n_vecs)
    gammas = np.stack(gammas)

    # Generate spinodoid structures with jit(vmap)
    rho_batch = generate_spinodoid_batch(n_vecs, gammas, target_vf_batch)

    # Compute homogenized stiffness tensors with jit(vmap)
    C_hom_batch = compute_C_hom_batch(rho_batch)

    # Store results for this batch
    batch_csv_data = []
    for i in range(batch_size):
        row = {
            # Random seed for this sample
            'seed': int(keys[i][0]),
            # Design parameters
            'theta1': float(theta_configs[i, 0]),
            'theta2': float(theta_configs[i, 1]),
            'theta3': float(theta_configs[i, 2]),
            'target_vf': float(target_vf_batch[i]),
            # Meta-parameters
            'mesh_size': 0.04,
            'radius': radius,
            'beta_heaviside': beta_heaviside,
            'beta_grf': beta_grf,
            'N': N,
            'E0': E0,
            'nu': nu_val,
            'p': p
        }
        # Add all C_hom components (6x6 Voigt notation)
        C = C_hom_batch[i]
        for ii in range(6):
            for jj in range(6):
                row[f'C{ii+1}{jj+1}'] = float(C[ii, jj])
        batch_csv_data.append(row)

    # Append to CSV after each batch
    df_batch = pd.DataFrame(batch_csv_data)
    if os.path.exists(csv_path):
        df_batch.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_batch.to_csv(csv_path, mode='w', header=True, index=False)

    # Save a few example structures from first batch only
    if batch_idx == 0:
        for i in range(min(3, batch_size)):
            fe.utils.save_sol(
                mesh, os.path.join(output_dir, f"spinodoid_example_{i}.vtu"),
                point_infos=[("density", rho_batch[i].reshape(-1, 1))]
            )

# Read final CSV to get total count
df_final = pd.read_csv(csv_path)
final_count = len(df_final)
