"""Reconstruct spinodoid structures from CSV parameters and save to VTU/OBJ.

Usage:
    python reconstruct_spinodoids.py              # Reconstruct all samples (VTU)
    python reconstruct_spinodoids.py --obj 0 5 10 # Reconstruct samples 0, 5, 10 (OBJ)
    python reconstruct_spinodoids.py --range 0 10 # Reconstruct samples 0-9 (VTU)
    python reconstruct_spinodoids.py --obj --range 0 10 # Reconstruct 0-9 (OBJ)
"""
import os
import sys
import jax
import jax.numpy as np
import numpy as onp
import pandas as pd
import feax as fe
import feax.flat as flat
from tqdm import tqdm
try:
    from skimage import measure
    import meshio
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Parse command line arguments
output_obj = False
args = sys.argv[1:]

# Check for --obj flag
if '--obj' in args:
    if not HAS_SKIMAGE:
        print("Error: --obj requires scikit-image. Install with: pip install scikit-image")
        sys.exit(1)
    output_obj = True
    args.remove('--obj')

# Parse sample IDs
if len(args) > 0:
    if args[0] == '--range':
        start_id = int(args[1])
        end_id = int(args[2])
        sample_ids = list(range(start_id, end_id))
    else:
        sample_ids = [int(arg) for arg in args]
else:
    sample_ids = None

# Load CSV
csv_path = os.path.join(os.path.dirname(__file__), "data", "vtk", "spinodoid_results.csv")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} samples from CSV")

if sample_ids is not None:
    # Validate IDs
    invalid_ids = [sid for sid in sample_ids if sid >= len(df)]
    if invalid_ids:
        print(f"Error: Invalid sample IDs (max {len(df)-1}): {invalid_ids}")
        sys.exit(1)
    print(f"Reconstructing {len(sample_ids)} specified samples: {sample_ids}")
else:
    sample_ids = list(range(len(df)))
    print(f"Reconstructing all {len(sample_ids)} samples")

if output_obj:
    print(f"Output format: OBJ (solid mesh)")
else:
    print(f"Output format: VTU (density field)")

# Check for meta-parameters in CSV
has_meta_params = 'mesh_size' in df.columns and 'seed' in df.columns
if has_meta_params:
    print("CSV contains meta-parameters - using them for reconstruction")
else:
    print("CSV missing meta-parameters - using default values")

# Generate spinodoid from parameters
def generate_spinodoid(row):
    # Extract meta-parameters from CSV or use defaults
    if has_meta_params:
        mesh_size = row['mesh_size']
        radius = row['radius']
        beta_heaviside = row['beta_heaviside']
        beta_grf = row['beta_grf']
        N = int(row['N'])
        seed = int(row['seed'])
    else:
        mesh_size = 0.04
        radius = 0.1
        beta_heaviside = 10.0
        beta_grf = 15.0
        N = 100
        seed = row.name

    # Setup mesh with correct size
    class SpinodoidUnitCell(flat.unitcell.UnitCell):
        def mesh_build(self, mesh_size):
            return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

    unitcell = SpinodoidUnitCell(mesh_size=mesh_size)
    mesh = unitcell.mesh
    pairings = flat.pbc.periodic_bc_3D(unitcell, vec=1, dim=3)
    P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
    cell_centers = np.mean(mesh.points[mesh.cells], axis=1)
    solver_opts = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg", verbose=False)

    # Extract design parameters
    theta1, theta2, theta3 = row['theta1'], row['theta2'], row['theta3']
    target_vf = row['target_vf']

    # Generate direction vectors and phases using stored seed
    key = jax.random.PRNGKey(seed)
    key_n, key_g = jax.random.split(key)
    n_vectors = flat.spinodoid.generate_direction_vectors(theta1, theta2, theta3, N, key_n)
    gamma = jax.random.uniform(key_g, shape=(N,), minval=0.0, maxval=2*np.pi)

    # Generate spinodoid
    rho = flat.spinodoid.evaluate_grf_field(cell_centers, n_vectors, gamma, beta_grf)
    rho = flat.filters.helmholtz_filter(rho, mesh, radius, P, solver_opts)
    rho = (rho - rho.min()) / (rho.max() - rho.min())
    thresh = flat.filters.compute_volume_fraction_threshold(rho, target_vf)
    rho = flat.filters.heaviside_projection(rho, beta_heaviside, thresh)

    return mesh, rho, mesh_size

def density_to_obj_mesh(mesh, rho, threshold=0.5, refinement=2):
    """Convert density field to closed surface mesh using marching cubes.

    Args:
        mesh: FE mesh
        rho: density field
        threshold: isosurface level
        refinement: upsampling factor for finer mesh (1=no refinement, 2=2x finer, etc.)

    Adds thin padding layer outside domain to ensure closed mesh.
    """
    from scipy.ndimage import zoom

    # Get mesh grid dimensions
    points = onp.array(mesh.points)

    # Sort and get unique coordinates
    x_unique = onp.sort(onp.unique(onp.round(points[:, 0], 10)))
    y_unique = onp.sort(onp.unique(onp.round(points[:, 1], 10)))
    z_unique = onp.sort(onp.unique(onp.round(points[:, 2], 10)))

    nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

    # Calculate spacing
    spacing = (
        x_unique[1]-x_unique[0] if len(x_unique) > 1 else 1.0,
        y_unique[1]-y_unique[0] if len(y_unique) > 1 else 1.0,
        z_unique[1]-z_unique[0] if len(z_unique) > 1 else 1.0
    )

    # Create mapping from points to grid indices
    rho_np = onp.array(rho)
    rho_3d = onp.zeros((nx, ny, nz))

    # Fill 3D grid with density values
    for i, point in enumerate(points):
        ix = onp.argmin(onp.abs(x_unique - point[0]))
        iy = onp.argmin(onp.abs(y_unique - point[1]))
        iz = onp.argmin(onp.abs(z_unique - point[2]))
        rho_3d[ix, iy, iz] = rho_np[i]

    # Upsample density field if refinement > 1
    if refinement > 1:
        rho_3d = zoom(rho_3d, refinement, order=3)  # cubic interpolation
        # Update spacing after refinement
        spacing = tuple(s / refinement for s in spacing)

    # Add padding layer
    rho_3d_padded = onp.pad(rho_3d, pad_width=1, mode='constant', constant_values=0.0)

    # Apply marching cubes to extract isosurface
    verts, faces, normals, values = measure.marching_cubes(
        rho_3d_padded,
        level=threshold,
        spacing=spacing
    )

    # Offset vertices to match original coordinates (account for padding offset)
    verts[:, 0] += x_unique[0] - spacing[0]
    verts[:, 1] += y_unique[0] - spacing[1]
    verts[:, 2] += z_unique[0] - spacing[2]

    return verts, faces

# Output directory
if output_obj:
    output_dir = os.path.join(os.path.dirname(__file__), "data", "obj_dataset")
else:
    output_dir = os.path.join(os.path.dirname(__file__), "data", "vtu_dataset")
os.makedirs(output_dir, exist_ok=True)

# Reconstruct and save
for sample_id in tqdm(sample_ids, desc="Reconstructing"):
    row = df.iloc[sample_id]

    # Extract C_hom
    C_hom = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            C_hom = C_hom.at[i, j].set(row[f'C{i+1}{j+1}'])

    # Generate spinodoid with exact parameters from CSV
    mesh, rho, mesh_size = generate_spinodoid(row)

    if output_obj:
        # Convert to OBJ mesh
        try:
            verts, faces = density_to_obj_mesh(mesh, rho, threshold=0.5)

            # Save as OBJ using meshio
            obj_path = os.path.join(output_dir, f"spinodoid_{sample_id:04d}.obj")
            mesh_obj = meshio.Mesh(
                points=verts,
                cells=[("triangle", faces)]
            )
            meshio.write(obj_path, mesh_obj)
        except Exception as e:
            print(f"\nWarning: Failed to convert sample {sample_id} to OBJ: {e}")
            continue
    else:
        # Save stiffness sphere visualization
        sphere_path = os.path.join(output_dir, f"stiffness_sphere_{sample_id:04d}.vtu")
        flat.utils.visualize_stiffness_sphere(C_hom, sphere_path, n_theta=30, n_phi=60)

        # Save spinodoid structure as VTU
        vtu_path = os.path.join(output_dir, f"spinodoid_{sample_id:04d}.vtu")
        fe.utils.save_sol(
            mesh, vtu_path,
            point_infos=[("density", rho.reshape(-1, 1))]
        )

print(f"\nReconstructed {len(sample_ids)} spinodoids → {output_dir}")
if output_obj:
    print(f"- OBJ solid meshes: spinodoid_XXXX.obj")
else:
    print(f"- Spinodoid structures: spinodoid_XXXX.vtu")
    print(f"- Stiffness spheres: stiffness_sphere_XXXX.vtu")
if has_meta_params:
    print("✓ Exact reconstruction using stored seeds and meta-parameters")
else:
    print("⚠ Approximate reconstruction (no seed/meta-parameters in CSV)")
