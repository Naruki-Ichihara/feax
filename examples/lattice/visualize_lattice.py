"""
Lattice Visualizer - Generate VTK files for lattice structures

This script takes a lattice ID from the dataset and generates:
1. Density field visualization (lattice structure as VTK)
2. Stiffness tensor visualization (directional stiffness sphere)

Usage:
    python visualize_lattice.py --lattice_id 0 --output_dir vtk_output
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import meshio

# FEAX imports
from feax import Problem, InternalVars, Mesh
from feax.mesh import box_mesh
from feax.lattice_toolkit.graph import create_lattice_function_from_adjmat
from feax.lattice_toolkit.utils import visualize_stiffness_sphere
from flax import nnx

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


def load_lattice_nodes(data_dir="datas"):
    """Load the lattice node coordinates."""
    nodes_df = pd.read_csv(Path(data_dir) / "lattice_nodes.csv")
    nodes = nodes_df[['x', 'y', 'z']].values
    return nodes


def load_dataset(data_dir="datas"):
    """Load the lattice dataset."""
    df = pd.read_csv(Path(data_dir) / "lattice_dataset.csv")

    # Extract adjacency matrix columns
    adj_cols = [col for col in df.columns if col.startswith('adj_')]

    # Extract stiffness matrix columns
    stiff_cols = [f'C{i}{j}' for i in range(1, 7) for j in range(i, 7)]
    stiff_cols = [col for col in stiff_cols if col in df.columns]

    adj_matrices = df[adj_cols].values
    stiffness_matrices = df[stiff_cols].values

    return adj_matrices, stiffness_matrices, len(df)


def reconstruct_adjacency_matrix(adj_vector, num_nodes=27):
    """Reconstruct symmetric adjacency matrix from flattened upper triangle."""
    adj_mat = np.zeros((num_nodes, num_nodes))

    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj_mat[i, j] = adj_vector[idx]
            adj_mat[j, i] = adj_vector[idx]  # Symmetric
            idx += 1

    return adj_mat


def reconstruct_stiffness_matrix(stiff_vector):
    """Reconstruct 6x6 symmetric stiffness matrix from Voigt notation vector."""
    C = np.zeros((6, 6))

    idx = 0
    for i in range(6):
        for j in range(i, 6):
            C[i, j] = stiff_vector[idx]
            C[j, i] = stiff_vector[idx]  # Symmetric
            idx += 1

    return C


def create_lattice_vtk(nodes, adj_matrix, output_path, radius=0.02):
    """Create VTK file for lattice structure as beams (lines with radius).

    Args:
        nodes: Node coordinates
        adj_matrix: Adjacency matrix
        output_path: Output file path
        radius: Beam radius for visualization
    """
    num_nodes = len(nodes)

    # Create list of edges
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0.5:  # Connection exists
                edges.append([i, j])

    if len(edges) == 0:
        print("Warning: No edges found in adjacency matrix")
        return

    edges = np.array(edges)

    # Create mesh with nodes (points) and edges (lines)
    cells = [("line", edges)]

    # Add radius as cell data for tube filter in ParaView
    edge_radii = np.ones(len(edges)) * radius

    mesh = meshio.Mesh(
        points=nodes,
        cells=cells,
        cell_data={"radius": [edge_radii], "connectivity": [np.ones(len(edges))]}
    )

    meshio.write(output_path, mesh)
    print(f"Saved beam lattice structure to {output_path}")


def create_density_field_vtk(nodes, adj_matrix, output_path, mesh_size=0.05, radius=0.15):
    """Create VTK file with continuous density field using FEAX."""
    # Create a FEAX mesh for the unit cube
    mesh = box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

    # Create density field from adjacency matrix
    nodes_jax = jnp.array(nodes)
    adj_jax = jnp.array(adj_matrix)

    # Use FEAX's lattice toolkit to create lattice function
    lattice_fn = create_lattice_function_from_adjmat(nodes_jax, adj_jax, radius=radius)

    # Evaluate density at mesh nodes (vectorized)
    mesh_points_jax = jnp.array(mesh.points)
    densities = jax.vmap(lattice_fn)(mesh_points_jax)
    densities = np.array(densities)

    # Create meshio mesh with density field
    # Convert FEAX mesh cells to meshio format
    cells = []
    if hasattr(mesh, 'cells') and mesh.cells is not None:
        if isinstance(mesh.cells, list):
            # Already in meshio format
            cells = mesh.cells
        else:
            # Convert from FEAX format
            cells = [("hexahedron", np.array(mesh.cells))]

    meshio_mesh = meshio.Mesh(
        points=np.array(mesh.points),
        cells=cells,
        point_data={"density": densities}
    )

    meshio.write(output_path, meshio_mesh)
    print(f"Saved density field to {output_path}")


def compute_directional_stiffness(C, direction):
    """Compute effective stiffness in a given direction.

    Args:
        C: 6x6 stiffness matrix in Voigt notation
        direction: 3D unit vector

    Returns:
        Effective stiffness (scalar)
    """
    # Normalize direction
    d = direction / (np.linalg.norm(direction) + 1e-10)

    # Create strain tensor in Voigt notation (uniaxial strain in direction d)
    # ε = d ⊗ d (outer product)
    eps_tensor = np.outer(d, d)

    # Convert to Voigt notation: [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12]
    eps_voigt = np.array([
        eps_tensor[0, 0],
        eps_tensor[1, 1],
        eps_tensor[2, 2],
        2 * eps_tensor[1, 2],
        2 * eps_tensor[0, 2],
        2 * eps_tensor[0, 1]
    ])

    # Compute stress: σ = C ε
    sigma_voigt = C @ eps_voigt

    # Effective stiffness: E_eff = d · σ · d (work done per unit strain)
    # Convert stress back to tensor form
    sigma_tensor = np.array([
        [sigma_voigt[0], sigma_voigt[5], sigma_voigt[4]],
        [sigma_voigt[5], sigma_voigt[1], sigma_voigt[3]],
        [sigma_voigt[4], sigma_voigt[3], sigma_voigt[2]]
    ])

    E_eff = d @ sigma_tensor @ d

    return E_eff


def create_stiffness_sphere_vtk(C, output_path, num_theta=30, num_phi=30):
    """Create VTK file showing directional stiffness as a sphere.

    Args:
        C: 6x6 stiffness matrix
        output_path: Path to save VTK file
        num_theta: Number of divisions in theta (polar angle)
        num_phi: Number of divisions in phi (azimuthal angle)
    """
    # Create spherical grid
    theta = np.linspace(0, np.pi, num_theta)
    phi = np.linspace(0, 2 * np.pi, num_phi)

    points = []
    stiffness_values = []

    for t in theta:
        for p in phi:
            # Spherical to Cartesian (unit sphere)
            direction = np.array([
                np.sin(t) * np.cos(p),
                np.sin(t) * np.sin(p),
                np.cos(t)
            ])

            # Compute directional stiffness
            E_eff = compute_directional_stiffness(C, direction)

            # Scale radius by stiffness (normalized)
            stiffness_values.append(E_eff)

    # Normalize stiffness values
    stiffness_values = np.array(stiffness_values)
    max_stiff = np.max(np.abs(stiffness_values)) + 1e-10
    normalized_stiff = stiffness_values / max_stiff

    # Create points scaled by stiffness
    for i, t in enumerate(theta):
        for j, p in enumerate(phi):
            idx = i * num_phi + j
            direction = np.array([
                np.sin(t) * np.cos(p),
                np.sin(t) * np.sin(p),
                np.cos(t)
            ])

            # Scale by normalized stiffness
            radius = 0.5 + 0.5 * normalized_stiff[idx]  # Range [0, 1]
            point = direction * radius
            points.append(point)

    points = np.array(points)

    # Create cells (quadrilaterals)
    cells = []
    for i in range(num_theta - 1):
        for j in range(num_phi - 1):
            # Quad connecting 4 neighboring points
            p0 = i * num_phi + j
            p1 = i * num_phi + (j + 1)
            p2 = (i + 1) * num_phi + (j + 1)
            p3 = (i + 1) * num_phi + j
            cells.append([p0, p1, p2, p3])

    # Handle wrap-around in phi
    for i in range(num_theta - 1):
        p0 = i * num_phi + (num_phi - 1)
        p1 = i * num_phi + 0
        p2 = (i + 1) * num_phi + 0
        p3 = (i + 1) * num_phi + (num_phi - 1)
        cells.append([p0, p1, p2, p3])

    cells = np.array(cells)

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("quad", cells)],
        point_data={"stiffness": stiffness_values}
    )

    meshio.write(output_path, mesh)
    print(f"Saved stiffness sphere to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lattice structures as VTK files (beams + density field)"
    )
    parser.add_argument("--lattice_id", type=int, required=True, help="Lattice ID from dataset")
    parser.add_argument("--data_dir", type=str, default="datas", help="Directory containing dataset")
    parser.add_argument("--output_dir", type=str, default="vtk_output", help="Output directory for VTK files")
    parser.add_argument("--mesh_size", type=float, default=0.05, help="Mesh size for density field (default: 0.1, smaller=finer)")
    parser.add_argument("--beam_radius", type=float, default=0.015, help="Beam radius for visualization (default: 0.015)")
    parser.add_argument("--density_radius", type=float, default=0.1, help="Strut radius for density field (default: 0.05)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading dataset...")
    nodes = load_lattice_nodes(args.data_dir)
    adj_matrices, stiffness_matrices, num_samples = load_dataset(args.data_dir)

    if args.lattice_id < 0 or args.lattice_id >= num_samples:
        raise ValueError(f"Lattice ID must be between 0 and {num_samples - 1}")

    print(f"Loaded {num_samples} lattice samples")
    print(f"Processing lattice {args.lattice_id}...")

    # Get specific lattice
    adj_vector = adj_matrices[args.lattice_id]
    stiff_vector = stiffness_matrices[args.lattice_id]

    # Reconstruct matrices
    adj_matrix = reconstruct_adjacency_matrix(adj_vector, num_nodes=len(nodes))
    stiffness_matrix = reconstruct_stiffness_matrix(stiff_vector)

    num_connections = int(np.sum(adj_matrix) / 2)  # Divide by 2 since symmetric
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of connections: {num_connections}")

    # Generate VTK files
    print("\nGenerating VTK files...")

    # 1. Lattice beam structure (lines with radius)
    beam_path = output_dir / f"lattice_{args.lattice_id}_beams.vtk"
    create_lattice_vtk(nodes, adj_matrix, beam_path, radius=args.beam_radius)
    print(f"   → Beam radius: {args.beam_radius}")
    print("   → Use 'Tube' filter in ParaView with 'radius' field for 3D beams")

    # 2. Density field (volumetric mesh)
    print(f"\nGenerating density field...")
    print(f"   → Mesh size: {args.mesh_size} (smaller = finer mesh, slower)")
    print(f"   → Strut radius: {args.density_radius}")
    density_path = output_dir / f"lattice_{args.lattice_id}_density.vtu"
    create_density_field_vtk(nodes, adj_matrix, density_path, mesh_size=args.mesh_size, radius=args.density_radius)

    # 3. Stiffness sphere
    stiffness_path = output_dir / f"lattice_{args.lattice_id}_stiffness_sphere.vtu"
    create_stiffness_sphere_vtk(stiffness_matrix, stiffness_path)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    print("\nStiffness matrix (Voigt notation):")
    print(stiffness_matrix)


if __name__ == "__main__":
    main()
