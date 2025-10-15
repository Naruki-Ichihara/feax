"""
Stiffness Visualizer - Compare actual vs predicted stiffness spheres

This script takes a lattice ID and generates VTK files showing:
1. Actual stiffness sphere (from FEA simulation)
2. Predicted stiffness sphere (from neural network)
3. Lattice structure

Usage:
    python visualize_stiffness.py --lattice_id 0 --model_path output/trained_model.pkl
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
from feax.lattice_toolkit.utils import visualize_stiffness_sphere
from flax import nnx

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Neural network definition (must match training script)
class SimpleNN(nnx.Module):
    """Simple 3-layer fully-connected neural network."""

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, rngs: nnx.Rngs, dropout_rate: float = 0.2):
        """Initialize the network layers."""
        self.layer1 = nnx.Linear(input_size, hidden_sizes[0], rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.layer2 = nnx.Linear(hidden_sizes[0], hidden_sizes[1], rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.layer3 = nnx.Linear(hidden_sizes[1], output_size, rngs=rngs)

    def __call__(self, x, *, deterministic: bool = False):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = nnx.relu(x)
        x = self.dropout1(x, deterministic=deterministic)

        x = self.layer2(x)
        x = nnx.relu(x)
        x = self.dropout2(x, deterministic=deterministic)

        x = self.layer3(x)
        return x


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


def load_model(model_path):
    """Load trained neural network model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Recreate model architecture
    model = SimpleNN(
        model_data['input_size'],
        model_data['hidden_sizes'],
        model_data['output_size'],
        rngs=nnx.Rngs(0),
        dropout_rate=0.0  # No dropout for inference
    )

    # Restore model state
    nnx.update(model, model_data['model_state'])

    print(f"Loaded model from {model_path}")
    print(f"  Architecture: {model_data['input_size']} -> {model_data['hidden_sizes']} -> {model_data['output_size']}")

    return model, model_data['X_mean'], model_data['X_std'], model_data['y_mean'], model_data['y_std']


def predict_stiffness(model, adj_vector, X_mean, X_std, y_mean, y_std):
    """Predict stiffness matrix from adjacency vector using trained model."""
    # Normalize input
    x_normalized = (adj_vector - X_mean) / X_std
    x_jax = jnp.array(x_normalized)

    # Predict (with dropout disabled)
    y_pred_normalized = model(x_jax, deterministic=True)

    # Denormalize output
    y_pred = np.array(y_pred_normalized) * y_std + y_mean

    return y_pred


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


def create_lattice_vtk(nodes, adj_matrix, output_path):
    """Create VTK file for lattice structure (nodes + edges)."""
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

    # Create density field (1.0 where connections exist)
    edge_density = np.ones(len(edges))

    mesh = meshio.Mesh(
        points=nodes,
        cells=cells,
        cell_data={"density": [edge_density]}
    )

    meshio.write(output_path, mesh)
    print(f"Saved lattice structure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize actual vs predicted stiffness spheres")
    parser.add_argument("--lattice_id", type=int, required=True, help="Lattice ID from dataset")
    parser.add_argument("--data_dir", type=str, default="datas", help="Directory containing dataset")
    parser.add_argument("--model_path", type=str, default="output/trained_model.pkl", help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="vtk_output", help="Output directory for VTK files")

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
    print(f"Processing lattice {args.lattice_id}...\n")

    # Get specific lattice
    adj_vector = adj_matrices[args.lattice_id]
    stiff_vector_actual = stiffness_matrices[args.lattice_id]

    # Reconstruct matrices
    adj_matrix = reconstruct_adjacency_matrix(adj_vector, num_nodes=len(nodes))
    stiffness_actual = reconstruct_stiffness_matrix(stiff_vector_actual)

    num_connections = int(np.sum(adj_matrix) / 2)  # Divide by 2 since symmetric
    print(f"Lattice properties:")
    print(f"  Number of nodes: {len(nodes)}")
    print(f"  Number of connections: {num_connections}")

    # Load model and predict stiffness
    print(f"\nLoading trained model from {args.model_path}...")
    model, X_mean, X_std, y_mean, y_std = load_model(args.model_path)

    print("Predicting stiffness...")
    stiff_vector_predicted = predict_stiffness(model, adj_vector, X_mean, X_std, y_mean, y_std)
    stiffness_predicted = reconstruct_stiffness_matrix(stiff_vector_predicted)

    # Compute error metrics
    mse = np.mean((stiff_vector_predicted - stiff_vector_actual) ** 2)
    mae = np.mean(np.abs(stiff_vector_predicted - stiff_vector_actual))
    rel_error = np.mean(np.abs((stiff_vector_predicted - stiff_vector_actual) / (stiff_vector_actual + 1e-10))) * 100

    print(f"\nPrediction quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Mean relative error: {rel_error:.2f}%")

    # Generate VTK files
    print("\nGenerating VTK files...")

    # 1. Lattice structure (edges)
    lattice_path = output_dir / f"lattice_{args.lattice_id}_structure.vtk"
    create_lattice_vtk(nodes, adj_matrix, lattice_path)

    # 2. Actual stiffness sphere
    actual_path = output_dir / f"lattice_{args.lattice_id}_stiffness_actual.vtu"
    print(f"\nGenerating actual stiffness sphere...")
    stats_actual = visualize_stiffness_sphere(jnp.array(stiffness_actual), str(actual_path))
    print(f"  E_max: {stats_actual['E_max']:.2f}")
    print(f"  E_min: {stats_actual['E_min']:.2f}")
    print(f"  Anisotropy ratio: {stats_actual['anisotropy_ratio']:.3f}")

    # 3. Predicted stiffness sphere
    predicted_path = output_dir / f"lattice_{args.lattice_id}_stiffness_predicted.vtu"
    print(f"\nGenerating predicted stiffness sphere...")
    stats_predicted = visualize_stiffness_sphere(jnp.array(stiffness_predicted), str(predicted_path))
    print(f"  E_max: {stats_predicted['E_max']:.2f}")
    print(f"  E_min: {stats_predicted['E_min']:.2f}")
    print(f"  Anisotropy ratio: {stats_predicted['anisotropy_ratio']:.3f}")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    print("\nActual Stiffness Matrix (Voigt notation):")
    print(stiffness_actual)
    print("\nPredicted Stiffness Matrix (Voigt notation):")
    print(stiffness_predicted)


if __name__ == "__main__":
    main()
