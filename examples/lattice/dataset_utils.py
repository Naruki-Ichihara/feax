"""Utility functions for loading and using compressed lattice dataset with Flax NNX."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple


def compress_symmetric_matrix(mat):
    """Extract upper triangle of symmetric matrix (excluding diagonal).

    Args:
        mat: Symmetric matrix of shape (..., n, n)

    Returns:
        Compressed vector of shape (..., n*(n-1)/2)
    """
    n = mat.shape[-1]
    indices = np.triu_indices(n, k=1)
    return mat[..., indices[0], indices[1]]


def decompress_symmetric_matrix(vec, n):
    """Reconstruct symmetric matrix from upper triangle vector.

    Args:
        vec: Compressed vector of shape (..., n*(n-1)/2)
        n: Matrix dimension

    Returns:
        Symmetric matrix of shape (..., n, n)
    """
    mat = jnp.zeros(vec.shape[:-1] + (n, n))
    indices = np.triu_indices(n, k=1)
    mat = mat.at[..., indices[0], indices[1]].set(vec)
    mat = mat.at[..., indices[1], indices[0]].set(vec)  # Mirror to lower triangle
    return mat


def compress_stiffness_voigt(C):
    """Compress 6x6 symmetric stiffness matrix to 21-component vector.

    Voigt notation order:
    [C11, C22, C33, C44, C55, C66,  # Diagonal (6)
     C12, C13, C23,                  # Upper block 3x3 off-diagonal (3)
     C14, C15, C16,                  # Row 1 off-diagonal (3)
     C24, C25, C26,                  # Row 2 off-diagonal (3)
     C34, C35, C36,                  # Row 3 off-diagonal (3)
     C45, C46, C56]                  # Rows 4-5 off-diagonal (3)

    Args:
        C: Stiffness matrix of shape (..., 6, 6)

    Returns:
        Compressed vector of shape (..., 21)
    """
    return np.array([
        C[..., 0, 0], C[..., 1, 1], C[..., 2, 2], C[..., 3, 3], C[..., 4, 4], C[..., 5, 5],  # Diagonal
        C[..., 0, 1], C[..., 0, 2], C[..., 1, 2],  # Upper 3x3 off-diagonal
        C[..., 0, 3], C[..., 0, 4], C[..., 0, 5],  # Row 0 off-diagonal
        C[..., 1, 3], C[..., 1, 4], C[..., 1, 5],  # Row 1 off-diagonal
        C[..., 2, 3], C[..., 2, 4], C[..., 2, 5],  # Row 2 off-diagonal
        C[..., 3, 4], C[..., 3, 5], C[..., 4, 5],  # Rows 3-4 off-diagonal
    ]).T


def decompress_stiffness_voigt(vec):
    """Reconstruct 6x6 stiffness matrix from 21-component Voigt vector.

    Args:
        vec: Compressed Voigt vector of shape (..., 21)

    Returns:
        Stiffness matrix of shape (..., 6, 6)
    """
    C = jnp.zeros(vec.shape[:-1] + (6, 6))
    # Diagonal
    C = C.at[..., 0, 0].set(vec[..., 0])
    C = C.at[..., 1, 1].set(vec[..., 1])
    C = C.at[..., 2, 2].set(vec[..., 2])
    C = C.at[..., 3, 3].set(vec[..., 3])
    C = C.at[..., 4, 4].set(vec[..., 4])
    C = C.at[..., 5, 5].set(vec[..., 5])
    # Upper triangle
    idx = 6
    for i in range(6):
        for j in range(i+1, 6):
            C = C.at[..., i, j].set(vec[..., idx])
            C = C.at[..., j, i].set(vec[..., idx])  # Symmetry
            idx += 1
    return C


def load_lattice_dataset(filepath: str) -> Dict:
    """Load compressed lattice dataset from NPZ file.

    Args:
        filepath: Path to .npz file

    Returns:
        Dictionary with JAX arrays:
            - adjacency_compressed: (num_samples, num_nodes*(num_nodes-1)/2)
            - stiffness_compressed: (num_samples, 21)
            - num_connections: (num_samples,)
            - node_positions: (num_nodes, 3)
    """
    data = np.load(filepath)
    return {k: jnp.array(v) for k, v in data.items()}


def create_dataloader(dataset: Dict, batch_size: int, shuffle: bool = True):
    """Create simple dataloader for NNX training.

    Args:
        dataset: Dataset dictionary from load_lattice_dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        Batches as dictionaries with 'inputs' and 'targets'
    """
    num_samples = dataset['adjacency_compressed'].shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        yield {
            'inputs': {
                'adjacency': dataset['adjacency_compressed'][batch_indices],
                'num_connections': dataset['num_connections'][batch_indices],
            },
            'targets': dataset['stiffness_compressed'][batch_indices]
        }


def get_full_matrices(batch: Dict, num_nodes: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Decompress batch to full adjacency and stiffness matrices.

    Args:
        batch: Batch from dataloader
        num_nodes: Number of nodes in lattice graph

    Returns:
        (adjacency_matrices, stiffness_matrices)
        - adjacency: (batch_size, num_nodes, num_nodes)
        - stiffness: (batch_size, 6, 6)
    """
    adj = decompress_symmetric_matrix(batch['inputs']['adjacency'], num_nodes)
    stiff = decompress_stiffness_voigt(batch['targets'])
    return adj, stiff


# Example usage
if __name__ == "__main__":
    # Load dataset
    dataset = load_lattice_dataset('lattice_dataset.npz')
    print("Dataset keys:", dataset.keys())
    print("Shapes:")
    for k, v in dataset.items():
        print(f"  {k}: {v.shape}")

    # Create dataloader
    batch_size = 4
    num_nodes = int(np.sqrt(2 * dataset['adjacency_compressed'].shape[1] + 0.25) + 0.5)

    print(f"\nCreating batches (batch_size={batch_size}, num_nodes={num_nodes})...")
    for i, batch in enumerate(create_dataloader(dataset, batch_size, shuffle=False)):
        print(f"\nBatch {i+1}:")
        print(f"  Input adjacency shape: {batch['inputs']['adjacency'].shape}")
        print(f"  Target stiffness shape: {batch['targets'].shape}")

        # Decompress to verify
        adj_full, stiff_full = get_full_matrices(batch, num_nodes)
        print(f"  Decompressed adjacency: {adj_full.shape}")
        print(f"  Decompressed stiffness: {stiff_full.shape}")

        if i >= 1:  # Show only 2 batches
            break
