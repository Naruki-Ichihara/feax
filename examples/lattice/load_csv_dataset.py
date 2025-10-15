"""Example script to load and use the CSV lattice dataset with Flax NNX."""

import pandas as pd
import jax.numpy as jnp
import numpy as np
from dataset_utils import decompress_symmetric_matrix, decompress_stiffness_voigt


def load_lattice_csv_dataset(
    data_file='lattice_dataset.csv',
    nodes_file='lattice_nodes.csv',
    metadata_file='lattice_metadata.csv'
):
    """Load lattice dataset from CSV files.

    Args:
        data_file: Path to main dataset CSV
        nodes_file: Path to node positions CSV
        metadata_file: Path to metadata CSV

    Returns:
        Dictionary with:
            - adjacency_compressed: (N, num_edges) JAX array
            - stiffness_compressed: (N, 21) JAX array
            - num_connections: (N,) JAX array
            - node_positions: (num_nodes, 3) JAX array
            - metadata: dict
    """
    # Load main dataset
    df = pd.read_csv(data_file)

    # Extract metadata
    sample_ids = df['sample_id'].values
    num_connections = df['num_connections'].values

    # Extract adjacency columns (all columns starting with 'adj_')
    adj_cols = [col for col in df.columns if col.startswith('adj_')]
    adjacency_compressed = df[adj_cols].values

    # Extract stiffness columns (all columns starting with 'C')
    stiff_cols = [col for col in df.columns if col.startswith('C')]
    stiffness_compressed = df[stiff_cols].values

    # Load node positions
    nodes_df = pd.read_csv(nodes_file)
    node_positions = nodes_df[['x', 'y', 'z']].values

    # Load metadata
    metadata_df = pd.read_csv(metadata_file)
    metadata = metadata_df.iloc[0].to_dict()

    return {
        'adjacency_compressed': jnp.array(adjacency_compressed),
        'stiffness_compressed': jnp.array(stiffness_compressed),
        'num_connections': jnp.array(num_connections),
        'node_positions': jnp.array(node_positions),
        'sample_ids': jnp.array(sample_ids),
        'metadata': metadata
    }


def create_train_val_split(dataset, train_ratio=0.8, seed=42):
    """Split dataset into training and validation sets.

    Args:
        dataset: Dataset dictionary from load_lattice_csv_dataset
        train_ratio: Ratio of training data (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset) tuple
    """
    num_samples = dataset['adjacency_compressed'].shape[0]
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    split_idx = int(num_samples * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    def split_dict(d, idx):
        return {
            k: v[idx] if k != 'metadata' and k != 'node_positions' else v
            for k, v in d.items()
        }

    return split_dict(dataset, train_indices), split_dict(dataset, val_indices)


def create_batches(dataset, batch_size, shuffle=True):
    """Create batches from dataset for training.

    Args:
        dataset: Dataset dictionary
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Yields:
        Batches with 'inputs' and 'targets' keys
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


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset from CSV...")
    dataset = load_lattice_csv_dataset()

    print("\n=== Dataset Info ===")
    print(f"Number of samples: {dataset['adjacency_compressed'].shape[0]}")
    print(f"Number of nodes: {dataset['node_positions'].shape[0]}")
    print(f"Adjacency compressed shape: {dataset['adjacency_compressed'].shape}")
    print(f"Stiffness compressed shape: {dataset['stiffness_compressed'].shape}")
    print(f"\nMetadata:")
    for k, v in dataset['metadata'].items():
        print(f"  {k}: {v}")

    # Split into train/val
    print("\n=== Train/Val Split ===")
    train_data, val_data = create_train_val_split(dataset, train_ratio=0.8)
    print(f"Training samples: {train_data['adjacency_compressed'].shape[0]}")
    print(f"Validation samples: {val_data['adjacency_compressed'].shape[0]}")

    # Create batches
    print("\n=== Sample Batches ===")
    batch_size = 8
    for i, batch in enumerate(create_batches(train_data, batch_size, shuffle=False)):
        print(f"\nBatch {i+1}:")
        print(f"  Input adjacency: {batch['inputs']['adjacency'].shape}")
        print(f"  Input connections: {batch['inputs']['num_connections'].shape}")
        print(f"  Target stiffness: {batch['targets'].shape}")

        # Test decompression
        num_nodes = int(dataset['metadata']['num_nodes'])
        adj_full = decompress_symmetric_matrix(batch['inputs']['adjacency'], num_nodes)
        stiff_full = decompress_stiffness_voigt(batch['targets'])
        print(f"  Decompressed adjacency: {adj_full.shape}")
        print(f"  Decompressed stiffness: {stiff_full.shape}")

        if i >= 1:  # Show only 2 batches
            break

    print("\n✓ Dataset loaded and verified successfully!")
