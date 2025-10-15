"""
Simple Neural Network for Lattice Stiffness Prediction

This script trains a 3-layer neural network that predicts
stiffness matrices from adjacency matrices of lattice structures.

Uses Flax NNX and Optax for modern JAX neural network training.

Data format:
- Input: Adjacency matrix (27x27 symmetric matrix, flattened to 351 values)
- Output: Stiffness matrix (6x6 symmetric, stored as 21 unique Voigt components)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import os

# JAX and neural network libraries
import jax
import jax.numpy as jnp
from flax import nnx
import optax

# Visualization utilities
from visualize_utils import plot_training_history, plot_actual_vs_predicted

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Output directory
OUTPUT_DIR = "output"

class SimpleNN(nnx.Module):
    """
    Simple 3-layer fully-connected neural network.

    Architecture:
    - Layer 1: Linear + ReLU
    - Layer 2: Linear + ReLU
    - Layer 3: Linear (output)
    """

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, rngs: nnx.Rngs):
        """Initialize the network layers."""
        self.layer1 = nnx.Linear(input_size, hidden_sizes[0], rngs=rngs)
        self.layer2 = nnx.Linear(hidden_sizes[0], hidden_sizes[1], rngs=rngs)
        self.layer3 = nnx.Linear(hidden_sizes[1], output_size, rngs=rngs)

    def __call__(self, x):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = nnx.relu(x)

        x = self.layer2(x)
        x = nnx.relu(x)

        x = self.layer3(x)
        return x

def load_data(data_dir="datas"):
    """Load lattice dataset from CSV files."""
    data_path = Path(data_dir)

    # Load metadata
    metadata = pd.read_csv(data_path / "lattice_metadata.csv")
    num_nodes = int(metadata['num_nodes'].values[0])
    print(f"Loaded metadata: {num_nodes} nodes per lattice")

    # Load dataset
    df = pd.read_csv(data_path / "lattice_dataset.csv")
    num_samples = len(df)
    print(f"Loaded {num_samples} samples")

    # Extract adjacency matrices (symmetric, so adj_i_j for i < j)
    adj_cols = [col for col in df.columns if col.startswith('adj_')]

    # Stiffness matrix components (Voigt notation: C11, C22, ..., C56)
    stiff_cols = [f'C{i}{j}' for i in range(1, 7) for j in range(i, 7)]

    # Verify all columns exist
    missing_stiff = [col for col in stiff_cols if col not in df.columns]
    if missing_stiff:
        print(f"Warning: Missing stiffness columns: {missing_stiff}")
        stiff_cols = [col for col in stiff_cols if col in df.columns]

    adj_matrices = df[adj_cols].values.astype(np.float32)
    stiffness_matrices = df[stiff_cols].values.astype(np.float32)

    print(f"Adjacency matrix shape: {adj_matrices.shape}")
    print(f"Stiffness matrix shape: {stiffness_matrices.shape}")

    return adj_matrices, stiffness_matrices


def loss_fn(model: SimpleNN, x_batch, y_batch):
    """Mean squared error loss."""
    predictions = nnx.vmap(model)(x_batch)
    return jnp.mean((predictions - y_batch) ** 2)


@nnx.jit
def train_step(model: SimpleNN, optimizer: nnx.Optimizer, x_batch, y_batch):
    """Single training step with gradient update."""
    loss, grads = nnx.value_and_grad(loss_fn)(model, x_batch, y_batch)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model: SimpleNN, x_batch, y_batch):
    """Evaluation step (no gradient computation)."""
    return loss_fn(model, x_batch, y_batch)


def train_network(X_train, y_train, X_val, y_val,
                  hidden_sizes=[128, 64],
                  learning_rate=0.001,
                  num_epochs=100,
                  batch_size=16):

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    num_samples = X_train.shape[0]

    print(f"\nTraining network:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Output size: {output_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")

    # Initialize model and optimizer
    model = SimpleNN(input_size, hidden_sizes, output_size, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    # Training loop
    train_losses = []
    val_losses = []

    key = jax.random.PRNGKey(42)

    for epoch in range(num_epochs):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_samples)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        # Mini-batch training
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]

            batch_loss = train_step(model, optimizer, batch_X, batch_y)
            epoch_loss += batch_loss
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        train_losses.append(float(avg_train_loss))

        # Validation loss
        val_loss = eval_step(model, X_val, y_val)
        val_losses.append(float(val_loss))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return model, train_losses, val_losses


def evaluate_model(model: SimpleNN, X_test, y_test):
    """Evaluate model on test set."""
    predictions = nnx.vmap(model)(X_test)
    mse = jnp.mean((predictions - y_test) ** 2)
    mae = jnp.mean(jnp.abs(predictions - y_test))

    # R-squared
    ss_res = jnp.sum((y_test - predictions) ** 2)
    ss_tot = jnp.sum((y_test - jnp.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nTest Set Evaluation:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2:  {r2:.6f}")

    return predictions


def save_model(model, X_mean, X_std, y_mean, y_std, save_dir=OUTPUT_DIR):
    """Save trained model and normalization parameters."""
    # Create output directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(save_dir) / 'trained_model.pkl'

    # Extract model parameters using Flax NNX
    model_state = nnx.state(model)

    model_data = {
        'model_state': model_state,
        'X_mean': np.array(X_mean),
        'X_std': np.array(X_std),
        'y_mean': np.array(y_mean),
        'y_std': np.array(y_std),
        'input_size': model.layer1.in_features,
        'hidden_sizes': [model.layer1.out_features, model.layer2.out_features],
        'output_size': model.layer3.out_features
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nSaved trained model to {save_path}")


def load_model(load_path='trained_model.pkl'):
    """Load trained model and normalization parameters."""
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)

    # Recreate model architecture
    model = SimpleNN(
        model_data['input_size'],
        model_data['hidden_sizes'],
        model_data['output_size'],
        rngs=nnx.Rngs(0)
    )

    # Restore model state
    nnx.update(model, model_data['model_state'])

    print(f"\nLoaded trained model from {load_path}")

    return model, model_data['X_mean'], model_data['X_std'], model_data['y_mean'], model_data['y_std']


def save_results_to_csv(train_losses, val_losses, y_test, y_pred, save_dir=OUTPUT_DIR):
    """Save training history and predictions to CSV files."""
    # Create output directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Save training history
    history_df = pd.DataFrame({
        'epoch': np.arange(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_path = Path(save_dir) / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    # Save predictions
    component_labels = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16',
                       'C22', 'C23', 'C24', 'C25', 'C26',
                       'C33', 'C34', 'C35', 'C36',
                       'C44', 'C45', 'C46',
                       'C55', 'C56', 'C66']

    # Create DataFrame for predictions
    pred_data = {}
    pred_data['sample_id'] = np.arange(len(y_test))

    # Add actual values
    for i, label in enumerate(component_labels):
        pred_data[f'actual_{label}'] = y_test[:, i]

    # Add predicted values
    for i, label in enumerate(component_labels):
        pred_data[f'predicted_{label}'] = y_pred[:, i]

    # Add errors
    for i, label in enumerate(component_labels):
        pred_data[f'error_{label}'] = y_pred[:, i] - y_test[:, i]
        pred_data[f'abs_error_{label}'] = np.abs(y_pred[:, i] - y_test[:, i])
        pred_data[f'rel_error_{label}'] = np.abs((y_pred[:, i] - y_test[:, i]) / (y_test[:, i] + 1e-10)) * 100

    pred_df = pd.DataFrame(pred_data)
    pred_path = Path(save_dir) / 'test_predictions.csv'
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved test predictions to {pred_path}")


def main():
    # Load data
    X, y = load_data()

    # Normalize data
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-8

    X_normalized = (X - X_mean) / X_std
    y_normalized = (y - y_mean) / y_std

    # Convert to JAX arrays
    X_jax = jnp.array(X_normalized)
    y_jax = jnp.array(y_normalized)

    # Train/validation/test split (70/15/15)
    num_samples = len(X_jax)
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)

    X_train, y_train = X_jax[:train_end], y_jax[:train_end]
    X_val, y_val = X_jax[train_end:val_end], y_jax[train_end:val_end]
    X_test, y_test = X_jax[val_end:], y_jax[val_end:]

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train network
    model, train_losses, val_losses = train_network(
        X_train, y_train, X_val, y_val,
        hidden_sizes=[64, 64],
        learning_rate=0.001,
        num_epochs=100,
        batch_size=10
    )

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Plot training history
    plot_training_history(train_losses, val_losses,
                         save_path=str(Path(OUTPUT_DIR) / 'training_history.png'))

    # Evaluate on test set
    y_pred_normalized = evaluate_model(model, X_test, y_test)

    # Denormalize predictions
    y_pred = y_pred_normalized * y_std + y_mean
    y_test_denorm = y_test * y_std + y_mean

    # Plot predictions
    if len(X_test) > 0:
        # Plot actual vs predicted with 45-degree line
        plot_actual_vs_predicted(
            np.array(y_test_denorm),
            np.array(y_pred),
            save_path=str(Path(OUTPUT_DIR) / 'test_actual_vs_predicted.png')
        )

    # Save trained model
    save_model(model, X_mean, X_std, y_mean, y_std, save_dir=OUTPUT_DIR)

    # Save results to CSV
    save_results_to_csv(train_losses, val_losses, y_test_denorm, y_pred, save_dir=OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
