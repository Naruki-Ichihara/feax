"""
Visualization utilities for lattice neural network results.

This module contains plotting functions for training history,
predictions, and actual vs predicted comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved training history plot to {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, num_samples=5, save_path='test_predictions.png'):
    """Plot comparison of true vs predicted stiffness matrices (TEST SET ONLY)."""
    num_samples = min(num_samples, len(y_true))
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

    # Add overall title
    fig.suptitle('Test Set: True vs Predicted Stiffness Components', fontsize=14, fontweight='bold', y=0.995)

    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_samples):
        # True values
        axes[0, i].bar(range(len(y_true[i])), y_true[i], color='#2E86AB')  # Deep blue
        axes[0, i].set_title(f'Sample {i+1}', fontsize=10)
        if len(y_true) > 0:
            axes[0, i].set_ylim([y_true.min(), y_true.max()])

        # Predicted values
        axes[1, i].bar(range(len(y_pred[i])), y_pred[i], color='#A23B72')  # Purple-magenta
        if len(y_true) > 0:
            axes[1, i].set_ylim([y_true.min(), y_true.max()])

    axes[0, 0].set_ylabel('True Stiffness', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Predicted Stiffness', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved test predictions plot to {save_path}")
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, save_path='test_actual_vs_predicted.png'):
    """Plot actual vs predicted scatter plot with 45-degree reference line (TEST SET ONLY)."""
    # Component labels (21 Voigt notation components)
    component_labels = ['C11', 'C12', 'C13', 'C14', 'C15', 'C16',
                       'C22', 'C23', 'C24', 'C25', 'C26',
                       'C33', 'C34', 'C35', 'C36',
                       'C44', 'C45', 'C46',
                       'C55', 'C56',
                       'C66']

    # Define custom colors: bold for diagonal (C11, C22, C33, C44, C55, C66), light for off-diagonal
    custom_colors = [
        '#0000FF',  # C11 - Blue (bold)
        '#B0C4DE',  # C12 - Light steel blue
        '#B0E0E6',  # C13 - Powder blue
        '#ADD8E6',  # C14 - Light blue
        '#87CEEB',  # C15 - Sky blue
        '#87CEFA',  # C16 - Light sky blue
        '#FF0000',  # C22 - Red (bold)
        '#FFB6C1',  # C23 - Light pink
        '#FFC0CB',  # C24 - Pink
        '#FFD700',  # C25 - Gold
        '#FFE4B5',  # C26 - Moccasin
        '#00FF00',  # C33 - Green (bold)
        '#90EE90',  # C34 - Light green
        '#98FB98',  # C35 - Pale green
        '#ADFF2F',  # C36 - Green yellow
        '#FF00FF',  # C44 - Magenta (bold)
        '#DDA0DD',  # C45 - Plum
        '#EE82EE',  # C46 - Violet
        '#FFA500',  # C55 - Orange (bold)
        '#FFDAB9',  # C56 - Peach puff
        '#800080'   # C66 - Purple (bold)
    ]

    # Create component index array for coloring
    num_samples = y_true.shape[0]
    num_components = y_true.shape[1]
    component_indices = np.tile(np.arange(num_components), num_samples)

    # Flatten all predictions and true values
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 9))

    # Map component indices to custom colors
    colors = [custom_colors[i] for i in component_indices]

    # Scatter plot of predictions, colored by component
    ax.scatter(y_true_flat, y_pred_flat,
               c=colors,
               alpha=0.7, s=40,
               edgecolors='black', linewidth=0.3)

    # 45-degree reference line (perfect prediction) - Gray
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color='gray', linestyle='--', linewidth=2.5,
            label='Perfect Prediction (45°)', zorder=1000, alpha=0.7)

    # Create custom legend with component colors
    diagonal_components = [0, 6, 11, 15, 18, 20]  # Indices for C11, C22, C33, C44, C55, C66
    legend_elements = []

    for i in diagonal_components:
        legend_elements.append(Patch(facecolor=custom_colors[i], edgecolor='black',
                                     label=component_labels[i] + ' (diagonal)'))

    legend_elements.append(Patch(facecolor='lightgray', edgecolor='black',
                                  label='Off-diagonal (light colors)'))
    legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5,
                                       label='Perfect Prediction (45°)'))

    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    # Labels and title
    ax.set_xlabel('Actual Stiffness Values (Test Set)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Stiffness Values (Test Set)', fontsize=13, fontweight='bold')
    ax.set_title('Test Set: Actual vs Predicted by Component', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Make axes equal
    ax.set_aspect('equal', adjustable='box')

    # Add R2 score and sample count as text
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    n_points = len(y_true_flat)

    text_str = f'R² = {r2:.4f}\nMSE = {mse:.4f}\nPoints = {n_points}'
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.9, edgecolor='#F18F01', linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved test actual vs predicted plot to {save_path}")
    plt.close()
