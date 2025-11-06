"""Utility functions for lattice toolkit visualization and analysis."""

import jax
import jax.numpy as np
import numpy as onp
import meshio


def compute_directional_youngs_modulus(C, n):
    """Compute Young's modulus in direction n from stiffness matrix C.

    For a given direction n, E(n) = 1/S(n) where S(n) = n^T @ S @ n
    and S is the compliance tensor in full 3x3x3x3 form.

    Args:
        C: 6x6 stiffness matrix in Voigt notation
        n: Direction vector (3,)

    Returns:
        float: Young's modulus in direction n
    """
    S = np.linalg.inv(C)

    # Convert Voigt notation to full tensor (compliance)
    S_tensor = np.zeros((3, 3, 3, 3))
    voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    for i in range(6):
        for j in range(6):
            I, J = voigt_map[i]
            K, L = voigt_map[j]
            factor = 1.0
            if i > 2:
                factor *= 2.0
            if j > 2:
                factor *= 2.0
            S_tensor = S_tensor.at[I, J, K, L].set(S[i, j] / factor)
            if I != J:
                S_tensor = S_tensor.at[J, I, K, L].set(S[i, j] / factor)
            if K != L:
                S_tensor = S_tensor.at[I, J, L, K].set(S[i, j] / factor)
            if I != J and K != L:
                S_tensor = S_tensor.at[J, I, L, K].set(S[i, j] / factor)

    # Compute S_ijkl * n_j * n_l (contracted with direction twice)
    S_n = np.einsum('ijkl,j,l->ik', S_tensor, n, n)
    # E(n) = 1 / (n_i * S_ik * n_k)
    compliance_n = np.einsum('i,ij,j->', n, S_n, n)
    return 1.0 / compliance_n


def visualize_stiffness_sphere(C, output_file, n_theta=30, n_phi=60):
    """Create 3D sphere visualization of directional Young's modulus.

    Creates a VTK file showing how Young's modulus varies with direction.
    The surface is shaped by E(n) values - a perfect sphere indicates isotropy.

    Args:
        C: 6x6 stiffness matrix in Voigt notation
        output_file: Path to output VTK file (e.g., 'stiffness_sphere.vtu')
        n_theta: Number of theta divisions (default 30)
        n_phi: Number of phi divisions (default 60)

    Returns:
        dict: Statistics including E_max, E_min, anisotropy_ratio
    """
    # Generate sphere of directions
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)

    # Direction vectors
    nx = np.sin(THETA) * np.cos(PHI)
    ny = np.sin(THETA) * np.sin(PHI)
    nz = np.cos(THETA)

    # Stack directions for vectorized computation
    directions = np.stack([nx.flatten(), ny.flatten(), nz.flatten()], axis=1)

    # Vectorized computation using vmap
    compute_E_vec = jax.vmap(lambda n: compute_directional_youngs_modulus(C, n))
    E_directional = compute_E_vec(directions).reshape(nx.shape)

    # Normalize and scale coordinates
    E_max = np.max(E_directional)
    E_min = np.min(E_directional)
    scale = E_directional / E_max

    x = scale * nx
    y = scale * ny
    z = scale * nz

    # Convert to numpy for meshio
    x_np = onp.array(x)
    y_np = onp.array(y)
    z_np = onp.array(z)
    E_np = onp.array(E_directional)

    # Create mesh points and cells
    points = onp.stack([x_np.flatten(), y_np.flatten(), z_np.flatten()], axis=1)

    # Create quad cells (structured grid connectivity)
    cells = []
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            idx0 = i * n_theta + j
            idx1 = i * n_theta + (j + 1)
            idx2 = (i + 1) * n_theta + (j + 1)
            idx3 = (i + 1) * n_theta + j
            cells.append([idx0, idx1, idx2, idx3])

    cells_np = onp.array(cells)

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("quad", cells_np)],
        point_data={"youngs_modulus": E_np.flatten()}
    )

    # Save to VTK
    meshio.write(output_file, mesh)

    # Compute statistics
    anisotropy_ratio = float(E_max / E_min)

    stats = {
        'E_max': float(E_max),
        'E_min': float(E_min),
        'anisotropy_ratio': anisotropy_ratio,
        'output_file': output_file
    }

    return stats
