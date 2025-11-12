"""
Spinodoid structure generation using Gaussian Random Fields.

This module provides tools for generating anisotropic Gaussian Random Fields (GRF)
for periodic spinodoid microstructures. For filtering and projection operations,
see the `filters` module.
"""

import jax
import jax.numpy as np

@jax.jit
def evaluate_grf_field(cell_centers, n_vectors, gamma, beta):
    """
    Evaluate Gaussian Random Field at given points (JIT-compiled).

    φ(x) = sqrt(2/N) * Σ cos(β*n_i·x + γ_i)

    Args:
        cell_centers: (num_cells, 3) array of evaluation points
        n_vectors: (N, 3) array of random unit vectors
        gamma: (N,) array of random phases
        beta: scalar wave number (controls characteristic length scale)

    Returns:
        (num_cells,) array of zero-mean GRF values
    """
    N = n_vectors.shape[0]

    def evaluate_grf(x):
        dot_products = np.dot(n_vectors, x)
        arguments = beta * dot_products + gamma
        return np.sqrt(2.0 / N) * np.sum(np.cos(arguments))

    source = jax.vmap(evaluate_grf)(cell_centers)
    return source - np.mean(source)


def generate_direction_vectors(theta1, theta2, theta3, N, key):
    """
    Generate random unit vectors in constrained cone regions.

    Vectors are distributed among cones around x, y, z axes based on
    the specified angle constraints.

    Args:
        theta1: Cone angle for x-axis (radians, 0 = no constraint)
        theta2: Cone angle for y-axis (radians, 0 = no constraint)
        theta3: Cone angle for z-axis (radians, 0 = no constraint)
        N: Total number of vectors to generate
        key: JAX random key

    Returns:
        (N, 3) array of unit vectors

    Note: At least one theta must be > 0.
    """
    thetas = np.array([theta1, theta2, theta3])
    active_mask = thetas > 0
    num_active = int(np.sum(active_mask))

    if num_active == 0:
        raise ValueError("At least one theta must be greater than 0")

    # Determine N for each axis (Python ints for static shapes)
    N_per_direction = N // num_active
    N_counts_base = [N_per_direction if active_mask[i] else 0 for i in range(3)]

    # Determine which axis gets the remainder
    case_id = int(active_mask[0]) * 4 + int(active_mask[1]) * 2 + int(active_mask[2])
    remainder = N - sum(N_counts_base)

    # Add remainder to last active axis
    last_active_map = {1: 2, 2: 1, 3: 2, 4: 0, 5: 2, 6: 1, 7: 2}
    if case_id in last_active_map:
        N_counts_base[last_active_map[case_id]] += remainder

    N0, N1, N2 = N_counts_base

    # Generate random values for each axis with static shapes
    keys = jax.random.split(key, 7)

    def gen_vecs(axis, theta, N_i, key_phi, key_theta):
        if N_i == 0:
            return np.zeros((0, 3))

        phi = jax.random.uniform(key_phi, shape=(N_i,), minval=0.0, maxval=2*np.pi)
        cos_theta_vals = jax.random.uniform(key_theta, shape=(N_i,),
                                             minval=np.cos(theta), maxval=1.0)
        sin_theta_vals = np.sqrt(1.0 - cos_theta_vals**2)

        vecs = np.zeros((N_i, 3))
        vecs = vecs.at[:, axis].set(cos_theta_vals)
        vecs = vecs.at[:, (axis + 1) % 3].set(sin_theta_vals * np.cos(phi))
        vecs = vecs.at[:, (axis + 2) % 3].set(sin_theta_vals * np.sin(phi))
        return vecs

    vec0 = gen_vecs(0, thetas[0], N0, keys[0], keys[1])
    vec1 = gen_vecs(1, thetas[1], N1, keys[2], keys[3])
    vec2 = gen_vecs(2, thetas[2], N2, keys[4], keys[5])

    return np.concatenate([vec0, vec1, vec2], axis=0)


def generate_grf_source(mesh, beta=10.0, N=100, theta1=None, theta2=None, theta3=None, seed=0):
    """
    Generate anisotropic Gaussian Random Field (GRF) source for spinodoid structures.

    φ(x) = sqrt(2/N) * Σ cos(β*n_i·x + γ_i)
    where n_i ~ U(constrained region on S²)

    Args:
        mesh: Mesh object with points and cells
        beta: Wave number (controls characteristic length scale, default 10.0)
        N: Number of random waves (default 100)
        theta1: Cone angle constraint for x-direction (radians, None = no constraint)
        theta2: Cone angle constraint for y-direction (radians, None = no constraint)
        theta3: Cone angle constraint for z-direction (radians, None = no constraint)
        seed: Random seed (default 0)

    Returns:
        (num_cells,) array of GRF values (zero mean)

    Note: At least one theta must be specified (not None).

    Example:
        >>> mesh = box_mesh(size=1.0, mesh_size=0.1)
        >>> source = generate_grf_source(mesh, beta=10.0, N=100,
        ...                              theta1=np.pi/4, theta2=np.pi/4, theta3=np.pi/4)
    """
    # Validate that at least one theta is specified
    if theta1 is None and theta2 is None and theta3 is None:
        raise ValueError("At least one theta (theta1, theta2, or theta3) must be specified")

    # Convert None to 0.0 for compatibility
    theta1 = theta1 if theta1 is not None else 0.0
    theta2 = theta2 if theta2 is not None else 0.0
    theta3 = theta3 if theta3 is not None else 0.0

    key = jax.random.PRNGKey(seed)
    key_n, key_gamma = jax.random.split(key)

    # Generate direction vectors
    n_vectors = generate_direction_vectors(theta1, theta2, theta3, N, key_n)

    # Generate random phases
    gamma = jax.random.uniform(key_gamma, shape=(N,), minval=0.0, maxval=2*np.pi)

    # Compute element centers
    cell_centers = np.mean(mesh.points[mesh.cells], axis=1)

    # Evaluate GRF using JIT-compiled function
    source = evaluate_grf_field(cell_centers, n_vectors, gamma, beta)

    return source
