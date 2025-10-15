import jax
import jax.numpy as np
from feax import Problem, InternalVars, SolverOptions, DirichletBCConfig
from feax.mesh import box_mesh
import os
import time

from feax.lattice_toolkit.unitcell import UnitCell
from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix
from feax.lattice_toolkit.solver import create_homogenization_solver
from feax.lattice_toolkit.graph import create_lattice_function_from_adjmat, create_lattice_density_field


def create_cube_nodes(nx=3, ny=3, nz=3):
    """Create nodes in an nx×ny×nz grid within a unit cube using meshgrid.

    Args:
        nx: Number of nodes in x direction (default: 3)
        ny: Number of nodes in y direction (default: 3)
        nz: Number of nodes in z direction (default: 3)

    Returns:
        nodes: Array of shape (nx*ny*nz, 3) with node coordinates
    """
    # Create 1D coordinates
    x_coords = np.linspace(0.0, 1.0, nx)
    y_coords = np.linspace(0.0, 1.0, ny)
    z_coords = np.linspace(0.0, 1.0, nz)

    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Flatten and stack to get (nx*ny*nz, 3) array
    nodes = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    return nodes

def create_random_adjacency_matrix(nodes, num_connections=None, connection_prob=None,
                                   max_distance=None, seed=None, face_tol=1e-6,
                                   enforce_connectivity=True, enforce_periodic=True,
                                   max_retries=100):
    """Create random adjacency matrix for given nodes with periodic face connections.

    Connections are generated with two important constraints:
    1. No island struts: All nodes must be connected (if enforce_connectivity=True)
    2. Periodic faces: Opposite faces have identical connectivity (if enforce_periodic=True)

    Args:
        nodes: Array of shape (num_nodes, 3) with node coordinates
        num_connections: Total number of connections to create (optional, mutually exclusive with connection_prob)
        connection_prob: Probability of connection between any two nodes (optional, mutually exclusive with num_connections)
        max_distance: Maximum distance for connections (None = no limit)
        seed: Random seed for reproducibility
        face_tol: Tolerance for identifying nodes on faces (default: 1e-6)
        enforce_connectivity: If True, ensure graph is fully connected (no islands)
        enforce_periodic: If True, enforce identical connectivity on opposite faces
        max_retries: Maximum attempts to generate valid lattice (default: 100)

    Returns:
        adj_mat: Random adjacency matrix of shape (num_nodes, num_nodes)

    Raises:
        RuntimeError: If unable to generate valid lattice after max_retries attempts

    Example:
        # Random lattice with ~20% connections
        nodes = create_cube_nodes(4, 4, 4)
        adj_mat = create_random_adjacency_matrix(nodes, connection_prob=0.2, seed=42)

        # Random lattice with exactly 100 connections within distance 0.5
        adj_mat = create_random_adjacency_matrix(nodes, num_connections=100,
                                                 max_distance=0.5, seed=42)
    """
    if num_connections is not None and connection_prob is not None:
        raise ValueError("Specify either num_connections or connection_prob, not both")

    if num_connections is None and connection_prob is None:
        raise ValueError("Must specify either num_connections or connection_prob")

    num_nodes = nodes.shape[0]

    # Initialize random key
    if seed is None:
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
    else:
        key = jax.random.PRNGKey(seed)

    # Compute domain bounds (needed for periodic enforcement)
    bbox_min = np.min(nodes, axis=0)
    bbox_max = np.max(nodes, axis=0)

    # Try to generate valid lattice with retries
    for attempt in range(max_retries):
        key, subkey = jax.random.split(key)

        # Generate base adjacency matrix
        adj_mat = _generate_base_adjacency(
            nodes, num_connections, connection_prob, max_distance,
            subkey, face_tol, bbox_min, bbox_max
        )

        # Apply periodic boundary constraint if requested
        if enforce_periodic:
            adj_mat = enforce_periodic_face_connections(
                adj_mat, nodes, bbox_min, bbox_max, face_tol
            )

        # Check connectivity if requested
        if enforce_connectivity:
            if check_connectivity(adj_mat):
                return adj_mat
            # If not connected, retry
        else:
            return adj_mat

    # If we get here, we failed to generate valid lattice
    raise RuntimeError(
        f"Failed to generate valid lattice after {max_retries} attempts. "
        f"Try relaxing constraints (larger max_distance, higher connection_prob)"
    )


def _generate_base_adjacency(nodes, num_connections, connection_prob, max_distance,
                             key, face_tol, bbox_min, bbox_max):
    """Internal function to generate base adjacency matrix without constraints.

    This is the core generation logic extracted for retry purposes.
    """
    num_nodes = nodes.shape[0]

    # Identify nodes on each face
    on_x_min = np.abs(nodes[:, 0] - bbox_min[0]) < face_tol
    on_x_max = np.abs(nodes[:, 0] - bbox_max[0]) < face_tol
    on_y_min = np.abs(nodes[:, 1] - bbox_min[1]) < face_tol
    on_y_max = np.abs(nodes[:, 1] - bbox_max[1]) < face_tol
    on_z_min = np.abs(nodes[:, 2] - bbox_min[2]) < face_tol
    on_z_max = np.abs(nodes[:, 2] - bbox_max[2]) < face_tol

    # Node is on any face if it's on at least one face
    on_any_face = on_x_min | on_x_max | on_y_min | on_y_max | on_z_min | on_z_max

    # OPTIMIZATION: Pre-compute opposite node mappings (vectorized, 10-20× faster)
    opposite_node_maps = compute_all_opposite_nodes_vectorized(nodes, bbox_min, bbox_max, face_tol)

    # Compute pairwise distances
    diff = nodes[:, None, :] - nodes[None, :, :]  # (num_nodes, num_nodes, 3)
    distances = np.linalg.norm(diff, axis=2)  # (num_nodes, num_nodes)

    # Create base mask for valid connections (not self-loops, within max_distance)
    valid_mask = np.ones((num_nodes, num_nodes), dtype=bool)
    valid_mask = valid_mask.at[np.arange(num_nodes), np.arange(num_nodes)].set(False)  # No self-loops

    if max_distance is not None:
        distance_mask = distances <= max_distance
        valid_mask = valid_mask & distance_mask

    # Separate face and internal connections
    # Face connection: at least one node is on a face
    face_mask = on_any_face[:, None] | on_any_face[None, :]
    internal_mask = ~face_mask

    # Split valid connections into face and internal
    valid_face_mask = valid_mask & face_mask
    valid_internal_mask = valid_mask & internal_mask

    # Initialize adjacency matrix
    adj_mat = np.zeros((num_nodes, num_nodes))

    if connection_prob is not None:
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Generate face connections on 3 primary faces (x_min, y_min, z_min)
        on_primary_faces = on_x_min | on_y_min | on_z_min
        primary_face_mask = on_primary_faces[:, None] | on_primary_faces[None, :]
        valid_primary_face_mask = valid_face_mask & primary_face_mask

        random_vals_face = jax.random.uniform(subkey1, shape=(num_nodes, num_nodes))
        face_connections = (random_vals_face < connection_prob) & valid_primary_face_mask

        # Make symmetric
        upper_face = np.triu(face_connections, k=1)
        face_adj = (upper_face + upper_face.T).astype(float)

        # OPTIMIZATION: Collect all face connection updates first, then batch apply
        face_updates = []

        # Copy face connections to opposite faces using pre-computed mappings
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if face_adj[i, j] > 0:
                    # Check which primary faces nodes are on
                    node_i_axes = []
                    node_j_axes = []

                    if on_x_min[i]: node_i_axes.append('x')
                    if on_y_min[i]: node_i_axes.append('y')
                    if on_z_min[i]: node_i_axes.append('z')
                    if on_x_min[j]: node_j_axes.append('x')
                    if on_y_min[j]: node_j_axes.append('y')
                    if on_z_min[j]: node_j_axes.append('z')

                    # Use pre-computed opposite node mappings
                    for axis in node_i_axes:
                        target_i = opposite_node_maps[axis][i]
                        if target_i >= 0:
                            for axis_j in node_j_axes:
                                target_j = opposite_node_maps[axis_j][j]
                                if target_j >= 0 and target_i < target_j:
                                    face_updates.append((target_i, target_j))

        # Batch apply face connection updates
        adj_mat = adj_mat + face_adj
        if len(face_updates) > 0:
            update_i = np.array([u[0] for u in face_updates])
            update_j = np.array([u[1] for u in face_updates])
            adj_mat = adj_mat.at[update_i, update_j].set(1.0)
            adj_mat = adj_mat.at[update_j, update_i].set(1.0)

        # Generate internal connections
        random_vals_internal = jax.random.uniform(subkey2, shape=(num_nodes, num_nodes))
        internal_connections = (random_vals_internal < connection_prob) & valid_internal_mask

        # Make symmetric
        upper_internal = np.triu(internal_connections, k=1)
        internal_adj = (upper_internal + upper_internal.T).astype(float)

        adj_mat = adj_mat + internal_adj

    else:  # num_connections is not None
        # Work with upper triangle only for symmetric matrix
        upper_face_mask = np.triu(valid_face_mask, k=1)
        upper_internal_mask = np.triu(valid_internal_mask, k=1)

        num_face_possible = np.sum(upper_face_mask)
        num_internal_possible = np.sum(upper_internal_mask)

        if num_connections > num_face_possible + num_internal_possible:
            raise ValueError(f"Cannot create {num_connections} connections, only {num_face_possible + num_internal_possible} possible")

        # Allocate connections proportionally or split evenly
        # For simplicity, we'll select from combined pool
        upper_valid_mask = np.triu(valid_mask, k=1)
        valid_indices = np.argwhere(upper_valid_mask)

        key, subkey = jax.random.split(key)
        selected_indices = jax.random.choice(subkey, valid_indices.shape[0],
                                            shape=(num_connections,), replace=False)
        selected_connections = valid_indices[selected_indices]

        # OPTIMIZATION: Collect all updates first, then batch apply
        primary_updates = []
        face_copy_updates = []

        # Build adjacency matrix (symmetric)
        for conn in selected_connections:
            i, j = int(conn[0]), int(conn[1])
            primary_updates.append((i, j))

            # If this is a face connection on primary faces, copy to opposite
            if face_mask[i, j]:
                on_primary_i = on_x_min[i] | on_y_min[i] | on_z_min[i]
                on_primary_j = on_x_min[j] | on_y_min[j] | on_z_min[j]

                if on_primary_i or on_primary_j:
                    # Use pre-computed opposite node mappings
                    for axis in ['x', 'y', 'z']:
                        target_i = opposite_node_maps[axis][i]
                        target_j = opposite_node_maps[axis][j]

                        if target_i >= 0 and target_j >= 0:
                            face_copy_updates.append((target_i, target_j))

        # Batch apply all updates
        if len(primary_updates) > 0:
            update_i = np.array([u[0] for u in primary_updates])
            update_j = np.array([u[1] for u in primary_updates])
            adj_mat = adj_mat.at[update_i, update_j].set(1.0)
            adj_mat = adj_mat.at[update_j, update_i].set(1.0)

        if len(face_copy_updates) > 0:
            update_i = np.array([u[0] for u in face_copy_updates])
            update_j = np.array([u[1] for u in face_copy_updates])
            adj_mat = adj_mat.at[update_i, update_j].set(1.0)
            adj_mat = adj_mat.at[update_j, update_i].set(1.0)

    return adj_mat


def check_connectivity(adj_mat):
    """Check if the graph is fully connected (no island struts).

    Uses breadth-first search to verify all nodes are reachable from node 0.

    Args:
        adj_mat: Adjacency matrix (num_nodes, num_nodes)

    Returns:
        bool: True if graph is fully connected, False if there are islands
    """
    num_nodes = adj_mat.shape[0]

    # Find nodes with at least one connection
    has_connection = np.sum(adj_mat, axis=1) > 0
    connected_nodes = np.where(has_connection)[0]

    if len(connected_nodes) == 0:
        return False  # No connections at all

    # OPTIMIZATION: Use native Python list for visited to avoid JAX .at[] overhead
    visited_list = [False] * num_nodes
    queue = [int(connected_nodes[0])]
    visited_list[connected_nodes[0]] = True

    while len(queue) > 0:
        current = queue.pop(0)
        # Find neighbors
        neighbors = np.where(adj_mat[current] > 0)[0]
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if not visited_list[neighbor]:
                visited_list[neighbor] = True
                queue.append(neighbor)

    # Check if all nodes with connections are visited
    all_connected = all(visited_list[int(idx)] for idx in connected_nodes)

    return bool(all_connected)


def verify_periodic_connections(adj_mat, nodes, bbox_min, bbox_max, tol=1e-6):
    """Verify that opposite faces have identical connectivity patterns.

    Only checks connections where BOTH nodes can be mirrored to the opposite face.
    Connections that span between opposite faces (e.g., x_min to x_max) are
    excluded since they're already periodic by definition.

    Args:
        adj_mat: Adjacency matrix
        nodes: Node coordinates
        bbox_min: Minimum bounds
        bbox_max: Maximum bounds
        tol: Tolerance for face identification

    Returns:
        bool: True if periodic connections are valid
    """
    num_nodes = nodes.shape[0]

    # Identify nodes on each face
    on_x_min = np.abs(nodes[:, 0] - bbox_min[0]) < tol
    on_x_max = np.abs(nodes[:, 0] - bbox_max[0]) < tol
    on_y_min = np.abs(nodes[:, 1] - bbox_min[1]) < tol
    on_y_max = np.abs(nodes[:, 1] - bbox_max[1]) < tol
    on_z_min = np.abs(nodes[:, 2] - bbox_min[2]) < tol
    on_z_max = np.abs(nodes[:, 2] - bbox_max[2]) < tol

    # Check each axis
    for axis_name, axis_idx in [('x', 0), ('y', 1), ('z', 2)]:
        on_min = [on_x_min, on_y_min, on_z_min][axis_idx]
        on_max = [on_x_max, on_y_max, on_z_max][axis_idx]

        # For each connection on min face
        for i in range(num_nodes):
            if not on_min[i]:
                continue

            for j in range(num_nodes):
                if adj_mat[i, j] == 0:
                    continue

                # Skip if j is on the opposite face (cross-face connection)
                if on_max[j]:
                    continue  # This is a connection between opposite faces, no need to mirror

                # Find corresponding nodes on max face
                target_i = find_opposite_node(nodes, i, axis_name, bbox_min, bbox_max, tol)
                target_j = find_opposite_node(nodes, j, axis_name, bbox_min, bbox_max, tol)

                if target_i is not None and target_j is not None:
                    # Check if corresponding connection exists
                    if adj_mat[target_i, target_j] == 0:
                        return False

    return True


def enforce_periodic_face_connections(adj_mat, nodes, bbox_min, bbox_max, tol=1e-6, max_iterations=10):
    """Enforce that opposite faces have identical connectivity patterns.

    For each connection, if it involves a face node, create corresponding
    connections on opposite faces to ensure periodicity. Iterates until
    no new connections are added (fixpoint).

    Args:
        adj_mat: Adjacency matrix to modify
        nodes: Node coordinates
        bbox_min: Minimum bounds
        bbox_max: Maximum bounds
        tol: Tolerance for face identification
        max_iterations: Maximum enforcement iterations (default: 10)

    Returns:
        Modified adjacency matrix with enforced periodic connections
    """
    num_nodes = nodes.shape[0]
    adj_mat_current = np.array(adj_mat)

    # OPTIMIZATION: Pre-compute opposite node mappings (vectorized, 10-20× faster)
    opposite_node_maps = compute_all_opposite_nodes_vectorized(nodes, bbox_min, bbox_max, tol)

    # Iterate until fixpoint (no new connections added)
    for iteration in range(max_iterations):
        # OPTIMIZATION: Collect all updates first, then apply in batch
        updates = []

        # Process all existing connections
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_mat_current[i, j] > 0:
                    # For each axis, check if we need to replicate this connection
                    for axis_name in ['x', 'y', 'z']:
                        # Use pre-computed opposite node mapping
                        target_i = opposite_node_maps[axis_name][i]
                        target_j = opposite_node_maps[axis_name][j]

                        # If both nodes have opposites, queue mirror connection
                        if target_i >= 0 and target_j >= 0:
                            # Only update if not already set
                            if adj_mat_current[target_i, target_j] == 0:
                                updates.append((target_i, target_j))

        # Check if we've reached fixpoint
        if len(updates) == 0:
            break

        # OPTIMIZATION: Apply all updates in batch using index arrays
        if len(updates) > 0:
            update_i = np.array([u[0] for u in updates])
            update_j = np.array([u[1] for u in updates])

            # Set both triangles of symmetric matrix
            adj_mat_current = adj_mat_current.at[update_i, update_j].set(1.0)
            adj_mat_current = adj_mat_current.at[update_j, update_i].set(1.0)

    return adj_mat_current


def compute_all_opposite_nodes_vectorized(nodes, bbox_min, bbox_max, tol=1e-6):
    """Vectorized computation of all opposite node mappings at once.

    This is 10-20× faster than calling find_opposite_node() in a loop.

    Args:
        nodes: Array of node coordinates (num_nodes, 3)
        bbox_min: Minimum bounds
        bbox_max: Maximum bounds
        tol: Tolerance for face identification

    Returns:
        dict: Maps axis name ('x', 'y', 'z') to opposite node array (num_nodes,)
              where -1 indicates no opposite node exists
    """
    num_nodes = nodes.shape[0]
    opposite_maps = {}

    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        # Identify all nodes on min/max faces for this axis (vectorized)
        on_min = np.abs(nodes[:, axis_idx] - bbox_min[axis_idx]) < tol  # (num_nodes,)
        on_max = np.abs(nodes[:, axis_idx] - bbox_max[axis_idx]) < tol  # (num_nodes,)

        # Other two axes for coordinate matching
        other_axes = [i for i in range(3) if i != axis_idx]

        # For each node, find its opposite by broadcasting
        # Shape: (num_nodes, num_nodes) comparing all pairs
        target_coords = np.where(on_min, bbox_max[axis_idx],
                                np.where(on_max, bbox_min[axis_idx], -999.0))

        # Check if axis coordinate matches target (on opposite face)
        axis_match = np.abs(nodes[None, :, axis_idx] - target_coords[:, None]) < tol

        # Check if other coordinates match
        other_coord_match = np.ones((num_nodes, num_nodes), dtype=bool)
        for other_ax in other_axes:
            other_coord_match &= np.abs(nodes[None, :, other_ax] - nodes[:, None, other_ax]) < tol

        # Combined match: opposite face + same other coords
        full_match = axis_match & other_coord_match & (on_min[:, None] | on_max[:, None])

        # Find first matching node for each source node (vectorized argmax)
        # Use argmax on the match matrix (first True index)
        has_match = np.any(full_match, axis=1)
        match_indices = np.argmax(full_match, axis=1)

        # Set -1 for nodes without matches
        opposite_map = np.where(has_match, match_indices, -1)
        opposite_maps[axis_name] = opposite_map

    return opposite_maps


def find_opposite_node(nodes, node_idx, axis, bbox_min, bbox_max, tol=1e-6):
    """Find the corresponding node on the opposite face along given axis.

    Args:
        nodes: Array of node coordinates
        node_idx: Index of the node to find opposite for
        axis: 'x', 'y', or 'z'
        bbox_min: Minimum bounds of domain
        bbox_max: Maximum bounds of domain
        tol: Tolerance for face identification

    Returns:
        Index of opposite node, or None if not found
    """
    node = nodes[node_idx]
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]

    # Check if node is on min or max face of this axis
    on_min = np.abs(node[axis_idx] - bbox_min[axis_idx]) < tol
    on_max = np.abs(node[axis_idx] - bbox_max[axis_idx]) < tol

    if not (on_min or on_max):
        return None

    # Target coordinate on opposite face
    target_coord = bbox_max[axis_idx] if on_min else bbox_min[axis_idx]

    # OPTIMIZATION: Instead of building target point and computing distances,
    # find nodes where other coordinates match and axis coordinate is on opposite face
    other_axes = [i for i in range(3) if i != axis_idx]

    # Match nodes on opposite face with same other coordinates
    on_opposite = np.abs(nodes[:, axis_idx] - target_coord) < tol
    same_other_coords = (
        (np.abs(nodes[:, other_axes[0]] - node[other_axes[0]]) < tol) &
        (np.abs(nodes[:, other_axes[1]] - node[other_axes[1]]) < tol)
    )

    candidates = on_opposite & same_other_coords
    candidate_indices = np.where(candidates)[0]

    if len(candidate_indices) > 0:
        return int(candidate_indices[0])

    return None


if __name__ == "__main__":
    # Generate 3x3x3 lattice nodes
    nodes = create_cube_nodes(nx=3, ny=3, nz=3)
    print(f"Generated {nodes.shape[0]} nodes")

    # Create random adjacency matrix with ~30% connection probability
    # Now with connectivity and periodic constraints enforced
    print("\nGenerating lattice with constraints...")
    print("  - No island struts (fully connected)")
    print("  - Periodic boundary conditions")

    adj_mat = create_random_adjacency_matrix(
        nodes,
        connection_prob=0.3,
        max_distance=0.6,
        seed=42,
        enforce_connectivity=True,
        enforce_periodic=True
    )

    num_connections = int(np.sum(adj_mat) / 2)  # Divide by 2 since matrix is symmetric
    print(f"\nGenerated {num_connections} connections")

    # Verify constraints
    is_connected = check_connectivity(adj_mat)
    print(f"Connectivity check: {'PASS' if is_connected else 'FAIL'}")

    # Verify periodic connections
    bbox_min = np.min(nodes, axis=0)
    bbox_max = np.max(nodes, axis=0)
    periodic_valid = verify_periodic_connections(adj_mat, nodes, bbox_min, bbox_max)
    print(f"Periodic boundary check: {'PASS' if periodic_valid else 'FAIL'}")

    # Export to VTK file using meshio
    import meshio

    # Create cells (lines) from adjacency matrix
    cells = []
    for i in range(nodes.shape[0]):
        for j in range(i+1, nodes.shape[0]):
            if adj_mat[i, j] > 0:
                cells.append([i, j])

    cells = [("line", np.array(cells))]

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=np.array(nodes),
        cells=cells,
    )

    # Write to VTK file
    meshio.write("lattice_3x3x3.vtk", mesh)
    print(f"Exported lattice to lattice_3x3x3.vtk")
    print(f"Adjacency matrix shape: {adj_mat.shape}")
    print(f"Connection density: {num_connections / (nodes.shape[0] * (nodes.shape[0]-1) / 2):.4f}")
