"""Graph-based lattice density field generation for FEAX.

This module provides functions for creating density fields from lattice structures
for finite element analysis and computational homogenization. It includes various
lattice topologies that can be evaluated on finite element meshes to create 
heterogeneous material distributions.

Key Features:
    - Multiple lattice structure types (FCC, BCC, simple cubic, etc.)
    - Element-based density field generation
    - JAX-compatible for GPU acceleration and differentiation
    - Integration with FEAX Problem and mesh structures

Supported Lattice Structures:
    - FCC (Face-Centered Cubic): High stiffness, common metallic structure
    - BCC (Body-Centered Cubic): Good strength-to-weight ratio
    - Simple Cubic: Basic cubic lattice structure
    - Custom: User-defined node/edge graphs

Example:
    Creating FCC lattice density field for FEAX problem:
    
    >>> from feax.lattice_toolkit.graph import create_fcc_density
    >>> from feax import InternalVars
    >>> 
    >>> # Create FCC density field
    >>> rho_fcc = create_fcc_density(problem, radius=0.1, 
    ...                              density_solid=1.0, density_void=0.1)
    >>> 
    >>> # Use in FEAX simulation
    >>> internal_vars = InternalVars(volume_vars=(rho_fcc,), surface_vars=[])
"""

import jax.numpy as np
from functools import partial
import jax
from typing import Callable, Tuple, Any


def _segment_distance(x: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Compute the minimum distance from a point to a line segment.
    
    Args:
        x: Query point with shape (..., spatial_dim)
        p0: First endpoint of line segment with shape (..., spatial_dim)
        p1: Second endpoint of line segment with shape (..., spatial_dim)
        
    Returns:
        Minimum distance from point to line segment with shape (...)
    """
    v = p1 - p0
    w = x - p0
    
    # Compute projection parameter, clipped to [0, 1] for segment bounds
    v_dot_v = np.dot(v, v)
    t = np.where(v_dot_v > 0, np.clip(np.dot(w, v) / v_dot_v, 0.0, 1.0), 0.0)
    
    # Find closest point on segment and compute distance
    proj = p0 + t * v
    return np.linalg.norm(x - proj)


def universal_graph(x: np.ndarray, nodes: np.ndarray, edges: np.ndarray, 
                   radius: float) -> np.ndarray:
    """Evaluate if a point lies within the graph structure defined by nodes and edges.
    
    Args:
        x: Query point with shape (spatial_dim,)
        nodes: Node coordinates with shape (num_nodes, spatial_dim)
        edges: Edge connectivity matrix with shape (num_edges, 2)
        radius: Distance threshold for point inclusion
        
    Returns:
        Binary indicator (0 or 1) as float. Returns 1.0 if point x is
        within radius distance of any edge, 0.0 otherwise.
    """
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")
        
    def check_edge(edge: np.ndarray) -> np.ndarray:
        """Check if point is within radius of a specific edge."""
        i, j = edge
        return _segment_distance(x, nodes[i], nodes[j]) <= radius
    
    # Handle empty edges case
    if edges.shape[0] == 0:
        return 0.0
    
    # Use vmap to check all edges in parallel
    edge_checks = jax.vmap(check_edge)(edges)
    return np.where(np.any(edge_checks), 1.0, 0.0)


def create_lattice_function(nodes: np.ndarray, edges: np.ndarray, radius: float) -> Callable:
    """Create a lattice evaluation function from nodes and edges.
    
    Args:
        nodes: Node coordinates with shape (num_nodes, spatial_dim)
        edges: Edge connectivity with shape (num_edges, 2) 
        radius: Radius for edge thickness
        
    Returns:
        Function that evaluates lattice at a point
    """
    return partial(universal_graph, nodes=nodes, edges=edges, radius=radius)


def edges2adjcentMat(edges: np.ndarray, num_nodes: int = None) -> np.ndarray:
    """Convert edge list to adjacency matrix representation.

    Args:
        edges: Edge connectivity array with shape (num_edges, 2) where each row
               contains indices [i, j] of connected nodes
        num_nodes: Number of nodes in the graph. If None, inferred as max(edges) + 1

    Returns:
        Adjacency matrix with shape (num_nodes, num_nodes) where element [i, j]
        is 1.0 if nodes i and j are connected, 0.0 otherwise. The matrix is
        symmetric for undirected graphs.

    Example:
        >>> edges = np.array([[0, 1], [1, 2], [0, 2]])
        >>> adj_mat = edges2adjcentMat(edges, num_nodes=3)
        >>> print(adj_mat)
        [[0. 1. 1.]
         [1. 0. 1.]
         [1. 1. 0.]]
    """
    if edges.shape[0] == 0:
        # Handle empty edge case
        n = num_nodes if num_nodes is not None else 0
        return np.zeros((n, n))

    # Infer number of nodes if not provided
    if num_nodes is None:
        num_nodes = int(np.max(edges)) + 1

    # Initialize adjacency matrix
    adj_mat = np.zeros((num_nodes, num_nodes))

    # Fill in edges (assumes undirected graph, so symmetric)
    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        adj_mat = adj_mat.at[i, j].set(1.0)
        adj_mat = adj_mat.at[j, i].set(1.0)

    return adj_mat


def adjcentMat2edges(adj_mat: np.ndarray) -> np.ndarray:
    """Convert adjacency matrix to edge list representation.

    Args:
        adj_mat: Adjacency matrix with shape (num_nodes, num_nodes) where
                 non-zero elements indicate connections between nodes

    Returns:
        Edge connectivity array with shape (num_edges, 2) where each row
        contains indices [i, j] of connected nodes. For undirected graphs,
        only the upper triangle is extracted (i < j) to avoid duplicates.

    Example:
        >>> adj_mat = np.array([[0., 1., 1.],
        ...                     [1., 0., 1.],
        ...                     [1., 1., 0.]])
        >>> edges = adjcentMat2edges(adj_mat)
        >>> print(edges)
        [[0 1]
         [0 2]
         [1 2]]
    """
    # Extract upper triangle to avoid duplicate edges (assumes undirected graph)
    n = adj_mat.shape[0]

    # Create meshgrid of indices for upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)

    # Get upper triangle values
    upper_tri_values = adj_mat[i_idx, j_idx]

    # Use where to find non-zero entries (JAX-compatible)
    edge_indices = np.where(upper_tri_values != 0, size=n*(n-1)//2, fill_value=-1)

    # Filter out padding (-1 values) and create edge list
    valid_edges = edge_indices[0]
    edges = np.stack([i_idx[valid_edges], j_idx[valid_edges]], axis=1)

    # Remove padding rows (where both indices are -1 after filtering)
    valid_mask = valid_edges != -1
    edges = edges[valid_mask]

    return edges


def universal_graph_from_adjmat(x: np.ndarray, nodes: np.ndarray,
                                adj_mat: np.ndarray, radius: float) -> np.ndarray:
    """Evaluate if a point lies within the graph structure defined by adjacency matrix.

    Args:
        x: Query point with shape (spatial_dim,)
        nodes: Node coordinates with shape (num_nodes, spatial_dim)
        adj_mat: Adjacency matrix with shape (num_nodes, num_nodes)
        radius: Distance threshold for point inclusion

    Returns:
        Binary indicator (0 or 1) as float. Returns 1.0 if point x is
        within radius distance of any edge, 0.0 otherwise.

    Example:
        >>> nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> adj_mat = np.array([[0., 1., 1.],
        ...                     [1., 0., 1.],
        ...                     [1., 1., 0.]])
        >>> x = np.array([0.5, 0.0])
        >>> result = universal_graph_from_adjmat(x, nodes, adj_mat, radius=0.1)
    """
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")

    n = adj_mat.shape[0]

    def check_edge_pair(i: int, j: int) -> np.ndarray:
        """Check if point is within radius of edge (i,j) if it exists."""
        # Only check upper triangle to avoid duplicates
        edge_exists = np.logical_and(i < j, adj_mat[i, j] != 0)
        distance = _segment_distance(x, nodes[i], nodes[j])
        within_radius = distance <= radius
        return np.logical_and(edge_exists, within_radius)

    # Create all pairs (i, j) for upper triangle
    i_indices = np.arange(n)
    j_indices = np.arange(n)

    # Use nested vmap to check all pairs
    def check_row(i: int) -> np.ndarray:
        return jax.vmap(lambda j: check_edge_pair(i, j))(j_indices)

    all_checks = jax.vmap(check_row)(i_indices)

    return np.where(np.any(all_checks), 1.0, 0.0)


def create_lattice_function_from_adjmat(nodes: np.ndarray, adj_mat: np.ndarray,
                                        radius: float) -> Callable:
    """Create a lattice evaluation function from nodes and adjacency matrix.

    Args:
        nodes: Node coordinates with shape (num_nodes, spatial_dim)
        adj_mat: Adjacency matrix with shape (num_nodes, num_nodes)
        radius: Radius for edge thickness

    Returns:
        Function that evaluates lattice at a point

    Example:
        >>> nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        >>> adj_mat = np.array([[0., 1., 1.],
        ...                     [1., 0., 0.],
        ...                     [1., 0., 0.]])
        >>> lattice_func = create_lattice_function_from_adjmat(nodes, adj_mat, radius=0.05)
        >>> result = lattice_func(np.array([0.5, 0.0, 0.0]))
    """
    return partial(universal_graph_from_adjmat, nodes=nodes, adj_mat=adj_mat, radius=radius)


def create_lattice_density_field(problem: Any, lattice_func: Callable,
                                density_solid: float = 1.0,
                                density_void: float = 1e-5) -> np.ndarray:
    """Create element-based density field from lattice function for FEAX problem.

    Args:
        problem: FEAX Problem instance
        lattice_func: Function that evaluates lattice at a point
        density_solid: Density value for solid regions (lattice struts)
        density_void: Density value for void regions

    Returns:
        Density array with shape (num_elements,) - one value per element
    """
    # Get mesh from problem (handle both single mesh and list of meshes)
    mesh = problem.mesh[0] if isinstance(problem.mesh, list) else problem.mesh

    # Compute element centroids
    centroids = np.mean(mesh.points[mesh.cells], axis=1)

    # Evaluate lattice function at each element centroid
    lattice_values = jax.vmap(lattice_func)(centroids)

    # Convert to density values (element-based, not quad-point based)
    element_densities = np.where(lattice_values > 0.5, density_solid, density_void)

    return element_densities
