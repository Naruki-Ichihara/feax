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


def fcc_unitcell(scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate nodes and edges for a Face-Centered Cubic (FCC) unit cell.
    
    Args:
        scale: Scaling factor for the unit cell dimensions
        
    Returns:
        Tuple of (nodes, edges) where:
        - nodes: Array of shape (14, 3) with FCC node coordinates
        - edges: Array of shape (24, 2) with edge connectivity
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    
    # Define all FCC node positions
    nodes = np.array([
        # Corner atoms (8 nodes)
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        # Face center atoms (6 nodes)
        [0.5, 0.5, 0.0], [0.5, 0.5, 1.0], [0.5, 0.0, 0.5], 
        [0.5, 1.0, 0.5], [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],
    ]) * scale
    
    # Edges: cube edges + corner-to-face center connections
    cube_edges = [
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], 
        [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
    ]
    
    corner_face_edges = [
        [0, 8], [0, 10], [0, 12], [1, 8], [1, 10], [1, 13],
        [2, 8], [2, 11], [2, 12], [3, 9], [3, 10], [3, 12],
        [4, 8], [4, 11], [4, 13], [5, 9], [5, 10], [5, 13],
        [6, 9], [6, 11], [6, 12], [7, 9], [7, 11], [7, 13]
    ]
    
    edges = np.array(cube_edges + corner_face_edges)
    return nodes, edges


def bcc_unitcell(scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate nodes and edges for a Body-Centered Cubic (BCC) unit cell.
    
    Args:
        scale: Scaling factor for the unit cell dimensions
        
    Returns:
        Tuple of (nodes, edges) where:
        - nodes: Array of shape (9, 3) with BCC node coordinates
        - edges: Array of shape (16, 2) with edge connectivity
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    
    # BCC nodes: 8 corners + 1 body center
    nodes = np.array([
        # Corner atoms (8 nodes)
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        # Body center (1 node)
        [0.5, 0.5, 0.5],
    ]) * scale
    
    # Edges: each corner connected to body center
    edges = np.array([[i, 8] for i in range(8)])
    return nodes, edges


def simple_cubic_unitcell(scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate nodes and edges for a Simple Cubic unit cell.
    
    Args:
        scale: Scaling factor for the unit cell dimensions
        
    Returns:
        Tuple of (nodes, edges) where:
        - nodes: Array of shape (8, 3) with corner node coordinates
        - edges: Array of shape (12, 2) with edge connectivity
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    
    # Simple cubic nodes: 8 corners only
    nodes = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ]) * scale
    
    # Edges: cube edges only
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], 
        [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
    ])
    
    return nodes, edges


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


def create_lattice_density_field(problem: Any, lattice_func: Callable, 
                                density_solid: float = 1.0, 
                                density_void: float = 0.1) -> np.ndarray:
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


# Convenience functions for specific lattice types
def create_fcc_density(problem: Any, radius: float = 0.1, scale: float = 1.0,
                      density_solid: float = 1.0, density_void: float = 0.1) -> np.ndarray:
    """Create FCC lattice density field for FEAX problem.
    
    Args:
        problem: FEAX Problem instance
        radius: Radius for lattice struts
        scale: Scale factor for unit cell
        density_solid: Density value for solid regions
        density_void: Density value for void regions
        
    Returns:
        FCC density field array with shape (num_elements,)
    """
    nodes, edges = fcc_unitcell(scale)
    lattice_func = create_lattice_function(nodes, edges, radius)
    return create_lattice_density_field(problem, lattice_func, density_solid, density_void)


def create_bcc_density(problem: Any, radius: float = 0.1, scale: float = 1.0,
                      density_solid: float = 1.0, density_void: float = 0.1) -> np.ndarray:
    """Create BCC lattice density field for FEAX problem.
    
    Args:
        problem: FEAX Problem instance  
        radius: Radius for lattice struts
        scale: Scale factor for unit cell
        density_solid: Density value for solid regions
        density_void: Density value for void regions
        
    Returns:
        BCC density field array
    """
    nodes, edges = bcc_unitcell(scale)
    lattice_func = create_lattice_function(nodes, edges, radius)
    return create_lattice_density_field(problem, lattice_func, density_solid, density_void)


def create_simple_cubic_density(problem: Any, radius: float = 0.1, scale: float = 1.0,
                               density_solid: float = 1.0, density_void: float = 0.1) -> np.ndarray:
    """Create Simple Cubic lattice density field for FEAX problem.
    
    Args:
        problem: FEAX Problem instance
        radius: Radius for lattice struts  
        scale: Scale factor for unit cell
        density_solid: Density value for solid regions
        density_void: Density value for void regions
        
    Returns:
        Simple cubic density field array
    """
    nodes, edges = simple_cubic_unitcell(scale)
    lattice_func = create_lattice_function(nodes, edges, radius)
    return create_lattice_density_field(problem, lattice_func, density_solid, density_void)