---
sidebar_label: graph
title: feax.flat.graph
---

Graph-based density field generation for periodic structures.

This module provides generic tools for creating density fields from user-defined
node-edge graphs for finite element analysis and computational homogenization.
It evaluates graph-based structures on finite element meshes to create
heterogeneous material distributions.

Key Features:
- Generic node-edge graph evaluation
- Element-based density field generation
- JAX-compatible for GPU acceleration and differentiation
- Integration with FEAX Problem and mesh structures
- Support for both edge-list and adjacency matrix representations

Main Functions:
- create_lattice_function: Create evaluation function from nodes/edges
- create_lattice_function_from_adjmat: Create function from adjacency matrix
- create_lattice_density_field: Generate density field for FEAX problem
- edges2adjcentMat / adjcentMat2edges: Convert between representations

**Example**:

  Creating density field from custom node-edge graph:
  
  &gt;&gt;&gt; from feax.flat.graph import create_lattice_function, create_lattice_density_field
  &gt;&gt;&gt; from feax import InternalVars
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Define nodes and edges for your structure
  &gt;&gt;&gt; nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
  &gt;&gt;&gt; edges = np.array([[0, 1], [0, 2], [0, 3]])
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Create lattice function
  &gt;&gt;&gt; lattice_func = create_lattice_function(nodes, edges, radius=0.1)
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Create density field
  &gt;&gt;&gt; rho = create_lattice_density_field(problem, lattice_func,
  ...                                    density_solid=1.0, density_void=0.1)
  &gt;&gt;&gt;
  &gt;&gt;&gt; # Use in FEAX simulation
  &gt;&gt;&gt; internal_vars = InternalVars(volume_vars=(rho,), surface_vars=[])

#### universal\_graph

```python
def universal_graph(x: np.ndarray, nodes: np.ndarray, edges: np.ndarray,
                    radius: float) -> np.ndarray
```

Evaluate if a point lies within the graph structure defined by nodes and edges.

**Arguments**:

- `x` - Query point with shape (spatial_dim,)
- `nodes` - Node coordinates with shape (num_nodes, spatial_dim)
- `edges` - Edge connectivity matrix with shape (num_edges, 2)
- `radius` - Distance threshold for point inclusion
  

**Returns**:

  Binary indicator (0 or 1) as float. Returns 1.0 if point x is
  within radius distance of any edge, 0.0 otherwise.

#### create\_lattice\_function

```python
def create_lattice_function(nodes: np.ndarray, edges: np.ndarray,
                            radius: float) -> Callable
```

Create a lattice evaluation function from nodes and edges.

**Arguments**:

- `nodes` - Node coordinates with shape (num_nodes, spatial_dim)
- `edges` - Edge connectivity with shape (num_edges, 2)
- `radius` - Radius for edge thickness
  

**Returns**:

  Function that evaluates lattice at a point

#### edges2adjcentMat

```python
def edges2adjcentMat(edges: np.ndarray, num_nodes: int = None) -> np.ndarray
```

Convert edge list to adjacency matrix representation.

**Arguments**:

- `edges` - Edge connectivity array with shape (num_edges, 2) where each row
  contains indices [i, j] of connected nodes
- `num_nodes` - Number of nodes in the graph. If None, inferred as max(edges) + 1
  

**Returns**:

  Adjacency matrix with shape (num_nodes, num_nodes) where element [i, j]
  is 1.0 if nodes i and j are connected, 0.0 otherwise. The matrix is
  symmetric for undirected graphs.
  

**Example**:

  &gt;&gt;&gt; edges = np.array([[0, 1], [1, 2], [0, 2]])
  &gt;&gt;&gt; adj_mat = edges2adjcentMat(edges, num_nodes=3)
  &gt;&gt;&gt; print(adj_mat)
  [[0. 1. 1.]
  [1. 0. 1.]
  [1. 1. 0.]]

#### adjcentMat2edges

```python
def adjcentMat2edges(adj_mat: np.ndarray) -> np.ndarray
```

Convert adjacency matrix to edge list representation.

**Arguments**:

- `adj_mat` - Adjacency matrix with shape (num_nodes, num_nodes) where
  non-zero elements indicate connections between nodes
  

**Returns**:

  Edge connectivity array with shape (num_edges, 2) where each row
  contains indices [i, j] of connected nodes. For undirected graphs,
  only the upper triangle is extracted (i &lt; j) to avoid duplicates.
  

**Example**:

  &gt;&gt;&gt; adj_mat = np.array([[0., 1., 1.],
  ...                     [1., 0., 1.],
  ...                     [1., 1., 0.]])
  &gt;&gt;&gt; edges = adjcentMat2edges(adj_mat)
  &gt;&gt;&gt; print(edges)
  [[0 1]
  [0 2]
  [1 2]]

#### universal\_graph\_from\_adjmat

```python
def universal_graph_from_adjmat(x: np.ndarray, nodes: np.ndarray,
                                adj_mat: np.ndarray,
                                radius: float) -> np.ndarray
```

Evaluate if a point lies within the graph structure defined by adjacency matrix.

**Arguments**:

- `x` - Query point with shape (spatial_dim,)
- `nodes` - Node coordinates with shape (num_nodes, spatial_dim)
- `adj_mat` - Adjacency matrix with shape (num_nodes, num_nodes)
- `radius` - Distance threshold for point inclusion
  

**Returns**:

  Binary indicator (0 or 1) as float. Returns 1.0 if point x is
  within radius distance of any edge, 0.0 otherwise.
  

**Example**:

  &gt;&gt;&gt; nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
  &gt;&gt;&gt; adj_mat = np.array([[0., 1., 1.],
  ...                     [1., 0., 1.],
  ...                     [1., 1., 0.]])
  &gt;&gt;&gt; x = np.array([0.5, 0.0])
  &gt;&gt;&gt; result = universal_graph_from_adjmat(x, nodes, adj_mat, radius=0.1)

#### create\_lattice\_function\_from\_adjmat

```python
def create_lattice_function_from_adjmat(nodes: np.ndarray, adj_mat: np.ndarray,
                                        radius: float) -> Callable
```

Create a lattice evaluation function from nodes and adjacency matrix.

**Arguments**:

- `nodes` - Node coordinates with shape (num_nodes, spatial_dim)
- `adj_mat` - Adjacency matrix with shape (num_nodes, num_nodes)
- `radius` - Radius for edge thickness
  

**Returns**:

  Function that evaluates lattice at a point
  

**Example**:

  &gt;&gt;&gt; nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
  &gt;&gt;&gt; adj_mat = np.array([[0., 1., 1.],
  ...                     [1., 0., 0.],
  ...                     [1., 0., 0.]])
  &gt;&gt;&gt; lattice_func = create_lattice_function_from_adjmat(nodes, adj_mat, radius=0.05)
  &gt;&gt;&gt; result = lattice_func(np.array([0.5, 0.0, 0.0]))

#### create\_lattice\_density\_field

```python
def create_lattice_density_field(problem: Any,
                                 lattice_func: Callable,
                                 density_solid: float = 1.0,
                                 density_void: float = 1e-5) -> np.ndarray
```

Create element-based density field from lattice function for FEAX problem.

**Arguments**:

- `problem` - FEAX Problem instance
- `lattice_func` - Function that evaluates lattice at a point
- `density_solid` - Density value for solid regions (lattice struts)
- `density_void` - Density value for void regions
  

**Returns**:

  Density array with shape (num_elements,) - one value per element

