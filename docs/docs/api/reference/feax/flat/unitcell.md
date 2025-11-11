---
sidebar_label: unitcell
title: feax.flat.unitcell
---

## UnitCell Objects

```python
class UnitCell(ABC)
```

#### \_\_init\_\_

```python
def __init__(atol: float = 1e-6, **kwargs: Any) -> None
```

Initialize the unit cell with mesh construction and geometric setup.

**Arguments**:

- `atol` _float, optional_ - Absolute tolerance for geometric comparisons.
  Used for boundary detection and point classification. Defaults to 1e-6.
- `**kwargs` - Additional keyword arguments passed to mesh_build().


**Raises**:

- `NotImplementedError` - If mesh_build() is not implemented in the subclass.
- `ValueError` - If mesh construction fails or produces invalid geometry.

#### mesh\_build

```python
@abstractmethod
def mesh_build(**kwargs: Any) -> Mesh
```

Abstract method to construct the finite element mesh for the unit cell.

This method must be implemented by concrete subclasses to define the specific
mesh generation strategy. The mesh should represent the computational domain
of the unit cell with appropriate element connectivity and boundary definition.

**Arguments**:

- `**kwargs` - Keyword arguments for mesh generation parameters such as:
  - Element density/resolution parameters
  - Geometric dimensions
  - Element type specifications
  - Boundary condition requirements


**Returns**:

- `Mesh` - A finite element mesh object containing:
  - points: Node coordinates
  - cells: Element connectivity
  - ele_type: Element type identifier


**Raises**:

- `NotImplementedError` - Always raised as this is an abstract method.


**Example**:

Implementation for a structured cube mesh:
```python
>>> def mesh_build(self, nx=10, ny=10, nz=10, **kwargs):
...     return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0)
```

#### cell\_centers

```python
@property
def cell_centers() -> np.ndarray
```

Get the geometric centers of all elements in the unit cell mesh.

Computes the centroid of each element by averaging the coordinates of
its constituent nodes. This is useful for element-based operations,
visualization, and material property assignment.

**Returns**:

- `np.ndarray` - Element centers with shape (num_elements, spatial_dim).
  Each row contains the [x, y, z, ...] coordinates of one element center.


**Example**:

```python
>>> centers = unit_cell.cell_centers
>>> print(f&quot;First element center: {`centers[0]`}&quot;)
```

#### bounds

```python
@property
def bounds() -> Tuple[np.ndarray, np.ndarray]
```

Get the bounding box of the unit cell mesh.

Returns the minimal and maximal coordinates that define the axis-aligned
bounding box containing all mesh nodes. This defines the computational
domain boundaries for periodic boundary conditions and coordinate mapping.

**Returns**:

  Tuple[np.ndarray, np.ndarray]: A tuple containing:
  - Lower bound: minimum coordinates [x_min, y_min, z_min, ...]
  - Upper bound: maximum coordinates [x_max, y_max, z_max, ...]


**Example**:

```python
>>> lb, ub = unit_cell.bounds
>>> print(f&quot;Unit cell spans from {`lb`} to {`ub`}&quot;)
>>> volume = np.prod(ub - lb)
```

#### corners

```python
@property
def corners() -> np.ndarray
```

Get all corner coordinates of the unit cell bounding box.

Computes all 2^N corner points of the N-dimensional bounding box defined
by the mesh bounds. These corners are essential for periodic boundary
condition enforcement and geometric transformations.

**Returns**:

- `np.ndarray` - Array of shape (2^N, N) containing all corner coordinates.
  Each row represents one corner point in N-dimensional space.
  For a 3D box: 8 corners, for 2D: 4 corners, etc.


**Example**:

```python
>>> corners = unit_cell.corners  # For 3D unit cell
>>> print(f&quot;3D unit cell has {`len(corners)`} corners&quot;)
>>> print(f&quot;Corner coordinates:
```
{`corners`}&quot;)

#### is\_corner

```python
def is_corner(point: np.ndarray) -> bool
```

Check if a point lies at a corner of the unit cell bounding box.

A corner point has coordinates that match either the minimum or maximum
bound value in ALL spatial dimensions. This is used for periodic boundary
condition identification and constraint application.

**Arguments**:

- `point` _np.ndarray_ - Point coordinates to test with shape (spatial_dim,).


**Returns**:

- `bool` - True if the point is at a unit cell corner, False otherwise.


**Example**:

```python
>>> # Test if origin is a corner
>>> is_corner = unit_cell.is_corner(np.array([0.0, 0.0, 0.0]))
>>> print(f&quot;Origin is corner: {`is_corner`}&quot;)
```

#### is\_edge

```python
def is_edge(point: np.ndarray) -> bool
```

Check if a point lies on an edge of the unit cell bounding box.

An edge point has coordinates that match boundary values in exactly
(N-1) dimensions, where N is the spatial dimension. Edge points are
distinct from corner points and are important for periodic boundary
condition enforcement.

**Arguments**:

- `point` _np.ndarray_ - Point coordinates to test with shape (spatial_dim,).


**Returns**:

- `bool` - True if the point is on a unit cell edge (but not a corner),
  False otherwise.


**Example**:

```python
>>> # Test if point is on an edge
>>> test_point = np.array([0.5, 0.0, 0.0])  # Middle of bottom edge
>>> is_edge = unit_cell.is_edge(test_point)
```

#### is\_face

```python
def is_face(point: np.ndarray) -> bool
```

Check if a point lies on a face of the unit cell bounding box.

A face point has coordinates that match boundary values in exactly
(N-2) dimensions, where N is the spatial dimension. Face points are
interior to faces (not on edges or corners) and are relevant for
surface-based boundary conditions.

**Arguments**:

- `point` _np.ndarray_ - Point coordinates to test with shape (spatial_dim,).


**Returns**:

- `bool` - True if the point is on a unit cell face (but not on edges
  or corners), False otherwise.


**Example**:

```python
>>> # Test if point is on a face interior
>>> test_point = np.array([0.5, 0.5, 0.0])  # Center of bottom face
>>> is_face = unit_cell.is_face(test_point)
```

#### corner\_mask

```python
@property
def corner_mask() -> np.ndarray
```

Boolean mask identifying all corner nodes in the mesh.

Applies the is_corner() test to all mesh nodes using JAX vectorization
for efficient computation. This mask is useful for applying boundary
conditions and constraints specifically to corner nodes.

**Returns**:

- `np.ndarray` - Boolean array with shape (num_nodes,) where True indicates
  the corresponding node is at a unit cell corner.


**Example**:

```python
>>> corner_mask = unit_cell.corner_mask
>>> corner_nodes = unit_cell.points[corner_mask]
>>> print(f&quot;Found {`np.sum(corner_mask)`} corner nodes&quot;)
```

#### edge\_mask

```python
@property
def edge_mask() -> np.ndarray
```

Boolean mask identifying all edge nodes in the mesh.

Applies the is_edge() test to all mesh nodes using JAX vectorization.
Edge nodes lie on unit cell edges but are not corner nodes. This mask
is useful for applying periodic boundary conditions along edges.

**Returns**:

- `np.ndarray` - Boolean array with shape (num_nodes,) where True indicates
  the corresponding node is on a unit cell edge (excluding corners).


**Example**:

```python
>>> edge_mask = unit_cell.edge_mask
>>> edge_nodes = unit_cell.points[edge_mask]
>>> print(f&quot;Found {`np.sum(edge_mask)`} edge nodes&quot;)
```

#### face\_mask

```python
@property
def face_mask() -> np.ndarray
```

Boolean mask identifying all face nodes in the mesh.

Applies the is_face() test to all mesh nodes using JAX vectorization.
Face nodes lie on unit cell faces but are not on edges or corners.
This mask is useful for applying surface-based boundary conditions.

**Returns**:

- `np.ndarray` - Boolean array with shape (num_nodes,) where True indicates
  the corresponding node is on a unit cell face (excluding edges
  and corners).


**Example**:

```python
>>> face_mask = unit_cell.face_mask
>>> face_nodes = unit_cell.points[face_mask]
>>> print(f&quot;Found {`np.sum(face_mask)`} face nodes&quot;)
```

#### face\_function

```python
def face_function(
        axis: int,
        value: float,
        excluding_edge: bool = False,
        excluding_corner: bool = False) -> Callable[[np.ndarray], bool]
```

Create a function to identify points on a specific face of the unit cell.

Generates a boolean test function for points lying on a particular face
defined by a constant coordinate value along a specified axis. The function
can optionally exclude edge and corner points from the face definition.

**Arguments**:

- `axis` _int_ - The coordinate axis defining the face (0=x, 1=y, 2=z, etc.).
  Must be in range [0, spatial_dim-1].
- `value` _float_ - The coordinate value along the specified axis that defines
  the face plane (e.g., 0.0 for minimum face, 1.0 for maximum face).
- `excluding_edge` _bool, optional_ - If True, exclude points that lie on
  edges of the unit cell. Defaults to False.
- `excluding_corner` _bool, optional_ - If True, exclude points that lie at
  corners of the unit cell. Defaults to False.


**Returns**:

  Callable[[np.ndarray], bool]: A function that takes a point coordinate
  array and returns True if the point lies on the specified face.


**Raises**:

- `ValueError` - If axis is out of range for the mesh dimensionality.


**Example**:

```python
>>> # Create function for left face (x=0), excluding corners
>>> left_face = unit_cell.face_function(axis=0, value=0.0, excluding_corner=True)
>>> test_point = np.array([0.0, 0.5, 0.5])
>>> on_face = left_face(test_point)
```

#### edge\_function

```python
def edge_function(
        axes: Iterable[int],
        values: Iterable[float],
        excluding_corner: bool = False) -> Callable[[np.ndarray], bool]
```

Create a function to identify points on a specific edge of the unit cell.

Generates a boolean test function for points lying on a particular edge
defined by constant coordinate values along multiple specified axes. The
function can optionally exclude corner points from the edge definition.

**Arguments**:

- `axes` _Iterable[int]_ - The coordinate axes that define the edge.
  For a 3D unit cell, an edge is defined by fixing 2 axes.
- `Example` - [0, 1] for an edge parallel to the z-axis.
- `values` _Iterable[float]_ - The coordinate values along the specified axes
  that define the edge. Must have same length as axes.
- `Example` - [0.0, 0.0] for the edge at x=0, y=0.
- `excluding_corner` _bool, optional_ - If True, exclude points that lie at
  corners of the unit cell. Defaults to False.


**Returns**:

  Callable[[np.ndarray], bool]: A function that takes a point coordinate
  array and returns True if the point lies on the specified edge.


**Example**:

```python
>>> # Create function for bottom-left edge (x=0, y=0) in 3D
>>> bottom_left_edge = unit_cell.edge_function([0, 1], [0.0, 0.0])
>>> test_point = np.array([0.0, 0.0, 0.5])
>>> on_edge = bottom_left_edge(test_point)
```

#### corner\_function

```python
def corner_function(values: Iterable[float]) -> Callable[[np.ndarray], bool]
```

Create a function to identify points at a specific corner of the unit cell.

Generates a boolean test function for points lying at a particular corner
defined by specific coordinate values in all spatial dimensions.

**Arguments**:

- `values` _Iterable[float]_ - The coordinate values that define the corner.
  Must contain exactly spatial_dim values corresponding to the
  [x, y, z, ...] coordinates of the corner.
- `Example` - [0.0, 0.0, 0.0] for the origin corner in 3D.


**Returns**:

  Callable[[np.ndarray], bool]: A function that takes a point coordinate
  array and returns True if the point lies at the specified corner.


**Raises**:

- `ValueError` - If the number of values doesn&#x27;t match mesh dimensionality.


**Example**:

```python
>>> # Create function for origin corner
>>> origin_corner = unit_cell.corner_function([0.0, 0.0, 0.0])
>>> test_point = np.array([0.0, 0.0, 0.0])
>>> at_corner = origin_corner(test_point)
```

#### mapping

```python
def mapping(
        master: Callable[[np.ndarray], bool],
        slave: Callable[[np.ndarray],
                        bool]) -> Callable[[np.ndarray], np.ndarray]
```

Create a mapping function from master boundary to slave boundary.

Generates a coordinate transformation function that maps points from a
master boundary (face, edge, or corner) to the corresponding points on
a slave boundary. This is essential for implementing periodic boundary
conditions where displacements on opposite boundaries must be related.

The mapping is computed by finding the geometric transformation (translation)
that relates corresponding points on the master and slave boundaries.

**Arguments**:

- `master` _Callable[[np.ndarray], bool]_ - Boolean filter function that
  identifies points on the master boundary. Can be created using
  face_function(), edge_function(), or corner_function().
- `slave` _Callable[[np.ndarray], bool]_ - Boolean filter function that
  identifies points on the slave boundary. Must identify the same
  number of points as the master function.


**Returns**:

  Callable[[np.ndarray], np.ndarray]: A mapping function that takes a
  point on the master boundary and returns the corresponding point
  on the slave boundary.


**Raises**:

- `ValueError` - If master and slave boundaries contain different numbers
  of points, indicating incompatible boundary definitions.


**Example**:

```python
>>> # Map left face to right face for periodic BC
>>> left_face = unit_cell.face_function(0, 0.0)  # x = 0
>>> right_face = unit_cell.face_function(0, 1.0)  # x = 1
>>> mapper = unit_cell.mapping(left_face, right_face)
>>>
>>> # Map a point from left to right
>>> left_point = np.array([0.0, 0.5, 0.5])
>>> right_point = mapper(left_point)  # Should be [1.0, 0.5, 0.5]
```

