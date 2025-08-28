import jax
import jax.numpy as np
from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Tuple, Iterable, Callable, Optional

from feax.mesh import Mesh

class UnitCell(ABC):
    def __init__(self, atol: float = 1e-6, **kwargs: Any) -> None:
        """Initialize the unit cell with mesh construction and geometric setup.
        
        Args:
            atol (float, optional): Absolute tolerance for geometric comparisons.
                Used for boundary detection and point classification. Defaults to 1e-6.
            **kwargs: Additional keyword arguments passed to mesh_build().
        
        Raises:
            NotImplementedError: If mesh_build() is not implemented in the subclass.
            ValueError: If mesh construction fails or produces invalid geometry.
        """
        self.mesh = self.mesh_build(**kwargs)
        self.cells = self.mesh.cells
        self.ele_type = self.mesh.ele_type
        self.points = self.mesh.points
        self.atol = atol
        
        # Compute bounding box
        if self.points.shape[0] > 0:
            self.lb = np.min(self.points, axis=0)  # Lower bound
            self.ub = np.max(self.points, axis=0)  # Upper bound
        else:
            # Handle empty mesh case
            self.lb = np.array([0.0] * self.points.shape[1])
            self.ub = np.array([1.0] * self.points.shape[1])

        self.num_dims = self.points.shape[1] if self.points.shape[0] > 0 else len(self.lb)

    @abstractmethod
    def mesh_build(self, **kwargs: Any) -> Mesh:
        """Abstract method to construct the finite element mesh for the unit cell.
        
        This method must be implemented by concrete subclasses to define the specific
        mesh generation strategy. The mesh should represent the computational domain
        of the unit cell with appropriate element connectivity and boundary definition.
        
        Args:
            **kwargs: Keyword arguments for mesh generation parameters such as:
                - Element density/resolution parameters
                - Geometric dimensions
                - Element type specifications
                - Boundary condition requirements
        
        Returns:
            Mesh: A finite element mesh object containing:
                - points: Node coordinates
                - cells: Element connectivity
                - ele_type: Element type identifier
        
        Raises:
            NotImplementedError: Always raised as this is an abstract method.
        
        Example:
            Implementation for a structured cube mesh:
            
            >>> def mesh_build(self, nx=10, ny=10, nz=10, **kwargs):
            ...     return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0)
        """
        raise NotImplementedError("Mesh needs to be defined.")
    
    @property
    def cell_centers(self) -> np.ndarray:
        """Get the geometric centers of all elements in the unit cell mesh.
        
        Computes the centroid of each element by averaging the coordinates of
        its constituent nodes. This is useful for element-based operations,
        visualization, and material property assignment.

        Returns:
            np.ndarray: Element centers with shape (num_elements, spatial_dim).
                Each row contains the [x, y, z, ...] coordinates of one element center.
        
        Example:
            >>> centers = unit_cell.cell_centers
            >>> print(f"First element center: {centers[0]}")
        """
        # Calculate the centers of the cells
        return np.mean(self.points[self.cells], axis=1)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the unit cell mesh.
        
        Returns the minimal and maximal coordinates that define the axis-aligned
        bounding box containing all mesh nodes. This defines the computational
        domain boundaries for periodic boundary conditions and coordinate mapping.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Lower bound: minimum coordinates [x_min, y_min, z_min, ...]
                - Upper bound: maximum coordinates [x_max, y_max, z_max, ...]
        
        Example:
            >>> lb, ub = unit_cell.bounds
            >>> print(f"Unit cell spans from {lb} to {ub}")
            >>> volume = np.prod(ub - lb)
        """
        # Get the boundary of the mesh
        return self.lb, self.ub

    @property
    def corners(self) -> np.ndarray:
        """Get all corner coordinates of the unit cell bounding box.
        
        Computes all 2^N corner points of the N-dimensional bounding box defined
        by the mesh bounds. These corners are essential for periodic boundary
        condition enforcement and geometric transformations.

        Returns:
            np.ndarray: Array of shape (2^N, N) containing all corner coordinates.
                Each row represents one corner point in N-dimensional space.
                For a 3D box: 8 corners, for 2D: 4 corners, etc.
        
        Example:
            >>> corners = unit_cell.corners  # For 3D unit cell
            >>> print(f"3D unit cell has {len(corners)} corners")
            >>> print(f"Corner coordinates:\n{corners}")
        """
        min_corner, max_corner = self.lb, self.ub
        dim = min_corner.shape[0]
        corner_list = []
        for bits in product([0, 1], repeat=dim):
            corner = np.where(np.array(bits), max_corner, min_corner)
            corner_list.append(corner)

        return np.array(corner_list)
    
    def is_corner(self, point: np.ndarray) -> bool:
        """Check if a point lies at a corner of the unit cell bounding box.
        
        A corner point has coordinates that match either the minimum or maximum
        bound value in ALL spatial dimensions. This is used for periodic boundary
        condition identification and constraint application.

        Args:
            point (np.ndarray): Point coordinates to test with shape (spatial_dim,).

        Returns:
            bool: True if the point is at a unit cell corner, False otherwise.
            
        Example:
            >>> # Test if origin is a corner
            >>> is_corner = unit_cell.is_corner(np.array([0.0, 0.0, 0.0]))
            >>> print(f"Origin is corner: {is_corner}")
        """
        close_0 = np.isclose(point, self.lb, atol=self.atol)
        close_1 = np.isclose(point, self.ub, atol=self.atol)
        count_of_hit_corners = np.sum(np.logical_or(close_0, close_1))
        return np.equal(count_of_hit_corners, point.shape[0])

    def is_edge(self, point: np.ndarray) -> bool:
        """Check if a point lies on an edge of the unit cell bounding box.
        
        An edge point has coordinates that match boundary values in exactly
        (N-1) dimensions, where N is the spatial dimension. Edge points are
        distinct from corner points and are important for periodic boundary
        condition enforcement.

        Args:
            point (np.ndarray): Point coordinates to test with shape (spatial_dim,).

        Returns:
            bool: True if the point is on a unit cell edge (but not a corner),
                False otherwise.
                
        Example:
            >>> # Test if point is on an edge
            >>> test_point = np.array([0.5, 0.0, 0.0])  # Middle of bottom edge
            >>> is_edge = unit_cell.is_edge(test_point)
        """
        close_0 = np.isclose(point, self.lb, atol=self.atol)
        close_1 = np.isclose(point, self.ub, atol=self.atol)
        boundary_hits = np.sum(np.logical_or(close_0, close_1))
        return np.logical_and(
            np.equal(boundary_hits, point.shape[0] - 1),
            np.logical_not(self.is_corner(point)),
        )

    def is_face(self, point: np.ndarray) -> bool:
        """Check if a point lies on a face of the unit cell bounding box.
        
        A face point has coordinates that match boundary values in exactly
        (N-2) dimensions, where N is the spatial dimension. Face points are
        interior to faces (not on edges or corners) and are relevant for
        surface-based boundary conditions.

        Args:
            point (np.ndarray): Point coordinates to test with shape (spatial_dim,).

        Returns:
            bool: True if the point is on a unit cell face (but not on edges
                or corners), False otherwise.
                
        Example:
            >>> # Test if point is on a face interior
            >>> test_point = np.array([0.5, 0.5, 0.0])  # Center of bottom face
            >>> is_face = unit_cell.is_face(test_point)
        """
        close_0 = np.isclose(point, self.lb, atol=self.atol)
        close_1 = np.isclose(point, self.ub, atol=self.atol)
        boundary_hits = np.sum(np.logical_or(close_0, close_1))
        return np.logical_and(
            np.equal(boundary_hits, point.shape[0] - 2),
            np.logical_and(
                np.logical_not(self.is_edge(point)),
                np.logical_not(self.is_corner(point)),
            ),
        )

    @property
    def corner_mask(self) -> np.ndarray:
        """Boolean mask identifying all corner nodes in the mesh.
        
        Applies the is_corner() test to all mesh nodes using JAX vectorization
        for efficient computation. This mask is useful for applying boundary
        conditions and constraints specifically to corner nodes.

        Returns:
            np.ndarray: Boolean array with shape (num_nodes,) where True indicates
                the corresponding node is at a unit cell corner.
                
        Example:
            >>> corner_mask = unit_cell.corner_mask
            >>> corner_nodes = unit_cell.points[corner_mask]
            >>> print(f"Found {np.sum(corner_mask)} corner nodes")
        """
        is_corner_vec = jax.vmap(self.is_corner)
        result = is_corner_vec(self.points)
        return result

    @property
    def edge_mask(self) -> np.ndarray:
        """Boolean mask identifying all edge nodes in the mesh.
        
        Applies the is_edge() test to all mesh nodes using JAX vectorization.
        Edge nodes lie on unit cell edges but are not corner nodes. This mask
        is useful for applying periodic boundary conditions along edges.

        Returns:
            np.ndarray: Boolean array with shape (num_nodes,) where True indicates
                the corresponding node is on a unit cell edge (excluding corners).
                
        Example:
            >>> edge_mask = unit_cell.edge_mask
            >>> edge_nodes = unit_cell.points[edge_mask]
            >>> print(f"Found {np.sum(edge_mask)} edge nodes")
        """
        is_edge_vec = jax.vmap(self.is_edge)
        result = is_edge_vec(self.points)
        return result

    @property
    def face_mask(self) -> np.ndarray:
        """Boolean mask identifying all face nodes in the mesh.
        
        Applies the is_face() test to all mesh nodes using JAX vectorization.
        Face nodes lie on unit cell faces but are not on edges or corners.
        This mask is useful for applying surface-based boundary conditions.

        Returns:
            np.ndarray: Boolean array with shape (num_nodes,) where True indicates
                the corresponding node is on a unit cell face (excluding edges
                and corners).
                
        Example:
            >>> face_mask = unit_cell.face_mask
            >>> face_nodes = unit_cell.points[face_mask]
            >>> print(f"Found {np.sum(face_mask)} face nodes")
        """
        is_face_vec = jax.vmap(self.is_face)
        result = is_face_vec(self.points)
        return result

    def face_function(
        self, axis: int, value: float, excluding_edge: bool = False, excluding_corner: bool = False
    ) -> Callable[[np.ndarray], bool]:
        """Create a function to identify points on a specific face of the unit cell.
        
        Generates a boolean test function for points lying on a particular face
        defined by a constant coordinate value along a specified axis. The function
        can optionally exclude edge and corner points from the face definition.

        Args:
            axis (int): The coordinate axis defining the face (0=x, 1=y, 2=z, etc.).
                Must be in range [0, spatial_dim-1].
            value (float): The coordinate value along the specified axis that defines
                the face plane (e.g., 0.0 for minimum face, 1.0 for maximum face).
            excluding_edge (bool, optional): If True, exclude points that lie on
                edges of the unit cell. Defaults to False.
            excluding_corner (bool, optional): If True, exclude points that lie at
                corners of the unit cell. Defaults to False.

        Returns:
            Callable[[np.ndarray], bool]: A function that takes a point coordinate
                array and returns True if the point lies on the specified face.
                
        Raises:
            ValueError: If axis is out of range for the mesh dimensionality.
                
        Example:
            >>> # Create function for left face (x=0), excluding corners
            >>> left_face = unit_cell.face_function(axis=0, value=0.0, excluding_corner=True)
            >>> test_point = np.array([0.0, 0.5, 0.5])
            >>> on_face = left_face(test_point)
        """
        # Validate axis parameter
        spatial_dim = self.points.shape[1] if self.points.shape[0] > 0 else len(self.lb)
        if axis < 0 or axis >= spatial_dim:
            raise ValueError(f"Axis {axis} is out of range for {spatial_dim}D mesh")

        def fn(point: np.ndarray) -> bool:
            # Base condition: point lies on the specified face
            on_face = np.isclose(point[axis], value, atol=self.atol)
            
            # Apply exclusions as needed
            if excluding_corner and excluding_edge:
                # Exclude both corners and edges
                return np.logical_and(
                    on_face,
                    np.logical_and(
                        np.logical_not(self.is_corner(point)),
                        np.logical_not(self.is_edge(point))
                    )
                )
            elif excluding_corner:
                # Exclude only corners
                return np.logical_and(on_face, np.logical_not(self.is_corner(point)))
            elif excluding_edge:
                # Exclude only edges (and implicitly corners since edges don't include corners)
                return np.logical_and(
                    on_face,
                    np.logical_and(
                        np.logical_not(self.is_edge(point)),
                        np.logical_not(self.is_corner(point))
                    )
                )
            else:
                # No exclusions
                return on_face

        return fn

    def edge_function(
        self, axes: Iterable[int], values: Iterable[float], excluding_corner: bool = False
    ) -> Callable[[np.ndarray], bool]:
        """Create a function to identify points on a specific edge of the unit cell.
        
        Generates a boolean test function for points lying on a particular edge
        defined by constant coordinate values along multiple specified axes. The
        function can optionally exclude corner points from the edge definition.

        Args:
            axes (Iterable[int]): The coordinate axes that define the edge.
                For a 3D unit cell, an edge is defined by fixing 2 axes.
                Example: [0, 1] for an edge parallel to the z-axis.
            values (Iterable[float]): The coordinate values along the specified axes
                that define the edge. Must have same length as axes.
                Example: [0.0, 0.0] for the edge at x=0, y=0.
            excluding_corner (bool, optional): If True, exclude points that lie at
                corners of the unit cell. Defaults to False.

        Returns:
            Callable[[np.ndarray], bool]: A function that takes a point coordinate
                array and returns True if the point lies on the specified edge.
                
        Example:
            >>> # Create function for bottom-left edge (x=0, y=0) in 3D
            >>> bottom_left_edge = unit_cell.edge_function([0, 1], [0.0, 0.0])
            >>> test_point = np.array([0.0, 0.0, 0.5])
            >>> on_edge = bottom_left_edge(test_point)
        """

        def fn(point: np.ndarray) -> bool:
            cond = np.ones((), dtype=bool)
            for axis, value in zip(axes, values):
                cond = np.logical_and(
                    cond, np.isclose(point[axis], value, atol=self.atol)
                )
            if excluding_corner:
                # Exclude corners from the edge mask
                return np.logical_and(cond, np.logical_not(self.is_corner(point)))
            else:
                return cond

        return fn

    def corner_function(self, values: Iterable[float]) -> Callable[[np.ndarray], bool]:
        """Create a function to identify points at a specific corner of the unit cell.
        
        Generates a boolean test function for points lying at a particular corner
        defined by specific coordinate values in all spatial dimensions.

        Args:
            values (Iterable[float]): The coordinate values that define the corner.
                Must contain exactly spatial_dim values corresponding to the
                [x, y, z, ...] coordinates of the corner.
                Example: [0.0, 0.0, 0.0] for the origin corner in 3D.

        Returns:
            Callable[[np.ndarray], bool]: A function that takes a point coordinate
                array and returns True if the point lies at the specified corner.
                
        Raises:
            ValueError: If the number of values doesn't match mesh dimensionality.
                
        Example:
            >>> # Create function for origin corner
            >>> origin_corner = unit_cell.corner_function([0.0, 0.0, 0.0])
            >>> test_point = np.array([0.0, 0.0, 0.0])
            >>> at_corner = origin_corner(test_point)
        """
        values = np.array(values)
        spatial_dim = self.points.shape[1] if self.points.shape[0] > 0 else len(self.lb)
        
        if len(values) != spatial_dim:
            raise ValueError(f"Number of values ({len(values)}) must match spatial dimension ({spatial_dim})")

        def fn(point: np.ndarray) -> bool:
            if len(point) != len(values):
                raise ValueError(f"Point dimension ({len(point)}) must match corner dimension ({len(values)})")
            return np.all(np.isclose(point, values, atol=self.atol))

        return fn

    def mapping(self, master: Callable[[np.ndarray], bool], slave: Callable[[np.ndarray], bool]) -> Callable[[np.ndarray], np.ndarray]:
        """Create a mapping function from master boundary to slave boundary.
        
        Generates a coordinate transformation function that maps points from a
        master boundary (face, edge, or corner) to the corresponding points on
        a slave boundary. This is essential for implementing periodic boundary
        conditions where displacements on opposite boundaries must be related.
        
        The mapping is computed by finding the geometric transformation (translation)
        that relates corresponding points on the master and slave boundaries.

        Args:
            master (Callable[[np.ndarray], bool]): Boolean filter function that
                identifies points on the master boundary. Can be created using
                face_function(), edge_function(), or corner_function().
            slave (Callable[[np.ndarray], bool]): Boolean filter function that
                identifies points on the slave boundary. Must identify the same
                number of points as the master function.

        Returns:
            Callable[[np.ndarray], np.ndarray]: A mapping function that takes a
                point on the master boundary and returns the corresponding point
                on the slave boundary.
                
        Raises:
            ValueError: If master and slave boundaries contain different numbers
                of points, indicating incompatible boundary definitions.
                
        Example:
            >>> # Map left face to right face for periodic BC
            >>> left_face = unit_cell.face_function(0, 0.0)  # x = 0
            >>> right_face = unit_cell.face_function(0, 1.0)  # x = 1
            >>> mapper = unit_cell.mapping(left_face, right_face)
            >>> 
            >>> # Map a point from left to right
            >>> left_point = np.array([0.0, 0.5, 0.5])
            >>> right_point = mapper(left_point)  # Should be [1.0, 0.5, 0.5]
        """
        master_mask = jax.vmap(master)(self.points)
        slave_mask = jax.vmap(slave)(self.points)

        master_pts = self.points[master_mask]
        slave_pts = self.points[slave_mask]

        if master_pts.shape[0] != slave_pts.shape[0]:
            raise ValueError(
                "Master and slave point sets must have the same number of points."
            )

        deltas = slave_pts - master_pts

        def fn(point: np.ndarray) -> np.ndarray:
            dists = np.linalg.norm(master_pts - point, axis=1)
            idx = np.argmin(dists)
            return point + deltas[idx]

        return fn