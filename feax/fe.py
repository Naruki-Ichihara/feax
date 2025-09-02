"""
Finite element class implementation for FEAX framework.

This module provides the FiniteElement class that handles shape functions,
quadrature rules, and geometric computations for individual variables in
finite element problems.
"""

import jax
import jax.numpy as np
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional, Union, TYPE_CHECKING
from feax.mesh import Mesh
from feax.basis import get_face_shape_vals_and_grads, get_shape_vals_and_grads

if TYPE_CHECKING:
    from numpy.typing import NDArray


np.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


@dataclass
class FiniteElement:
    """Finite element class for a single variable with shape functions and quadrature.
    
    This class handles all geometric and computational aspects for one variable in
    a finite element problem, including shape functions, quadrature rules, and
    transformations between reference and physical domains.
    
    The variable can be:
    - Scalar-valued (vec=1): temperature, pressure, concentration
    - Vector-valued (vec>1): displacement, velocity, electric field
    
    Parameters
    ----------
    mesh : Mesh
        Finite element mesh containing node coordinates and element connectivity
    vec : int
        Number of vector components in solution (e.g., 3 for 3D displacement, 1 for temperature)
    dim : int
        Spatial dimension of the problem (2D or 3D)
    ele_type : str, optional
        Element type identifier (default: 'HEX8')
        Supported: 'TET4', 'TET10', 'HEX8', 'HEX20', 'HEX27', 'TRI3', 'TRI6', 'QUAD4', 'QUAD8'
    gauss_order : Optional[int], optional
        Gaussian quadrature order (default: determined by element type)
        
    Attributes
    ----------
    num_cells : int
        Number of elements in the mesh
    num_nodes : int
        Number of nodes per element
    num_total_nodes : int
        Total number of nodes in the mesh
    num_total_dofs : int
        Total degrees of freedom for this variable
    num_quads : int
        Number of quadrature points per element
    shape_vals : np.ndarray
        Shape function values at quadrature points
    shape_grads : np.ndarray
        Shape function gradients in physical coordinates
    JxW : np.ndarray
        Jacobian determinant times quadrature weights
        
    Notes
    -----
    The class pre-computes shape functions, gradients, and Jacobian data for
    efficient assembly operations. All computations are JAX-compatible for
    automatic differentiation and JIT compilation.
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = 'HEX8'
    gauss_order: Optional[int] = None

    def __post_init__(self) -> None:
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.mesh.points)
        self.num_total_dofs = self.num_total_nodes * self.vec

        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
        = get_face_shape_vals_and_grads(self.ele_type, self.gauss_order)
        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.shape_grads, self.JxW = self.get_shape_grads()
        # Initialize empty BC lists - these are no longer used since BC is handled separately
        self.node_inds_list = []
        self.vec_inds_list = []
        self.vals_list = []
        
        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]
        self.num_face_quads = self.face_quad_weights.shape[1]

    def get_shape_grads(self) -> Tuple['NDArray', 'NDArray']:
        """Compute shape function gradients in physical coordinates.

        Transforms shape function gradients from reference coordinates to physical
        coordinates using the Jacobian transformation. Also computes the Jacobian
        determinant times quadrature weights for integration.
        
        References
        ----------
        Hughes, Thomas JR. The finite element method: linear static and dynamic 
        finite element analysis. Courier Corporation, 2012. Page 147, Eq. (3.9.3)

        Returns
        -------
        shape_grads_physical : np.ndarray
            Shape function gradients in physical coordinates.
            Shape: (num_cells, num_quads, num_nodes, dim)
        JxW : np.ndarray
            Jacobian determinant times quadrature weights for integration.
            Shape: (num_cells, num_quads)
        """
        assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes, self.dim)
        physical_coos = np.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] *
                                   self.shape_grads_ref[None, :, :, None, :], axis=2, keepdims=True)
        jacobian_det = np.linalg.det(jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * self.quad_weights[None, :]
        return shape_grads_physical, JxW

    def get_face_shape_grads(self, boundary_inds: 'NDArray') -> Tuple['NDArray', 'NDArray']:
        """Compute face shape function gradients and surface integration scaling.
        
        Uses Nanson's formula to transform surface integrals from physical domain
        to reference domain. Computes shape function gradients on boundary faces
        and the scaling factor needed for surface integration.
        
        References
        ----------
        Wikiversity: Continuum mechanics/Volume change and area change
        https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Parameters
        ----------
        boundary_inds : np.ndarray
            Boundary face indices with shape (num_selected_faces, 2).
            First column: element index, Second column: local face index

        Returns
        -------
        face_shape_grads_physical : np.ndarray
            Face shape function gradients in physical coordinates.
            Shape: (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale : np.ndarray
            Surface integration scaling factor (Jacobian * weights).
            Shape: (num_selected_faces, num_face_quads)
        """
        physical_coos = np.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]]  # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = np.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = np.linalg.det(jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]

        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = np.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        selected_weights = self.face_quad_weights[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self) -> 'NDArray':
        """Compute physical coordinates of quadrature points.
        
        Maps quadrature points from reference element to physical coordinates
        using shape function interpolation.

        Returns
        -------
        physical_quad_points : np.ndarray
            Physical coordinates of quadrature points.
            Shape: (num_cells, num_quads, dim)
        """
        physical_coos = np.take(self.points, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim)
        physical_quad_points = np.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        return physical_quad_points

    def get_physical_surface_quad_points(self, boundary_inds: 'NDArray') -> 'NDArray':
        """Compute physical coordinates of surface quadrature points.
        
        Maps surface quadrature points from reference faces to physical coordinates
        using face shape function interpolation.

        Parameters
        ----------
        boundary_inds : np.ndarray
            Boundary face indices with shape (num_selected_faces, 2).
            First column: element index, Second column: local face index

        Returns
        -------
        physical_surface_quad_points : np.ndarray
            Physical coordinates of surface quadrature points.
            Shape: (num_selected_faces, num_face_quads, dim)
        """
        physical_coos = np.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim)
        physical_surface_quad_points = np.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return physical_surface_quad_points

    def Dirichlet_boundary_conditions(self, dirichlet_bc_info: Optional[List]) -> Tuple[List['NDArray'], List['NDArray'], List['NDArray']]:
        """Extract node indices and values for Dirichlet boundary conditions.
        
        Note: This method is deprecated. Use DirichletBC class from DCboundary module instead.

        Parameters
        ----------
        dirichlet_bc_info : list or None
            Legacy BC specification: [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_list : list[np.ndarray]
            Node indices for each BC. Values range from 0 to num_total_nodes - 1
        vec_inds_list : list[np.ndarray]  
            Vector component indices for each BC. Values range from 0 to vec - 1
        vals_list : list[np.ndarray]
            Prescribed values for each BC
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []
        if dirichlet_bc_info is not None:
            location_fns, vecs, value_fns = dirichlet_bc_info
            assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:
                    location_fn = lambda point, ind: location_fns[i](point)
                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}")

                node_inds = np.argwhere(jax.vmap(location_fn)(self.mesh.points, np.arange(self.num_total_nodes))).reshape(-1)
                vec_inds = np.ones_like(node_inds, dtype=np.int32) * vecs[i]
                values = jax.vmap(value_fns[i])(self.mesh.points[node_inds].reshape(-1, self.dim)).reshape(-1)
                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)
        return node_inds_list, vec_inds_list, vals_list

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info: List) -> None:
        """Update Dirichlet boundary conditions for time-dependent problems.
        
        Note: This method is deprecated. Use DirichletBC class from DCboundary module instead.
        
        Parameters
        ----------
        dirichlet_bc_info : list
            Legacy BC specification: [location_fns, vecs, value_fns]
        """
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(dirichlet_bc_info)

    def get_boundary_conditions_inds(self, location_fns: Optional[List[Callable]]) -> List['NDArray']:
        """Identify boundary faces that satisfy location function conditions.
        
        Determines which element faces lie on boundaries defined by location functions.
        Used internally for surface integral computations.

        Parameters
        ----------
        location_fns : list[callable] or None
            Location functions that define boundary regions. Each function takes
            a point coordinate and optionally a node index, returning True if the
            point is on the boundary.
            
            Examples:
            - Single argument: lambda x: np.isclose(x[0], 0.)
            - Two arguments: lambda x, ind: np.isclose(x[0], 0.) & np.isin(ind, [1, 3, 10])

        Returns
        -------
        boundary_inds_list : list[np.ndarray]
            List of boundary face indices for each location function.
            Each array has shape (num_selected_faces, 2) where:
            - [:, 0]: global element index
            - [:, 1]: local face index within element
        """

        # TODO: assume this works for all variables, and return the same result
        cell_points = np.take(self.points, self.cells, axis=0)  # (num_cells, num_nodes, dim)
        cell_face_points = np.take(cell_points, self.face_inds, axis=1)  # (num_cells, num_faces, num_face_vertices, dim)
        cell_face_inds = np.take(self.cells, self.face_inds, axis=1) # (num_cells, num_faces, num_face_vertices)
        boundary_inds_list = []
        if location_fns is not None:
            for i in range(len(location_fns)):
                num_args = location_fns[i].__code__.co_argcount
                if num_args == 1:
                    location_fn = lambda point, ind: location_fns[i](point)
                elif num_args == 2:
                    location_fn = location_fns[i]
                else:
                    raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, get {num_args}")

                vmap_location_fn = jax.vmap(location_fn)
                def on_boundary(cell_points, cell_inds):
                    boundary_flag = vmap_location_fn(cell_points, cell_inds)
                    return np.all(boundary_flag)

                vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
                boundary_flags = vvmap_on_boundary(cell_face_points, cell_face_inds)
                boundary_inds = np.argwhere(boundary_flags)  # (num_selected_faces, 2)
                boundary_inds_list.append(boundary_inds)

        return boundary_inds_list

    def convert_from_dof_to_quad(self, sol: 'NDArray') -> 'NDArray':
        """Interpolate nodal solution values to quadrature points.
        
        Uses shape functions to interpolate solution from nodes to quadrature
        points within each element.

        Parameters
        ----------
        sol : np.ndarray
            Nodal solution values with shape (num_total_nodes, vec)

        Returns
        -------
        u : np.ndarray
            Solution values at quadrature points.
            Shape: (num_cells, num_quads, vec)
        """
        # (num_total_nodes, vec) -> (num_cells, num_nodes, vec)
        cells_sol = sol[self.cells]
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.shape_vals[None, :, :, None], axis=2)
        return u

    def convert_from_dof_to_face_quad(self, sol: 'NDArray', boundary_inds: 'NDArray') -> 'NDArray':
        """Interpolate nodal solution to surface quadrature points.
        
        Uses face shape functions to interpolate solution from nodes to
        quadrature points on boundary faces.

        Parameters
        ----------
        sol : np.ndarray
            Nodal solution values with shape (num_total_nodes, vec)
        boundary_inds : np.ndarray
            Boundary face indices with shape (num_selected_faces, 2)

        Returns
        -------
        u : np.ndarray
            Solution values at surface quadrature points.
            Shape: (num_selected_faces, num_face_quads, vec)
        """
        cells_old_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        selected_cell_sols = cells_old_sol[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) 
        # -> (num_selected_faces, num_face_quads, vec)
        u = np.sum(selected_cell_sols[:, None, :, :] * selected_face_shape_vals[:, :, :, None], axis=2)
        return u

    def sol_to_grad(self, sol: 'NDArray') -> 'NDArray':
        """Compute solution gradients at quadrature points.
        
        Uses shape function gradients to compute spatial derivatives of the
        solution at quadrature points within each element.

        Parameters
        ----------
        sol : np.ndarray
            Nodal solution values with shape (num_total_nodes, vec)

        Returns
        -------
        u_grads : np.ndarray
            Solution gradients at quadrature points.
            Shape: (num_cells, num_quads, vec, dim)
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)
        return u_grads

    def print_BC_info(self) -> None:
        """Print boundary condition information for debugging.
        
        Note: This method is deprecated and may not work correctly.
        Use DirichletBC class from DCboundary module for BC handling.
        """
        if hasattr(self, 'neumann_boundary_inds_list'):
            print(f"\n\n### Neumann B.C. is specified")
            for i in range(len(self.neumann_boundary_inds_list)):
                print(f"\nNeumann Boundary part {i + 1} information:")
                print(self.neumann_boundary_inds_list[i])
                print(
                    f"Array.shape = (num_selected_faces, 2) = {self.neumann_boundary_inds_list[i].shape}"
                )
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the global cell index of the ith selected face"
                )
                print(
                    f"    Array[i, 1] returns the local face index of the ith selected face"
                )
        else:
            print(f"\n\n### No Neumann B.C. found.")

        if len(self.node_inds_list) != 0:
            print(f"\n\n### Dirichlet B.C. is specified")
            for i in range(len(self.node_inds_list)):
                print(f"\nDirichlet Boundary part {i + 1} information:")
                bc_array = np.stack([
                    self.node_inds_list[i], self.vec_inds_list[i],
                    self.vals_list[i]
                ]).T
                print(bc_array)
                print(
                    f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}")
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the node index of the ith selected dof"
                )
                print(
                    f"    Array[i, 1] returns the vec index of the ith selected dof"
                )
                print(
                    f"    Array[i, 2] returns the value assigned to ith selected dof"
                )
        else:
            print(f"\n\n### No Dirichlet B.C. found.")
