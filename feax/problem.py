"""
Problem class with modular design separating FE structure from material parameters.

This module provides the core Problem class that defines finite element problem
structure independent of material parameters, enabling efficient optimization
and parameter studies through JAX transformations.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Callable, Any, TYPE_CHECKING

from feax.mesh import Mesh
from feax.fe import FiniteElement

if TYPE_CHECKING:
    from feax.types import TensorMap, MassMap, SurfaceMap


@dataclass
class Problem:
    """Finite element problem definition with separated structure and material parameters.
    
    This class defines the finite element problem structure (mesh, elements, quadrature)
    independently of material parameters, enabling efficient optimization and parameter
    studies through JAX transformations.
    
    The design separates:
    - Structure: mesh, elements, boundary conditions (Problem class)
    - Parameters: material properties, loads (InternalVars class)
    
    This separation allows the same Problem to be used with different material
    parameters while maintaining efficiency through pre-computed geometric data.
    
    Parameters
    ----------
    mesh : Union[Mesh, List[Mesh]]
        Finite element mesh(es). Single mesh for single-variable problems,
        list of meshes for multi-variable problems
    vec : Union[int, List[int]] 
        Number of vector components per variable. Single int for single-variable,
        list of ints for multi-variable problems
    dim : int
        Spatial dimension of the problem (2D or 3D)
    ele_type : Union[str, List[str]], optional
        Element type identifier(s). Default 'HEX8'
    gauss_order : Union[int, List[int]], optional
        Gaussian quadrature order(s). Default determined by element type
    location_fns : Optional[List[Callable]], optional
        Functions defining boundary locations for surface integrals
    additional_info : Tuple[Any, ...], optional
        Additional problem-specific information passed to custom_init()
        
    Attributes
    ----------
    num_vars : int
        Number of variables in the problem
    fes : List[FiniteElement] 
        Finite element objects for each variable
    num_cells : int
        Total number of elements
    num_total_dofs_all_vars : int
        Total degrees of freedom across all variables
    I, J : np.ndarray
        Sparse matrix indices for assembly
    unflatten_fn_sol_list : Callable
        Function to unflatten solution vector to per-variable arrays
        
    Notes
    -----
    Subclasses should implement:
    - get_tensor_map(): Returns function for gradient-based physics 
    - get_mass_map(): Returns function for mass/reaction terms (optional)
    - get_surface_maps(): Returns functions for surface loads (optional)
    - custom_init(): Additional initialization if needed (optional)
    """
    mesh: Union[Mesh, List[Mesh]]
    vec: Union[int, List[int]]
    dim: int
    ele_type: Union[str, List[str]] = 'HEX8'
    gauss_order: Optional[Union[int, List[int]]] = None
    location_fns: Optional[List[Callable]] = None
    additional_info: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Initialize all state data for the finite element problem.
        
        This method handles the conversion of single variables to lists for
        uniform processing, creates finite element objects, computes assembly
        indices, and pre-computes geometric data for efficient assembly.
        
        The initialization process:
        1. Normalizes input parameters to list format
        2. Creates FiniteElement objects for each variable  
        3. Computes sparse matrix assembly indices (I, J)
        4. Pre-computes shape functions and Jacobian data
        5. Sets up boundary condition data structures
        6. Calls custom_init() for problem-specific setup
        """
        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.gauss_order = [self.gauss_order]

        self.num_vars = len(self.mesh)

        self.fes = [FiniteElement(mesh=self.mesh[i], 
                                  vec=self.vec[i], 
                                  dim=self.dim, 
                                  ele_type=self.ele_type[i], 
                                  gauss_order=self.gauss_order[i] if type(self.gauss_order) == type([]) else self.gauss_order) \
                    for i in range(self.num_vars)] 

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0] 
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                x[i].reshape(-1)
                crt_ind = self.fes[i].vec * x[i][:, None] + np.arange(self.fes[i].vec)[None, :] + self.offset[i]
                inds.append(crt_ind.reshape(-1))

            return np.hstack(inds)

        # (num_cells, num_nodes*vec + ...)
        inds = np.array(jax.vmap(find_ind)(*self.cells_list))
        self.I = np.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
        self.J = np.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list] # [(num_selected_faces, num_nodes), ...]
            inds_face = np.array(jax.vmap(find_ind)(*cells_list_face)) # (num_selected_faces, num_nodes*vec + ...)
            I_face = np.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
            J_face = np.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
            self.I = np.hstack((self.I, I_face))
            self.J = np.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)
     
        self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*self.cells_list) # (num_cells, num_nodes + ...)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)
        
        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
        self.num_total_dofs_all_vars = len(dumb_dofs)

        self.num_nodes_cumsum = np.cumsum(np.array([0] + [fe.num_nodes for fe in self.fes]))
        # (num_cells, num_vars, num_quads)
        self.JxW = np.transpose(np.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2)) 
        # (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = np.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # TODO: assert all vars quad points be the same
        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points()  

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for fe in self.fes:
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds)  
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))  
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds) 

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # TODO: assert all vars face quad points be the same
            self.physical_surface_quad_points.append(physical_surface_quad_points)

        # Initialize without internal_vars - kernels will be created separately
        self.custom_init(*self.additional_info)

    def custom_init(self, *args: Any) -> None:
        """Custom initialization for problem-specific setup.
        
        Subclasses should override this method to perform additional
        initialization using the additional_info parameters.
        
        Parameters
        ----------
        *args : Any
            Arguments passed from additional_info tuple
        """
        pass

    def get_tensor_map(self) -> 'TensorMap':
        """Get tensor map function for gradient-based physics.
        
        This method must be implemented by subclasses to define the constitutive
        relationship between gradients and stress/flux tensors.
        
        Returns
        -------
        TensorMap
            Function that maps gradients to stress/flux tensors
            Signature: (u_grad: Array, *internal_vars) -> stress_tensor: Array
            
        Raises
        ------
        NotImplementedError
            If not implemented by subclass
            
        Examples
        --------
        For linear elasticity:
        >>> def tensor_map(u_grad, E, nu):
        ...     # Compute stress from displacement gradient
        ...     return stress_tensor
        """
        raise NotImplementedError("Subclass must implement get_tensor_map")
    
    def get_surface_maps(self) -> List['SurfaceMap']:
        """Get surface map functions for boundary loads.
        
        Override this method to define surface tractions, pressures, or fluxes
        applied to boundaries identified by location_fns.
        
        Returns
        -------
        List[SurfaceMap]
            List of functions for surface loads. Each function has signature:
            (u: Array, x: Array, *internal_vars) -> traction: Array
            
        Notes
        -----
        The number of surface maps should match the number of location_fns
        provided to the Problem constructor.
        """
        return []
    
    def get_mass_map(self) -> Optional['MassMap']:
        """Get mass map function for inertia/reaction terms.
        
        Override this method to define mass matrix contributions or reaction terms
        that don't involve gradients (e.g., inertia, damping, reactions).
        
        Returns
        -------
        Optional[MassMap]
            Function for mass/reaction terms with signature:
            (u: Array, x: Array, *internal_vars) -> mass_term: Array
            Returns None if no mass terms are present
        """
        return None


# Register as JAX PyTree for use with JAX transformations
def _problem_tree_flatten(obj: Problem) -> Tuple[Tuple, dict]:
    """Flatten Problem object for JAX pytree registration.
    
    Since Problem objects contain only static structure information
    (no JAX arrays), all data goes into the static part.
    
    Parameters
    ----------
    obj : Problem
        Problem object to flatten
        
    Returns
    -------
    Tuple[Tuple, dict]
        (dynamic_data, static_data) where dynamic_data is empty
        and static_data contains all Problem fields
    """
    # No dynamic parts - everything is static structure
    dynamic = ()
    
    # All data is static geometric/structural information
    static = {
        'mesh': obj.mesh,
        'vec': obj.vec,
        'dim': obj.dim,
        'ele_type': obj.ele_type,
        'gauss_order': obj.gauss_order,
        'location_fns': obj.location_fns,
        'additional_info': obj.additional_info,
    }
    return dynamic, static


def _problem_tree_unflatten(static: dict, dynamic: Tuple) -> Problem:
    """Reconstruct Problem object from flattened parts.
    
    Parameters
    ----------
    static : dict
        Static data containing Problem constructor arguments
    dynamic : Tuple
        Dynamic data (empty for Problem objects)
        
    Returns
    -------
    Problem
        Reconstructed Problem instance
    """
    # Create instance with original constructor parameters
    instance = Problem(
        mesh=static['mesh'],
        vec=static['vec'],
        dim=static['dim'],
        ele_type=static['ele_type'],
        gauss_order=static['gauss_order'],
        location_fns=static['location_fns'],
        additional_info=static['additional_info'],
    )
    
    return instance


jax.tree_util.register_pytree_node(
    Problem,
    _problem_tree_flatten,
    _problem_tree_unflatten
)