"""
Dataclass for managing internal variables in a JAX-compatible way.
"""

import jax
import jax.numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass(frozen=True)
class InternalVars:
    """Container for internal variables used in FE computations.
    
    This dataclass separates material properties and loading parameters from the
    Problem structure, making it easier to vary these parameters in optimization
    or sensitivity analysis while keeping the FE structure fixed.
    
    Attributes
    ----------
    volume_vars : tuple of arrays
        Internal variables for volume integrals. Each array has shape 
        (num_cells, num_quads) and represents values at quadrature points.
    surface_vars : list of tuples
        Internal variables for surface integrals. One entry per surface/location_fn.
        Each entry is a tuple of arrays with shape (num_surface_faces, num_face_quads).
    """
    volume_vars: Tuple[np.ndarray, ...] = ()
    surface_vars: List[Tuple[np.ndarray, ...]] = None
    
    def __post_init__(self):
        # Initialize surface_vars as empty list if None
        if self.surface_vars is None:
            object.__setattr__(self, 'surface_vars', [])
    
    @staticmethod
    def create_uniform_volume_var(problem, value: float, var_index: int = 0) -> np.ndarray:
        """Create a uniform volume variable array for all quadrature points.
        
        Parameters
        ----------
        problem : Problem
            The FE problem to get dimensions from
        value : float
            The uniform value to set
        var_index : int
            Which FE variable to use (for multi-physics problems)
            
        Returns
        -------
        array : np.ndarray
            Array of shape (num_cells, num_quads) filled with value
        """
        num_cells = problem.num_cells
        num_quads = problem.fes[var_index].num_quads
        return np.full((num_cells, num_quads), value)
    
    @staticmethod
    def create_uniform_surface_var(problem, value: float, surface_index: int = 0) -> np.ndarray:
        """Create a uniform surface variable array for all surface quadrature points.
        
        Parameters
        ----------
        problem : Problem
            The FE problem to get dimensions from
        value : float
            The uniform value to set
        surface_index : int
            Which surface/location_fn to use
            
        Returns
        -------
        array : np.ndarray
            Array of shape (num_surface_faces, num_face_quads) filled with value
        """
        num_surface_faces = len(problem.boundary_inds_list[surface_index])
        num_face_quads = problem.fes[0].face_shape_vals.shape[1]
        return np.full((num_surface_faces, num_face_quads), value)
    
    @staticmethod
    def create_spatially_varying_volume_var(problem, var_fn, var_index: int = 0) -> np.ndarray:
        """Create a spatially varying volume variable using a function.
        
        Parameters
        ----------
        problem : Problem
            The FE problem to get quadrature points from
        var_fn : callable
            Function that takes position (x, y, z) and returns variable value
        var_index : int
            Which FE variable to use
            
        Returns
        -------
        array : np.ndarray
            Array of shape (num_cells, num_quads) with spatially varying values
        """
        quad_points = problem.physical_quad_points  # (num_cells, num_quads, dim)
        return jax.vmap(jax.vmap(var_fn))(quad_points)
    
    @staticmethod
    def create_spatially_varying_surface_var(problem, var_fn, surface_index: int = 0) -> np.ndarray:
        """Create a spatially varying surface variable using a function.
        
        Parameters
        ----------
        problem : Problem
            The FE problem to get surface quadrature points from
        var_fn : callable
            Function that takes position (x, y, z) and returns variable value
        surface_index : int
            Which surface to use
            
        Returns
        -------
        array : np.ndarray
            Array of shape (num_surface_faces, num_face_quads) with spatially varying values
        """
        surface_quad_points = problem.physical_surface_quad_points[surface_index]
        return jax.vmap(jax.vmap(var_fn))(surface_quad_points)
    
    def replace_volume_var(self, index: int, new_var: np.ndarray) -> 'InternalVars':
        """Create a new InternalVars with one volume variable replaced.
        
        Parameters
        ----------
        index : int
            Index of volume variable to replace
        new_var : np.ndarray
            New variable array
            
        Returns
        -------
        InternalVars
            New instance with updated variable
        """
        volume_vars_list = list(self.volume_vars)
        volume_vars_list[index] = new_var
        return InternalVars(tuple(volume_vars_list), self.surface_vars)
    
    def replace_surface_var(self, surface_index: int, var_index: int, new_var: np.ndarray) -> 'InternalVars':
        """Create a new InternalVars with one surface variable replaced.
        
        Parameters
        ----------
        surface_index : int
            Index of surface (location_fn)
        var_index : int
            Index of variable within that surface
        new_var : np.ndarray
            New variable array
            
        Returns
        -------
        InternalVars
            New instance with updated variable
        """
        surface_vars_list = list(self.surface_vars)
        surface_tuple_list = list(surface_vars_list[surface_index])
        surface_tuple_list[var_index] = new_var
        surface_vars_list[surface_index] = tuple(surface_tuple_list)
        return InternalVars(self.volume_vars, surface_vars_list)


# Register as JAX PyTree for automatic differentiation and transformations
def _internal_vars_flatten(iv):
    """Flatten InternalVars into leaves and treedef."""
    # Flatten all arrays into a single list
    leaves = list(iv.volume_vars)
    for surface_tuple in iv.surface_vars:
        leaves.extend(surface_tuple)
    
    # Auxiliary data to reconstruct structure
    aux_data = (len(iv.volume_vars), [len(st) for st in iv.surface_vars])
    return leaves, aux_data


def _internal_vars_unflatten(aux_data, leaves):
    """Reconstruct InternalVars from leaves and treedef."""
    num_volume_vars, surface_var_counts = aux_data
    
    # Extract volume vars
    volume_vars = tuple(leaves[:num_volume_vars])
    leaves = leaves[num_volume_vars:]
    
    # Extract surface vars
    surface_vars = []
    for count in surface_var_counts:
        surface_vars.append(tuple(leaves[:count]))
        leaves = leaves[count:]
    
    return InternalVars(volume_vars, surface_vars)


# Register the PyTree
jax.tree_util.register_pytree_node(
    InternalVars,
    _internal_vars_flatten,
    _internal_vars_unflatten
)