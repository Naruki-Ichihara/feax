"""
Internal variables management for FEAX finite element framework.

This module provides the InternalVars dataclass for managing material properties
and loading parameters separately from problem structure. This separation enables
efficient optimization and parameter studies through JAX transformations.

The design philosophy separates:
- Problem: Geometric structure (mesh, elements, quadrature)  
- InternalVars: Material parameters and loads
"""

import jax
import jax.numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from feax.problem import Problem


@dataclass(frozen=True)
class InternalVars:
    """Container for internal variables used in finite element computations.

    This dataclass holds material properties, loading parameters, and other
    variables that change during optimization or parameter studies, while keeping
    the finite element structure (Problem) fixed. This separation enables efficient
    JAX transformations for optimization and sensitivity analysis.

    The data structure mirrors the physics kernel organization:
    - volume_vars: Parameters for volume integrals (tensor maps, mass maps)
    - surface_vars: Parameters for surface integrals (boundary loads, tractions)

    Parameters
    ----------
    volume_vars : Tuple[np.ndarray, ...], optional
        Tuple of arrays for volume integral parameters. Each array can have shape:
        - (num_nodes,) for node-based variables (most memory efficient)
        - (num_cells,) for cell-based/element-wise variables
        - (num_cells, num_quads) for quad-point based variables (legacy)
        The assembler automatically interpolates to quadrature points.
        Common examples: material properties (E, nu), density, source terms
    surface_vars : Optional[List[Tuple[np.ndarray, ...]]], optional
        List of tuples for surface integral parameters. One entry per surface/location_fn.
        Each tuple contains arrays with shape (num_surface_faces, num_face_quads).
        Common examples: surface tractions, pressures, heat fluxes

    Notes
    -----
    This class is registered as a JAX PyTree for automatic differentiation and
    transformations like vmap, grad, jit. The frozen dataclass ensures immutability
    required for functional programming patterns.

    Examples
    --------
    Creating material parameters for elasticity (node-based):
    >>> E = InternalVars.create_node_var(problem, 210e9)  # Young's modulus at nodes
    >>> nu = InternalVars.create_node_var(problem, 0.3)   # Poisson's ratio at nodes
    >>> internal_vars = InternalVars(volume_vars=(E, nu))

    Creating cell-based material properties:
    >>> rho = InternalVars.create_cell_var(problem, 0.5)  # Density per element
    >>> internal_vars = InternalVars(volume_vars=(rho,))

    Adding spatial variation:
    >>> def E_field(x): return 200e9 + 50e9 * x[0]  # Varies with x-coordinate
    >>> E_varying = InternalVars.create_node_var_from_fn(problem, E_field)
    >>> internal_vars = InternalVars(volume_vars=(E_varying,))
    """
    volume_vars: Tuple[np.ndarray, ...] = ()
    surface_vars: Optional[List[Tuple[np.ndarray, ...]]] = None
    
    def __post_init__(self) -> None:
        """Initialize default values for optional fields.
        
        This method is called automatically after dataclass initialization
        to handle default values for optional fields.
        """
        # Initialize surface_vars as empty list if None
        if self.surface_vars is None:
            object.__setattr__(self, 'surface_vars', [])
    
    @staticmethod
    def create_node_var(problem: 'Problem', value: float, var_index: int = 0) -> np.ndarray:
        """Create uniform node-based variable (most memory efficient).

        Node-based variables are the most efficient representation for material
        properties. The assembler automatically interpolates to quadrature points
        using finite element shape functions.

        Parameters
        ----------
        problem : Problem
            The finite element problem to get dimensions from
        value : float
            The uniform value to assign to all nodes
        var_index : int, optional
            Which finite element variable to use for multi-variable problems.
            Default is 0 (first variable)

        Returns
        -------
        array : np.ndarray
            Array of shape (num_nodes,) filled with the specified value

        Examples
        --------
        Create uniform material properties:
        >>> E = InternalVars.create_node_var(problem, 210e9)  # Young's modulus
        >>> rho = InternalVars.create_node_var(problem, 7800)  # Density
        """
        num_nodes = problem.fes[var_index].num_total_nodes
        return np.full(num_nodes, value)

    @staticmethod
    def create_cell_var(problem: 'Problem', value: float, var_index: int = 0) -> np.ndarray:
        """Create uniform cell-based (element-wise) variable.

        Cell-based variables are useful for element-wise constant properties,
        such as in topology optimization where each element has a single density.
        The assembler automatically expands to quadrature points.

        Parameters
        ----------
        problem : Problem
            The finite element problem to get dimensions from
        value : float
            The uniform value to assign to all cells
        var_index : int, optional
            Which finite element variable to use for multi-variable problems.
            Default is 0 (first variable)

        Returns
        -------
        array : np.ndarray
            Array of shape (num_cells,) filled with the specified value

        Examples
        --------
        Create uniform element properties:
        >>> rho = InternalVars.create_cell_var(problem, 0.5)  # Topology density per element
        >>> E = InternalVars.create_cell_var(problem, 70e3)  # Young's modulus per element
        """
        num_cells = problem.num_cells
        return np.full(num_cells, value)

    @staticmethod
    def create_uniform_volume_var(problem: 'Problem', value: float, var_index: int = 0) -> np.ndarray:
        """Create uniform volume variable array for all quadrature points (legacy).

        NOTE: This method is deprecated. Use create_node_var() or create_cell_var() instead
        for better memory efficiency.

        Parameters
        ----------
        problem : Problem
            The finite element problem to get dimensions from
        value : float
            The uniform value to assign to all quadrature points
        var_index : int, optional
            Which finite element variable to use for multi-variable problems.
            Default is 0 (first variable)

        Returns
        -------
        array : np.ndarray
            Array of shape (num_cells, num_quads) filled with the specified value

        Examples
        --------
        Create uniform material properties:
        >>> E = InternalVars.create_uniform_volume_var(problem, 210e9)  # Young's modulus
        >>> rho = InternalVars.create_uniform_volume_var(problem, 7800)  # Density
        """
        num_cells = problem.num_cells
        num_quads = problem.fes[var_index].num_quads
        return np.full((num_cells, num_quads), value)
    
    @staticmethod
    def create_uniform_surface_var(problem: 'Problem', value: float, surface_index: int = 0) -> np.ndarray:
        """Create uniform surface variable array for all surface quadrature points.
        
        Used to create surface parameters like uniform tractions, pressures,
        or boundary fluxes that are constant over a boundary surface.
        
        Parameters
        ----------
        problem : Problem
            The finite element problem to get surface dimensions from
        value : float
            The uniform value to assign to all surface quadrature points
        surface_index : int, optional
            Which surface/location_fn to use. Default is 0 (first surface)
            
        Returns
        -------
        array : np.ndarray
            Array of shape (num_surface_faces, num_face_quads) filled with the specified value
            
        Examples
        --------
        Create uniform surface loads:
        >>> pressure = InternalVars.create_uniform_surface_var(problem, -1000)  # Pressure load
        >>> heat_flux = InternalVars.create_uniform_surface_var(problem, 50.0)  # Heat flux
        """
        num_surface_faces = len(problem.boundary_inds_list[surface_index])
        num_face_quads = problem.fes[0].face_shape_vals.shape[1]
        return np.full((num_surface_faces, num_face_quads), value)
    
    @staticmethod
    def create_node_var_from_fn(problem: 'Problem', var_fn: Callable[[np.ndarray], float],
                                var_index: int = 0) -> np.ndarray:
        """Create spatially varying node-based variable using a function.

        This method evaluates a user-defined function at all node positions
        to create spatially varying material properties or parameters.

        Parameters
        ----------
        problem : Problem
            The finite element problem to get node positions from
        var_fn : Callable[[np.ndarray], float]
            Function that takes position coordinates (x, y, z) and returns variable value.
            Function signature: (coordinates: np.ndarray) -> float
        var_index : int, optional
            Which finite element variable to use. Default is 0 (first variable)

        Returns
        -------
        array : np.ndarray
            Array of shape (num_nodes,) with spatially varying values

        Examples
        --------
        Create spatially varying material properties:
        >>> def E_gradient(x): return 200e9 + 50e9 * x[0]  # Varies with x-coordinate
        >>> E_varying = InternalVars.create_node_var_from_fn(problem, E_gradient)

        >>> def density_field(x): return 7800 * (1 + 0.1 * np.sin(x[0]))  # Sinusoidal variation
        >>> rho_varying = InternalVars.create_node_var_from_fn(problem, density_field)
        """
        node_points = problem.fes[var_index].points  # (num_nodes, dim)
        return jax.vmap(var_fn)(node_points)

    @staticmethod
    def create_cell_var_from_fn(problem: 'Problem', var_fn: Callable[[np.ndarray], float],
                                var_index: int = 0) -> np.ndarray:
        """Create spatially varying cell-based variable using a function.

        This method evaluates a user-defined function at element centroids
        to create spatially varying material properties or parameters.

        Parameters
        ----------
        problem : Problem
            The finite element problem to get element centroids from
        var_fn : Callable[[np.ndarray], float]
            Function that takes position coordinates (x, y, z) and returns variable value.
            Function signature: (coordinates: np.ndarray) -> float
        var_index : int, optional
            Which finite element variable to use. Default is 0 (first variable)

        Returns
        -------
        array : np.ndarray
            Array of shape (num_cells,) with spatially varying values

        Examples
        --------
        Create spatially varying material properties:
        >>> def density_field(x): return 0.5 * (1 + np.tanh(x[0]))  # Smooth transition
        >>> rho_varying = InternalVars.create_cell_var_from_fn(problem, density_field)
        """
        # Compute cell centroids
        cells = problem.fes[var_index].cells
        points = problem.fes[var_index].points
        centroids = jax.vmap(lambda cell: np.mean(points[cell], axis=0))(cells)
        return jax.vmap(var_fn)(centroids)

    @staticmethod
    def create_spatially_varying_volume_var(problem: 'Problem', var_fn: Callable[[np.ndarray], float],
                                          var_index: int = 0) -> np.ndarray:
        """Create spatially varying volume variable using a function (legacy).

        NOTE: This method is deprecated. Use create_node_var_from_fn() or
        create_cell_var_from_fn() instead for better memory efficiency.

        This method evaluates a user-defined function at all quadrature points
        to create spatially varying material properties or parameters.

        Parameters
        ----------
        problem : Problem
            The finite element problem to get quadrature points from
        var_fn : Callable[[np.ndarray], float]
            Function that takes position coordinates (x, y, z) and returns variable value.
            Function signature: (coordinates: np.ndarray) -> float
        var_index : int, optional
            Which finite element variable to use. Default is 0 (first variable)

        Returns
        -------
        array : np.ndarray
            Array of shape (num_cells, num_quads) with spatially varying values

        Examples
        --------
        Create spatially varying material properties:
        >>> def E_gradient(x): return 200e9 + 50e9 * x[0]  # Varies with x-coordinate
        >>> E_varying = InternalVars.create_spatially_varying_volume_var(problem, E_gradient)

        >>> def density_field(x): return 7800 * (1 + 0.1 * np.sin(x[0]))  # Sinusoidal variation
        >>> rho_varying = InternalVars.create_spatially_varying_volume_var(problem, density_field)
        """
        quad_points = problem.physical_quad_points  # (num_cells, num_quads, dim)
        return jax.vmap(jax.vmap(var_fn))(quad_points)
    
    @staticmethod
    def create_spatially_varying_surface_var(problem: 'Problem', var_fn: Callable[[np.ndarray], float], 
                                           surface_index: int = 0) -> np.ndarray:
        """Create spatially varying surface variable using a function.
        
        This method evaluates a user-defined function at all surface quadrature points
        to create spatially varying surface loads, tractions, or boundary conditions.
        
        Parameters
        ----------
        problem : Problem
            The finite element problem to get surface quadrature points from
        var_fn : Callable[[np.ndarray], float]
            Function that takes position coordinates (x, y, z) and returns variable value.
            Function signature: (coordinates: np.ndarray) -> float
        surface_index : int, optional
            Which surface to use. Default is 0 (first surface)
            
        Returns
        -------
        array : np.ndarray
            Array of shape (num_surface_faces, num_face_quads) with spatially varying values
            
        Examples
        --------
        Create spatially varying surface loads:
        >>> def pressure_gradient(x): return 1000 * x[1]  # Hydrostatic pressure
        >>> pressure = InternalVars.create_spatially_varying_surface_var(problem, pressure_gradient)
        
        >>> def traction_field(x): return 500 * np.sin(np.pi * x[0])  # Sinusoidal traction
        >>> traction = InternalVars.create_spatially_varying_surface_var(problem, traction_field)
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
def _internal_vars_flatten(iv: InternalVars) -> Tuple[List[np.ndarray], Tuple[int, List[int]]]:
    """Flatten InternalVars into leaves and auxiliary data for JAX pytree.
    
    This function extracts all JAX arrays (leaves) and structural information
    needed to reconstruct the InternalVars object.
    
    Parameters
    ----------
    iv : InternalVars
        InternalVars object to flatten
        
    Returns
    -------
    Tuple[List[np.ndarray], Tuple[int, List[int]]]
        (leaves, aux_data) where leaves contains all arrays and aux_data
        contains structure information for reconstruction
    """
    # Flatten all arrays into a single list
    leaves = list(iv.volume_vars)
    for surface_tuple in iv.surface_vars:
        leaves.extend(surface_tuple)
    
    # Auxiliary data to reconstruct structure
    aux_data = (len(iv.volume_vars), [len(st) for st in iv.surface_vars])
    return leaves, aux_data


def _internal_vars_unflatten(aux_data: Tuple[int, List[int]], leaves: List[np.ndarray]) -> InternalVars:
    """Reconstruct InternalVars from flattened leaves and auxiliary data.
    
    This function rebuilds the InternalVars structure from the flat list of
    arrays and structural information.
    
    Parameters
    ----------
    aux_data : Tuple[int, List[int]]
        Structural information: (num_volume_vars, surface_var_counts)
    leaves : List[np.ndarray]
        Flat list of all arrays
        
    Returns
    -------
    InternalVars
        Reconstructed InternalVars object
    """
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