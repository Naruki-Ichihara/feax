"""
Dirichlet boundary condition implementation for FEAX finite element framework.

This module provides the core DirichletBC class for efficient boundary condition
application and dataclass-based BC specification classes for type-safe definition.

Key Classes:
- DirichletBC: JAX-compatible BC class with efficient apply methods  
- DirichletBCSpec: Dataclass for specifying individual boundary conditions
- DirichletBCConfig: Container for multiple BC specifications with convenience methods
"""

import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from dataclasses import dataclass, field
from typing import Callable, List, Union, Tuple, Optional, TYPE_CHECKING
from jax.tree_util import register_pytree_node

if TYPE_CHECKING:
    from feax.problem import Problem


@dataclass(frozen=True)
class DirichletBC:
    """JAX-compatible dataclass for Dirichlet boundary conditions.
    
    This class pre-computes and stores all BC information as static JAX arrays,
    making it suitable for JIT compilation.
    """
    bc_rows: np.ndarray  # All boundary condition row indices
    bc_mask: np.ndarray  # Boolean mask for BC rows (size: total_dofs)
    bc_vals: np.ndarray  # Boundary condition values for each BC row
    total_dofs: int
    
    @staticmethod
    def from_specs(problem: 'Problem', specs: List['DirichletBCSpec']) -> 'DirichletBC':
        """Create DirichletBC directly from a list of DirichletBCSpec objects.
        
        This is a convenient factory method that creates a DirichletBC without 
        needing to create an intermediate DirichletBCConfig object.
        
        Parameters
        ----------
        problem : Problem
            The finite element problem instance
        specs : List[DirichletBCSpec]
            List of boundary condition specifications
            
        Returns
        -------
        DirichletBC
            The compiled boundary condition object
            
        Examples
        --------
        >>> bc = DirichletBC.from_specs(problem, [
        ...     DirichletBCSpec(left_boundary, 'all', 0.0),
        ...     DirichletBCSpec(right_boundary, 'x', 0.1)
        ... ])
        """
        config = DirichletBCConfig(specs)
        return config.create_bc(problem)


# Register DirichletBC as a JAX pytree
def _dirichletbc_flatten(bc: DirichletBC) -> Tuple[Tuple[np.ndarray, ...], int]:
    """Flatten DirichletBC into a list of arrays and auxiliary data.
    
    Parameters
    ----------
    bc : DirichletBC
        The DirichletBC object to flatten
        
    Returns
    -------
    Tuple[Tuple[np.ndarray, ...], int]
        A tuple of (arrays, aux_data) where arrays contains the JAX arrays
        and aux_data contains static information
    """
    # Arrays go in the first return value
    arrays = (bc.bc_rows, bc.bc_mask, bc.bc_vals)
    # Static data goes in the second return value
    aux_data = bc.total_dofs
    return arrays, aux_data


def _dirichletbc_unflatten(aux_data: int, arrays: Tuple[np.ndarray, ...]) -> DirichletBC:
    """Reconstruct DirichletBC from flattened representation.
    
    Parameters
    ----------
    aux_data : int
        Static auxiliary data (total_dofs)
    arrays : Tuple[np.ndarray, ...]
        Tuple of JAX arrays (bc_rows, bc_mask, bc_vals)
        
    Returns
    -------
    DirichletBC
        Reconstructed DirichletBC object
    """
    bc_rows, bc_mask, bc_vals = arrays
    total_dofs = aux_data
    return DirichletBC(bc_rows=bc_rows, bc_mask=bc_mask, bc_vals=bc_vals, total_dofs=total_dofs)


# Register the pytree
register_pytree_node(
    DirichletBC,
    _dirichletbc_flatten,
    _dirichletbc_unflatten
)


def apply_boundary_to_J(bc: DirichletBC, J: BCOO) -> BCOO:
    """Apply Dirichlet boundary conditions to Jacobian matrix J using row elimination.
    
    This function modifies the Jacobian matrix to enforce Dirichlet boundary conditions
    by zeroing out all entries in boundary condition rows and setting diagonal entries
    to 1.0 for those rows. This transforms the system to enforce u[bc_dof] = bc_val.
    
    The algorithm:
    1. Zero out all entries in BC rows (both on-diagonal and off-diagonal)
    2. Set diagonal entries to 1.0 for all BC rows
    3. Handle potential duplicates by concatenation (JAX sparse solvers handle this)
    
    Parameters
    ----------
    bc : DirichletBC
        Pre-computed boundary condition information containing:
        - bc_rows: DOF indices where BCs are applied
        - bc_mask: Boolean mask for fast BC row identification
        - bc_vals: Prescribed values (not used in Jacobian modification)
        - total_dofs: Total number of DOFs in the system
    J : jax.experimental.sparse.BCOO
        The sparse Jacobian matrix in BCOO format with shape (total_dofs, total_dofs)
        
    Returns
    -------
    J_bc : jax.experimental.sparse.BCOO
        The Jacobian matrix with boundary conditions applied, same shape as input
        
    Notes
    -----
    This function is JAX-JIT compatible and designed for efficient use in Newton solvers.
    The returned matrix may have duplicate entries (original zeros + new diagonal ones),
    but JAX sparse solvers handle this correctly by summing duplicates.
    """
    # Get the data and indices from the BCOO matrix
    data = J.data
    indices = J.indices
    shape = J.shape
    # Get row and column indices from sparse matrix
    row_indices = indices[:, 0]
    
    # Create mask for BC rows using pre-computed bc_mask
    is_bc_row = bc.bc_mask[row_indices]
    
    # The algorithm:
    # 1. Zero out all BC row entries 
    # 2. Add diagonal entries for ALL BC rows with value 1.0
    
    # Step 1: Zero out all BC row entries
    bc_row_mask = is_bc_row
    data_modified = np.where(bc_row_mask, 0.0, data)
    
    # Step 2: Add diagonal entries for ALL BC rows
    # Direct approach that works with JIT: always add all BC diagonal entries
    # This may create duplicates, but most JAX sparse solvers handle this correctly
    
    bc_diag_indices = np.stack([bc.bc_rows, bc.bc_rows], axis=-1)
    bc_diag_data = np.ones_like(bc.bc_rows, dtype=data.dtype)
    
    # Concatenate all data
    all_indices = np.concatenate([indices, bc_diag_indices], axis=0)
    all_data = np.concatenate([data_modified, bc_diag_data], axis=0)
    
    # Create final BCOO matrix
    J_bc = BCOO((all_data, all_indices), shape=shape)
    
    # Skip sorting for large matrices to avoid slow compilation
    # Most JAX sparse solvers can handle unsorted matrices with duplicates
    # The duplicates will be handled correctly by summing during solve
    # (BC diagonal entries: 0 + 1 = 1, which is what we want)
    
    return J_bc



def apply_boundary_to_res(bc: DirichletBC, res_vec: np.ndarray, sol_vec: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Apply Dirichlet boundary conditions to residual vector using row elimination.
    
    This function modifies the residual vector to enforce Dirichlet boundary conditions
    by setting residual entries at BC DOFs to: res[bc_dof] = sol[bc_dof] - bc_val * scale
    This ensures that the Newton step will drive the solution towards the prescribed values.
    
    The modified residual enforces the constraint that after the Newton update:
    sol_new[bc_dof] = bc_val * scale
    
    Parameters
    ----------
    bc : DirichletBC
        Pre-computed boundary condition information containing:
        - bc_rows: DOF indices where BCs are applied  
        - bc_vals: Prescribed values at boundary DOFs
        - bc_mask: Boolean mask (not used in this function)
        - total_dofs: Total number of DOFs (for validation)
    res_vec : np.ndarray
        The residual vector (flattened) with shape (total_dofs,)
    sol_vec : np.ndarray  
        The current solution vector (flattened) with shape (total_dofs,)
    scale : float, optional
        Scaling factor for boundary condition values, by default 1.0
        Useful for ramping up BCs or unit conversion
        
    Returns
    -------
    np.ndarray
        The residual vector with boundary conditions applied, same shape as input
        
    Notes
    -----
    This function is JAX-JIT compatible and creates a copy of the input residual
    to avoid modifying the original array. The boundary condition enforcement
    follows the standard penalty method approach for constraint enforcement
    in Newton-Raphson solvers.
    """
    # Create a copy of the residual vector to modify
    res_modified = res_vec.copy()
    
    # For each boundary condition row:
    # res[bc_row] = sol[bc_row] - bc_val * scale
    # This is equivalent to the reference implementation
    
    # Apply BC: set residual at BC nodes to solution minus BC values
    bc_residual_values = sol_vec[bc.bc_rows] - bc.bc_vals * scale
    res_modified = res_modified.at[bc.bc_rows].set(bc_residual_values)
    
    return res_modified


# =============================================================================
# Boundary Condition Specification API
# =============================================================================

@dataclass
class DirichletBCSpec:
    """Specification for a single Dirichlet boundary condition.
    
    This dataclass provides a clear, type-safe way to specify boundary conditions.
    
    Parameters
    ----------
    location : Callable[[np.ndarray], bool]
        Function that takes a point (x, y, z) and returns True if the point
        is on the boundary where this BC should be applied
    component : Union[int, str]
        Which component to constrain:
        - For scalar problems: must be 0 or 'all'
        - For vector problems: 0='x', 1='y', 2='z', or 'all' for all components
    value : Union[float, Callable[[np.ndarray], float]]
        The prescribed value, either:
        - A constant float value
        - A function that takes a point and returns the value at that point
    
    Examples
    --------
    >>> # Fix left boundary in x-direction to zero
    >>> bc1 = DirichletBCSpec(
    ...     location=lambda pt: np.isclose(pt[0], 0.0),
    ...     component='x',  # or component=0
    ...     value=0.0
    ... )
    
    >>> # Apply varying displacement on right boundary
    >>> bc2 = DirichletBCSpec(
    ...     location=lambda pt: np.isclose(pt[0], 1.0),
    ...     component='y',
    ...     value=lambda pt: 0.1 * pt[2]  # varies with z-coordinate
    ... )
    
    >>> # Fix all components on a boundary
    >>> bc3 = DirichletBCSpec(
    ...     location=lambda pt: np.isclose(pt[1], 0.0),
    ...     component='all',
    ...     value=0.0
    ... )
    """
    location: Callable[[np.ndarray], bool]
    component: Union[int, str]
    value: Union[float, Callable[[np.ndarray], float]]
    
    def __post_init__(self) -> None:
        """Validate and normalize the component specification.
        
        This method is automatically called after __init__ to:
        1. Convert string component names ('x', 'y', 'z', 'all') to integers
        2. Validate integer component indices are non-negative  
        3. Convert constant values to functions for uniform interface
        
        Raises
        ------
        ValueError
            If component string is invalid or integer component is negative
        """
        # Convert string components to integers
        if isinstance(self.component, str):
            component_map = {'x': 0, 'y': 1, 'z': 2, 'all': 'all'}
            if self.component.lower() not in component_map:
                raise ValueError(f"Invalid component string: {self.component}. "
                               "Must be 'x', 'y', 'z', or 'all'")
            self.component = component_map[self.component.lower()]
        
        # Validate integer components
        elif isinstance(self.component, int):
            if self.component < 0:
                raise ValueError(f"Component index must be non-negative, got {self.component}")
        
        # Convert constant values to functions for uniform interface
        if isinstance(self.value, (int, float)):
            const_val = float(self.value)
            self.value = lambda pt: const_val


@dataclass
class DirichletBCConfig:
    """Configuration for all Dirichlet boundary conditions in a problem.
    
    This dataclass holds a collection of DirichletBCSpec objects and provides
    methods to convert to the format expected by DirichletBC.from_bc_info.
    
    Parameters
    ----------
    specs : List[DirichletBCSpec]
        List of boundary condition specifications
        
    Examples
    --------
    >>> # Create BC configuration for elasticity problem
    >>> bc_config = DirichletBCConfig([
    ...     DirichletBCSpec(
    ...         location=lambda pt: np.isclose(pt[0], 0.0),
    ...         component='all',
    ...         value=0.0
    ...     ),
    ...     DirichletBCSpec(
    ...         location=lambda pt: np.isclose(pt[0], 1.0), 
    ...         component='x',
    ...         value=0.1
    ...     )
    ... ])
    >>> 
    >>> # Create DirichletBC from config
    >>> bc = bc_config.create_bc(problem)
    """
    specs: List[DirichletBCSpec] = field(default_factory=list)
    
    def add(self, 
            location: Callable[[np.ndarray], bool], 
            component: Union[int, str], 
            value: Union[float, Callable[[np.ndarray], float]]) -> 'DirichletBCConfig':
        """Add a boundary condition specification to the configuration.
        
        This method allows for fluent-style chaining when building BC configurations.
        
        Parameters
        ----------
        location : Callable[[np.ndarray], bool]
            Function that takes a point coordinate array and returns True if 
            the point is on the boundary where this BC should be applied
        component : Union[int, str]
            Which component to constrain:
            - For scalar problems: 0 or 'all'
            - For vector problems: 0/'x', 1/'y', 2/'z', or 'all'
        value : Union[float, Callable[[np.ndarray], float]]
            The prescribed value, either a constant or a spatial function
            
        Returns
        -------
        self : DirichletBCConfig
            Returns self for method chaining, allowing: config.add(...).add(...)
            
        Examples
        --------
        >>> config = DirichletBCConfig()
        >>> config.add(lambda pt: np.isclose(pt[0], 0.0), 'all', 0.0)
        >>> config.add(lambda pt: np.isclose(pt[0], 1.0), 'x', 0.1)
        """
        self.specs.append(DirichletBCSpec(location, component, value))
        return self
    
    
    def create_bc(self, problem: 'Problem') -> 'DirichletBC':
        """Create a DirichletBC object from this configuration.
        
        This method directly processes the BC specification without intermediate format conversion.
        
        Parameters
        ----------
        problem : Problem
            The finite element problem instance containing mesh information
            and vector dimension specifications
            
        Returns
        -------
        bc : DirichletBC
            The compiled boundary condition object ready for use in solvers
            
        Notes
        -----
        This method automatically detects the vector dimension from the problem
        and handles both single-variable and multi-variable problems.
        """
        if not self.specs:
            return DirichletBC(
                bc_rows=np.array([], dtype=np.int32),
                bc_mask=np.zeros(problem.num_total_dofs_all_vars, dtype=bool),
                bc_vals=np.array([], dtype=np.float64),
                total_dofs=problem.num_total_dofs_all_vars
            )
        
        # Get vec from problem - handle both single and multi-variable problems
        if hasattr(problem, 'vec') and not isinstance(problem.vec, list):
            vec = problem.vec
        else:
            vec = problem.vec[0] if isinstance(problem.vec, list) else problem.vec
            
        bc_rows_list = []
        bc_vals_list = []
        
        for ind, fe in enumerate(problem.fes):
            for spec in self.specs:
                # Handle 'all' component expansion
                if spec.component == 'all':
                    components = list(range(vec))
                else:
                    if spec.component >= vec:
                        raise ValueError(f"Component {spec.component} is out of range for vec={vec} problem")
                    components = [spec.component]
                
                for component in components:
                    # Handle location functions with 1 or 2 arguments
                    num_args = spec.location.__code__.co_argcount
                    if num_args == 1:
                        location_fn = lambda point, ind_unused: spec.location(point)
                    elif num_args == 2:
                        location_fn = spec.location
                    else:
                        raise ValueError(f"Wrong number of arguments for location_fn: must be 1 or 2, got {num_args}")
                    
                    # Find nodes that satisfy the boundary condition
                    node_inds = np.argwhere(
                        jax.vmap(location_fn)(fe.mesh.points, np.arange(fe.num_total_nodes))
                    ).reshape(-1)
                    
                    if len(node_inds) > 0:
                        # Create vector component indices
                        vec_inds = np.ones_like(node_inds, dtype=np.int32) * component
                        
                        # Calculate DOF indices
                        bc_indices = node_inds * fe.vec + vec_inds + problem.offset[ind]
                        bc_rows_list.append(bc_indices)
                        
                        # Calculate BC values at the nodes
                        values = jax.vmap(spec.value)(fe.mesh.points[node_inds].reshape(-1, fe.dim)).reshape(-1)
                        bc_vals_list.append(values)
        
        if bc_rows_list:
            bc_rows = np.concatenate(bc_rows_list)
            bc_vals = np.concatenate(bc_vals_list)
            
            # Sort by row indices to maintain consistency
            sort_idx = np.argsort(bc_rows)
            bc_rows = bc_rows[sort_idx]
            bc_vals = bc_vals[sort_idx]
            
            # Handle duplicates by keeping first occurrence
            unique_rows, unique_idx = np.unique(bc_rows, return_index=True)
            bc_rows = unique_rows
            bc_vals = bc_vals[unique_idx]
        else:
            bc_rows = np.array([], dtype=np.int32)
            bc_vals = np.array([], dtype=np.float64)
        
        # Create a boolean mask for faster lookup
        total_dofs = problem.num_total_dofs_all_vars
        bc_mask = np.zeros(total_dofs, dtype=bool)
        if bc_rows.shape[0] > 0:
            bc_mask = bc_mask.at[bc_rows].set(True)
        
        return DirichletBC(
            bc_rows=bc_rows,
            bc_mask=bc_mask,
            bc_vals=bc_vals,
            total_dofs=total_dofs
        )


def dirichlet_bc_config(*specs: DirichletBCSpec) -> DirichletBCConfig:
    """Convenience function to create a DirichletBCConfig from multiple specs.
    
    This function provides a concise way to create BC configurations without
    explicitly instantiating the DirichletBCConfig class.
    
    Parameters
    ----------
    *specs : DirichletBCSpec
        Variable number of boundary condition specifications
        
    Returns
    -------
    config : DirichletBCConfig
        The BC configuration containing all provided specifications
        
    Examples
    --------
    >>> # Create BC config with multiple specifications
    >>> config = dirichlet_bc_config(
    ...     DirichletBCSpec(left_boundary, 'all', 0.0),
    ...     DirichletBCSpec(right_boundary, 'x', 0.1),
    ...     DirichletBCSpec(top_boundary, 'y', lambda pt: 0.01 * pt[0])
    ... )
    >>> bc = config.create_bc(problem)
    
    See Also
    --------
    DirichletBCConfig : The main configuration class
    DirichletBCSpec : Individual BC specification
    """
    return DirichletBCConfig(list(specs))