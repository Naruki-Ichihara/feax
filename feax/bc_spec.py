"""
Boundary condition specification dataclasses for cleaner BC definition.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Union
import jax.numpy as np


@dataclass
class DirichletBCSpec:
    """Specification for a single Dirichlet boundary condition.
    
    This dataclass provides a clear, type-safe way to specify boundary conditions
    instead of using nested lists.
    
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
    
    def __post_init__(self):
        """Validate and normalize the component specification."""
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
    
    def add(self, location: Callable, component: Union[int, str], 
            value: Union[float, Callable]) -> 'DirichletBCConfig':
        """Add a boundary condition specification.
        
        Parameters
        ----------
        location : Callable
            Boundary location function
        component : Union[int, str]
            Component to constrain
        value : Union[float, Callable]
            Prescribed value
            
        Returns
        -------
        self : DirichletBCConfig
            Returns self for method chaining
        """
        self.specs.append(DirichletBCSpec(location, component, value))
        return self
    
    def to_bc_info(self, vec: int) -> List:
        """Convert to the format expected by DirichletBC.from_bc_info.
        
        Parameters
        ----------
        vec : int
            Number of vector components in the problem
            
        Returns
        -------
        bc_info : List
            [location_fns, vecs, value_fns] format
        """
        if not self.specs:
            return None
            
        location_fns = []
        vecs = []
        value_fns = []
        
        for spec in self.specs:
            if spec.component == 'all':
                # Expand 'all' to individual components
                for comp in range(vec):
                    location_fns.append(spec.location)
                    vecs.append(comp)
                    value_fns.append(spec.value)
            else:
                # Single component
                if spec.component >= vec:
                    raise ValueError(f"Component {spec.component} is out of range "
                                   f"for vec={vec} problem")
                location_fns.append(spec.location)
                vecs.append(spec.component)
                value_fns.append(spec.value)
        
        return [location_fns, vecs, value_fns]
    
    def create_bc(self, problem) -> 'DirichletBC':
        """Create a DirichletBC object from this configuration.
        
        Parameters
        ----------
        problem : Problem
            The FE problem instance
            
        Returns
        -------
        bc : DirichletBC
            The boundary condition object
        """
        from feax.DCboundary import DirichletBC
        
        # Get vec from problem - handle both single and multi-variable problems
        if hasattr(problem, 'vec') and not isinstance(problem.vec, list):
            vec = problem.vec
        else:
            # For multi-variable problems, use the first variable's vec
            # This might need refinement for complex multi-physics problems
            vec = problem.vec[0] if isinstance(problem.vec, list) else problem.vec
            
        bc_info = self.to_bc_info(vec)
        return DirichletBC.from_bc_info(problem, bc_info)


def dirichlet_bc_config(*specs: DirichletBCSpec) -> DirichletBCConfig:
    """Convenience function to create a DirichletBCConfig from specs.
    
    Parameters
    ----------
    *specs : DirichletBCSpec
        Variable number of BC specifications
        
    Returns
    -------
    config : DirichletBCConfig
        The BC configuration
        
    Examples
    --------
    >>> config = dirichlet_bc_config(
    ...     DirichletBCSpec(left_boundary, 'all', 0.0),
    ...     DirichletBCSpec(right_boundary, 'x', 0.1)
    ... )
    """
    return DirichletBCConfig(list(specs))