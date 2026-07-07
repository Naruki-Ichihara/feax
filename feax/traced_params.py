"""
Internal variables management for FEAX finite element framework.

This module provides the TracedParams dataclass for handling dynamic parameters
separately from problem structure.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as np

from feax.problem import Problem


def _is_solution(v) -> bool:
    from feax.solution import Solution
    return isinstance(v, Solution)


def _coerce_volume_var(v):
    """Unwrap a :class:`feax.Solution` used as a coefficient.

    Single-field Solutions become node-based volume vars — ``(num_nodes,)``
    when the solved field is scalar (vec=1), ``(num_nodes, vec)`` otherwise
    (the assembler interpolates both via shape functions). Multi-field
    Solutions are ambiguous: pick one with ``sol.field(i)`` /
    ``sol.node_var(var_index=i)``.
    """
    if not _is_solution(v):
        return v
    if v.num_fields != 1:
        raise ValueError(
            f"Solution with {v.num_fields} fields is ambiguous as a "
            "coefficient; pass sol.field(i) or sol.node_var(var_index=i).")
    u = v.field(0)                                    # (..., num_nodes, vec)
    return u[..., 0] if v.layout[0][1] == 1 else u


@dataclass(frozen=True)
@jax.tree_util.register_pytree_node_class
class TracedParams:
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
        - (num_nodes, k) for node-based vector variables
        - (num_cells,) for cell-based/element-wise variables
        - (num_cells, num_quads) for quad-point based variables (legacy)
        The assembler automatically interpolates to quadrature points.
        Common examples: material properties (E, nu), density, source terms.
        A :class:`feax.Solution` may be passed directly — it is converted to
        its node-based field (staggered chaining: the previous solve's result
        becomes this solve's coefficient).
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
    >>> E = TracedParams.create_node_var(problem, 210e9)  # Young's modulus at nodes
    >>> nu = TracedParams.create_node_var(problem, 0.3)   # Poisson's ratio at nodes
    >>> traced_params = TracedParams(volume_vars=(E, nu))
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
        # A feax.Solution passed directly as a coefficient becomes its
        # node-based field: (num_nodes,) for a scalar solve, (num_nodes, vec)
        # for a vector solve (both layouts are native node vars to the
        # assembler). This is the staggered-multiphysics shortcut:
        #     tp_mech = TracedParams(volume_vars=(E_cells, sol_thermal))
        if any(_is_solution(v) for v in self.volume_vars):
            object.__setattr__(self, 'volume_vars',
                               tuple(_coerce_volume_var(v)
                                     for v in self.volume_vars))

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
        >>> E = TracedParams.create_node_var(problem, 210e9)  # Young's modulus
        >>> rho = TracedParams.create_node_var(problem, 7800)  # Density
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
        >>> rho = TracedParams.create_cell_var(problem, 0.5)  # Topology density per element
        >>> E = TracedParams.create_cell_var(problem, 70e3)  # Young's modulus per element
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
        >>> E = TracedParams.create_uniform_volume_var(problem, 210e9)  # Young's modulus
        >>> rho = TracedParams.create_uniform_volume_var(problem, 7800)  # Density
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
        >>> pressure = TracedParams.create_uniform_surface_var(problem, -1000)  # Pressure load
        >>> heat_flux = TracedParams.create_uniform_surface_var(problem, 50.0)  # Heat flux
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
        >>> E_varying = TracedParams.create_node_var_from_fn(problem, E_gradient)

        >>> def density_field(x): return 7800 * (1 + 0.1 * np.sin(x[0]))  # Sinusoidal variation
        >>> rho_varying = TracedParams.create_node_var_from_fn(problem, density_field)
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
        >>> rho_varying = TracedParams.create_cell_var_from_fn(problem, density_field)
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
        >>> E_varying = TracedParams.create_spatially_varying_volume_var(problem, E_gradient)

        >>> def density_field(x): return 7800 * (1 + 0.1 * np.sin(x[0]))  # Sinusoidal variation
        >>> rho_varying = TracedParams.create_spatially_varying_volume_var(problem, density_field)
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
        >>> pressure = TracedParams.create_spatially_varying_surface_var(problem, pressure_gradient)
        
        >>> def traction_field(x): return 500 * np.sin(np.pi * x[0])  # Sinusoidal traction
        >>> traction = TracedParams.create_spatially_varying_surface_var(problem, traction_field)
        """
        surface_quad_points = problem.physical_surface_quad_points[surface_index]
        return jax.vmap(jax.vmap(var_fn))(surface_quad_points)

    @staticmethod
    def node_var_from_solution(problem: 'Problem', sol: np.ndarray,
                               component: Optional[int] = None,
                               var_index: int = 0) -> np.ndarray:
        """Turn a solved field into a node-based volume var for the NEXT solve.

        This is the chaining bridge for staggered multiphysics: the flat
        solution of one solver becomes ``volume_vars`` input of another Problem
        on the same mesh (e.g. thermal ``T`` driving mechanical thermal strain).

        Parameters
        ----------
        problem : Problem
            The problem the solution belongs to (defines the DOF layout).
        sol : ndarray
            Flat solution vector as returned by a feax solver.
        component : int, optional
            Which component to extract when ``vec > 1`` (e.g. 0/1/2 for a
            displacement). Required for vector fields; ignored for scalars.
        var_index : int, optional
            Which variable of a multi-variable problem. Default 0.

        Returns
        -------
        ndarray, (num_nodes,)
            Node-based variable — directly usable in ``TracedParams``
            (``volume_vars``) of a problem sharing the mesh. Differentiable.

        Examples
        --------
        >>> T = solver_thermal(tp_thermal)                       # flat solution
        >>> T_nodes = TracedParams.node_var_from_solution(thermal_problem, T)
        >>> tp_mech = tp_mech.replace_volume_var(0, T_nodes)     # chain
        >>> u = solver_mech(tp_mech)

        See also :meth:`feax.Solution.node_var` — the same bridge as a method
        when the solver was built with ``return_solution=True``.
        """
        sol = getattr(sol, "dofs", sol)               # accept a feax.Solution
        sol_list = problem.unflatten_fn_sol_list(sol)
        u = sol_list[var_index]                       # (num_nodes, vec)
        if u.shape[1] == 1:
            return u[:, 0]
        if component is None:
            raise ValueError(
                f"solution has vec={u.shape[1]} components; pass component=")
        return u[:, component]

    def replace_volume_var(self, index: int, new_var: np.ndarray) -> 'TracedParams':
        """Create a new TracedParams with one volume variable replaced.
        
        Parameters
        ----------
        index : int
            Index of volume variable to replace
        new_var : np.ndarray
            New variable array
            
        Returns
        -------
        TracedParams
            New instance with updated variable
        """
        volume_vars_list = list(self.volume_vars)
        volume_vars_list[index] = new_var
        return TracedParams(tuple(volume_vars_list), self.surface_vars)

    def replace_surface_var(self, surface_index: int, var_index: int, new_var: np.ndarray) -> 'TracedParams':
        """Create a new TracedParams with one surface variable replaced.
        
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
        TracedParams
            New instance with updated variable
        """
        surface_vars_list = list(self.surface_vars)
        surface_tuple_list = list(surface_vars_list[surface_index])
        surface_tuple_list[var_index] = new_var
        surface_vars_list[surface_index] = tuple(surface_tuple_list)
        return TracedParams(self.volume_vars, surface_vars_list)

    def tree_flatten(self) -> Tuple[List[np.ndarray], Tuple[int, List[int]]]:
        """Flatten TracedParams into leaves and auxiliary data for JAX pytree.

        This method extracts all JAX arrays (leaves) and structural information
        needed to reconstruct the TracedParams object.

        Returns
        -------
        Tuple[List[np.ndarray], Tuple[int, List[int]]]
            (leaves, aux_data) where leaves contains all arrays and aux_data
            contains structure information for reconstruction
        """
        # Flatten all arrays into a single list
        leaves = list(self.volume_vars)
        for surface_tuple in self.surface_vars:
            leaves.extend(surface_tuple)

        # Auxiliary data to reconstruct structure
        aux_data = (len(self.volume_vars), [len(st) for st in self.surface_vars])
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[int, List[int]], leaves: List[np.ndarray]) -> 'TracedParams':
        """Reconstruct TracedParams from flattened leaves and auxiliary data.

        This method rebuilds the TracedParams structure from the flat list of
        arrays and structural information.

        Parameters
        ----------
        aux_data : Tuple[int, List[int]]
            Structural information: (num_volume_vars, surface_var_counts)
        leaves : List[np.ndarray]
            Flat list of all arrays

        Returns
        -------
        TracedParams
            Reconstructed TracedParams object
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

        return cls(volume_vars, surface_vars)
