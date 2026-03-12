"""
Problem class with modular design separating FE structure from material parameters.

This module provides the core Problem class that defines finite element problem
structure independent of material parameters, enabling efficient optimization
and parameter studies through JAX transformations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp

from feax.fe import FiniteElement
from feax.mesh import Mesh


class MatrixView(Enum):
    """Matrix storage format for sparse assembly.

    Controls which entries are stored in the assembled matrix:
    - FULL: Store all entries (default, backward compatible)
    - UPPER: Store only upper triangular entries (j >= i)
    - LOWER: Store only lower triangular entries (j <= i)

    For symmetric problems, UPPER or LOWER reduces memory by ~50%
    and enables optimized solvers like Cholesky factorization.
    """
    FULL = 0
    UPPER = 1
    LOWER = 2

@dataclass
@jax.tree_util.register_pytree_node_class
class Problem:
    """Finite element problem definition.
    
    This class defines the finite element problem structure.
    
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
    matrix_view : Union[MatrixView, str], optional
        Matrix storage format: 'FULL' (default), 'UPPER', or 'LOWER'.
        Use UPPER for symmetric problems to reduce memory by ~50%.
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
    matrix_view: Union[MatrixView, str] = MatrixView.FULL
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
        # Convert matrix_view string to enum if needed
        if isinstance(self.matrix_view, str):
            try:
                object.__setattr__(self, 'matrix_view', MatrixView[self.matrix_view.upper()])
            except KeyError:
                raise ValueError(f"Invalid matrix_view: {self.matrix_view}. Must be 'FULL', 'UPPER', or 'LOWER'")

        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.gauss_order = [self.gauss_order]

        self.num_vars = len(self.mesh)

        # Build FiniteElement objects with deduplication.
        # When two variables share the same mesh object, vec, ele_type, and
        # gauss_order the expensive JAX operations (shape_grads, linalg.inv)
        # are identical.  Reuse the first FE instead of recomputing.
        _fe_cache: dict = {}
        fes = []
        for i in range(self.num_vars):
            go_i = (self.gauss_order[i]
                    if type(self.gauss_order) == type([])
                    else self.gauss_order)
            cache_key = (id(self.mesh[i]), self.vec[i], self.ele_type[i], go_i)
            if cache_key not in _fe_cache:
                _fe_cache[cache_key] = FiniteElement(
                    mesh=self.mesh[i],
                    vec=self.vec[i],
                    dim=self.dim,
                    ele_type=self.ele_type[i],
                    gauss_order=go_i,
                )
            fes.append(_fe_cache[cache_key])
        self.fes = fes

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0]
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def _build_inds(cells_list_local):
            """Build per-element global DOF index array using pure numpy.

            Avoids JAX JIT compilation overhead for index arithmetic, which
            has no gradient and does not need to be traced.

            Returns ndarray of shape (num_elems, total_dofs_per_elem).
            """
            parts = []
            for i, cells_i in enumerate(cells_list_local):
                c = onp.asarray(cells_i)             # (E, nn)
                vec_i = self.fes[i].vec
                # c * vec_i expands node index to DOF base; arange adds component offset
                dof = (vec_i * c[:, :, None]         # (E, nn, 1)
                       + onp.arange(vec_i)[None, None, :]  # (1, 1, vec)
                       + self.offset[i])              # scalar
                parts.append(dof.reshape(len(c), -1))  # (E, nn*vec)
            return np.array(onp.concatenate(parts, axis=1))  # (E, total_dofs)

        # (num_cells, total_dofs_per_elem)
        inds = _build_inds(self.cells_list)
        ndof = inds.shape[1]
        self.I = np.array(onp.repeat(onp.asarray(inds)[:, :, None], ndof, axis=2).reshape(-1))
        self.J = np.array(onp.repeat(onp.asarray(inds)[:, None, :], ndof, axis=1).reshape(-1))

        # Note: I and J are kept as FULL for assembly
        # Filtering to UPPER/LOWER happens in get_J() after computing values

        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list]
            inds_face = _build_inds(cells_list_face)
            nf = inds_face.shape[1]
            I_face = np.array(onp.repeat(onp.asarray(inds_face)[:, :, None], nf, axis=2).reshape(-1))
            J_face = np.array(onp.repeat(onp.asarray(inds_face)[:, None, :], nf, axis=1).reshape(-1))

            self.I = np.hstack((self.I, I_face))
            self.J = np.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)

        # Pre-compute filtering indices for UPPER/LOWER views (for JIT compatibility)
        # Store integer indices instead of boolean mask to enable JIT compilation
        if self.matrix_view == MatrixView.UPPER:
            mask = self.J >= self.I
            self.filter_indices = np.where(mask)[0]
            self.I_filtered = self.I[mask]
            self.J_filtered = self.J[mask]
        elif self.matrix_view == MatrixView.LOWER:
            mask = self.J <= self.I
            self.filter_indices = np.where(mask)[0]
            self.I_filtered = self.I[mask]
            self.J_filtered = self.J[mask]
        else:  # FULL
            self.filter_indices = None
            self.I_filtered = self.I
            self.J_filtered = self.J

        # Concatenate cell connectivity for all variables along the node axis.
        # Pure numpy avoids a JAX vmap JIT compilation for a trivial concat.
        self.cells_flat = np.array(
            onp.concatenate([onp.asarray(c) for c in self.cells_list], axis=1)
        )  # (num_cells, total_nodes_per_elem)

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

        # Validate multi-variable problem setup
        self._validate_multi_variable_setup()

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

    def _validate_multi_variable_setup(self) -> None:
        """Validate that multi-variable problems are set up correctly.

        Checks that multi-variable problems (num_vars > 1) use universal kernels
        instead of the single-variable laplace/mass/surface kernels.

        Raises
        ------
        ValueError
            If multi-variable problem incorrectly uses single-variable kernels
        """
        if self.num_vars <= 1:
            return  # Single-variable - all kernels allowed

        # Check for problematic kernel definitions (call methods to check return values)
        has_tensor_map = self.get_tensor_map() is not None
        has_energy_density = self.get_energy_density() is not None
        has_mass_map = self.get_mass_map() is not None
        has_surface_maps = len(self.get_surface_maps()) > 0
        has_universal = hasattr(self, 'get_universal_kernel') and callable(self.get_universal_kernel)
        has_weak_form = self.get_weak_form() is not None

        # For multi-var, must have universal kernel or weak_form
        if not has_universal and not has_weak_form:
            raise ValueError(
                f"Multi-variable problem (num_vars={self.num_vars}) requires get_universal_kernel() "
                f"or get_weak_form() implementation. "
                f"The standard laplace/mass/surface kernels only support single-variable problems."
            )

        # Warn if single-variable kernels are defined (they will be ignored)
        single_var_methods = []
        if has_tensor_map:
            single_var_methods.append("get_tensor_map()")
        if has_energy_density:
            single_var_methods.append("get_energy_density()")
        if has_mass_map:
            single_var_methods.append("get_mass_map()")
        if has_surface_maps:
            single_var_methods.append("get_surface_maps()")

        if single_var_methods:
            import warnings as warn_module
            warn_module.warn(
                f"Multi-variable problem (num_vars={self.num_vars}) defines {', '.join(single_var_methods)} "
                f"which will be IGNORED. Only get_universal_kernel() is used for multi-variable problems. "
                f"Remove these methods or move their logic into get_universal_kernel().",
                UserWarning,
                stacklevel=3
            )

    def get_tensor_map(self) -> Optional[Callable]:
        """Get tensor map function for gradient-based physics.

        Override this method to define the constitutive relationship between
        gradients and stress/flux tensors directly.

        Alternatively, override :meth:`get_energy_density` to define a scalar
        energy density — the stress tensor will be derived automatically via
        ``jax.grad``.

        Returns
        -------
        Optional[Callable]
            Function that maps gradients to stress/flux tensors.
            Signature: ``(u_grad, *internal_vars) -> stress_tensor``
            Returns ``None`` if not defined (default).

        Examples
        --------
        For linear elasticity:

        ```python
        def get_tensor_map(self):
            def stress(u_grad):
                eps = 0.5 * (u_grad + u_grad.T)
                return lmbda * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
            return stress
        ```
        """
        return None

    def get_energy_density(self) -> Optional[Callable]:
        """Get energy density function for gradient-based physics.

        Override this method to define the strain energy density as a scalar
        function of the displacement gradient. The stress tensor is derived
        automatically via ``jax.grad``:

        ```python
        σ = ∂ψ/∂(∇u)
        ```

        This is an alternative to :meth:`get_tensor_map`. If both are defined,
        ``get_tensor_map`` takes precedence.

        Returns
        -------
        Optional[Callable]
            Scalar energy density function.
            Signature: ``(u_grad, *internal_vars) -> scalar``
            Returns ``None`` if not defined (default).

        Examples
        --------
        For Neo-Hookean hyperelasticity:

        ```python
        def get_energy_density(self):
            def psi(F):
                C = F.T @ F
                I1 = jnp.trace(C)
                J = jnp.linalg.det(F)
                return mu/2 * (I1 - 3) - mu * jnp.log(J) + lmbda/2 * jnp.log(J)**2
            return psi
        ```
        """
        return None

    def get_surface_maps(self) -> List[Callable]:
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

    def get_mass_map(self) -> Optional[Callable]:
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

    def get_weak_form(self) -> Optional[Callable]:
        """Get weak form function for multi-variable problems.

        Override this method to define coupled physics at a single quadrature
        point. The framework automatically handles solution interpolation,
        gradient computation, and integration. This is the recommended
        interface for multi-variable problems.

        The function is automatically ``jax.vmap``-ed over quadrature points.

        Returns
        -------
        Optional[Callable]
            Weak form function with signature:

            ```python
            (vals, grads, x, *internal_vars) -> (mass_terms, grad_terms)
            ```

            where:

            - ``vals[i]``: solution of variable *i*, shape ``(vec_i,)``
            - ``grads[i]``: gradient of variable *i*, shape ``(vec_i, dim)``
            - ``x``: physical coordinate, shape ``(dim,)``
            - ``mass_terms[i]``: residual integrated as ``∫ · v dΩ``, shape ``(vec_i,)``
            - ``grad_terms[i]``: residual integrated as ``∫ · ∇v dΩ``, shape ``(vec_i, dim)``

            Returns ``None`` if not defined (default).

        Examples
        --------
        Cahn-Hilliard with mixed (c, μ) formulation:

        ```python
        def get_weak_form(self):
            def weak_form(vals, grads, x, c_old):
                c, mu = vals[0], vals[1]
                grad_c, grad_mu = grads[0], grads[1]
                return ([(c - c_old) / dt, mu - (c**3 - c)],
                        [M * grad_mu, -kappa * grad_c])
            return weak_form
        ```
        """
        return None

    def get_surface_weak_forms(self) -> List[Callable]:
        """Get surface weak form functions for multi-variable boundary loads.

        Override this method to define surface tractions/fluxes at a single
        surface quadrature point. The framework handles solution interpolation
        and integration automatically. This is the recommended interface for
        multi-variable problems with boundary conditions.

        The function is automatically ``jax.vmap``-ed over surface quadrature
        points.

        Returns
        -------
        List[Callable]
            List of surface weak form functions, one per boundary (matching
            ``location_fns``). Each function has signature:

            ```python
            (vals, x, *internal_vars) -> tractions
            ```

            where:

            - ``vals[i]``: solution of variable *i*, shape ``(vec_i,)``
            - ``x``: physical coordinate, shape ``(dim,)``
            - ``tractions[i]``: surface load integrated as ``∫ t_i · v_i dΓ``,
              shape ``(vec_i,)``

        Examples
        --------
        Pressure BC on a Stokes problem (u: vec=2, p: vec=1):

        ```python
        def get_surface_weak_forms(self):
            def inlet_pressure(vals, x):
                return [np.array([p_in, 0.]), np.zeros(1)]
            return [inlet_pressure]
        ```
        """
        return []

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        """Flatten Problem object for JAX pytree registration.

        Since Problem objects contain only static structure information
        (no JAX arrays), all data goes into the static part.

        Parameters
        ----------
        self : Problem
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
            'mesh': self.mesh,
            'vec': self.vec,
            'dim': self.dim,
            'ele_type': self.ele_type,
            'gauss_order': self.gauss_order,
            'location_fns': self.location_fns,
            'matrix_view': self.matrix_view,
            'additional_info': self.additional_info,
        }
        return dynamic, static

    @classmethod
    def tree_unflatten(cls, static: dict, _dynamic: Tuple) -> 'Problem':
        """Reconstruct Problem object from flattened parts.

        Parameters
        ----------
        static : dict
            Static data containing Problem constructor arguments
        _dynamic : Tuple
            Dynamic data (empty for Problem objects, unused)

        Returns
        -------
        Problem
            Reconstructed Problem instance
        """
        # Create instance with original constructor parameters
        instance = cls(
            mesh=static['mesh'],
            vec=static['vec'],
            dim=static['dim'],
            ele_type=static['ele_type'],
            gauss_order=static['gauss_order'],
            location_fns=static['location_fns'],
            matrix_view=static.get('matrix_view', MatrixView.FULL),  # Default for backward compatibility
            additional_info=static['additional_info'],
        )

        return instance
