"""
Assembler functions that work with Problem and InternalVars.

This module provides the main assembler API for finite element analysis with
separated internal variables. It handles the assembly of residual vectors and
Jacobian matrices for both volume and surface integrals, supporting various
physics kernels (Laplace, mass, surface, and universal).
"""

import functools
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import jax
import jax.flatten_util
import jax.numpy as np
from jax.experimental import sparse

if TYPE_CHECKING:
    from feax.DCboundary import DirichletBC
    from feax.problem import Problem

from feax.internal_vars import InternalVars


class Operator:
    """Element-level operator for quadrature-point computations.

    Provides methods for interpolating solutions, computing gradients,
    interpolating internal variables, and integrating over quadrature points.
    Used internally by kernel functions to eliminate code duplication.

    Parameters
    ----------
    problem : Problem
        The finite element problem.
    fe_index : int
        Index of the finite element variable (default 0).
    """

    def __init__(self, problem: 'Problem', fe_index: int = 0):
        self._problem = problem
        self._fe = problem.fes[fe_index]
        self._shape_vals = self._fe.shape_vals    # (num_quads, num_nodes)
        self._num_quads = self._fe.num_quads
        self._num_nodes = self._fe.num_nodes
        self._vec = self._fe.vec

    def eval(self, cell_sol: np.ndarray, shape_vals: np.ndarray = None) -> np.ndarray:
        """Interpolate nodal solution to quadrature points.

        Parameters
        ----------
        cell_sol : np.ndarray
            Nodal solution values, shape (num_nodes, vec).
        shape_vals : np.ndarray, optional
            Shape function values. Uses volume shape functions if None.

        Returns
        -------
        np.ndarray
            Solution at quadrature points, shape (num_quads, vec).
        """
        sv = shape_vals if shape_vals is not None else self._shape_vals
        return np.sum(cell_sol[None, :, :] * sv[:, :, None], axis=1)

    def grad(self, cell_sol: np.ndarray, cell_shape_grads: np.ndarray) -> np.ndarray:
        """Compute solution gradient at quadrature points.

        Parameters
        ----------
        cell_sol : np.ndarray
            Nodal solution values, shape (num_nodes, vec).
        cell_shape_grads : np.ndarray
            Shape function gradients, shape (num_quads, num_nodes, dim).

        Returns
        -------
        np.ndarray
            Gradient at quadrature points, shape (num_quads, vec, dim).
        """
        u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
        return np.sum(u_grads, axis=1)

    def interpolate_var(self, var: np.ndarray) -> np.ndarray:
        """Interpolate a single internal variable to quadrature points.

        Handles node-based (shape function interpolation), cell-based
        (broadcast), and quad-based (pass-through) variables.

        Parameters
        ----------
        var : np.ndarray
            Internal variable. Shape determines interpolation strategy:
            - scalar: broadcast to all quad points
            - (num_nodes,): node-based, interpolate via shape functions
            - (1,) or cell-based: broadcast to all quad points
            - (num_quads,): quad-based, pass through

        Returns
        -------
        np.ndarray
            Values at quadrature points, shape (num_quads,).
        """
        if var.ndim == 0:
            return np.full(self._num_quads, var)
        elif var.ndim == 1:
            is_node_based = any(
                var.shape[0] == fe.num_nodes for fe in self._problem.fes)
            if is_node_based:
                return np.dot(self._shape_vals, var)
            elif var.shape[0] == 1:
                return np.full(self._num_quads, var[0])
            elif var.shape[0] == self._num_quads:
                return var
            else:
                return np.full(self._num_quads, var[0])
        else:
            return var

    def interpolate_vars(self, cell_internal_vars: Tuple[np.ndarray, ...]) -> List[np.ndarray]:
        """Interpolate all internal variables to quadrature points.

        Parameters
        ----------
        cell_internal_vars : tuple of np.ndarray
            Internal variables for a single element.

        Returns
        -------
        list of np.ndarray
            Interpolated values at quadrature points.
        """
        return [self.interpolate_var(v) for v in cell_internal_vars]

    def integrate_grad(self, quad_values: np.ndarray,
                       cell_v_grads_JxW: np.ndarray) -> np.ndarray:
        """Integrate tensor quad-point values against test function gradients.

        Computes: sum_q sigma(q) : grad_v(q) * JxW(q)

        Parameters
        ----------
        quad_values : np.ndarray
            Physics values at quad points, shape (num_quads, vec, dim).
        cell_v_grads_JxW : np.ndarray
            Pre-multiplied test function gradients × JxW,
            shape (num_quads, num_nodes, vec, dim).

        Returns
        -------
        np.ndarray
            Element contribution, shape (num_nodes, vec).
        """
        return np.sum(
            quad_values[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))

    def integrate_val(self, quad_values: np.ndarray,
                      cell_JxW: np.ndarray,
                      shape_vals: np.ndarray = None) -> np.ndarray:
        """Integrate scalar/vector quad-point values with shape functions.

        Computes: sum_q f(q) * N(q) * JxW(q)

        Parameters
        ----------
        quad_values : np.ndarray
            Physics values at quad points, shape (num_quads, vec).
        cell_JxW : np.ndarray
            Jacobian determinant × quadrature weights, shape (num_quads,).
        shape_vals : np.ndarray, optional
            Shape function values. Uses volume shape functions if None.

        Returns
        -------
        np.ndarray
            Element contribution, shape (num_nodes, vec).
        """
        sv = shape_vals if shape_vals is not None else self._shape_vals
        return np.sum(
            quad_values[:, None, :] * sv[:, :, None] * cell_JxW[:, None, None],
            axis=0)

    @staticmethod
    def gather_internal_vars(problem: 'Problem',
                             internal_vars: Tuple[np.ndarray, ...]
                             ) -> List[np.ndarray]:
        """Gather global internal variables to per-cell format.

        Transforms node-based variables from global (num_nodes,) arrays to
        per-cell (num_cells, num_nodes_per_elem) arrays using element
        connectivity. Cell-based and quad-based variables are passed through.

        Parameters
        ----------
        problem : Problem
            The finite element problem with connectivity information.
        internal_vars : tuple of np.ndarray
            Global internal variables.

        Returns
        -------
        list of np.ndarray
            Per-cell internal variables ready for vmapped kernels.
        """
        result = []
        for var in internal_vars:
            if var.ndim == 1:
                if var.shape[0] == problem.num_cells:
                    result.append(var)
                else:
                    fe_idx = 0
                    for i, fe in enumerate(problem.fes):
                        if var.shape[0] == fe.num_total_nodes:
                            fe_idx = i
                            break
                    result.append(var[problem.fes[fe_idx].cells])
            elif var.ndim == 2:
                if var.shape[0] == problem.num_cells:
                    result.append(var)
                else:
                    fe_idx = 0
                    for i, fe in enumerate(problem.fes):
                        if var.shape[0] == fe.num_total_nodes:
                            fe_idx = i
                            break
                    result.append(var[problem.fes[fe_idx].cells])
            else:
                result.append(var)
        return result


def interpolate_to_quad_points(var: np.ndarray, shape_vals: np.ndarray, num_cells: int, num_quads: int) -> np.ndarray:
    """Interpolate node-based or cell-based values to quadrature points.

    .. deprecated::
        Use :meth:`Operator.interpolate_var` instead.

    This function handles three cases:
    1. Node-based: shape (num_nodes,) -> interpolate using shape functions
    2. Cell-based: shape (num_cells,) -> broadcast to all quad points in cell
    3. Quad-based: shape (num_cells, num_quads) -> pass through (legacy)

    Parameters
    ----------
    var : np.ndarray
        Variable to interpolate. Can be:
        - (num_nodes,) for node-based
        - (num_cells,) for cell-based
        - (num_cells, num_quads) for quad-based (legacy)
    shape_vals : np.ndarray
        Shape function values at quadrature points, shape (num_quads, num_nodes)
    num_cells : int
        Number of cells/elements
    num_quads : int
        Number of quadrature points per cell

    Returns
    -------
    np.ndarray
        Values at quadrature points, shape (num_quads,)
    """
    if var.ndim == 1:
        if var.shape[0] == num_cells:
            # Cell-based: broadcast single value to all quad points
            return np.full(num_quads, var[0])  # For single cell, var[0] is the cell value
        else:
            # Node-based: interpolate using shape functions
            # var has shape (num_nodes,), need to extract cell nodes
            # This is handled by the caller passing cell_var_nodal
            return np.dot(shape_vals, var)  # (num_quads, num_nodes) @ (num_nodes,) -> (num_quads,)
    elif var.ndim == 2:
        # Quad-based (legacy): shape (num_cells, num_quads)
        # Return just this cell's quad values
        return var[0]  # Assumes var is already sliced for this cell
    else:
        raise ValueError(f"Variable has unexpected shape: {var.shape}")


def get_laplace_kernel(problem: 'Problem', tensor_map: Callable) -> Callable:
    """Create Laplace kernel function for gradient-based physics.

    The Laplace kernel handles gradient-based terms in the weak form, such as
    those arising in elasticity, heat conduction, and diffusion problems. It
    implements the integral term: ∫ σ(∇u) : ∇v dΩ where σ is the stress/flux
    tensor computed from the gradient.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and element information.
    tensor_map : Callable
        Function that maps gradient tensor to stress/flux tensor.
        Signature: (u_grad: ndarray, *internal_vars) -> ndarray
        where u_grad has shape (vec, dim) and returns (vec, dim).

    Returns
    -------
    Callable
        Laplace kernel function that computes the contribution to the weak form
        from gradient-based physics.
    """
    op = Operator(problem)

    def laplace_kernel(cell_sol_flat: np.ndarray,
                      cell_shape_grads: np.ndarray,
                      cell_v_grads_JxW: np.ndarray,
                      *cell_internal_vars: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_shape_grads = cell_shape_grads[:, :op._num_nodes, :]
        cell_v_grads_JxW = cell_v_grads_JxW[:, :op._num_nodes, :, :]

        u_grads = op.grad(cell_sol, cell_shape_grads)  # (num_quads, vec, dim)
        vars_quad = op.interpolate_vars(cell_internal_vars)
        u_physics = jax.vmap(tensor_map)(u_grads, *vars_quad)

        val = op.integrate_grad(u_physics, cell_v_grads_JxW)
        return jax.flatten_util.ravel_pytree(val)[0]

    return laplace_kernel


def get_mass_kernel(problem: 'Problem', mass_map: Callable) -> Callable:
    """Create mass kernel function for non-gradient terms.

    The mass kernel handles terms without derivatives in the weak form, such as
    mass matrices, reaction terms, or body forces. It implements the integral
    term: ∫ m(u, x) · v dΩ where m is a mass-like term.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and element information.
    mass_map : Callable
        Function that computes the mass term.
        Signature: (u: ndarray, x: ndarray, *internal_vars) -> ndarray
        where u has shape (vec,), x has shape (dim,), and returns (vec,).

    Returns
    -------
    Callable
        Mass kernel function that computes the contribution to the weak form
        from non-gradient physics.
    """
    op = Operator(problem)

    def mass_kernel(cell_sol_flat: np.ndarray,
                   x: np.ndarray,
                   cell_JxW: np.ndarray,
                   *cell_internal_vars: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_JxW = cell_JxW[0]

        u = op.eval(cell_sol)  # (num_quads, vec)
        vars_quad = op.interpolate_vars(cell_internal_vars)
        u_physics = jax.vmap(mass_map)(u, x, *vars_quad)  # (num_quads, vec)

        val = op.integrate_val(u_physics, cell_JxW)
        return jax.flatten_util.ravel_pytree(val)[0]

    return mass_kernel


def get_surface_kernel(problem: 'Problem', surface_map: Callable) -> Callable:
    """Create surface kernel function for boundary integrals.

    The surface kernel handles boundary integrals in the weak form, such as
    surface tractions, pressures, or fluxes. It implements the integral term:
    ∫ t(u, x) · v dΓ where t is the surface load/flux.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and element information.
    surface_map : Callable
        Function that computes the surface traction/flux.
        Signature: (u: ndarray, x: ndarray, *internal_vars) -> ndarray
        where u has shape (vec,), x has shape (dim,), and returns (vec,).

    Returns
    -------
    Callable
        Surface kernel function that computes the contribution to the weak form
        from boundary loads/fluxes.
    """
    op = Operator(problem)

    def surface_kernel(cell_sol_flat: np.ndarray,
                      x: np.ndarray,
                      face_shape_vals: np.ndarray,
                      face_shape_grads: np.ndarray,
                      face_nanson_scale: np.ndarray,
                      *cell_internal_vars_surface: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        face_shape_vals = face_shape_vals[:, :op._num_nodes]
        face_nanson_scale = face_nanson_scale[0]

        u = op.eval(cell_sol, shape_vals=face_shape_vals)  # (num_face_quads, vec)
        u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)

        val = op.integrate_val(u_physics, face_nanson_scale, shape_vals=face_shape_vals)
        return jax.flatten_util.ravel_pytree(val)[0]

    return surface_kernel


def _build_surface_kernel_from_weak_form(problem: 'Problem', surface_weak_form: Callable) -> Callable:
    """Build a surface kernel from a high-level surface weak form function.

    Converts a user-defined surface weak form (operating at a single surface
    quadrature point) into a full element-level surface kernel.

    Parameters
    ----------
    problem : Problem
        The finite element problem.
    surface_weak_form : Callable
        Surface weak form function with signature:
        ``(vals, x, *internal_vars) -> tractions``
        where vals[i] is (vec_i,), tractions[i] is (vec_i,).
        Automatically vmapped over surface quadrature points.

    Returns
    -------
    Callable
        Element-level surface kernel compatible with the assembler.
    """
    ops = [Operator(problem, i) for i in range(problem.num_vars)]
    # Compute static node offsets (Python ints, JIT-safe)
    nc = [0]
    for op in ops:
        nc.append(nc[-1] + op._num_nodes)

    def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
               face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)

        # Interpolate solutions to surface quad points per variable
        vals = [ops[i].eval(cell_sol_list[i],
                            shape_vals=face_shape_vals[:, nc[i]:nc[i+1]])
                for i in range(problem.num_vars)]

        # Call surface weak_form vmapped over surface quad points
        tractions = jax.vmap(surface_weak_form)(
            vals, physical_surface_quad_points, *cell_internal_vars_surface)

        # Integrate per variable: ∫ t_i · v_i dΓ
        residuals = []
        for i in range(problem.num_vars):
            fsv = face_shape_vals[:, nc[i]:nc[i+1]]
            res_i = ops[i].integrate_val(tractions[i], face_nanson_scale[i],
                                         shape_vals=fsv)
            residuals.append(res_i)

        return jax.flatten_util.ravel_pytree(residuals)[0]

    return kernel


def _build_kernel_from_weak_form(problem: 'Problem', weak_form: Callable) -> Callable:
    """Build a volume kernel from a high-level weak form function.

    Converts a user-defined weak form (operating at a single quadrature point)
    into a full element-level kernel that handles solution interpolation,
    gradient computation, internal variable interpolation, and integration.

    Parameters
    ----------
    problem : Problem
        The finite element problem.
    weak_form : Callable
        Weak form function with signature:
        ``(vals, grads, x, *internal_vars) -> (mass_terms, grad_terms)``
        where vals[i] is (vec_i,), grads[i] is (vec_i, dim),
        mass_terms[i] is (vec_i,), grad_terms[i] is (vec_i, dim).
        Automatically vmapped over quadrature points.

    Returns
    -------
    Callable
        Element-level kernel compatible with the assembler.
    """
    ops = [Operator(problem, i) for i in range(problem.num_vars)]
    # Compute static node offsets from Operator instances (Python ints, JIT-safe)
    nc = [0]
    for op in ops:
        nc.append(nc[-1] + op._num_nodes)

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
               cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)

        # Interpolate solutions and gradients to quad points
        vals = [ops[i].eval(cell_sol_list[i]) for i in range(problem.num_vars)]
        grads = [ops[i].grad(cell_sol_list[i], cell_shape_grads[:, nc[i]:nc[i+1], :])
                 for i in range(problem.num_vars)]

        # Interpolate internal vars to quad points
        vars_quad = ops[0].interpolate_vars(cell_internal_vars)

        # Call weak_form vmapped over quadrature points
        mass_terms, grad_terms = jax.vmap(weak_form)(
            vals, grads, physical_quad_points, *vars_quad)

        # Integrate per variable
        residuals = []
        for i in range(problem.num_vars):
            vg = cell_v_grads_JxW[:, nc[i]:nc[i+1], :, :]
            res_i = ops[i].integrate_val(mass_terms[i], cell_JxW[i]) + \
                    ops[i].integrate_grad(grad_terms[i], vg)
            residuals.append(res_i)

        return jax.flatten_util.ravel_pytree(residuals)[0]

    return kernel


def create_volume_kernel(problem: 'Problem') -> Callable:
    """Create unified volume kernel combining all volume physics.

    This function creates a kernel that combines contributions from all volume
    integral terms: Laplace (gradient-based), mass (non-gradient), universal
    (custom), and weak form kernels. The resulting kernel is used for both
    residual and Jacobian assembly.

    Parameters
    ----------
    problem : Problem
        The finite element problem that may define get_tensor_map(),
        get_mass_map(), get_weak_form(), and/or get_universal_kernel() methods.

    Returns
    -------
    Callable
        Combined volume kernel function that sums contributions from all
        applicable physics kernels.

    Notes
    -----
    The kernel checks for the existence of each physics method in the problem
    and only includes contributions from those that are defined. This allows
    for flexible problem definitions with any combination of physics terms.

    For multi-variable problems, the priority order is:
    1. ``get_universal_kernel()`` — full low-level control
    2. ``get_weak_form()`` — high-level quad-point physics definition
    """

    # Pre-build multi-variable kernels outside the traced function
    _multi_var_kernel = None
    if problem.num_vars > 1:
        if hasattr(problem, 'get_universal_kernel'):
            _multi_var_kernel = problem.get_universal_kernel()
        else:
            weak_form = problem.get_weak_form()
            if weak_form is not None:
                _multi_var_kernel = _build_kernel_from_weak_form(problem, weak_form)
            else:
                raise ValueError("Multi-variable problems require get_universal_kernel() or get_weak_form()")

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        # For multi-variable problems, use pre-built kernel
        if problem.num_vars > 1:
            return _multi_var_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW,
                cell_v_grads_JxW, *cell_internal_vars)

        # Single-variable: use laplace/mass/universal kernels
        mass_val = 0.
        if hasattr(problem, 'get_mass_map') and problem.get_mass_map() is not None:
            mass_kernel = get_mass_kernel(problem, problem.get_mass_map())
            mass_val = mass_kernel(cell_sol_flat, physical_quad_points, cell_JxW, *cell_internal_vars)

        laplace_val = 0.
        # Resolve tensor_map: explicit get_tensor_map takes precedence,
        # otherwise derive from get_energy_density via jax.grad
        tensor_map = None
        if hasattr(problem, 'get_tensor_map'):
            tensor_map = problem.get_tensor_map()
        if tensor_map is None and hasattr(problem, 'get_energy_density'):
            energy_density = problem.get_energy_density()
            if energy_density is not None:
                tensor_map = jax.grad(energy_density)
        if tensor_map is not None:
            laplace_kernel = get_laplace_kernel(problem, tensor_map)
            laplace_val = laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)

        universal_val = 0.
        if hasattr(problem, 'get_universal_kernel'):
            universal_kernel = problem.get_universal_kernel()
            universal_val = universal_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW,
                cell_v_grads_JxW, *cell_internal_vars)

        return laplace_val + mass_val + universal_val

    return kernel


def create_surface_kernel(problem: 'Problem', surface_index: int) -> Callable:
    """Create unified surface kernel for a specific boundary.

    This function creates a kernel that combines contributions from standard
    surface maps and universal surface kernels for a specific boundary surface
    identified by surface_index.

    Parameters
    ----------
    problem : Problem
        The finite element problem that may define get_surface_maps() and/or
        get_universal_kernels_surface() methods.
    surface_index : int
        Index identifying which boundary surface this kernel is for.
        Corresponds to the index in problem.location_fns.

    Returns
    -------
    Callable
        Combined surface kernel function for the specified boundary.

    Notes
    -----
    Multiple boundaries can have different physics. The surface_index
    parameter selects which surface map and universal kernel to use.

    For multi-variable problems, only universal_kernels_surface should be used,
    as get_surface_maps() only supports single-variable problems.
    """

    # Pre-build multi-variable surface kernel outside traced function
    _multi_var_surface_kernel = None
    if problem.num_vars > 1:
        if hasattr(problem, 'get_universal_kernels_surface') and len(problem.get_universal_kernels_surface()) > surface_index:
            _multi_var_surface_kernel = problem.get_universal_kernels_surface()[surface_index]
        elif hasattr(problem, 'get_surface_weak_forms') and callable(getattr(problem, 'get_surface_weak_forms', None)):
            swf_list = problem.get_surface_weak_forms()
            if len(swf_list) > surface_index:
                _multi_var_surface_kernel = _build_surface_kernel_from_weak_form(problem, swf_list[surface_index])

    def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
        # For multi-variable problems, use pre-built kernel
        if problem.num_vars > 1:
            if _multi_var_surface_kernel is not None:
                return _multi_var_surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                    face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)
            elif hasattr(problem, 'get_surface_maps') and len(problem.get_surface_maps()) > surface_index:
                raise ValueError(
                    f"Multi-variable problems (num_vars={problem.num_vars}) cannot use get_surface_maps(). "
                    f"Please implement get_surface_weak_forms() instead."
                )
            else:
                # No surface kernel defined - return zeros with proper shape
                return np.zeros(problem.num_total_dofs_all_vars)

        # Single-variable: use surface_maps and/or universal kernels
        surface_val = 0.
        if hasattr(problem, 'get_surface_maps') and len(problem.get_surface_maps()) > surface_index:
            surface_kernel = get_surface_kernel(problem, problem.get_surface_maps()[surface_index])
            surface_val = surface_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)

        universal_val = 0.
        if hasattr(problem, 'get_universal_kernels_surface') and len(problem.get_universal_kernels_surface()) > surface_index:
            universal_kernel = problem.get_universal_kernels_surface()[surface_index]
            universal_val = universal_kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals,
                face_shape_grads, face_nanson_scale, *cell_internal_vars_surface)

        return surface_val + universal_val

    return kernel


def split_and_compute_cell(problem: 'Problem',
                           cells_sol_flat: np.ndarray,
                           jac_flag: bool,
                           internal_vars_volume: Tuple[np.ndarray, ...]) -> Any:
    """Compute volume integrals for residual or Jacobian assembly.
    
    This function evaluates volume integrals over all elements, optionally
    computing the Jacobian via forward-mode automatic differentiation. It
    uses batching to manage memory for large meshes.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and quadrature data.
    cells_sol_flat : np.ndarray
        Flattened solution values at element nodes.
        Shape: (num_cells, num_nodes * vec).
    jac_flag : bool
        If True, compute both values and Jacobian. If False, compute only values.
    internal_vars_volume : tuple of np.ndarray
        Material properties at quadrature points for each variable.
        Each array has shape (num_cells, num_quads).
    
    Returns
    -------
    np.ndarray or tuple of np.ndarray
        If jac_flag is False: weak form values with shape (num_cells, num_dofs).
        If jac_flag is True: tuple of (values, jacobian) where jacobian has
        shape (num_cells, num_dofs, num_dofs).
    
    Notes
    -----
    The function splits computation into batches (default 20) to avoid memory
    issues with large meshes. This is particularly important for 3D problems.
    """

    def value_and_jacfwd(f: Callable, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pushfwd = functools.partial(jax.jvp, f, (x, ))
        basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
        return y, jac

    kernel = create_volume_kernel(problem)

    if jac_flag:
        def kernel_jac(cell_sol_flat, *args):
            kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
            return value_and_jacfwd(kernel_partial, cell_sol_flat)
        vmap_fn = jax.vmap(kernel_jac)
    else:
        vmap_fn = jax.vmap(kernel)

    # Prepare input collection
    # Adaptive batch size based on problem size to manage memory
    # Smaller batches for larger problems to avoid OOM
    if problem.num_cells > 5000:
        num_cuts = min(100, problem.num_cells)  # More cuts for very large problems
    elif problem.num_cells > 1000:
        num_cuts = min(50, problem.num_cells)   # Medium number of cuts
    else:
        num_cuts = min(20, problem.num_cells)   # Original behavior for small problems

    batch_size = problem.num_cells // num_cuts

    internal_vars_per_cell = Operator.gather_internal_vars(
        problem, internal_vars_volume)

    input_collection = [cells_sol_flat, problem.physical_quad_points, problem.shape_grads,
                       problem.JxW, problem.v_grads_JxW, *internal_vars_per_cell]

    if jac_flag:
        values = []
        jacs = []
        for i in range(num_cuts):
            if i < num_cuts - 1:
                input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
            else:
                input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:], input_collection)

            val, jac = vmap_fn(*input_col)
            values.append(val)
            jacs.append(jac)
        # Use concatenate instead of vstack to avoid memory overhead
        values = np.concatenate(values, axis=0) if len(values) > 1 else values[0]
        jacs = np.concatenate(jacs, axis=0) if len(jacs) > 1 else jacs[0]
        return values, jacs
    else:
        values = []
        for i in range(num_cuts):
            if i < num_cuts - 1:
                input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:(i + 1) * batch_size], input_collection)
            else:
                input_col = jax.tree_util.tree_map(lambda x: x[i * batch_size:], input_collection)

            val = vmap_fn(*input_col)
            values.append(val)
        # Use concatenate instead of vstack to avoid memory overhead
        values = np.concatenate(values, axis=0) if len(values) > 1 else values[0]
        return values


def compute_face(problem: 'Problem',
                cells_sol_flat: np.ndarray,
                jac_flag: bool,
                internal_vars_surfaces: List[Tuple[np.ndarray, ...]]) -> Any:
    """Compute surface integrals for residual or Jacobian assembly.
    
    This function evaluates surface integrals over all boundary faces,
    optionally computing the Jacobian via forward-mode automatic differentiation.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing boundary information.
    cells_sol_flat : np.ndarray
        Flattened solution values at element nodes.
        Shape: (num_cells, num_nodes * vec).
    jac_flag : bool
        If True, compute both values and Jacobian. If False, compute only values.
    internal_vars_surfaces : list of tuple of np.ndarray
        Surface variables for each boundary. Each entry corresponds to one
        boundary surface and contains arrays with shape
        (num_surface_faces, num_face_quads).
    
    Returns
    -------
    list of np.ndarray or list of tuple
        If jac_flag is False: list of weak form values for each boundary.
        If jac_flag is True: list of (values, jacobian) tuples for each boundary.
    
    Notes
    -----
    Each boundary surface can have different loading conditions or physics,
    handled through separate surface kernels and internal variables.
    """

    def value_and_jacfwd(f: Callable, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pushfwd = functools.partial(jax.jvp, f, (x, ))
        basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
        return y, jac

    if jac_flag:
        values = []
        jacs = []
        for i, boundary_inds in enumerate(problem.boundary_inds_list):
            kernel = create_surface_kernel(problem, i)
            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(kernel_partial, cell_sol_flat)
            vmap_fn = jax.vmap(kernel_jac)

            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]

            # Handle case where internal_vars_surfaces might be empty or insufficient
            surface_vars_for_boundary = internal_vars_surfaces[i] if i < len(internal_vars_surfaces) else ()

            input_collection = [selected_cell_sols_flat, problem.physical_surface_quad_points[i],
                              problem.selected_face_shape_vals[i], problem.selected_face_shape_grads[i],
                              problem.nanson_scale[i], *surface_vars_for_boundary]

            val, jac = vmap_fn(*input_collection)
            values.append(val)
            jacs.append(jac)
        return values, jacs
    else:
        values = []
        for i, boundary_inds in enumerate(problem.boundary_inds_list):
            kernel = create_surface_kernel(problem, i)
            vmap_fn = jax.vmap(kernel)

            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]

            # Handle case where internal_vars_surfaces might be empty or insufficient
            surface_vars_for_boundary = internal_vars_surfaces[i] if i < len(internal_vars_surfaces) else ()

            input_collection = [selected_cell_sols_flat, problem.physical_surface_quad_points[i],
                              problem.selected_face_shape_vals[i], problem.selected_face_shape_grads[i],
                              problem.nanson_scale[i], *surface_vars_for_boundary]
            val = vmap_fn(*input_collection)
            values.append(val)
        return values


def compute_residual_vars_helper(problem: 'Problem',
                                 weak_form_flat: np.ndarray,
                                 weak_form_face_flat: List[np.ndarray]) -> List[np.ndarray]:
    """Assemble residual from element and face contributions.
    
    This helper function assembles the global residual vector by accumulating
    contributions from volume and surface integrals at the appropriate nodes.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing connectivity information.
    weak_form_flat : np.ndarray
        Flattened weak form values from volume integrals.
        Shape: (num_cells, num_dofs_per_cell).
    weak_form_face_flat : list of np.ndarray
        Weak form values from surface integrals for each boundary.
        Each array has shape (num_boundary_faces, num_dofs_per_face).
    
    Returns
    -------
    list of np.ndarray
        Global residual for each solution variable.
        Each array has shape (num_total_nodes, vec).
    
    Notes
    -----
    Uses JAX's at[].add() for scatter-add operations to accumulate
    contributions from multiple elements sharing the same nodes.
    """
    res_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in problem.fes]
    weak_form_list = jax.vmap(lambda x: problem.unflatten_fn_dof(x))(weak_form_flat) # [(num_cells, num_nodes, vec), ...]
    res_list = [res_list[i].at[problem.cells_list[i].reshape(-1)].add(weak_form_list[i].reshape(-1, problem.fes[i].vec)) for i in range(len(res_list))]

    for j, boundary_inds in enumerate(problem.boundary_inds_list):
        weak_form_face_list = jax.vmap(lambda x: problem.unflatten_fn_dof(x))(weak_form_face_flat[j]) # [(num_selected_faces, num_nodes, vec), ...]
        res_list = [res_list[i].at[problem.cells_list_face_list[j][i].reshape(-1)].add(weak_form_face_list[i].reshape(-1, problem.fes[i].vec)) for i in range(len(res_list))]

    return res_list


def _get_J(problem: 'Problem',
           sol_list: List[np.ndarray],
           internal_vars: InternalVars) -> sparse.BCOO:
    """Internal function to compute Jacobian matrix.

    WARNING: This is a private function. Do not call directly from user code.
    When used with JIT-compiled solvers and cuDSS backend, calling this
    function from outside the JIT boundary can cause GPU memory conflicts.

    Use get_jacobian_info() for safe access to Jacobian statistics.

    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and physics definitions.
    sol_list : list of np.ndarray
        Solution arrays for each variable.
    internal_vars : InternalVars
        Container with material properties and loading parameters.

    Returns
    -------
    sparse.BCOO
        Sparse Jacobian matrix in JAX BCOO format.
    """
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)

    # Compute Jacobian values from volume integrals
    _, cells_jac_flat = split_and_compute_cell(problem, cells_sol_flat, True, internal_vars.volume_vars)

    # Collect all Jacobian arrays to avoid repeated concatenation
    V_arrays = [cells_jac_flat.reshape(-1)]

    # Add Jacobian values from surface integrals
    _, cells_jac_face_flat = compute_face(problem, cells_sol_flat, True, internal_vars.surface_vars)
    for cells_jac_f_flat in cells_jac_face_flat:
        V_arrays.append(cells_jac_f_flat.reshape(-1))

    # Single concatenation to avoid memory overhead
    V = np.concatenate(V_arrays) if len(V_arrays) > 1 else np.array(V_arrays[0])

    # Use pre-computed filtered indices (JIT-compatible)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    if hasattr(problem, 'filter_indices') and problem.filter_indices is not None:
        # Use pre-computed integer indices for filtering (JIT-compatible)
        filtered_data = V[problem.filter_indices]
        filtered_indices = np.stack([problem.I_filtered, problem.J_filtered], axis=1)
        J = sparse.BCOO((filtered_data, filtered_indices), shape=shape)
    else:
        # FULL - use pre-filtered indices (which are same as I, J for FULL)
        indices = np.stack([problem.I_filtered, problem.J_filtered], axis=1)
        J = sparse.BCOO((V, indices), shape=shape)

    return J


def get_jacobian_info(problem: 'Problem',
                      sol_list: List[np.ndarray],
                      internal_vars: InternalVars) -> dict:
    """Get Jacobian matrix information without full matrix construction.

    This function provides safe access to Jacobian statistics that works
    correctly with JIT-compiled solvers and cuDSS backend. Unlike the
    internal _get_J function, this does not cause GPU memory conflicts.

    Parameters
    ----------
    problem : Problem
        The finite element problem definition.
    sol_list : list of np.ndarray
        Solution arrays for each variable.
    internal_vars : InternalVars
        Internal variables container.

    Returns
    -------
    dict
        Dictionary containing:
        - 'nnz': Number of non-zero entries (int)
        - 'shape': Matrix shape (tuple)
        - 'matrix_view': Matrix storage format (MatrixView enum)

    Examples
    --------
    >>> info = get_jacobian_info(problem, sol_list, internal_vars)
    >>> print(f"Jacobian NNZ: {info['nnz']:,}")
    >>> print(f"Matrix view: {info['matrix_view'].name}")

    Notes
    -----
    This function is safe to call from user code and does not interfere
    with JIT-compiled solvers using cuDSS backend.
    """
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)

    # Calculate NNZ based on matrix view
    if hasattr(problem, 'filter_indices') and problem.filter_indices is not None:
        nnz = len(problem.filter_indices)
    elif hasattr(problem, 'I_filtered'):
        nnz = len(problem.I_filtered)
    else:
        nnz = len(problem.I)

    return {
        'nnz': nnz,
        'shape': shape,
        'matrix_view': problem.matrix_view if hasattr(problem, 'matrix_view') else None
    }


def get_res(problem: 'Problem',
            sol_list: List[np.ndarray],
            internal_vars: InternalVars) -> List[np.ndarray]:
    """Compute residual vector with separated internal variables.
    
    Assembles the global residual vector by evaluating the weak form at the
    current solution state. Includes contributions from both volume and
    surface integrals.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem containing mesh and physics definitions.
    sol_list : list of np.ndarray
        Solution arrays for each variable.
        Each array has shape (num_total_nodes, vec).
    internal_vars : InternalVars
        Container with material properties and loading parameters.
    
    Returns
    -------
    list of np.ndarray
        Residual arrays for each solution variable.
        Each array has shape (num_total_nodes, vec).
    
    Examples
    --------
    >>> residual = get_res(problem, [solution], internal_vars)
    >>> res_norm = np.linalg.norm(jax.flatten_util.ravel_pytree(residual)[0])
    >>> print(f"Residual norm: {res_norm}")
    
    Notes
    -----
    The residual represents the imbalance in the weak form equations.
    For converged solutions, the residual should be near zero.
    """
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)

    # Compute weak form values from volume integrals
    weak_form_flat = split_and_compute_cell(problem, cells_sol_flat, False, internal_vars.volume_vars)

    # Add weak form values from surface integrals
    weak_form_face_flat = compute_face(problem, cells_sol_flat, False, internal_vars.surface_vars)

    return compute_residual_vars_helper(problem, weak_form_flat, weak_form_face_flat)


def create_J_bc_function(problem: 'Problem', bc: 'DirichletBC') -> Callable[[np.ndarray, InternalVars], sparse.BCOO]:
    """Create Jacobian function with Dirichlet BC applied.
    
    Returns a function that computes the Jacobian matrix with Dirichlet
    boundary conditions enforced. The BC application modifies the matrix
    to enforce constraints.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem definition.
    bc : DirichletBC
        Dirichlet boundary condition specifications.
    
    Returns
    -------
    Callable
        Function with signature (sol_flat, internal_vars) -> sparse.BCOO
        that returns the BC-modified Jacobian matrix.
    
    Notes
    -----
    The returned function is suitable for use in Newton solvers and
    can be differentiated for sensitivity analysis.
    """
    from feax.DCboundary import apply_boundary_to_J

    def J_bc_func(sol_flat, internal_vars: InternalVars):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        J = _get_J(problem, sol_list, internal_vars)
        return apply_boundary_to_J(bc, J)

    return J_bc_func


def create_res_bc_function(problem: 'Problem', bc: 'DirichletBC') -> Callable[[np.ndarray, InternalVars], np.ndarray]:
    """Create residual function with Dirichlet BC applied.
    
    Returns a function that computes the residual vector with Dirichlet
    boundary conditions enforced. The BC application zeros out residuals
    at constrained DOFs.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem definition.
    bc : DirichletBC
        Dirichlet boundary condition specifications.
    
    Returns
    -------
    Callable
        Function with signature (sol_flat, internal_vars) -> np.ndarray
        that returns the BC-modified residual vector.
    
    Notes
    -----
    The returned function is used in Newton solvers to find solutions
    that satisfy both the weak form equations and boundary conditions.
    """
    from feax.DCboundary import apply_boundary_to_res

    def res_bc_func(sol_flat, internal_vars: InternalVars):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_list, internal_vars)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)

    return res_bc_func
