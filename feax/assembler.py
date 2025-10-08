"""
Assembler functions that work with Problem and InternalVars.

This module provides the main assembler API for finite element analysis with
separated internal variables. It handles the assembly of residual vectors and
Jacobian matrices for both volume and surface integrals, supporting various
physics kernels (Laplace, mass, surface, and universal).

The assembler separates the finite element structure (Problem) from material
properties and loading parameters (InternalVars), enabling efficient
optimization and sensitivity analysis.
"""

import jax
import jax.numpy as np
from jax.experimental import sparse
import jax.flatten_util
import functools
from typing import List, Tuple, Any, TYPE_CHECKING
from feax.internal_vars import InternalVars
from feax.types import (
    TensorMap, MassMap, SurfaceMap, LaplaceKernel, MassKernel, SurfaceKernel,
    VolumeKernel, JacobianFunction, ResidualFunction
)

if TYPE_CHECKING:
    from feax.problem import Problem
    from feax.DCboundary import DirichletBC


def interpolate_to_quad_points(var: np.ndarray, shape_vals: np.ndarray, num_cells: int, num_quads: int) -> np.ndarray:
    """Interpolate node-based or cell-based values to quadrature points.

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


def get_laplace_kernel(problem: 'Problem', tensor_map: TensorMap) -> LaplaceKernel:
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
    
    Notes
    -----
    The kernel operates on a single element and is vectorized over quadrature
    points for efficiency. The tensor_map function is applied via vmap to all
    quadrature points simultaneously.
    """
    
    def laplace_kernel(cell_sol_flat: np.ndarray,
                      cell_shape_grads: np.ndarray,
                      cell_v_grads_JxW: np.ndarray,
                      *cell_internal_vars: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_shape_grads = cell_shape_grads[:, :problem.fes[0].num_nodes, :]
        cell_sol = cell_sol_list[0]
        cell_v_grads_JxW = cell_v_grads_JxW[:, :problem.fes[0].num_nodes, :, :]
        vec = problem.fes[0].vec
        num_quads = problem.fes[0].num_quads
        shape_vals = problem.fes[0].shape_vals

        # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
        u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
        u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
        u_grads_reshape = u_grads.reshape(-1, vec, problem.dim)  # (num_quads, vec, dim)

        # Interpolate internal variables to quadrature points
        cell_internal_vars_quad = []
        for var in cell_internal_vars:
            if var.ndim == 0:
                # Scalar (cell-based): broadcast to all quad points
                var_quad = np.full(num_quads, var)
            elif var.ndim == 1:
                if var.shape[0] == problem.fes[0].num_nodes:
                    # Node-based: interpolate using shape functions
                    var_quad = np.dot(shape_vals, var)  # (num_quads, num_nodes) @ (num_nodes,) -> (num_quads,)
                elif var.shape[0] == 1:
                    # Cell-based (single element): broadcast to all quad points
                    var_quad = np.full(num_quads, var[0])
                elif var.shape[0] == num_quads:
                    # Quad-based (legacy): already has quad point values
                    var_quad = var
                else:
                    # Unknown, assume cell-based
                    var_quad = np.full(num_quads, var[0])
            else:
                # Unknown format, pass through
                var_quad = var
            cell_internal_vars_quad.append(var_quad)

        # Apply tensor map with internal variables at quad points
        u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars_quad).reshape(u_grads.shape)

        # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
        val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
        val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
        return val

    return laplace_kernel


def get_mass_kernel(problem: 'Problem', mass_map: MassMap) -> MassKernel:
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
    
    Notes
    -----
    This kernel is useful for time-dependent problems (inertia terms),
    reaction-diffusion equations, or adding body forces/sources.
    """
    
    def mass_kernel(cell_sol_flat: np.ndarray,
                   x: np.ndarray,
                   cell_JxW: np.ndarray,
                   *cell_internal_vars: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_JxW = cell_JxW[0]
        vec = problem.fes[0].vec
        num_quads = problem.fes[0].num_quads
        shape_vals = problem.fes[0].shape_vals

        # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
        u = np.sum(cell_sol[None, :, :] * shape_vals[:, :, None], axis=1)

        # Interpolate internal variables to quadrature points
        cell_internal_vars_quad = []
        for var in cell_internal_vars:
            if var.ndim == 0:
                # Scalar (cell-based): broadcast to all quad points
                var_quad = np.full(num_quads, var)
            elif var.ndim == 1:
                if var.shape[0] == problem.fes[0].num_nodes:
                    # Node-based: interpolate using shape functions
                    var_quad = np.dot(shape_vals, var)  # (num_quads, num_nodes) @ (num_nodes,) -> (num_quads,)
                elif var.shape[0] == 1:
                    # Cell-based (single element): broadcast to all quad points
                    var_quad = np.full(num_quads, var[0])
                elif var.shape[0] == num_quads:
                    # Quad-based (legacy): already has quad point values
                    var_quad = var
                else:
                    # Unknown, assume cell-based
                    var_quad = np.full(num_quads, var[0])
            else:
                # Unknown format, pass through
                var_quad = var
            cell_internal_vars_quad.append(var_quad)

        u_physics = jax.vmap(mass_map)(u, x, *cell_internal_vars_quad)  # (num_quads, vec)

        # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
        val = np.sum(u_physics[:, None, :] * shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
        val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
        return val

    return mass_kernel


def get_surface_kernel(problem: 'Problem', surface_map: SurfaceMap) -> SurfaceKernel:
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
    
    Notes
    -----
    The Nanson scale factor accounts for the transformation from reference to
    physical surface elements, including the Jacobian and surface normal.
    """
    
    def surface_kernel(cell_sol_flat: np.ndarray, 
                      x: np.ndarray, 
                      face_shape_vals: np.ndarray, 
                      face_shape_grads: np.ndarray, 
                      face_nanson_scale: np.ndarray, 
                      *cell_internal_vars_surface: np.ndarray) -> np.ndarray:
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        face_shape_vals = face_shape_vals[:, :problem.fes[0].num_nodes]
        face_nanson_scale = face_nanson_scale[0]

        # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
        u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
        u_physics = jax.vmap(surface_map)(u, x, *cell_internal_vars_surface)  # (num_face_quads, vec)
        
        # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
        val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] * face_nanson_scale[:, None, None], axis=0)

        return jax.flatten_util.ravel_pytree(val)[0]

    return surface_kernel


def create_volume_kernel(problem: 'Problem') -> VolumeKernel:
    """Create unified volume kernel combining all volume physics.
    
    This function creates a kernel that combines contributions from all volume
    integral terms: Laplace (gradient-based), mass (non-gradient), and universal
    (custom) kernels. The resulting kernel is used for both residual and
    Jacobian assembly.
    
    Parameters
    ----------
    problem : Problem
        The finite element problem that may define get_tensor_map(),
        get_mass_map(), and/or get_universal_kernel() methods.
    
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
    """
    
    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        mass_val = 0.
        if hasattr(problem, 'get_mass_map') and problem.get_mass_map() is not None:
            mass_kernel = get_mass_kernel(problem, problem.get_mass_map())
            mass_val = mass_kernel(cell_sol_flat, physical_quad_points, cell_JxW, *cell_internal_vars)

        laplace_val = 0.
        if hasattr(problem, 'get_tensor_map'):
            laplace_kernel = get_laplace_kernel(problem, problem.get_tensor_map())
            laplace_val = laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars)

        universal_val = 0.
        if hasattr(problem, 'get_universal_kernel'):
            universal_kernel = problem.get_universal_kernel()
            universal_val = universal_kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, 
                cell_v_grads_JxW, *cell_internal_vars)

        return laplace_val + mass_val + universal_val

    return kernel


def create_surface_kernel(problem: 'Problem', surface_index: int) -> VolumeKernel:
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
    """
    
    def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
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
    
    def value_and_jacfwd(f: VolumeKernel, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    # Transform internal vars to per-cell format
    # For node-based: extract nodes for each cell
    # For cell-based: keep as-is
    # For quad-based: keep as-is (legacy)
    internal_vars_per_cell = []
    for var in internal_vars_volume:
        if var.ndim == 1:
            if var.shape[0] == problem.num_cells:
                # Cell-based: already per-cell, just needs to be indexable
                internal_vars_per_cell.append(var)
            else:
                # Node-based: extract nodes for each cell
                # var shape: (num_nodes,), cells shape: (num_cells, num_nodes_per_elem)
                var_per_cell = var[problem.fes[0].cells]  # (num_cells, num_nodes_per_elem)
                internal_vars_per_cell.append(var_per_cell)
        elif var.ndim == 2:
            # Quad-based (legacy): already (num_cells, num_quads)
            internal_vars_per_cell.append(var)
        else:
            # Unknown format, pass through
            internal_vars_per_cell.append(var)

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
    
    def value_and_jacfwd(f: VolumeKernel, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def get_J(problem: 'Problem', 
          sol_list: List[np.ndarray], 
          internal_vars: InternalVars) -> sparse.BCOO:
    """Compute Jacobian matrix with separated internal variables.
    
    Assembles the global Jacobian matrix by computing derivatives of the weak
    form with respect to the solution variables. Uses forward-mode automatic
    differentiation for element-level Jacobians.
    
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
    sparse.BCOO
        Sparse Jacobian matrix in JAX BCOO format.
        Shape: (num_total_dofs, num_total_dofs).
    
    Examples
    --------
    >>> J = get_J(problem, [solution], internal_vars)
    >>> print(f"Jacobian shape: {J.shape}, nnz: {J.nnz}")
    
    Notes
    -----
    The Jacobian is assembled in sparse format for memory efficiency,
    particularly important for large 3D problems.
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

    # Build BCOO sparse matrix
    indices = np.stack([problem.I, problem.J], axis=1)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    J = sparse.BCOO((V, indices), shape=shape)
    
    return J


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


def create_J_bc_function(problem: 'Problem', bc: 'DirichletBC') -> JacobianFunction:
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
        J = get_J(problem, sol_list, internal_vars)
        return apply_boundary_to_J(bc, J)
    
    return J_bc_func


def create_res_bc_function(problem: 'Problem', bc: 'DirichletBC') -> ResidualFunction:
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