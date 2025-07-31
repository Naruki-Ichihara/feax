"""
Assembly functions for finite element problems.
"""

import jax
import jax.numpy as np
from jax.experimental import sparse
import jax.flatten_util
import functools
from feax import logger


def get_laplace_kernel(problem, tensor_map):
    """Create laplace kernel function for the given tensor map."""
    
    def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
        # cell_sol_flat: (num_nodes*vec + ...,)
        # cell_sol_list: [(num_nodes, vec), ...]
        # cell_shape_grads: (num_quads, num_nodes + ..., dim)
        # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_shape_grads = cell_shape_grads[:, :problem.fes[0].num_nodes, :]
        cell_sol = cell_sol_list[0]
        cell_v_grads_JxW = cell_v_grads_JxW[:, :problem.fes[0].num_nodes, :, :]
        vec = problem.fes[0].vec

        # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
        u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
        u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
        u_grads_reshape = u_grads.reshape(-1, vec, problem.dim)  # (num_quads, vec, dim)
        # (num_quads, vec, dim)
        u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)
        # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
        val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
        val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
        return val

    return laplace_kernel


def get_mass_kernel(problem, mass_map):
    """Create mass kernel function for the given mass map."""
    
    def mass_kernel(cell_sol_flat, x, cell_JxW, *cell_internal_vars):
        # cell_sol_flat: (num_nodes*vec + ...,)
        # cell_sol_list: [(num_nodes, vec), ...]
        # x: (num_quads, dim)
        # cell_JxW: (num_vars, num_quads)

        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_JxW = cell_JxW[0]
        vec = problem.fes[0].vec
        # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
        u = np.sum(cell_sol[None, :, :] * problem.fes[0].shape_vals[:, :, None], axis=1)
        u_physics = jax.vmap(mass_map)(u, x, *cell_internal_vars)  # (num_quads, vec)
        # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
        val = np.sum(u_physics[:, None, :] * problem.fes[0].shape_vals[:, :, None] * cell_JxW[:, None, None], axis=0)
        val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
        return val

    return mass_kernel


def get_surface_kernel(problem, surface_map):
    """Create surface kernel function for the given surface map."""
    
    def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
        # face_shape_vals: (num_face_quads, num_nodes + ...)
        # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
        # x: (num_face_quads, dim)
        # face_nanson_scale: (num_vars, num_face_quads)

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


def split_and_compute_cell(problem, cells_sol_flat, jac_flag, internal_vars):
    """Volume integral in weak form
    
    Parameters
    ----------
    problem : Problem
        Problem instance containing all state data
    cells_sol_flat : np.ndarray
        Flattened cell solutions
    jac_flag : bool
        Whether to compute Jacobian
    internal_vars : tuple
        Internal variables
        
    Returns
    -------
    values : np.ndarray
        Computed values
    jacs : np.ndarray, optional
        Jacobian values if jac_flag is True
    """
    vmap_fn = problem.kernel_jac if jac_flag else problem.kernel
    num_cuts = 20
    if num_cuts > problem.num_cells:
        num_cuts = problem.num_cells
    batch_size = problem.num_cells // num_cuts
    input_collection = [cells_sol_flat, problem.physical_quad_points, problem.shape_grads, 
                       problem.JxW, problem.v_grads_JxW, *internal_vars]

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
        values = np.vstack(values)
        jacs = np.vstack(jacs)

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
        values = np.vstack(values)
        return values


def compute_face(problem, cells_sol_flat, jac_flag, internal_vars_surfaces):
    """Surface integral in weak form
    
    Parameters
    ----------
    problem : Problem
        Problem instance containing all state data
    cells_sol_flat : np.ndarray
        Flattened cell solutions
    jac_flag : bool
        Whether to compute Jacobian
    internal_vars_surfaces : list
        Internal variables for surfaces
        
    Returns
    -------
    values : list
        Computed values for each surface
    jacs : list, optional
        Jacobian values if jac_flag is True
    """
    if jac_flag:
        values = []
        jacs = []
        for i, boundary_inds in enumerate(problem.boundary_inds_list):
            vmap_fn = problem.kernel_jac_face[i]
            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
            input_collection = [selected_cell_sols_flat, problem.physical_surface_quad_points[i], 
                              problem.selected_face_shape_vals[i], problem.selected_face_shape_grads[i], 
                              problem.nanson_scale[i], *internal_vars_surfaces[i]]

            val, jac = vmap_fn(*input_collection)
            values.append(val)
            jacs.append(jac)
        return values, jacs
    else:
        values = []
        for i, boundary_inds in enumerate(problem.boundary_inds_list):
            vmap_fn = problem.kernel_face[i]
            selected_cell_sols_flat = cells_sol_flat[boundary_inds[:, 0]]  # (num_selected_faces, num_nodes*vec + ...))
            input_collection = [selected_cell_sols_flat, problem.physical_surface_quad_points[i], 
                              problem.selected_face_shape_vals[i], problem.selected_face_shape_grads[i], 
                              problem.nanson_scale[i], *internal_vars_surfaces[i]]
            val = vmap_fn(*input_collection)
            values.append(val)
        return values


def compute_residual_vars_helper(problem, weak_form_flat, weak_form_face_flat):
    """Helper function to compute residual variables
    
    Parameters
    ----------
    problem : Problem
        Problem instance containing all state data
    weak_form_flat : np.ndarray
        Flattened weak form values
    weak_form_face_flat : list
        Weak form values for faces
        
    Returns
    -------
    res_list : list
        Residual list
    """
    res_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in problem.fes]
    weak_form_list = jax.vmap(lambda x: problem.unflatten_fn_dof(x))(weak_form_flat) # [(num_cells, num_nodes, vec), ...]
    res_list = [res_list[i].at[problem.cells_list[i].reshape(-1)].add(weak_form_list[i].reshape(-1, 
        problem.fes[i].vec)) for i in range(problem.num_vars)]

    for ind, cells_list_face in enumerate(problem.cells_list_face_list):
        weak_form_face_list = jax.vmap(lambda x: problem.unflatten_fn_dof(x))(weak_form_face_flat[ind]) # [(num_selected_faces, num_nodes, vec), ...]
        res_list = [res_list[i].at[cells_list_face[i].reshape(-1)].add(weak_form_face_list[i].reshape(-1, 
            problem.fes[i].vec)) for i in range(problem.num_vars)]   

    return res_list


def compute_residual_vars(problem, sol_list, internal_vars, internal_vars_surfaces):
    """Compute residual variables
    
    Parameters
    ----------
    problem : Problem
        Problem instance containing all state data
    sol_list : list
        Solution list
    internal_vars : tuple
        Internal variables
    internal_vars_surfaces : list
        Internal variables for surfaces
        
    Returns
    -------
    res_list : list
        Residual list
    """
    logger.debug(f"Computing cell residual...")
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list) # (num_cells, num_nodes*vec + ...)
    weak_form_flat = split_and_compute_cell(problem, cells_sol_flat, False, internal_vars)  # (num_cells, num_nodes*vec + ...)
    weak_form_face_flat = compute_face(problem, cells_sol_flat, False, internal_vars_surfaces)  # [(num_selected_faces, num_nodes*vec + ...), ...]
    return compute_residual_vars_helper(problem, weak_form_flat, weak_form_face_flat)


def get_res(problem, sol_list):
    """Compute residual list (pure function version).

    Parameters
    ----------
    problem : Problem
        Problem instance containing all state data
    sol_list : list
        A list of JaxArray with the shape being (num_total_nodes, vec).

    Returns
    -------
    res_list : list
        Same shape as sol_list.
    """
    return compute_residual_vars(problem, sol_list, problem.internal_vars, problem.internal_vars_surfaces)


def get_J(problem, sol_list):
    """Compute Jacobian matrix and return as JAX BCOO sparse matrix (pure function version).

    Parameters
    ----------
    problem : Problem
        Problem instance containing all state data
    sol_list : list
        A list of JaxArray with the shape being (num_total_nodes, vec).

    Returns
    -------
    J : jax.experimental.sparse.BCOO
        Sparse Jacobian matrix.
    """
    logger.debug(f"Computing Jacobian matrix...")
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
    
    # Compute Jacobian values from volume integrals
    _, cells_jac_flat = split_and_compute_cell(problem, cells_sol_flat, True, problem.internal_vars)
    V = np.array(cells_jac_flat.reshape(-1))

    # Add Jacobian values from surface integrals
    _, cells_jac_face_flat = compute_face(problem, cells_sol_flat, True, problem.internal_vars_surfaces)
    for cells_jac_f_flat in cells_jac_face_flat:
        V = np.hstack((V, np.array(cells_jac_f_flat.reshape(-1))))

    # Build BCOO sparse matrix
    indices = np.stack([problem.I, problem.J], axis=1)  # (nnz, 2)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    J = sparse.BCOO((V, indices), shape=shape)
    
    return J