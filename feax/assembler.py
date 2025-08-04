"""
Assembler functions that work with Problem and InternalVars.
This is the main assembler API with separated internal variables.
"""

import jax
import jax.numpy as np
from jax.experimental import sparse
import jax.flatten_util
import functools
from feax import logger
from feax.internal_vars import InternalVars


def get_laplace_kernel(problem, tensor_map):
    """Create laplace kernel function for problem (without internal_vars in problem)."""
    
    def laplace_kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars):
        cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
        cell_shape_grads = cell_shape_grads[:, :problem.fes[0].num_nodes, :]
        cell_sol = cell_sol_list[0]
        cell_v_grads_JxW = cell_v_grads_JxW[:, :problem.fes[0].num_nodes, :, :]
        vec = problem.fes[0].vec

        # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
        u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
        u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
        u_grads_reshape = u_grads.reshape(-1, vec, problem.dim)  # (num_quads, vec, dim)
        
        # Apply tensor map with internal variables
        u_physics = jax.vmap(tensor_map)(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)
        
        # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
        val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
        val = jax.flatten_util.ravel_pytree(val)[0] # (num_nodes*vec + ...,)
        return val

    return laplace_kernel


def get_mass_kernel(problem, mass_map):
    """Create mass kernel function for problem."""
    
    def mass_kernel(cell_sol_flat, x, cell_JxW, *cell_internal_vars):
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
    """Create surface kernel function for problem."""
    
    def surface_kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
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


def create_volume_kernel(problem):
    """Create volume kernel function that accepts internal_vars."""
    
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


def create_surface_kernel(problem, surface_index):
    """Create surface kernel function that accepts internal_vars."""
    
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


def split_and_compute_cell(problem, cells_sol_flat, jac_flag, internal_vars_volume):
    """Volume integral computation with problem and internal_vars."""
    
    def value_and_jacfwd(f, x):
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
    num_cuts = 20
    if num_cuts > problem.num_cells:
        num_cuts = problem.num_cells
    batch_size = problem.num_cells // num_cuts
    input_collection = [cells_sol_flat, problem.physical_quad_points, problem.shape_grads, 
                       problem.JxW, problem.v_grads_JxW, *internal_vars_volume]

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
    """Surface integral computation with problem and internal_vars."""
    
    def value_and_jacfwd(f, x):
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


def compute_residual_vars_helper(problem, weak_form_flat, weak_form_face_flat):
    """Compute residual variables helper function."""
    res_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in problem.fes]
    weak_form_list = jax.vmap(lambda x: problem.unflatten_fn_dof(x))(weak_form_flat) # [(num_cells, num_nodes, vec), ...]
    res_list = [res_list[i].at[problem.cells_list[i].reshape(-1)].add(weak_form_list[i].reshape(-1, problem.fes[i].vec)) for i in range(len(res_list))]
    
    for j, boundary_inds in enumerate(problem.boundary_inds_list):
        weak_form_face_list = jax.vmap(lambda x: problem.unflatten_fn_dof(x))(weak_form_face_flat[j]) # [(num_selected_faces, num_nodes, vec), ...]
        res_list = [res_list[i].at[problem.cells_list_face_list[j][i].reshape(-1)].add(weak_form_face_list[i].reshape(-1, problem.fes[i].vec)) for i in range(len(res_list))]
    
    return res_list


def get_J(problem, sol_list, internal_vars: InternalVars):
    """Compute Jacobian matrix with separated internal variables."""
    logger.debug(f"Computing Jacobian matrix...")
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
    
    # Compute Jacobian values from volume integrals
    _, cells_jac_flat = split_and_compute_cell(problem, cells_sol_flat, True, internal_vars.volume_vars)
    V = np.array(cells_jac_flat.reshape(-1))

    # Add Jacobian values from surface integrals
    _, cells_jac_face_flat = compute_face(problem, cells_sol_flat, True, internal_vars.surface_vars)
    for cells_jac_f_flat in cells_jac_face_flat:
        V = np.hstack((V, np.array(cells_jac_f_flat.reshape(-1))))

    # Build BCOO sparse matrix
    indices = np.stack([problem.I, problem.J], axis=1)
    shape = (problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)
    J = sparse.BCOO((V, indices), shape=shape)
    
    return J


def get_res(problem, sol_list, internal_vars: InternalVars):
    """Compute residual with separated internal variables."""
    cells_sol_list = [sol[cells] for cells, sol in zip(problem.cells_list, sol_list)]
    cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list)
    
    # Compute weak form values from volume integrals
    weak_form_flat = split_and_compute_cell(problem, cells_sol_flat, False, internal_vars.volume_vars)
    
    # Add weak form values from surface integrals
    weak_form_face_flat = compute_face(problem, cells_sol_flat, False, internal_vars.surface_vars)
    
    return compute_residual_vars_helper(problem, weak_form_flat, weak_form_face_flat)


def create_J_bc_function(problem, bc):
    """Create Jacobian function with BC applied."""
    from feax.DCboundary import apply_boundary_to_J
    
    def J_bc_func(sol_flat, internal_vars: InternalVars):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        J = get_J(problem, sol_list, internal_vars)
        return apply_boundary_to_J(bc, J)
    
    return J_bc_func


def create_res_bc_function(problem, bc):
    """Create residual function with BC applied."""
    from feax.DCboundary import apply_boundary_to_res
    
    def res_bc_func(sol_flat, internal_vars: InternalVars):
        sol_list = problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(problem, sol_list, internal_vars)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)
    
    return res_bc_func