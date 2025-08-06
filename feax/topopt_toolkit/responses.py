"""
Response function generators for topology optimization and analysis.
"""

import jax
import jax.numpy as np
import functools


def create_compliance_fn(problem, surface_load_params=None):
    """
    Creates a universal JIT-compiled compliance function for a given problem.
    Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
    where u is displacement and f is traction on each loaded boundary.
    
    Args:
        problem: FEAX Problem instance
        surface_load_params: Optional list/array of parameters for each surface map.
                           If None, defaults to scalar 1.0 for each surface.
                           If single value, applies to all surfaces.
                           If list/array, must match number of surfaces.
        
    Returns:
        compliance_fn: JIT-compiled function that takes solution and returns compliance value
    """
    # Get all surface information from the problem
    surface_maps = problem.get_surface_maps()
    num_surfaces = len(problem.boundary_inds_list)
    
    if num_surfaces == 0:
        # No surface loads, return zero compliance function
        @jax.jit
        def compliance_fn(sol):
            return 0.0
        return compliance_fn
    
    # Handle surface load parameters
    if surface_load_params is None:
        surface_load_params = [1.0] * num_surfaces
    elif not isinstance(surface_load_params, (list, tuple, np.ndarray)):
        # Single value for all surfaces
        surface_load_params = [surface_load_params] * num_surfaces
    elif len(surface_load_params) != num_surfaces:
        raise ValueError(f"surface_load_params length {len(surface_load_params)} must match number of surfaces {num_surfaces}")
    
    # Pre-compute data for all surfaces
    surface_data = []
    for i in range(num_surfaces):
        boundary_inds = problem.boundary_inds_list[i]
        _, nanson_scale = problem.fes[0].get_face_shape_grads(boundary_inds)
        subset_quad_points = problem.physical_surface_quad_points[i]
        surface_fn = surface_maps[i] if i < len(surface_maps) else lambda u, x, p: np.zeros_like(u)
        
        # Pre-compute indices for gathering
        cells = problem.fes[0].cells
        face_shape_vals = problem.fes[0].face_shape_vals
        cell_indices = cells[boundary_inds[:, 0]]
        face_indices = boundary_inds[:, 1]
        selected_face_shape_vals = face_shape_vals[face_indices]
        
        surface_data.append({
            'cell_indices': cell_indices,
            'selected_face_shape_vals': selected_face_shape_vals,
            'nanson_scale': nanson_scale,
            'subset_quad_points': subset_quad_points,
            'surface_fn': surface_fn,
            'load_param': surface_load_params[i]
        })
    
    # Get solution shape info for unflattening
    num_nodes_per_var = [fe.num_total_nodes for fe in problem.fes]
    vec_per_var = [fe.vec for fe in problem.fes]
    total_nodes_var0 = num_nodes_per_var[0]
    vec_var0 = vec_per_var[0]
    
    @jax.jit
    def compliance_fn(sol):
        """Compute compliance for the given solution over all surfaces."""
        # Manually unflatten just the first variable (displacement)
        displacement = sol[:total_nodes_var0 * vec_var0].reshape((total_nodes_var0, vec_var0))
        
        total_compliance = 0.0
        
        # Loop over all surfaces
        for surface in surface_data:
            # Extract displacements on the boundary faces
            u_face_nodes = displacement[surface['cell_indices']]  # (num_selected_faces, num_nodes, vec)
            
            # Apply shape functions
            u_face = np.sum(surface['selected_face_shape_vals'][:, :, :, None] * u_face_nodes[:, None, :, :], axis=2)
            
            # Apply surface map to get traction vector
            # Note: negative sign follows the convention that external work is positive
            traction = -jax.vmap(jax.vmap(lambda u, x: surface['surface_fn'](u, x, surface['load_param'])))(
                u_face, surface['subset_quad_points'])
            
            # Compute compliance contribution from this surface
            surface_compliance = np.sum(traction * u_face * surface['nanson_scale'][:, :, None])
            total_compliance += surface_compliance
        
        return total_compliance
    
    return compliance_fn


def create_volume_fn(problem):
    """
    Creates a JIT-compiled volume fraction calculation function for a given problem.
    Returns a function that computes the volume fraction of material in the domain.
    
    Args:
        problem: FEAX Problem instance
        
    Returns:
        volume_fn: JIT-compiled function that takes density array and returns volume fraction
    """
    # Pre-compute problem-specific data as static values
    fe = problem.fes[0]
    # Get quadrature weights and Jacobian determinants for volume integration
    JxW = fe.JxW  # Jacobian times quadrature weights
    num_cells = fe.num_cells
    num_quads = fe.num_quads
    domain_volume = float(np.sum(JxW))  # Pre-compute domain volume as Python float
    
    def volume_fn_inner(rho_array, num_quads):
        """Compute volume fraction for the given density distribution."""
        # rho_array shape: (num_cells, 1) or (num_cells*num_quads, 1)
        # Check if rho is already at quadrature points or needs expansion
        if rho_array.shape[0] == num_cells:
            # Expand to quadrature points: (num_cells, num_quads, 1)
            rho_quads = np.repeat(rho_array[:, None, :], num_quads, axis=1)
            # Integrate density over the domain: integral rho dOmega
            total_volume = np.sum(rho_quads[:, :, 0] * JxW)
        else:
            # Already at quadrature points (num_cells * num_quads, 1)
            # Reshape to (num_cells, num_quads) for integration
            rho_reshaped = rho_array.reshape((num_cells, num_quads))
            total_volume = np.sum(rho_reshaped * JxW)
        
        # Return volume fraction
        return total_volume / domain_volume
    
    # Return a wrapper that fixes num_quads
    def volume_fn(rho_array):
        return volume_fn_inner(rho_array, num_quads)
    
    return volume_fn