"""
Response function generators for topology optimization and analysis.
"""


import jax
import jax.numpy as np


def create_compliance_fn(problem, surface_load_params=None):
    """
    Creates a universal pure-JAX compliance function for a given problem.
    Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
    where u is displacement and f is traction on each loaded boundary.
    
    Args:
        problem: FEAX Problem instance
        surface_load_params: Optional list/array of parameters for each surface map.
                           If None, defaults to scalar 1.0 for each surface.
                           If single value, applies to all surfaces.
                           If list/array, must match number of surfaces. An
                           entry is passed to the surface map as its parameter;
                           a tuple entry is unpacked into several parameters
                           (``surface_fn(u, x, *entry)``). An entry (or a tuple
                           element) may carry a leading batch axis, (B, ...):
                           the compliance of case ``i`` then uses its ``[i]``
                           (one load per case, as in a ``feax.vmap`` batch).

    Returns:
        compliance_fn: pure-JAX function (sol) -> compliance; jit the
            composed objective at the call site. ``sol`` may be one flat
            solution (n,) or a batch (B, n) (a batched ``feax.Solution``'s
            ``dofs``), which gives (B,) compliances.
    """
    # Get all surface information from the problem
    surface_maps = problem.get_surface_maps()
    num_surfaces = len(problem.boundary_inds_list)

    if num_surfaces == 0:
        # No surface loads, return zero compliance function
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
    total_nodes_var0 = problem.fes[0].num_total_nodes
    vec_var0 = problem.fes[0].vec

    # Load parameters given per case (a leading batch axis) are vmapped
    # together with a batched solution; scalars are shared by every case. A
    # tuple is unpacked into the surface map's parameters.
    def _axis(p):
        return 0 if np.ndim(p) >= 1 else None
    load_params = [s['load_param'] for s in surface_data]
    param_axes = [tuple(_axis(q) for q in p) if isinstance(p, tuple) else _axis(p)
                  for p in load_params]
    any_batched = any((ax is not None) if not isinstance(ax, tuple) else any(a is not None for a in ax)
                      for ax in param_axes)

    def _apply_map(surface_fn, u, x, p):
        return surface_fn(u, x, *p) if isinstance(p, tuple) else surface_fn(u, x, p)

    def _compliance_one(sol, params):
        """Compliance of one flat solution with one set of surface parameters."""
        # Manually unflatten just the first variable (displacement)
        displacement = sol[:total_nodes_var0 * vec_var0].reshape((total_nodes_var0, vec_var0))

        total_compliance = 0.0

        # Loop over all surfaces
        for surface, load_param in zip(surface_data, params):
            # Extract displacements on the boundary faces
            u_face_nodes = displacement[surface['cell_indices']]  # (num_selected_faces, num_nodes, vec)

            # Apply shape functions
            u_face = np.sum(surface['selected_face_shape_vals'][:, :, :, None] * u_face_nodes[:, None, :, :], axis=2)

            # Apply surface map to get traction vector
            # Note: negative sign follows the convention that external work is positive
            traction = -jax.vmap(
                jax.vmap(
                    lambda u, x: _apply_map(surface['surface_fn'], u, x, load_param)
                )
            )(u_face, surface['subset_quad_points'])

            # Compute compliance contribution from this surface
            surface_compliance = np.sum(traction * u_face * surface['nanson_scale'][:, :, None])
            total_compliance += surface_compliance

        return total_compliance

    def compliance_fn(sol):
        """Compute compliance for the given solution(s) over all surfaces."""
        if sol.ndim == 2:
            return jax.vmap(_compliance_one, in_axes=(0, param_axes))(sol, load_params)
        if any_batched:
            raise ValueError(
                "surface_load_params have a batch axis, so compliance_fn needs a "
                "batch of solutions (B, n).")
        return _compliance_one(sol, load_params)

    return compliance_fn


def create_dynamic_compliance_fn(problem):
    """
    Creates a universal pure-JAX compliance function for a given problem.
    Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
    where u is displacement and f is traction on each loaded boundary.

    Unlike ``create_compliance_fn``, the surface load parameters are not fixed when
    the function is created. They are inferred from ``surface_vars`` at runtime and
    passed into each surface map when evaluating the traction field.

    Args:
        problem: FEAX Problem instance

    Returns:
        compliance_fn: pure-JAX function that takes ``sol`` and
            ``surface_vars`` and returns the compliance value.
    """
    surface_maps = problem.get_surface_maps()
    num_surfaces = len(problem.boundary_inds_list)

    if num_surfaces == 0:
        def compliance_fn(sol, surface_vars):
            return 0.0
        return compliance_fn

    # Pre-compute data for all surfaces
    surface_data = []
    for i in range(num_surfaces):
        boundary_inds = problem.boundary_inds_list[i]
        _, nanson_scale = problem.fes[0].get_face_shape_grads(boundary_inds)
        subset_quad_points = problem.physical_surface_quad_points[i]
        surface_fn = surface_maps[i] if i < len(surface_maps) else lambda u, x, *p: np.zeros_like(u)


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
        })

    total_nodes_var0 = problem.fes[0].num_total_nodes
    vec_var0 = problem.fes[0].vec

    def compliance_fn(sol, surface_vars):
        """Compute compliance for the given solution over all surfaces."""
        # Manually unflatten just the first variable (displacement)

        displacement = sol[:total_nodes_var0 * vec_var0].reshape((total_nodes_var0, vec_var0))

        total_compliance = 0.0

        # Loop over all surfaces
        for surface, surface_vars_i in zip(surface_data, surface_vars):
            # Extract displacements on the boundary faces
            u_face_nodes = displacement[surface['cell_indices']]  # (num_selected_faces, num_nodes, vec)
            # Apply shape functions
            u_face = np.sum(surface['selected_face_shape_vals'][:, :, :, None] * u_face_nodes[:, None, :, :], axis=2)

            # Apply surface map to get traction vector
            # Note: negative sign follows the convention that external work is positive
            traction = -jax.vmap(
                jax.vmap(
                    lambda u, x, svars: surface['surface_fn'](u, x, *svars),
                    in_axes=(0, 0, 0),
                )
            )(u_face, surface['subset_quad_points'], surface_vars_i)

            # Compute compliance contribution from this surface
            surface_compliance = np.sum(traction * u_face * surface['nanson_scale'][:, :, None])
            total_compliance += surface_compliance

        return total_compliance

    return compliance_fn


def create_volume_fn(problem):
    """
    Creates a pure-JAX volume fraction calculation function for a given problem.
    Returns a function that computes the volume fraction of material in the domain.

    Supports node-based, cell-based, and quad-based density arrays.

    Args:
        problem: FEAX Problem instance

    Returns:
        volume_fn: pure-JAX function that takes density array and returns volume fraction
                   Accepts: (num_nodes,) node-based, (num_cells,) cell-based,
                           or (num_cells, num_quads) quad-based density arrays,
                           and a batch of node- or cell-based fields,
                           (B, num_nodes) / (B, num_cells), which gives (B,)
                           volume fractions.
    """
    # Pre-compute problem-specific data as static values
    fe = problem.fes[0]
    # Get quadrature weights and Jacobian determinants for volume integration
    JxW = fe.JxW  # Jacobian times quadrature weights (num_cells, num_quads)
    num_cells = fe.num_cells
    num_quads = fe.num_quads
    num_nodes = fe.num_total_nodes
    shape_vals = fe.shape_vals  # (num_quads, num_nodes_per_cell)
    cells = fe.cells  # (num_cells, num_nodes_per_cell)
    domain_volume = float(np.sum(JxW))  # Pre-compute domain volume as Python float

    def volume_fn(rho_array):
        """Compute volume fraction for the given density distribution.

        Args:
            rho_array: Density field. Can be:
                - (num_nodes,) node-based density
                - (num_cells,) cell-based density
                - (num_cells, num_quads) quad-based density
                - (B, num_nodes) or (B, num_cells): a batch of fields

        Returns:
            Volume fraction (scalar between 0 and 1), or (B,) for a batch
        """
        # A 2-D input is a batch of node/cell fields unless it is exactly the
        # quad-based layout (num_cells, num_quads).
        if rho_array.ndim == 2 and rho_array.shape[1] in (num_nodes, num_cells) \
                and rho_array.shape != (num_cells, num_quads):
            return jax.vmap(volume_fn)(rho_array)
        rho_flat = rho_array.ravel()

        # Determine input type based on size
        if rho_flat.shape[0] == num_nodes:
            # Node-based density: interpolate to quadrature points using shape functions
            # Gather node values for each cell: (num_cells, num_nodes_per_cell)
            rho_cell_nodes = rho_flat[cells]
            # Interpolate to quad points: (num_cells, num_quads)
            # shape_vals: (num_quads, num_nodes_per_cell)
            # rho_cell_nodes: (num_cells, num_nodes_per_cell)
            rho_quads = np.einsum('qn,cn->cq', shape_vals, rho_cell_nodes)
            total_volume = np.sum(rho_quads * JxW)
        elif rho_flat.shape[0] == num_cells:
            # Cell-based density: expand to quadrature points
            rho_quads = np.repeat(rho_flat[:, None], num_quads, axis=1)
            total_volume = np.sum(rho_quads * JxW)
        else:
            # Already at quadrature points (num_cells * num_quads,)
            rho_reshaped = rho_flat.reshape((num_cells, num_quads))
            total_volume = np.sum(rho_reshaped * JxW)

        # Return volume fraction
        return total_volume / domain_volume

    return volume_fn
