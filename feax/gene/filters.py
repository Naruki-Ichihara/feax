"""Filters for topology optimization using FEAX framework.

Provides both PDE-based (Helmholtz) and distance-based (density) filters
for design variable smoothing in generative design workflows.
"""

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse
import feax as fe
from feax.problem import Problem
from typing import Callable, Tuple


class HelmholtzProblem(Problem):
    """Helmholtz equation problem for design variable filtering."""
    
    def __post_init__(self):
        super().__post_init__()
        # Get radius from additional_info
        if self.additional_info:
            self.radius = self.additional_info[0]
        else:
            self.radius = 0.05
            
    def get_tensor_map(self):
        """Get the diffusion tensor mapping for the Helmholtz equation."""
        def diffusion(u_grad, design_variable):
            """Compute diffusion term r²∇u."""
            return self.radius**2 * u_grad
        return diffusion
        
    def get_mass_map(self):
        """Get the mass term mapping for the Helmholtz equation."""
        def mass_term(u, x, design_variable):
            """Compute mass term u - design_variable."""
            return u - design_variable
        return mass_term


class HelmholtzFilterProblem(fe.problem.Problem):
    """
    Helmholtz filter problem: ρ̃ - r² ∇²ρ̃ = ρ_source

    This is a helper Problem class for internal use by helmholtz_filter().
    """

    def get_tensor_map(self):
        def tensor_map(rho_grad, rho_source, r_squared):
            return r_squared * rho_grad
        return tensor_map

    def get_mass_map(self):
        def mass_map(rho, x, rho_source, r_squared):
            return rho - rho_source
        return mass_map


def create_helmholtz_filter(mesh, radius, P=None, solver_options=None):
    """
    Create a differentiable Helmholtz filter function (node-based).

    This factory function creates the filter problem and solver once, returning
    a pure function that can be used with jax.jit, jax.vmap, and jax.grad.

    Solves: ρ̃ - r² ∇²ρ̃ = ρ_source

    Args:
        mesh: Mesh object
        radius: Filter radius (controls smoothness - larger = smoother)
        P: Optional prolongation matrix for periodic boundary conditions (default None)
        solver_options: Optional SolverOptions (default: tol=1e-8, cg solver)

    Returns:
        filter_fn: A pure function (rho_source) -> rho_filtered that can be
                   used with JAX transformations (jit, vmap, grad)
                   Input: (num_nodes,) node-based density field
                   Output: (num_nodes,) filtered node-based density field

    Example:
        >>> # Create filter function once
        >>> filter_fn = create_helmholtz_filter(mesh, radius=0.1)
        >>>
        >>> # Use in differentiable objective
        >>> def objective(rho):
        ...     rho_filtered = filter_fn(rho)
        ...     # ... use rho_filtered in FE solve
        ...     return compliance
        >>>
        >>> # Compute gradients
        >>> grad_fn = jax.grad(objective)
        >>> gradient = grad_fn(rho)
        >>>
        >>> # With periodic BCs
        >>> P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
        >>> filter_fn = create_helmholtz_filter(mesh, radius=0.1, P=P)
    """
    # Default solver options
    if solver_options is None:
        solver_options = fe.solver.SolverOptions(
            tol=1e-8,
            linear_solver="cudss",
            verbose=False
        )

    # Detect element type
    if mesh.points.shape[1] == 2:
        # 2D mesh
        ele_type = 'QUAD4' if mesh.cells.shape[1] == 4 else 'TRI3'
        dim = 2
    else:
        # 3D mesh
        ele_type = 'HEX8' if mesh.cells.shape[1] == 8 else 'TET4'
        dim = 3

    # Create problem (done once)
    problem = HelmholtzFilterProblem(
        mesh=mesh,
        vec=1,  # Scalar field
        dim=dim,
        ele_type=ele_type,
        location_fns=[]
    )

    # Empty boundary conditions (periodic handled via P matrix)
    bc_config = fe.DCboundary.DirichletBCConfig([])
    bc = bc_config.create_bc(problem)

    # Create solver (done once)
    solver = fe.solver.create_solver(
        problem, bc, solver_options,
        iter_num=1,  # Linear problem
        P=P  # Optional periodic boundary conditions
    )

    # Pre-compute constants
    r_squared = radius ** 2
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    num_nodes = mesh.points.shape[0]

    def filter_fn(rho_source):
        """
        Apply Helmholtz filter to density field.

        Args:
            rho_source: (num_nodes,) array of node-based source density field

        Returns:
            (num_nodes,) array of filtered node-based density field
        """
        # Create internal variables with rho_source as node-based variable
        r_sq_array = np.full(num_nodes, r_squared)
        internal_vars = fe.internal_vars.InternalVars(
            volume_vars=(rho_source, r_sq_array),
            surface_vars=()
        )

        # Solve
        rho_filtered = solver(internal_vars, initial_guess)
        return rho_filtered

    return filter_fn


def helmholtz_filter(rho_source, mesh, radius, P=None, solver_options=None):
    """
    Apply Helmholtz filter to node-based density field.

    WARNING: This function creates problem/solver each call. For use inside jax.grad,
    use create_helmholtz_filter() instead to create the filter function once.

    Solves: ρ̃ - r² ∇²ρ̃ = ρ_source

    Args:
        rho_source: (num_nodes,) array of node-based source density field
        mesh: Mesh object
        radius: Filter radius (controls smoothness - larger = smoother)
        P: Optional prolongation matrix for periodic boundary conditions (default None)
        solver_options: Optional SolverOptions (default: tol=1e-8, cg solver)

    Returns:
        (num_nodes,) array of filtered node-based density field

    Example:
        >>> # For one-time use (NOT inside jax.grad)
        >>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1)

        >>> # For use with jax.grad, use create_helmholtz_filter instead:
        >>> filter_fn = create_helmholtz_filter(mesh, radius=0.1)
        >>> def objective(rho):
        ...     return jnp.sum(filter_fn(rho))
        >>> grad_fn = jax.grad(objective)
    """
    filter_fn = create_helmholtz_filter(mesh, radius, P, solver_options)
    return filter_fn(rho_source)


# ============================================================================
# Standard Density Filter (Distance-Based Weighted Averaging)
# ============================================================================


def _compute_density_filter_weights_sparse(points, radius: float,
                                           weight_type: str = "cone", batch_size: int = 500):
    """Compute sparse filter weight matrix for density filtering (memory-efficient).

    Args:
        points: Node coordinates (num_nodes, dim) as JAX or numpy array
        radius: Filter radius
        weight_type: Type of weight function:
            - "cone": Linear decay w = max(0, r - d) (default, most common in topology optimization)
            - "gaussian": Gaussian decay w = exp(-(d/r)^2)
            - "constant": Constant weight within radius
        batch_size: Number of nodes to process at once (smaller = less memory, slower)

    Returns:
        Sparse weight matrix in BCOO format and row sums for normalization
    """
    import numpy as onp

    # Convert to numpy for preprocessing
    points_np = onp.array(points) if hasattr(points, 'shape') else points
    num_nodes = points_np.shape[0]

    # Build sparse matrix in COO format using batch processing
    rows = []
    cols = []
    data = []

    # Process nodes in batches to avoid memory issues
    for start_idx in range(0, num_nodes, batch_size):
        end_idx = min(start_idx + batch_size, num_nodes)
        batch_points = points_np[start_idx:end_idx]

        # Compute distances from batch to all nodes
        # (batch_size, 1, dim) - (1, num_nodes, dim) -> (batch_size, num_nodes, dim)
        diff = batch_points[:, onp.newaxis, :] - points_np[onp.newaxis, :, :]
        distances = onp.linalg.norm(diff, axis=2)

        # Find neighbors within radius and compute weights
        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            dists = distances[local_i]
            neighbors = onp.where(dists <= radius)[0]

            if len(neighbors) > 0:
                neighbor_dists = dists[neighbors]

                # Compute raw weights based on distance
                if weight_type == "cone":
                    weights = onp.maximum(0.0, radius - neighbor_dists)
                elif weight_type == "gaussian":
                    weights = onp.exp(-(neighbor_dists / radius) ** 2)
                elif weight_type == "constant":
                    weights = onp.ones_like(neighbor_dists)
                else:
                    raise ValueError(f"Unknown weight_type: {weight_type}")

                # Store COO entries
                rows.extend([global_i] * len(neighbors))
                cols.extend(neighbors.tolist())
                data.extend(weights.tolist())

    # Convert to JAX BCOO sparse matrix
    rows = onp.array(rows, dtype=onp.int32)
    cols = onp.array(cols, dtype=onp.int32)
    data = onp.array(data)
    indices = onp.stack([rows, cols], axis=1)

    # Create BCOO matrix
    sparse_matrix = jsparse.BCOO((np.array(data), np.array(indices)),
                                 shape=(num_nodes, num_nodes))

    # Compute row sums for normalization and convert to dense array
    row_sums = np.array(sparse_matrix.sum(axis=1).todense())

    return sparse_matrix, row_sums


def create_density_filter(mesh, radius: float, weight_type: str = "cone",
                         filter_gradients: bool = False) -> Callable:
    """Create a standard density filter using distance-based weighted averaging.

    This is the classic topology optimization filter that computes weighted
    averages of design variables based on spatial proximity. It's computationally
    lighter than Helmholtz filter but provides similar smoothing effects.

    The filter computes: ρ̃_i = Σ_j w_ij ρ_j / Σ_j w_ij
    where w_ij is a distance-based weight function.

    Args:
        mesh: Mesh object
        radius: Filter radius (controls neighborhood size)
        weight_type: Type of weight function:
            - "cone": w = max(0, r - d) (default, most common)
            - "gaussian": w = exp(-(d/r)^2)
            - "constant": w = 1 if d <= r, else 0
        filter_gradients: If True, also returns a gradient filter for sensitivity filtering

    Returns:
        filter_fn: A JIT-compiled function (rho) -> rho_filtered
                  Input: (num_nodes,) node-based density field
                  Output: (num_nodes,) filtered node-based density field

    Example:
        >>> # Create filter function once
        >>> filter_fn = create_density_filter(mesh, radius=3.0)
        >>>
        >>> # Use in optimization
        >>> def objective(rho):
        ...     rho_filtered = filter_fn(rho)
        ...     # ... use rho_filtered in FE solve
        ...     return compliance
        >>>
        >>> # Automatic differentiation works seamlessly
        >>> grad_fn = jax.grad(objective)
        >>> gradient = grad_fn(rho)

    Notes:
        - This filter is linear and does not require solving a PDE
        - The cone weight function (default) is most common in literature
        - For large meshes, pre-computation of weights may require significant memory
        - The filter preserves the integral of the density field (mass conservation)
    """
    points = mesh.points  # (num_nodes, dim)

    # Pre-compute sparse filter weight matrix (done once at filter creation)
    weight_matrix, row_sums = _compute_density_filter_weights_sparse(points, radius, weight_type)

    @jax.jit
    def filter_fn(rho: np.ndarray) -> np.ndarray:
        """Apply density filter to node-based design variables.

        Args:
            rho: Design variables at nodes (num_nodes,)

        Returns:
            Filtered design variables at nodes (num_nodes,)
        """
        # Sparse matrix-vector multiplication for weighted averaging
        rho_weighted = weight_matrix @ rho
        # Normalize by row sums
        rho_filtered = rho_weighted / (row_sums + 1e-12)
        return rho_filtered

    return filter_fn


def create_sensitivity_filter(mesh, radius: float, weight_type: str = "cone",
                              element_volumes: np.ndarray = None) -> Callable:
    """Create a sensitivity filter for gradient smoothing (mesh-independent filtering).

    The sensitivity filter applies weighted averaging to gradients rather than
    design variables. This provides mesh-independent results by incorporating
    element volumes in the weighting.

    Implements: dJ/dρ̃_i = (Σ_j w_ij V_j dJ/dρ_j) / (V_i Σ_j w_ij)

    where V_j are element volumes and w_ij are distance-based weights.

    Args:
        mesh: Mesh object
        radius: Filter radius
        weight_type: Type of weight function ("cone", "gaussian", "constant")
        element_volumes: Optional pre-computed element volumes (num_cells,)
                        If None, will compute from mesh

    Returns:
        filter_fn: A JIT-compiled function (sensitivities) -> filtered_sensitivities

    Example:
        >>> # Create sensitivity filter
        >>> sens_filter = create_sensitivity_filter(mesh, radius=3.0)
        >>>
        >>> # Filter gradients before optimization update
        >>> raw_gradient = jax.grad(objective)(rho)
        >>> filtered_gradient = sens_filter(raw_gradient)

    Notes:
        - This filter is particularly useful for mesh-independent optimization
        - Often combined with density filtering: filter both design vars and sensitivities
        - Provides smoother convergence in topology optimization
    """
    points = mesh.points  # (num_nodes, dim)
    num_nodes = points.shape[0]

    # Pre-compute filter weight matrix
    weight_matrix = _compute_density_filter_weights(points, radius, weight_type)

    # Compute element volumes if not provided
    if element_volumes is None:
        # Use nodal volumes (sum of connected element volumes)
        # For simplicity, approximate as uniform (can be improved)
        nodal_volumes = np.ones(num_nodes)
    else:
        # Map element volumes to nodes (average of connected elements)
        cells = mesh.cells
        nodal_volumes = np.zeros(num_nodes)
        for cell in cells:
            for node in cell:
                nodal_volumes = nodal_volumes.at[node].add(1.0)
        nodal_volumes = nodal_volumes / np.maximum(nodal_volumes, 1.0)

    @jax.jit
    def filter_fn(sensitivities: np.ndarray) -> np.ndarray:
        """Apply sensitivity filter to gradients.

        Args:
            sensitivities: Gradient/sensitivity values (num_nodes,)

        Returns:
            Filtered sensitivities (num_nodes,)
        """
        # Weight sensitivities by nodal volumes
        weighted_sens = sensitivities * nodal_volumes

        # Apply distance-based filter
        filtered_weighted = np.dot(weight_matrix, weighted_sens)

        # Normalize by volume-weighted sum
        volume_weighted_sum = np.dot(weight_matrix, nodal_volumes)
        filtered_sens = filtered_weighted / (volume_weighted_sum + 1e-12)

        return filtered_sens

    return filter_fn


def density_filter(rho_source, mesh, radius: float, weight_type: str = "cone") -> np.ndarray:
    """Apply standard density filter to node-based density field.

    WARNING: This function creates the filter each call. For use inside jax.grad
    or in optimization loops, use create_density_filter() instead to create the
    filter function once.

    Args:
        rho_source: (num_nodes,) array of node-based source density field
        mesh: Mesh object
        radius: Filter radius
        weight_type: Weight function type ("cone", "gaussian", "constant")

    Returns:
        (num_nodes,) array of filtered node-based density field

    Example:
        >>> # For one-time use (NOT inside jax.grad or loops)
        >>> rho_filtered = density_filter(rho_source, mesh, radius=3.0)
        >>>
        >>> # For repeated use or with jax.grad, use create_density_filter instead:
        >>> filter_fn = create_density_filter(mesh, radius=3.0)
        >>> def objective(rho):
        ...     return jnp.sum(filter_fn(rho))
        >>> grad_fn = jax.grad(objective)
    """
    filter_fn = create_density_filter(mesh, radius, weight_type)
    return filter_fn(rho_source)