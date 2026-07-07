"""Filters for topology optimization using FEAX framework.

Provides both PDE-based (Helmholtz) and distance-based (density) filters
for design variable smoothing in generative design workflows.
"""

from typing import Callable

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as np

import feax as fe


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
        solver_options: Optional KrylovSolverOptions (default: cg, tol=1e-8)

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
        solver_options = fe.KrylovSolverOptions(
            solver="cg",
            tol=1e-8,
            verbose=False,
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

    # Build a TracedStructure for the filter problem so the internal solve runs
    # on the recommended (non-deprecated) assembly path instead of emitting the
    # no-TracedStructure FutureWarning. The reduced/periodic (P) solver does not
    # accept a traced_structure, so only use one on the standard path.
    ts = fe.TracedStructure.from_problem(problem) if P is None else None

    # Create solver (done once)
    solver = fe.solver.create_solver(
        problem, bc, solver_options,
        linear=True,  # Linear problem
        P=P,  # Optional periodic boundary conditions
        traced_structure=ts,
        return_solution=False,  # internal: raw flat field
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
        traced_params = fe.traced_params.TracedParams(
            volume_vars=(rho_source, r_sq_array),
            surface_vars=()
        )

        # Solve (route through the TracedStructure path when available)
        if ts is not None:
            rho_filtered = solver(traced_params, initial_guess, traced_structure=ts)
        else:
            rho_filtered = solver(traced_params, initial_guess)
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
        solver_options: Optional KrylovSolverOptions (default: cg, tol=1e-8)

    Returns:
        (num_nodes,) array of filtered node-based density field

    Example:
        >>> # For one-time use (NOT inside jax.grad)
        >>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1)

        >>> # For use with jax.grad, use create_helmholtz_filter instead:
        >>> filter_fn = create_helmholtz_filter(mesh, radius=0.1)
        >>> def objective(rho):
        ...     return np.sum(filter_fn(rho))
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


def create_density_filter(mesh, radius: float, weight_type: str = "cone") -> Callable:
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

    Returns:
        filter_fn: A pure-JAX function (rho) -> rho_filtered; compose it
            under the caller's ``jax.jit`` (feax.gene.optimizer does this)
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
        ...     return np.sum(filter_fn(rho))
        >>> grad_fn = jax.grad(objective)
    """
    filter_fn = create_density_filter(mesh, radius, weight_type)
    return filter_fn(rho_source)


# ============================================================================
# Projection and Threshold Functions
# ============================================================================


def heaviside_projection(rho, beta=10.0, threshold=0.5):
    """
    Apply Heaviside projection to density field for sharp void/solid boundaries.

    H(ρ) = (tanh(β*(ρ-threshold)) + 1) / 2

    This function is pure JAX; jit it (or the loss containing it) at the
    call site for efficient batched processing.

    Args:
        rho: Density field (normalized to [0, 1])
        beta: Sharpness parameter (higher = sharper transition, default 10.0)
        threshold: Transition threshold (default 0.5)

    Returns:
        Projected density field with sharp boundaries

    Example:
        >>> # Single field
        >>> rho_sharp = heaviside_projection(rho_smooth, beta=10.0, threshold=0.5)

        >>> # Vectorized batch processing
        >>> rho_batch_sharp = jax.vmap(lambda r: heaviside_projection(r, beta=10.0))(rho_batch)
    """
    return (np.tanh(beta * (rho - threshold)) + 1.0) / 2.0


def compute_volume_fraction_threshold(rho, target_volume_fraction):
    """
    Compute density threshold to achieve target volume fraction.

    Uses percentile-based approach to ensure exact volume fraction after Heaviside projection.

    Args:
        rho: Density field
        target_volume_fraction: Target solid volume fraction (0.0 to 1.0)

    Returns:
        Threshold value for Heaviside projection

    Example:
        >>> # Normalize density to [0, 1]
        >>> rho_normalized = (rho - rho.min()) / (rho.max() - rho.min())
        >>>
        >>> # Compute threshold for 50% volume fraction
        >>> threshold = compute_volume_fraction_threshold(rho_normalized, 0.5)
        >>>
        >>> # Apply Heaviside projection
        >>> rho_projected = heaviside_projection(rho_normalized, beta=10.0, threshold=threshold)
    """
    percentile = (1.0 - target_volume_fraction) * 100.0
    return np.percentile(rho, percentile)


def create_sensitivity_filter(grid, rmin: float):
    """Narrow-band-native sensitivity filter (88-line ``ft=1``) for a StructuredGrid.

    Unlike the Helmholtz / density filters above (node-based, mesh-based, and not
    designed for a cell-based moving band), this one is CELL-based and O(band): for
    each active cell it averages the sensitivity over its in-radius ACTIVE
    neighbours using cone weights, addressing neighbours by grid arithmetic +
    ``searchsorted`` into the active set — no full-grid array, so it follows the
    moving band. Intended for the analytic-sensitivity + OC workflow (it modifies
    the sensitivity heuristically; it is not a differentiable design transform —
    use Helmholtz/density for ``jax.grad`` design filtering).

    Returns ``filt(active_cells, rho_active, dc_active) -> dc_filtered`` (all arrays
    over the active band cells), computing

        dc_filt_e = Σ_n w_en ρ_n dc_n / ( Hs_e · max(1e-3, ρ_e) )

    with Hs_e summed over all IN-GRID neighbours (active or void).
    """
    import numpy as onp

    r = int(onp.ceil(rmin)) - 1
    offs = [(di, dj, dk) for di in range(-r, r + 1)
            for dj in range(-r, r + 1) for dk in range(-r, r + 1)]
    w0 = onp.array([max(0.0, rmin - onp.sqrt(di * di + dj * dj + dk * dk))
                    for di, dj, dk in offs])
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def filt(active_cells, rho_active, dc_active):
        ac = onp.asarray(active_cells, onp.int64)
        na = ac.size
        ex, ey, ez = grid.cell_ijk(ac)
        rho_pad = onp.concatenate([onp.asarray(rho_active, float), [0.0]])   # sink at na
        dc_pad = onp.concatenate([onp.asarray(dc_active, float), [0.0]])
        num = onp.zeros(na)
        Hs = onp.zeros(na)
        for (di, dj, dk), w in zip(offs, w0):
            if w <= 0.0:
                continue
            jx, jy, jz = ex + di, ey + dj, ez + dk
            inb = ((jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
                   & (jz >= 0) & (jz < nz))
            Hs += onp.where(inb, w, 0.0)
            cell = grid.cell_id(onp.clip(jx, 0, nx - 1), onp.clip(jy, 0, ny - 1),
                                onp.clip(jz, 0, nz - 1))
            pos = onp.clip(onp.searchsorted(ac, cell), 0, na - 1)
            hit = inb & (ac[pos] == cell)
            idx = onp.where(hit, pos, na)                # miss -> sink (0)
            num += w * rho_pad[idx] * dc_pad[idx]
        return num / (Hs * onp.maximum(1e-3, onp.asarray(rho_active, float)) + 1e-30)

    return filt
