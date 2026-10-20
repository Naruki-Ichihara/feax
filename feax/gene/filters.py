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
        projection=P,  # Optional periodic boundary conditions
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
                                           weight_type: str = "cone"):
    """Compute sparse filter weight matrix for density filtering.

    The neighbours within ``radius`` come from a k-d tree, so the cost is
    O(N k) for N points with k neighbours each (an all-pairs distance pass is
    O(N^2): minutes at 10^5 nodes).

    Args:
        points: Node coordinates (num_nodes, dim) as JAX or numpy array
        radius: Filter radius
        weight_type: Type of weight function:
            - "cone": Linear decay w = max(0, r - d) (default, most common in topology optimization)
            - "gaussian": Gaussian decay w = exp(-(d/r)^2)
            - "constant": Constant weight within radius

    Returns:
        Sparse weight matrix in BCOO format (row-major, every pair with
        d <= radius, self included) and row sums for normalization
    """
    import numpy as onp
    from scipy.spatial import cKDTree

    if weight_type not in ("cone", "gaussian", "constant"):
        raise ValueError(f"Unknown weight_type: {weight_type}")
    points_np = onp.asarray(points, dtype=float)
    num_nodes = points_np.shape[0]

    tree = cKDTree(points_np)
    pairs = tree.sparse_distance_matrix(tree, radius, output_type="ndarray")
    order = onp.lexsort((pairs["j"], pairs["i"]))
    rows = pairs["i"][order].astype(onp.int32)
    cols = pairs["j"][order].astype(onp.int32)
    dists = pairs["v"][order]

    if weight_type == "cone":
        data = onp.maximum(0.0, radius - dists)
    elif weight_type == "gaussian":
        data = onp.exp(-(dists / radius) ** 2)
    else:
        data = onp.ones_like(dists)

    indices = onp.stack([rows, cols], axis=1)
    sparse_matrix = jsparse.BCOO((np.array(data), np.array(indices)),
                                 shape=(num_nodes, num_nodes))
    row_sums = np.asarray(onp.bincount(rows, weights=data, minlength=num_nodes))

    return sparse_matrix, row_sums


@jax.tree_util.register_pytree_node_class
class DensityFilter:
    """The filter :func:`create_density_filter` returns: ``filt(rho)``.

    A pytree whose leaves are the sparse weights, so it can be an *argument*
    of a jitted function. Closed over instead, the weights (~10^7 entries at
    10^5 nodes) are baked into the executable as constants, which makes XLA
    compilation slow at scale; pass the filter in::

        @jax.jit
        def objective(X, filt):
            rho = filt(X)
            ...
    """

    def __init__(self, weights, row_sums):
        self.weights = weights
        self.row_sums = row_sums

    def __call__(self, rho):
        """Apply density filter to node-based design variables.

        Args:
            rho: Design variables at nodes (num_nodes,), or a batch (B, num_nodes)

        Returns:
            Filtered design variables, same shape
        """
        if rho.ndim == 2:
            return jax.vmap(self)(rho)
        # Sparse matrix-vector multiplication for weighted averaging,
        # normalized by row sums
        return (self.weights @ rho) / (self.row_sums + 1e-12)

    def tree_flatten(self):
        return (self.weights, self.row_sums), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


def create_density_filter(mesh, radius: float, weight_type: str = "cone",
                          points=None) -> Callable:
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
        points: Optional (N, dim) array of filter sites, replacing the mesh
            nodes. Pass element centroids to filter an ELEMENT-wise design
            field; the returned function then maps (num_cells,) to
            (num_cells,). Everything else is unchanged -- the filter only ever
            needed coordinates.

    Returns:
        filter_fn: A :class:`DensityFilter`, pure-JAX (rho) -> rho_filtered;
            compose it under the caller's ``jax.jit`` (feax.gene.optimizer
            does this). It is a pytree: on a large mesh pass it to the jitted
            function as an argument rather than closing over it, so the
            weights are not compiled in as constants.
                  Input: (num_nodes,) node-based density field, or a batch
                         of them, (B, num_nodes) (e.g. the designs of a
                         ``feax.vmap`` batch)
                  Output: the filtered field(s), same shape

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
        - The weights hold every neighbour pair within the radius: ~(4/3 pi (r/h)^3)
          entries per node in 3-D
        - The filter preserves the integral of the density field (mass conservation)
    """
    if points is None:
        points = mesh.points  # (num_nodes, dim)

    # Pre-compute sparse filter weight matrix (done once at filter creation)
    weight_matrix, row_sums = _compute_density_filter_weights_sparse(points, radius, weight_type)
    return DensityFilter(weight_matrix, row_sums)


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

    This function is pure JAX and elementwise, so it applies to a single
    field or to a batch (B, n) alike; jit it (or the loss containing it) at
    the call site.

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
