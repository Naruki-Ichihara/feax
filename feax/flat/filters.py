"""
Density filtering utilities for periodic structures.

This module provides filtering functions commonly used in topology optimization
and microstructure design, with support for periodic boundary conditions.
"""

import jax
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
            linear_solver="cg",
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


@jax.jit
def heaviside_projection(rho, beta=10.0, threshold=0.5):
    """
    Apply Heaviside projection to density field for sharp void/solid boundaries.

    H(ρ) = (tanh(β*(ρ-threshold)) + 1) / 2

    This function is pure and JIT-compiled for efficient batched processing.

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