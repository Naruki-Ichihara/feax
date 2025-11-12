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


def helmholtz_filter(rho_source, mesh, radius, P=None, solver_options=None):
    """
    Apply Helmholtz filter to density field with optional periodic boundary conditions.

    Solves: ρ̃ - r² ∇²ρ̃ = ρ_source

    This function is pure (no side effects) and can be used with jax.jit and jax.vmap
    when P and solver_options are provided as static arguments.

    Args:
        rho_source: (num_cells,) array of source density field
        mesh: Mesh object
        radius: Filter radius (controls smoothness - larger = smoother)
        P: Optional prolongation matrix for periodic boundary conditions (default None)
        solver_options: Optional SolverOptions (default: tol=1e-8, cg solver)

    Returns:
        (num_nodes,) array of filtered density field

    Example:
        >>> # Without periodic BCs
        >>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1)

        >>> # With periodic BCs
        >>> P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
        >>> rho_filtered = helmholtz_filter(rho_source, mesh, radius=0.1, P=P)

        >>> # Vectorized filtering with vmap
        >>> filter_fn = lambda rho: helmholtz_filter(rho, mesh, radius=0.1, P=P)
        >>> rho_batch_filtered = jax.vmap(filter_fn)(rho_batch)
    """
    # Default solver options
    if solver_options is None:
        solver_options = fe.solver.SolverOptions(
            tol=1e-8,
            linear_solver="cg",
            verbose=False
        )

    # Create problem
    problem = HelmholtzFilterProblem(
        mesh=mesh,
        vec=1,  # Scalar field
        dim=3,
        ele_type=mesh.cells.shape[1] == 8 and mesh.points.shape[1] == 3 and 'HEX8' or 'TET4',
        location_fns=[]
    )

    # Empty boundary conditions (periodic handled via P matrix)
    bc_config = fe.DCboundary.DirichletBCConfig([])
    bc = bc_config.create_bc(problem)

    # Internal variables
    r_squared = radius ** 2
    rho_source_array = fe.internal_vars.InternalVars.create_cell_var(problem, rho_source)
    r_sq_array = fe.internal_vars.InternalVars.create_cell_var(problem, r_squared)
    internal_vars = fe.internal_vars.InternalVars(
        volume_vars=(rho_source_array, r_sq_array),
        surface_vars=()
    )

    # Create solver
    solver = fe.solver.create_solver(
        problem, bc, solver_options,
        iter_num=1,  # Linear problem
        P=P  # Optional periodic boundary conditions
    )

    # Solve
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    rho_filtered = solver(internal_vars, initial_guess)

    return rho_filtered


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