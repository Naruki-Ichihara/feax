"""Computational homogenization solver for periodic unit cell analysis.

Computes the homogenized stiffness matrix C_hom for a linear elastic unit
cell using asymptotic homogenization theory with periodic boundary conditions.

The homogenized stiffness relates volume-averaged stress to volume-averaged
strain:

    <σ> = C_hom : <ε>

For each of the n independent unit strain cases ε^(k), the periodic
fluctuation problem is solved and the volume-averaged stress response is
assembled into C_hom.
"""

from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as np

from ..solver import create_solver
from ..solvers.options import IterativeSolverOptions

# ---------------------------------------------------------------------------
# Unit strain tensors in Voigt order
# ---------------------------------------------------------------------------

# 3D: ε11, ε22, ε33, γ23 (=2ε23), γ13 (=2ε13), γ12 (=2ε12)
UNIT_STRAINS_3D = np.array([
    [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]],
    [[0., 0., 0.], [0., 0., .5], [0., .5, 0.]],
    [[0., 0., .5], [0., 0., 0.], [.5, 0., 0.]],
    [[0., .5, 0.], [.5, 0., 0.], [0., 0., 0.]],
])

# 2D: ε11, ε22, γ12 (=2ε12)
UNIT_STRAINS_2D = np.array([
    [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
    [[0., .5, 0.], [.5, 0., 0.], [0., 0., 0.]],
])


# ---------------------------------------------------------------------------
# Helpers (public)
# ---------------------------------------------------------------------------

def macro_displacement(mesh, epsilon_macro: np.ndarray) -> np.ndarray:
    """Affine displacement field u_i = ε_ij X_j for each mesh node.

    Parameters
    ----------
    mesh :
        FEAX mesh with ``points`` array of shape ``(num_nodes, 2)`` or
        ``(num_nodes, 3)``.
    epsilon_macro : ndarray, shape (3, 3)
        Symmetric macroscopic strain tensor.

    Returns
    -------
    ndarray
        Flattened affine displacement vector with the same number of
        components per node as the mesh spatial dimension.
    """
    pts = mesh.points
    ndim = pts.shape[1]
    if ndim < 3:
        # Pad 2D points to 3D, compute full product, keep first ndim components
        pts_3d = np.concatenate([pts, np.zeros((pts.shape[0], 3 - ndim))], axis=1)
        return (pts_3d @ epsilon_macro.T)[:, :ndim].flatten()
    return (pts @ epsilon_macro.T).flatten()


def average_stress(problem, u_total: np.ndarray, internal_vars, dim: int) -> np.ndarray:
    """Compute the volume-averaged Cauchy stress in Voigt notation.

    Parameters
    ----------
    problem :
        FEAX Problem instance.
    u_total : ndarray, shape (num_dofs,)
        Total displacement field.
    internal_vars :
        FEAX InternalVars.
    dim : int
        Problem dimension (2 or 3).

    Returns
    -------
    ndarray
        Volume-averaged stress in Voigt notation.
        Shape ``(3,)`` for 2D: ``[σ11, σ22, σ12]``.
        Shape ``(6,)`` for 3D: ``[σ11, σ22, σ33, σ23, σ13, σ12]``.
    """
    tensor_map = problem.get_tensor_map()

    sol_list = problem.unflatten_fn_sol_list(u_total)
    # cell_sol: (num_cells, num_nodes_per_cell, vec)
    cell_sol = sol_list[0][problem.cells_list[0]]

    # shape_grads: (num_cells, num_quads, num_nodes_per_cell, dim)
    shape_grads = problem.shape_grads

    # JxW: (num_cells, num_quads) — squeeze out any size-1 axes
    JxW = problem.JxW
    if JxW.ndim == 3:
        JxW = JxW[:, 0, :]

    total_volume = JxW.sum()

    # Displacement gradient at every quad point
    # u_grads: (num_cells, num_quads, vec, dim)
    u_grads = np.einsum('cqnd,cnv->cqvd', shape_grads, cell_sol)

    # Broadcast volume internal vars to (num_cells, num_quads)
    num_cells = u_grads.shape[0]
    num_quads = JxW.shape[1]

    # shape_vals: (num_quads, num_nodes_per_cell) — for node-based interpolation
    shape_vals    = problem.fes[0].shape_vals
    cell_node_ids = problem.cells_list[0]   # (num_cells, nodes_per_cell)

    vol_vars_quad = []
    for var in internal_vars.volume_vars:
        if var.ndim == 1:
            if var.shape[0] == num_cells:
                # cell-based: same value at every quad point
                vol_vars_quad.append(np.tile(var[:, None], (1, num_quads)))
            else:
                # node-based: interpolate to Gauss points via shape functions
                var_cell_nodes = var[cell_node_ids]                          # (num_cells, nodes_per_cell)
                var_at_quads   = np.einsum('qn,cn->cq', shape_vals, var_cell_nodes)  # (num_cells, num_quads)
                vol_vars_quad.append(var_at_quads)
        else:
            vol_vars_quad.append(var)

    def cell_weighted_stress(u_grads_c, JxW_c, *vars_c):
        """Sum of JxW-weighted stresses over quad points for one cell."""
        def quad_stress(u_grad_q, *vars_q):
            return tensor_map(u_grad_q, *vars_q)
        stresses = jax.vmap(quad_stress)(u_grads_c, *vars_c)  # (num_quads, vec, dim)
        return np.einsum('q,qvd->vd', JxW_c, stresses)

    weighted = jax.vmap(cell_weighted_stress)(u_grads, JxW, *vol_vars_quad)
    sigma = np.sum(weighted, axis=0) / total_volume  # (vec, dim)

    if dim == 2:
        return np.array([sigma[0, 0], sigma[1, 1], sigma[0, 1]])
    else:
        return np.array([
            sigma[0, 0], sigma[1, 1], sigma[2, 2],
            sigma[1, 2], sigma[0, 2], sigma[0, 1],
        ])


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
_macro_displacement = macro_displacement
_average_stress = average_stress
_UNIT_STRAINS_3D = UNIT_STRAINS_3D
_UNIT_STRAINS_2D = UNIT_STRAINS_2D


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class HomogenizationResult(NamedTuple):
    """Result of a computational homogenization analysis.

    Attributes
    ----------
    C_hom : ndarray, shape (3, 3) or (6, 6)
        Homogenized stiffness matrix in Voigt notation.
    u_totals : tuple of ndarray
        Total displacement fields for each unit strain case.
        ``u_totals[k]`` has shape ``(num_dofs,)``.
    u_macros : tuple of ndarray
        Macroscopic (affine) displacement fields for each unit strain case.
    unit_strains : ndarray
        The unit strain tensors used, shape ``(n_cases, 3, 3)``.
    labels : tuple of str
        Labels for each strain case (e.g. ``('eps11', 'eps22', ...)``)
    """
    C_hom: np.ndarray
    u_totals: Tuple[np.ndarray, ...]
    u_macros: Tuple[np.ndarray, ...]
    unit_strains: np.ndarray
    labels: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_homogenization_solver(
    problem: Any,
    bc: Any,
    P: Any,
    mesh: Any,
    solver_options: IterativeSolverOptions = None,
    dim: int = 3,
):
    """Create a computational homogenization solver for a linear elastic unit cell.

    For each of the n independent unit strain cases the periodic fluctuation
    problem is solved, the volume-averaged stress is computed, and the results
    are assembled into the homogenized stiffness matrix C_hom:

        C_hom[:, k] = <σ>  when  ε^(k) is applied

    where ε^(k) is the k-th unit strain in Voigt order.

    Parameters
    ----------
    problem :
        FEAX Problem instance (linear elasticity).
    bc :
        FEAX DirichletBC (typically empty for periodic unit cells).
    P :
        Prolongation matrix from ``feax.flat.pbc.prolongation_matrix``.
        Shape ``(num_dofs, num_reduced_dofs)``.
    mesh :
        FEAX mesh of the unit cell.
    solver_options : IterativeSolverOptions, optional
        Iterative solver configuration.
    dim : int
        Problem dimension. ``2`` or ``3``. Default: ``3``.

    Returns
    -------
    callable
        ``solve(internal_vars) -> HomogenizationResult``

    Examples
    --------
    >>> solve = create_homogenization_solver(problem, bc, P, mesh, dim=3)
    >>> result = solve(internal_vars)
    >>> result.C_hom   # ndarray, shape (6, 6)
    >>> result.u_totals[0]  # displacement field for first strain case
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    unit_strains_arr = UNIT_STRAINS_3D if dim == 3 else UNIT_STRAINS_2D
    n_cases = len(unit_strains_arr)
    labels = (
        ('eps11', 'eps22', 'eps33', 'gam23', 'gam13', 'gam12') if dim == 3
        else ('eps11', 'eps22', 'gam12')
    )

    if solver_options is None:
        solver_options = IterativeSolverOptions()
    _verbose = solver_options.verbose

    fluctuation_solver = create_solver(problem, bc, solver_options, P=P)

    def solve(internal_vars) -> HomogenizationResult:
        columns = []
        u_totals = []
        u_macros = []
        for k in range(n_cases):
            if _verbose:
                print(f"  [homogenization] strain case {k + 1}/{n_cases}: {labels[k]}")
            u_macro = macro_displacement(mesh, unit_strains_arr[k])
            u_fluct = fluctuation_solver(internal_vars, u_macro)
            u_total = u_fluct + u_macro
            sigma_voigt = average_stress(problem, u_total, internal_vars, dim)
            columns.append(sigma_voigt)
            u_totals.append(u_total)
            u_macros.append(u_macro)
        C_hom = np.stack(columns, axis=1)
        return HomogenizationResult(
            C_hom=C_hom,
            u_totals=tuple(u_totals),
            u_macros=tuple(u_macros),
            unit_strains=unit_strains_arr,
            labels=labels,
        )

    return solve
