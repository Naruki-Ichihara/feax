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

import jax
import jax.numpy as np
from typing import Any

from feax.solver import create_solver
from feax.solvers.options import IterativeSolverOptions


# ---------------------------------------------------------------------------
# Unit strain tensors in Voigt order
# ---------------------------------------------------------------------------

# 3D: ε11, ε22, ε33, γ23 (=2ε23), γ13 (=2ε13), γ12 (=2ε12)
_UNIT_STRAINS_3D = np.array([
    [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]],
    [[0., 0., 0.], [0., 0., .5], [0., .5, 0.]],
    [[0., 0., .5], [0., 0., 0.], [.5, 0., 0.]],
    [[0., .5, 0.], [.5, 0., 0.], [0., 0., 0.]],
])

# 2D: ε11, ε22, γ12 (=2ε12)
_UNIT_STRAINS_2D = np.array([
    [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
    [[0., .5, 0.], [.5, 0., 0.], [0., 0., 0.]],
])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _macro_displacement(mesh, epsilon_macro: np.ndarray) -> np.ndarray:
    """Affine displacement field u_i = ε_ij X_j for each mesh node.

    Parameters
    ----------
    mesh :
        FEAX mesh with ``points`` array of shape ``(num_nodes, 3)``.
    epsilon_macro : ndarray, shape (3, 3)
        Symmetric macroscopic strain tensor.

    Returns
    -------
    ndarray, shape (num_nodes * 3,)
        Flattened affine displacement vector.
    """
    return (mesh.points @ epsilon_macro.T).flatten()


def _average_stress(problem, u_total: np.ndarray, internal_vars, dim: int) -> np.ndarray:
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

    Internally uses ``feax.solver.create_solver`` with the prolongation matrix
    P.  For each macroscopic strain ε^(k), the fluctuation solver returns the
    periodic fluctuation P u'_red, and the total displacement is reconstructed
    as u_total = P u'_red + u_macro.

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
        Iterative solver configuration. Default: ``IterativeSolverOptions()``
        which uses ``solver="auto"`` (→ CG, since P^T K P is SPD),
        ``tol=1e-10``, ``atol=1e-10``, ``maxiter=10000``.
    dim : int
        Problem dimension. ``2`` → output shape ``(3, 3)``;
        ``3`` → output shape ``(6, 6)``. Default: ``3``.

    Returns
    -------
    callable
        ``compute_C_hom(internal_vars) -> ndarray`` of shape
        ``(3, 3)`` (2D) or ``(6, 6)`` (3D).

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or 3.

    Examples
    --------
    >>> from feax.flat.pbc import periodic_bc_3D, prolongation_matrix
    >>> from feax.solvers.options import IterativeSolverOptions
    >>> pbc = periodic_bc_3D(unitcell, vec=3, dim=3)
    >>> P = prolongation_matrix(pbc, mesh, vec=3)
    >>> opts = IterativeSolverOptions(solver="cg", tol=1e-10, maxiter=5000)
    >>> compute_C_hom = create_homogenization_solver(
    ...     problem, bc, P, mesh, solver_options=opts, dim=3
    ... )
    >>> C_hom = compute_C_hom(internal_vars)  # ndarray, shape (6, 6)
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    unit_strains = _UNIT_STRAINS_3D if dim == 3 else _UNIT_STRAINS_2D
    n_cases = len(unit_strains)
    _labels = (
        ['ε11', 'ε22', 'ε33', 'γ23', 'γ13', 'γ12'] if dim == 3
        else ['ε11', 'ε22', 'γ12']
    )

    if solver_options is None:
        solver_options = IterativeSolverOptions()
    _verbose = solver_options.verbose

    # create_solver with P routes to _create_reduced_solver.
    # Calling fluctuation_solver(internal_vars, u_macro) returns P @ u'_red,
    # the periodic fluctuation part only.
    fluctuation_solver = create_solver(problem, bc, solver_options, P=P)

    def compute_C_hom(internal_vars):
        columns = []
        for k in range(n_cases):
            if _verbose:
                print(f"  [homogenization] strain case {k + 1}/{n_cases}: {_labels[k]}")
            u_macro = _macro_displacement(mesh, unit_strains[k])
            u_fluct = fluctuation_solver(internal_vars, u_macro)
            u_total = u_fluct + u_macro
            sigma_voigt = _average_stress(problem, u_total, internal_vars, dim)
            columns.append(sigma_voigt)
        # C_hom[:, k] = stress response to unit strain k
        return np.stack(columns, axis=1)

    return compute_C_hom
