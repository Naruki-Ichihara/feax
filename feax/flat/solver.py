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

import warnings
from typing import Any, NamedTuple, Tuple

import numpy as onp
import jax
import jax.numpy as np

from ..solver import create_solver
from ..solvers.options import DirectSolverOptions, KrylovSolverOptions, has_cudss

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


def average_stress(problem, u_total: np.ndarray, traced_params, dim: int) -> np.ndarray:
    """Compute the volume-averaged Cauchy stress in Voigt notation.

    Parameters
    ----------
    problem :
        FEAX Problem instance.
    u_total : ndarray, shape (num_dofs,)
        Total displacement field.
    traced_params :
        FEAX TracedParams.
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
    for var in traced_params.volume_vars:
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
    u_totals : ndarray, shape (n_cases, num_dofs)
        Total displacement fields for each unit strain case.
        ``u_totals[k]`` has shape ``(num_dofs,)``.
    u_macros : ndarray, shape (n_cases, num_dofs)
        Macroscopic (affine) displacement fields for each unit strain case.
    """
    C_hom: np.ndarray
    u_totals: np.ndarray
    u_macros: np.ndarray


# ---------------------------------------------------------------------------
# Direct (factor-once / solve-many) homogenization path
# ---------------------------------------------------------------------------
# The fully-periodic reduced operator ``PᵀKP`` is SHARED by all n unit-strain
# cases (only the RHS — the macroscopic load — changes), so the ideal solve is a
# single factorization reused across every case (and every adjoint). ``PᵀKP`` is
# also SINGULAR (rigid-body translations survive periodicity), which a direct
# factorization cannot handle, so we pin one reduced dof per translation
# direction ("zero-mean" gauge: C_hom is translation-invariant, and the output
# fluctuation is re-centred to zero mean). The custom_vjp factors once and reuses
# that factorization for the n forward solves AND the n adjoint solves.


def _translation_pin_dofs(P, vec, dim):
    """Reduced dofs to pin (one per translation direction) to remove the rigid-
    body null space of ``PᵀKP``. A reduced dof carries a single spatial component
    (``full_dof % vec``); pinning one dof of each component c<dim kills the
    corresponding translation while leaving the operator symmetric positive
    definite."""
    coo = onp.asarray(P.indices)                 # (nnz, 2): [full_dof, reduced_dof]
    comp = coo[:, 0] % vec
    red = coo[:, 1]
    pins = []
    for c in range(dim):
        cand = red[comp == c]
        if cand.size == 0:
            raise ValueError(
                f"No reduced dof carries component {c}; cannot pin translation.")
        pins.append(int(cand.min()))
    pins = sorted(set(pins))
    if len(pins) != dim:
        raise ValueError("Translation pins are not distinct — unexpected P layout.")
    return onp.array(pins, dtype=onp.int64)


def _create_direct_homogenization_solve(problem, bc, P, mesh, dim,
                                        unit_strains_arr, labels, solver_options):
    """Build the differentiable factor-once / solve-many homogenization solve."""
    from ..assembler import (create_matfree_res_J_parametric,
                             create_res_bc_parametric)
    from ..asd import (connectivity_pattern, operator_assembler,
                       reduced_operator_pattern)
    from spineax.cudss.factor_solve import factorize, solve_with

    vec = problem.fes[0].vec
    ndof = problem.num_total_dofs_all_vars

    matfree_res_J = create_matfree_res_J_parametric(problem, symmetric=True)
    res_bc_parametric = create_res_bc_parametric(problem)

    # Reduced operator pattern (values-independent) → static CSR structure.
    R_pat = reduced_operator_pattern(P, connectivity_pattern(problem))
    assemble = operator_assembler(R_pat)
    indptr = np.asarray(R_pat.indptr)
    indices = np.asarray(R_pat.indices)

    # Symmetric pin: zero the pinned rows AND columns, set unit diagonal. Built
    # from the static pattern so it is a cheap masked update of the CSR values.
    pins = _translation_pin_dofs(P, vec, dim)
    ip = onp.asarray(R_pat.indptr); jc = onp.asarray(R_pat.indices)
    rows_np = onp.repeat(onp.arange(len(ip) - 1), onp.diff(ip))
    keep = np.asarray(~(onp.isin(rows_np, pins) | onp.isin(jc, pins)))
    diag_slots = []
    for p in pins:
        hit = onp.where((rows_np == p) & (jc == p))[0]
        if hit.size == 0:
            raise ValueError(f"Diagonal slot for pinned dof {p} missing from the "
                             "reduced pattern.")
        diag_slots.append(int(hit[0]))
    diag_slots = np.asarray(diag_slots)
    pins_j = np.asarray(pins)

    _mtype, _mview = 1, 0                         # cuDSS SYMMETRIC, FULL storage

    def _pinned_data(K_data):
        return np.where(keep, K_data, 0.0).at[diag_slots].set(1.0)

    # custom_vjp linear solve: factor ONCE, solve every RHS, and reuse the same
    # factorization for the adjoint (PᵀKP is symmetric ⇒ Kᵀ = K).
    @jax.custom_vjp
    def _solve_many(data, rhs_batch):
        token = factorize(data, indptr, indices, mtype_id=_mtype, mview_id=_mview)
        return jax.vmap(lambda b: solve_with(token, b))(rhs_batch)

    def _solve_fwd(data, rhs_batch):
        token = factorize(data, indptr, indices, mtype_id=_mtype, mview_id=_mview)
        x = jax.vmap(lambda b: solve_with(token, b))(rhs_batch)
        return x, (token, x)

    def _solve_bwd(res, x_bar):
        token, x = res                           # reuse the forward factorization
        lam = jax.vmap(lambda b: solve_with(token, b))(x_bar)
        # A x = b ⇒ ∂/∂A = -Σ_case λ ⊗ x, gathered onto the CSR slots.
        data_bar = -(lam[:, np.asarray(rows_np)] * x[:, indices]).sum(axis=0)
        return data_bar, lam

    _solve_many.defvjp(_solve_fwd, _solve_bwd)

    def _reduced_operator_data(traced_params):
        # Linear elasticity: J is constant, evaluate the reduced operator at u=0.
        _, J_matvec = matfree_res_J(np.zeros(ndof), traced_params, bc, None)
        K = assemble(lambda v: P.T @ J_matvec(P @ v))
        return _pinned_data(K.data)

    def _case_rhs(unit_strain, traced_params):
        u_macro = macro_displacement(mesh, unit_strain)
        rhs = -(P.T @ res_bc_parametric(u_macro, traced_params, bc, None))
        return rhs.at[pins_j].set(0.0), u_macro

    def _zero_mean(fluct):
        # Re-centre each spatial component to zero volume-mean (nodal mean is a
        # fine gauge fix; C_hom is unaffected by the constant translation).
        f = fluct.reshape(-1, vec)
        return (f - f.mean(axis=0)).reshape(-1)

    def solve(traced_params) -> HomogenizationResult:
        data = _reduced_operator_data(traced_params)
        rhs_batch, u_macros = jax.vmap(
            lambda s: _case_rhs(s, traced_params))(unit_strains_arr)
        sol_reduced = _solve_many(data, rhs_batch)          # (n_cases, n_red)

        def _finish(u_macro, sol_red):
            fluct = _zero_mean(P @ sol_red)
            u_total = u_macro + fluct
            return average_stress(problem, u_total, traced_params, dim), u_total

        columns, u_totals = jax.vmap(_finish)(u_macros, sol_reduced)
        return HomogenizationResult(C_hom=columns.T, u_totals=u_totals,
                                    u_macros=u_macros)

    solve.labels = labels
    solve.unit_strains = unit_strains_arr
    return solve


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_homogenization_solver(
    problem: Any,
    bc: Any,
    P: Any,
    mesh: Any,
    solver_options: KrylovSolverOptions = None,
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
        FEAX DirichletBC. Leave EMPTY for periodic unit cells: the reduced
        system is singular (rigid translations) but consistent, and the Krylov
        solve handles it. Do NOT pin a node to suppress the translations — the
        affine initial guess ``u_macro`` does not satisfy such a pin, which
        corrupts the fluctuation field around the pinned node (invisible for a
        homogeneous cell, wrong C_hom for a heterogeneous one).
    P :
        Prolongation matrix from ``feax.flat.pbc.prolongation_matrix``.
        Shape ``(num_dofs, num_reduced_dofs)``.
    mesh :
        FEAX mesh of the unit cell.
    solver_options : KrylovSolverOptions or DirectSolverOptions, optional
        Solver configuration. Default (``None``) is matrix-free Krylov (CG),
        which handles the singular periodic system directly. Pass
        ``DirectSolverOptions`` (GPU/cuDSS) to use the factor-once / solve-many
        direct path — one zero-mean-regularized factorization of ``PᵀKP`` reused
        across all strain cases and adjoints (machine-precise, faster at
        moderate-to-large cells; falls back to Krylov if cuDSS is unavailable).
    dim : int
        Problem dimension. ``2`` or ``3``. Default: ``3``.

    Returns
    -------
    callable
        ``solve(traced_params) -> HomogenizationResult``

    Examples
    --------
    >>> solve = create_homogenization_solver(problem, bc, P, mesh, dim=3)
    >>> result = solve(traced_params)
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

    # Default: matrix-free Krylov (CG) — robust on the singular periodic system
    # and light on memory at any scale. Passing DirectSolverOptions instead
    # selects the direct factor-once / solve-many path: the reduced operator
    # PᵀKP is identical across all unit-strain cases, so a single (zero-mean
    # regularized) factorization is reused for every case AND every adjoint —
    # machine-precise C_hom, competitive at moderate-to-large cells.
    if solver_options is None:
        solver_options = KrylovSolverOptions()

    if isinstance(solver_options, DirectSolverOptions):
        if has_cudss():
            return _create_direct_homogenization_solve(
                problem, bc, P, mesh, dim, unit_strains_arr, labels, solver_options)
        warnings.warn(
            "DirectSolverOptions requested for homogenization but cuDSS is "
            "unavailable; falling back to matrix-free Krylov (CG).",
            RuntimeWarning)
        solver_options = KrylovSolverOptions()

    _verbose = solver_options.verbose

    # Internal: the raw flat vector feeds average_stress / vmapped strain cases.
    fluctuation_solver = create_solver(problem, bc, solver_options, P=P,
                                       return_solution=False)

    def _single_case(unit_strain, traced_params):
        u_macro = macro_displacement(mesh, unit_strain)
        u_total = fluctuation_solver(traced_params, u_macro)
        sigma_voigt = average_stress(problem, u_total, traced_params, dim)
        return sigma_voigt, u_total, u_macro

    def solve(traced_params) -> HomogenizationResult:
        # lax.map, NOT vmap: the strain cases converge at different iteration
        # counts, and a vmapped Krylov while_loop keeps iterating every case
        # until ALL converge. On a singular reduced system (fully periodic,
        # no Dirichlet pin — rigid translations) those extra iterations
        # numerically corrupt the already-converged cases. lax.map gives each
        # case its own while_loop (still jittable and differentiable).
        columns, u_totals, u_macros = jax.lax.map(
            lambda s: _single_case(s, traced_params), unit_strains_arr)
        # columns: (n_cases, voigt_dim), need transpose for C_hom
        C_hom = columns.T
        return HomogenizationResult(
            C_hom=C_hom,
            u_totals=u_totals,
            u_macros=u_macros,
        )

    solve.labels = labels
    solve.unit_strains = unit_strains_arr

    return solve
