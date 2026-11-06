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


class _ReducedSystem(NamedTuple):
    """The pinned, periodic-reduced homogenization system of one unit cell,
    as functions of the traced parameters (shared by the single and the
    batched solvers)."""
    operator_data: Any      # tp -> (nnz,) CSR values of the pinned P^T K P
    case_rhs: Any           # (unit_strain, tp) -> ((n_red,) rhs, (ndof,) u_macro)
    zero_mean: Any          # (ndof,) fluctuation -> zero-mean fluctuation
    indptr: Any             # static CSR pattern of the reduced operator
    indices: Any
    rows: Any               # (nnz,) row of every CSR slot (numpy)
    diag_slots: Any         # (n_red,) CSR slot of every diagonal entry
    pins: Any               # pinned reduced dofs (one per translation)


def _reduced_system(problem, bc, P, mesh, dim) -> _ReducedSystem:
    """Build the reduced periodic operator on its static CSR pattern with the
    rigid translations pinned (zero-mean gauge): symmetric positive definite,
    so it can be factorized or solved by CG."""
    from ..assembler import (create_matfree_res_J_parametric,
                             create_res_bc_parametric)
    from ..asd import (connectivity_pattern, operator_assembler,
                       reduced_operator_pattern)

    from ..traced_structure import TracedStructure

    vec = problem.fes[0].vec
    ndof = problem.num_total_dofs_all_vars

    matfree_res_J = create_matfree_res_J_parametric(problem, symmetric=True)
    res_bc_parametric = create_res_bc_parametric(problem)
    # Assemble on the TracedStructure path (the closure path is deprecated).
    # ``free_scratch=False`` keeps the host slot maps so the same problem can
    # still be assembled elsewhere (get_jacobian, create_solver auto-detect).
    ts = TracedStructure.from_problem(problem, free_scratch=False)

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
    diag_all = onp.full(len(ip) - 1, -1, dtype=onp.int64)
    on_diag = onp.flatnonzero(rows_np == jc)
    diag_all[rows_np[on_diag]] = on_diag
    if onp.any(diag_all < 0):
        raise ValueError("The reduced pattern misses diagonal slots.")
    diag_slots = np.asarray(diag_all[pins])
    pins_j = np.asarray(pins)

    def _pinned_data(K_data):
        return np.where(keep, K_data, 0.0).at[diag_slots].set(1.0)

    def operator_data(traced_params):
        # Linear elasticity: J is constant, evaluate the reduced operator at u=0.
        _, J_matvec = matfree_res_J(np.zeros(ndof), traced_params, bc, ts)
        K = assemble(lambda v: P.T @ J_matvec(P @ v))
        return _pinned_data(K.data)

    def case_rhs(unit_strain, traced_params):
        u_macro = macro_displacement(mesh, unit_strain)
        rhs = -(P.T @ res_bc_parametric(u_macro, traced_params, bc, ts))
        return rhs.at[pins_j].set(0.0), u_macro

    def zero_mean(fluct):
        # Re-centre each spatial component to zero volume-mean (nodal mean is a
        # fine gauge fix; C_hom is unaffected by the constant translation).
        f = fluct.reshape(-1, vec)
        return (f - f.mean(axis=0)).reshape(-1)

    return _ReducedSystem(operator_data, case_rhs, zero_mean, indptr, indices, rows_np,
                          np.asarray(diag_all), pins)


def _create_direct_homogenization_solve(problem, bc, P, mesh, dim,
                                        unit_strains_arr, labels, solver_options):
    """Build the differentiable factor-once / solve-many homogenization solve."""
    from feax.solvers._cudss_compat import factorize, solve_with

    rs = _reduced_system(problem, bc, P, mesh, dim)
    indptr, indices, rows_np = rs.indptr, rs.indices, rs.rows

    _mtype, _mview = 1, 0                         # cuDSS SYMMETRIC, FULL storage

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

    def solve(traced_params) -> HomogenizationResult:
        data = rs.operator_data(traced_params)
        rhs_batch, u_macros = jax.vmap(
            lambda s: rs.case_rhs(s, traced_params))(unit_strains_arr)
        sol_reduced = _solve_many(data, rhs_batch)          # (n_cases, n_red)

        def _finish(u_macro, sol_red):
            fluct = rs.zero_mean(P @ sol_red)
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
        ``solve(traced_params) -> HomogenizationResult``. :func:`feax.vmap`
        batches it over designs (``fe.vmap(solve, in_axes=...,
        preconditioner=fe.shared_factor(scaling=True), share=...)``), giving
        the :class:`BatchedHomogenizationSolver` of
        :func:`create_batched_homogenization_solver`.

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
            solve = _create_direct_homogenization_solve(
                problem, bc, P, mesh, dim, unit_strains_arr, labels, solver_options)
            solve._homogenization = (problem, bc, P, mesh, dim)
            return solve
        warnings.warn(
            "DirectSolverOptions requested for homogenization but cuDSS is "
            "unavailable; falling back to matrix-free Krylov (CG).",
            RuntimeWarning)
        solver_options = KrylovSolverOptions()

    _verbose = solver_options.verbose

    # Internal: the raw flat vector feeds average_stress / vmapped strain cases.
    # TracedStructure path (the closure path is deprecated); keep the host slot
    # maps so the same problem can still be assembled elsewhere.
    from ..traced_structure import TracedStructure
    ts = TracedStructure.from_problem(problem, free_scratch=False)
    fluctuation_solver = create_solver(problem, bc, solver_options, projection=P,
                                       return_solution=False, traced_structure=ts)

    def _single_case(unit_strain, traced_params):
        u_macro = macro_displacement(mesh, unit_strain)
        u_total = fluctuation_solver(traced_params, u_macro, traced_structure=ts)
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
    solve._homogenization = (problem, bc, P, mesh, dim)   # lets fe.vmap batch this solver

    return solve


# ---------------------------------------------------------------------------
# Batched homogenization: many unit cells, one shared factorization
# ---------------------------------------------------------------------------

class BatchedHomogenizationResult(NamedTuple):
    """Result of :class:`BatchedHomogenizationSolver`.

    Attributes
    ----------
    C_hom : ndarray, shape (B, 3, 3) or (B, 6, 6)
        Homogenized stiffness of every design (Voigt).
    iterations : ndarray, shape (B, n_cases)
        CG iterations of every design and unit-strain case.
    converged : ndarray, shape (B, n_cases)
        Whether each case reached the tolerance.
    u_totals : ndarray (B, n_cases, num_dofs) or None
        Total displacement fields (only with ``return_fields=True``).
    """
    C_hom: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    u_totals: Any = None


class BatchedHomogenizationSolver:
    """Homogenized stiffness of a batch of unit cells that share one mesh and
    differ in their parameters (density, modulus, ... fields).

    Built by :func:`create_batched_homogenization_solver`. Every design has its
    own reduced periodic operator ``PᵀK_bP`` (translations pinned, SPD) and the
    ``n_cases`` unit-strain right-hand sides. The batch is a
    :class:`feax.LinearSystemBatch` solved by :func:`feax.vmap`'s lhs engine:
    all ``B × n_cases`` systems by conjugate gradients, preconditioned -- per
    group of ``share`` designs -- by ONE cuDSS factorization of the group's
    mean operator (:func:`feax.shared_factor`, diagonally scaled per design by
    default), or by Jacobi without cuDSS.

    ``solver(traced_params, factors=None) -> BatchedHomogenizationResult`` is
    jit-compiled and differentiable w.r.t. ``traced_params``;
    ``solver.evaluate`` is the same function unjitted (to compose with a
    caller's own ``jax.grad`` / ``jax.jit``). ``factorize(traced_params)``
    returns the group factorizations for reuse over many evaluations;
    ``operator_signature(traced_params)`` gives a per-design feature vector for
    :func:`feax.gene.similarity_order`.
    """

    def __init__(self, problem, bc, P, mesh, dim, *, in_axes, share, chunk, preconditioner,
                 tol, atol, maxiter, return_fields, scaling=True):
        from ..solvers.preconditioners import jacobi, shared_factor
        from ..vmap import LinearSystemBatch, _axes_tree, _is_axis_leaf, _to_front, vmap

        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        pc = None
        if callable(preconditioner):                 # fe.shared_factor(...) / fe.jacobi(...)
            pc = preconditioner
            if getattr(pc, "scaling", None) is not None:
                preconditioner, scaling = "shared_factor", pc.scaling
            else:
                preconditioner = "jacobi"
        if preconditioner is None:
            preconditioner = "shared_factor" if has_cudss() else "jacobi"
        if preconditioner not in ("shared_factor", "jacobi"):
            raise ValueError("preconditioner must be 'shared_factor', 'jacobi', or a "
                             "preconditioner object (feax.shared_factor / feax.jacobi).")
        if preconditioner == "shared_factor" and not has_cudss():
            raise ImportError("preconditioner='shared_factor' needs cuDSS (spineax); use 'jacobi'.")

        self.problem, self.P, self.mesh, self.dim = problem, P, mesh, dim
        self.in_axes, self.share, self.chunk = in_axes, share, chunk
        self.preconditioner, self.scaling = preconditioner, bool(scaling)
        self.return_fields = return_fields
        self.unit_strains = UNIT_STRAINS_3D if dim == 3 else UNIT_STRAINS_2D
        self.labels = (('eps11', 'eps22', 'eps33', 'gam23', 'gam13', 'gam12') if dim == 3
                       else ('eps11', 'eps22', 'gam12'))
        self._axes_tree, self._to_front, self._is_axis_leaf = _axes_tree, _to_front, _is_axis_leaf

        rs = _reduced_system(problem, bc, P, mesh, dim)
        self._rs = rs
        self._u_macros = jax.vmap(lambda s_: macro_displacement(mesh, s_))(self.unit_strains)
        strains = self.unit_strains

        def front_of(tp):
            return jax.tree_util.tree_map(lambda ax: None if ax is None else 0,
                                          _axes_tree(in_axes, tp), is_leaf=_is_axis_leaf)

        def values(tp_c, u0, ts):
            return jax.vmap(rs.operator_data, in_axes=(front_of(tp_c),))(tp_c)

        def rhs(tp_c, u0, ts):
            return jax.vmap(lambda t: jax.vmap(lambda e: rs.case_rhs(e, t)[0])(strains),
                            in_axes=(front_of(tp_c),))(tp_c)

        spec = LinearSystemBatch(values=values, rhs=rhs,
                                 pattern=lambda ts: (rs.indptr, rs.indices),
                                 diag_slots=lambda ts: rs.diag_slots,
                                 n_rhs=len(strains))
        if pc is None:
            pc = (shared_factor("spd", scaling=self.scaling) if preconditioner == "shared_factor"
                  else jacobi())
        from ..solvers.options import KrylovSolverOptions as _KO
        self._batched = vmap(spec, in_axes=in_axes, preconditioner=pc, share=share, chunk=chunk,
                             options=_KO(solver="cg", tol=tol, atol=atol, maxiter=maxiter),
                             return_info=True)
        self._front_of = front_of
        self._jit = {}

    def evaluate(self, traced_params, factors=None) -> BatchedHomogenizationResult:
        """The batched homogenization, unjitted (differentiable)."""
        X, info = self._batched.solve(traced_params, factors)          # (B, K, n_red)
        tp, front, _ = self._to_front(traced_params, self._axes_tree(self.in_axes, traced_params))
        rs, P, dim, problem = self._rs, self.P, self.dim, self.problem

        def finish(x_b, t):                                           # x_b (K, n)
            def case(x, u_macro):
                u_total = u_macro + rs.zero_mean(P @ x)
                return average_stress(problem, u_total, t, dim), u_total
            cols, u_tot = jax.vmap(case)(x_b, self._u_macros)
            return cols.T, u_tot
        C, U = jax.vmap(finish, in_axes=(0, front))(X, tp)
        return BatchedHomogenizationResult(C, info.iterations, info.converged,
                                           U if self.return_fields else None)

    def __call__(self, traced_params, factors=None) -> BatchedHomogenizationResult:
        """Homogenized stiffness of every design. ``factors`` (from
        :meth:`factorize`, same batch size and ``share``) precondition without
        assembling a mean or factorizing."""
        if factors is not None and self.preconditioner != "shared_factor":
            raise ValueError("factors= needs preconditioner='shared_factor'.")
        key = factors is None
        if key not in self._jit:
            self._jit[key] = jax.jit(self.evaluate)
        result = self._jit[key](traced_params, factors)
        # Safety net: a run whose CG did not converge is re-evaluated once. On
        # one platform (GB10, driver 580, XLA 0.11-dev) the compiled program
        # intermittently -- mostly on its first execution -- produced a
        # corrupted mean operator, visible as a non-converging CG; a second
        # execution was correct, and XLA_FLAGS=--xla_gpu_deterministic_ops=true
        # makes it rare (not impossible). A genuinely ill-conditioned batch
        # fails both times and raises.
        if isinstance(result.converged, jax.core.Tracer):      # under jit / grad: no host check
            return result
        if not bool(onp.all(onp.asarray(result.converged))):
            warnings.warn(
                "feax batched homogenization: the batched CG did not converge for every "
                "design (max iterations "
                f"{int(onp.asarray(result.iterations).max())}); re-evaluating once. If this "
                "recurs, run with XLA_FLAGS=--xla_gpu_deterministic_ops=true (see "
                "create_batched_homogenization_solver) or raise maxiter / lower share.",
                stacklevel=2)
            result = self._jit[key](traced_params, factors)
            if not bool(onp.all(onp.asarray(result.converged))):
                raise RuntimeError(
                    "feax batched homogenization: CG did not converge for every design in two "
                    "evaluations (max iterations "
                    f"{int(onp.asarray(result.iterations).max())}). Raise maxiter, lower share "
                    "(or use preconditioner='jacobi'), or set "
                    "XLA_FLAGS=--xla_gpu_deterministic_ops=true if the failure is intermittent.")
        return result

    def factorize(self, traced_params):
        """One factorization of the mean reduced operator per group of
        ``share`` designs (one cuDSS entry each, kept alive), for
        ``solver(traced_params, factors=...)``. Keep the batch order and size
        while reusing them; set ``SPINEAX_FACTOR_CACHE`` (before spineax is
        imported) to at least the number of groups."""
        if self.preconditioner != "shared_factor":
            raise ValueError("factorize needs preconditioner='shared_factor'.")
        return self._batched.factorize(traced_params)

    def operator_signature(self, traced_params):
        """``(B, n_red)``: log of the diagonal of every design's reduced
        operator -- a parametrization-independent feature vector for
        :func:`feax.gene.similarity_order`."""
        return self._batched.operator_signature(traced_params)


def create_batched_homogenization_solver(problem, bc, P, mesh, dim: int = 3, *, in_axes=0,
                                         share: int = None, chunk: int = None, preconditioner: str = None,
                                         tol: float = 1e-8, atol: float = 0.0, maxiter: int = 2000,
                                         return_fields: bool = False,
                                         scaling: bool = True) -> BatchedHomogenizationSolver:
    """Batched computational homogenization: ``C_hom`` of many unit cells at once.

    The cells share the mesh, the periodic prolongation ``P`` and the material
    law; they differ in ``traced_params`` (e.g. a lattice density or modulus
    field per design), which carries the batch along ``in_axes`` (as in
    :func:`feax.vmap`; e.g. ``fe.TracedParams(volume_vars=(0, None))`` for a
    batched modulus and a shared Poisson ratio).

    Instead of factorizing every design's reduced operator ``PᵀK_bP`` (B
    factorizations, memory-bound quickly), the ``B × n_cases`` unit-strain
    systems are solved together by CG preconditioned by one factorization of
    the mean operator per group of ``share`` designs -- the mean-based
    preconditioner of :func:`feax.shared_factor`, run by :func:`feax.vmap`
    on a :class:`feax.LinearSystemBatch` (the same engine as a batch of
    topology-optimization designs). It is exact for identical
    designs and degrades as they spread: watch ``result.iterations`` and lower
    ``share`` (after ordering the batch by :meth:`operator_signature` and
    :func:`feax.gene.similarity_order`) when the counts grow.

    Parameters
    ----------
    problem, bc, P, mesh, dim :
        As for :func:`create_homogenization_solver` (fully periodic cell,
        empty Dirichlet BC; the translations are pinned internally).
    in_axes : int, None or a TracedParams-shaped prefix
        Which leaves of ``traced_params`` carry the batch axis.
    share : int, optional
        Designs per shared factorization (default: the whole batch).
    chunk : int, optional
        Designs assembled and solved at a time (memory bound: ``chunk``
        reduced operators held at once; the group mean is accumulated chunk by
        chunk, as in :func:`feax.vmap`). Default: ``share``.
    preconditioner : {"shared_factor", "jacobi"} or preconditioner object, optional
        Default: ``"shared_factor"`` with cuDSS, else ``"jacobi"``. A
        :func:`feax.shared_factor` / :func:`feax.jacobi` object is used as is
        (its ``scaling`` wins over the argument below).
    tol, atol, maxiter :
        CG stopping criterion per case: ``||r|| <= max(tol ||b||, atol)``.
    return_fields : bool
        Also return the total displacement fields ``(B, n_cases, num_dofs)``.
        The solver call checks ``converged`` and re-evaluates once when a
        design did not converge (see the note on platform reliability below);
        ``solver.evaluate`` (for use under your own ``jax.grad`` / ``jax.jit``)
        performs no such check -- inspect ``result.converged`` yourself.
    scaling : bool
        Shared factorization with a per-design symmetric diagonal scaling,
        ``M_b = S_b K̄⁻¹ S_b`` with ``S_b = sqrt(diag K̄ / diag K_b)`` (default).
        It costs one diagonal product per CG iteration and keeps the shared
        preconditioner effective when the designs' stiffness differs by orders
        of magnitude in places (lattice struts vs void); ``False`` is the plain
        mean preconditioner of :func:`feax.shared_factor`.

    Returns
    -------
    BatchedHomogenizationSolver
        ``solver(traced_params, factors=None) -> BatchedHomogenizationResult``,
        plus ``solver.factorize(traced_params)`` and
        ``solver.operator_signature(traced_params)``.

    Notes
    -----
    On one platform (NVIDIA GB10 with driver 580 and a JAX 0.11 development
    build) the compiled gradient of this solver intermittently -- mostly on the
    first execution of the executable -- produced a corrupted mean operator
    (a race in XLA:GPU kernels outside feax; feax's own assembly uses no
    atomics), visible as a non-converging batched CG. It was seen with the
    operators assembled inside the solve's custom VJP; the shipped structure
    (operators assembled outside, see :func:`feax.vmap`) has not reproduced
    it. ``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` makes it rare at some
    cost in speed; the solver call also re-evaluates once when ``converged``
    is not all true.
    """
    return BatchedHomogenizationSolver(problem, bc, P, mesh, dim, in_axes=in_axes, share=share,
                                       chunk=chunk, preconditioner=preconditioner, tol=tol, atol=atol,
                                       maxiter=maxiter, return_fields=return_fields,
                                       scaling=scaling)
