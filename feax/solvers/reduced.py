"""Reduced solver path for periodic constraints.

This module contains the P-matrix reduced solve path used by
``feax.solver.create_solver`` when ``P is not None``.

Two operator representations:

* **matrix-free** (``KrylovSolverOptions``): the reduced operator ``PᵀJP`` is
  applied as three matvecs ``Pᵀ(J(P v))`` — never assembled.
* **assembled** (``DirectSolverOptions`` / ``AMGSolverOptions``): the reduced
  pattern is computed by a boolean triple product of the known ``P`` and
  connectivity patterns (:func:`feax.asd.reduced_operator_pattern`), and the
  operator is materialized from its matrix-free action by colored probing
  (:func:`feax.asd.operator_assembler`) — enabling direct factorization and
  AMG preconditioning for periodic problems.
"""

import dataclasses

import numpy as onp

import jax
import jax.numpy as np

from ..assembler import (
    create_matfree_res_J_parametric,
    create_res_bc_parametric,
)
from ..DCboundary import DirichletBC
from .common import (
    _safe_negate,
    create_iterative_solve_fn,
    create_linear_solve_fn,
)
from .options import (
    AMGSolverOptions,
    DirectSolverOptions,
    KrylovSolverOptions,
    MatrixProperty,
    resolve_direct_solver,
    resolve_iterative_solver,
)


def _validate_bc_pairing(P, bc):
    """Reject Dirichlet rows that land on part of a periodic equivalence class.

    Symmetric elimination replaces a Dirichlet row/column by identity in the
    FULL system; the reduction ``Pᵀ(·)P`` then sums that identity row with the
    physical rows of the dof's periodic partners, so the constraint is
    silently diluted instead of enforced. A pin is exact only when the dof is
    unpaired (its reduced column has a single member — e.g. an interior node)
    or when its WHOLE class is pinned (every partner row is an identity row
    too, as with a Dirichlet face crossing a periodic pairing).
    """
    bc_rows = onp.asarray(bc.bc_rows).ravel()
    if bc_rows.size == 0:
        return
    idx = onp.asarray(P.indices)
    rows, cols = idx[:, 0], idx[:, 1]
    col_of = onp.full(P.shape[0], -1, onp.int64)
    col_of[rows] = cols
    class_size = onp.bincount(cols, minlength=P.shape[1])
    bc_mask = onp.zeros(P.shape[0], bool)
    bc_mask[bc_rows] = True
    bc_per_class = onp.bincount(cols[bc_mask[rows]], minlength=P.shape[1])

    c = col_of[bc_rows]
    bad = bc_rows[(c >= 0) & (class_size[c] > 1) & (bc_per_class[c] < class_size[c])]
    if bad.size:
        raise ValueError(
            f"Dirichlet BC on periodically paired dof(s) {bad[:10].tolist()}"
            f"{'...' if bad.size > 10 else ''}: the PᵀJP reduction folds the "
            "eliminated row together with its periodic partners, so the "
            "constraint is NOT enforced. Pin an unpaired dof (e.g. an "
            "interior node) or constrain the entire periodic class (all "
            "paired faces/edges/corners of that dof).")


def _project_reduced_nullspace(P, B_full):
    """Project a full-space near-null-space onto the reduced space.

    ``B_red = diag(1/colsum(P)) PᵀB`` — the (weighted) pseudo-inverse pullback,
    exact for the 0/1 master-slave prolongations produced by
    ``feax.flat.pbc.prolongation_matrix``.
    """
    import scipy.sparse as sp
    idx = onp.asarray(P.indices)
    P_sp = sp.coo_matrix((onp.asarray(P.data), (idx[:, 0], idx[:, 1])),
                         shape=P.shape).tocsr()
    w = onp.asarray(P_sp.T @ onp.ones(P.shape[0]))
    return (P_sp.T @ onp.asarray(B_full)) / onp.maximum(w, 1e-30)[:, None]


def _reduced_amg_options(options, problem, bc, P, n_red):
    """AMG options with the near-null-space mapped into the reduced space."""
    from .amg import rigid_body_modes
    ns = options.near_nullspace
    if ns is None or (isinstance(ns, str) and ns.lower() == "rigid_body"):
        B = rigid_body_modes(problem, bc)
        ns_red = _project_reduced_nullspace(P, B) if B is not None else "constant"
    elif isinstance(ns, str):
        ns_red = ns                              # "constant" / "adaptive_sa"
    else:
        ns = onp.asarray(ns)
        if ns.shape[0] == P.shape[0]:
            ns_red = _project_reduced_nullspace(P, ns)
        elif ns.shape[0] == n_red:
            ns_red = ns
        else:
            raise ValueError(
                f"near_nullspace has {ns.shape[0]} rows; expected the full "
                f"({P.shape[0]}) or reduced ({n_red}) DOF count.")
    return dataclasses.replace(options, near_nullspace=ns_red)


def _make_reduced_side_solve(problem, bc, P, options, n_red, namespace):
    """One side (forward/adjoint) of the reduced solve.

    Returns ``(solve, assembled)``: ``assembled=True`` means ``solve(K, b)``
    consumes the materialized reduced ``CSRMatrix``; ``False`` means
    ``solve(matvec, b)`` runs matrix-free Krylov.
    """
    if isinstance(options, DirectSolverOptions):
        # Explicit solver choices are trusted (as in create_solver); only
        # "auto" goes through backend detection.
        resolved = options if options.solver != "auto" else resolve_direct_solver(
            options, MatrixProperty.SPD, matrix_view=problem.matrix_view)
        fn = create_linear_solve_fn(resolved, cache_namespace=namespace)
        return (lambda K, b: fn(K, b, None)), True
    if isinstance(options, AMGSolverOptions):
        import jax.scipy.sparse.linalg as jsla
        from .amg import build_amg_preconditioner
        opts_red = _reduced_amg_options(options, problem, bc, P, n_red)
        name = opts_red.solver if opts_red.solver != "auto" else "cg"
        krylov = getattr(jsla, name)

        def solve(K, b):
            M = build_amg_preconditioner(problem, bc, K, opts_red)
            kw = dict(x0=np.zeros_like(b), M=M, tol=opts_red.tol,
                      atol=opts_red.atol, maxiter=opts_red.maxiter)
            if name == "gmres":
                kw["restart"] = opts_red.restart or 50
            x, _ = krylov(lambda v: K @ v, b, **kw)
            return x
        return solve, True
    # KrylovSolverOptions: matrix-free
    resolved = resolve_iterative_solver(options, MatrixProperty.SPD)
    fn = create_iterative_solve_fn(resolved)
    return (lambda matvec, b: fn(matvec, b, np.zeros_like(b))), False


def create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options):
    """Create the reduced solver for periodic boundary conditions.

    ``KrylovSolverOptions`` runs fully matrix-free (three matvecs per Krylov
    iteration). ``DirectSolverOptions`` / ``AMGSolverOptions`` assemble the
    reduced operator ``PᵀJP``: its sparsity is the boolean triple product of
    the known patterns, its values come from colored probes of the matrix-free
    action (``num_colors`` matvecs, once per solve) — see :mod:`feax.asd`.
    Periodic operators are symmetric after symmetric elimination, so the
    adjoint reuses the same (assembled or matrix-free) operator.
    """
    if not isinstance(solver_options,
                      (KrylovSolverOptions, DirectSolverOptions, AMGSolverOptions)):
        raise TypeError(
            "Reduced solver requires Krylov/Direct/AMG solver options for the "
            f"forward solve, got {type(solver_options).__name__}.")
    if not isinstance(adjoint_solver_options,
                      (KrylovSolverOptions, DirectSolverOptions, AMGSolverOptions)):
        raise TypeError(
            "Reduced solver requires Krylov/Direct/AMG solver options for the "
            f"adjoint solve, got {type(adjoint_solver_options).__name__}.")

    _validate_bc_pairing(P, bc)

    _fwd_assembled_req = not isinstance(solver_options, KrylovSolverOptions)
    _adj_assembled_req = not isinstance(adjoint_solver_options, KrylovSolverOptions)
    if _fwd_assembled_req or _adj_assembled_req:
        return _create_assembled_reduced_solver(
            problem, bc, P, solver_options, adjoint_solver_options)

    # The reduced problem is always Krylov and never extracts a preconditioner
    # from the operator, so the BC-applied tangent is supplied matrix-free (a
    # residual JVP) — no Jacobian assembly. Periodic problems are SPD after
    # symmetric Dirichlet elimination, so the same matvec serves the adjoint
    # (Jᵀ = J).
    matfree_res_J = create_matfree_res_J_parametric(problem, symmetric=True)
    res_bc_parametric = create_res_bc_parametric(problem)

    _default_bc = bc

    resolved_solver_options = resolve_iterative_solver(solver_options, MatrixProperty.SPD)
    if adjoint_solver_options is solver_options:
        resolved_adjoint_options = resolved_solver_options
    else:
        resolved_adjoint_options = resolve_iterative_solver(adjoint_solver_options, MatrixProperty.SPD)

    fwd_linear_solve_fn = create_iterative_solve_fn(resolved_solver_options)
    if resolved_adjoint_options is resolved_solver_options:
        adj_linear_solve_fn = fwd_linear_solve_fn
    else:
        adj_linear_solve_fn = create_iterative_solve_fn(resolved_adjoint_options)

    def reduced_solve_fn(traced_params, initial_guess_full, effective_bc, ts):
        # One matfree pass returns the BC-applied residual and the tangent
        # matvec (J_bc @ w via JVP); the reduced operator is Pᵀ J_bc P.
        res_full, J_matvec = matfree_res_J(
            initial_guess_full, traced_params, effective_bc, ts
        )
        res_reduced = P.T @ res_full

        def J_reduced_matvec(v_reduced):
            return P.T @ J_matvec(P @ v_reduced)

        x0 = np.zeros(P.shape[1])
        sol_reduced = fwd_linear_solve_fn(J_reduced_matvec, -res_reduced, x0)
        sol_full = initial_guess_full + P @ sol_reduced
        return sol_full, None

    @jax.custom_vjp
    def differentiable_solve(traced_params, initial_guess, effective_bc, ts):
        return reduced_solve_fn(traced_params, initial_guess, effective_bc, ts)[0]

    def f_fwd(traced_params, initial_guess, effective_bc, ts):
        sol = differentiable_solve(traced_params, initial_guess, effective_bc, ts)
        return sol, (traced_params, sol, initial_guess, effective_bc, ts)

    def f_bwd(res, v):
        traced_params, sol, initial_guess, effective_bc, ts = res

        # sol already includes initial_guess (total solution). Symmetric BC ⇒
        # J_bc is symmetric ⇒ Jᵀ = J, so the forward matvec serves the adjoint.
        _, J_matvec = matfree_res_J(sol, traced_params, effective_bc, ts)
        rhs_reduced = P.T @ v

        def adjoint_matvec(adjoint_reduced):
            return P.T @ J_matvec(P @ adjoint_reduced)

        x0_reduced = np.zeros_like(rhs_reduced)
        adjoint_reduced = adj_linear_solve_fn(adjoint_matvec, rhs_reduced, x0_reduced)

        adjoint_full = P @ adjoint_reduced

        # VJP of residual w.r.t. traced_params and bc
        u_total_list = problem.unflatten_fn_sol_list(sol)
        adjoint_list = problem.unflatten_fn_sol_list(adjoint_full)

        def res_fn(tp, bc_arg):
            dofs = jax.flatten_util.ravel_pytree(u_total_list)[0]
            return problem.unflatten_fn_sol_list(
                res_bc_parametric(dofs, tp, bc_arg, ts)
            )

        _, f_vjp = jax.vjp(res_fn, traced_params, effective_bc)
        vjp_iv, vjp_bc = f_vjp(adjoint_list)
        vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
        vjp_bc = jax.tree_util.tree_map(_safe_negate, vjp_bc)

        return (vjp_iv, None, vjp_bc, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    from ..utils import zero_like_initial_guess
    default_initial_guess = zero_like_initial_guess(problem, bc)

    def solver_wrapper(traced_params, initial_guess=None, bc=None, traced_structure=None):
        effective_bc = bc if isinstance(bc, DirichletBC) else _default_bc
        ig = default_initial_guess if initial_guess is None else initial_guess
        return differentiable_solve(traced_params, ig, effective_bc, traced_structure)

    return solver_wrapper


def _create_assembled_reduced_solver(problem, bc, P, solver_options,
                                     adjoint_solver_options):
    """Reduced solver that ASSEMBLES ``PᵀJP`` (direct / AMG solves).

    Pattern: boolean triple product of the concrete ``P`` and the mesh
    connectivity (:func:`feax.asd.reduced_operator_pattern` — an exact
    superset). Values: colored probing of the matrix-free action
    ``v ↦ Pᵀ(J(Pv))`` (:func:`feax.asd.operator_assembler`,
    ``num_colors ≈ max row nnz`` matvecs per assembly). Runs eagerly (host
    orchestration); each solve re-materializes the current operator.
    """
    from ..asd import (
        connectivity_pattern,
        operator_assembler,
        reduced_operator_pattern,
    )

    matfree_res_J = create_matfree_res_J_parametric(problem, symmetric=True)
    res_bc_parametric = create_res_bc_parametric(problem)

    n_red = int(P.shape[1])
    R_pat = reduced_operator_pattern(P, connectivity_pattern(problem))
    assemble = operator_assembler(R_pat)

    fwd_solve, fwd_assembled = _make_reduced_side_solve(
        problem, bc, P, solver_options, n_red, "reduced-forward")
    if adjoint_solver_options is solver_options:
        adj_solve, adj_assembled = fwd_solve, fwd_assembled
    else:
        adj_solve, adj_assembled = _make_reduced_side_solve(
            problem, bc, P, adjoint_solver_options, n_red, "reduced-adjoint")

    def _operator(sol_full, traced_params, effective_bc, ts):
        res_full, J_matvec = matfree_res_J(sol_full, traced_params, effective_bc, ts)
        matvec = lambda v: P.T @ J_matvec(P @ v)
        return res_full, matvec

    def reduced_solve_fn(traced_params, initial_guess_full, effective_bc, ts):
        res_full, matvec = _operator(initial_guess_full, traced_params, effective_bc, ts)
        res_reduced = P.T @ res_full
        op = assemble(matvec) if fwd_assembled else matvec
        sol_reduced = fwd_solve(op, -res_reduced)
        return initial_guess_full + P @ sol_reduced, None

    @jax.custom_vjp
    def differentiable_solve(traced_params, initial_guess, effective_bc, ts):
        return reduced_solve_fn(traced_params, initial_guess, effective_bc, ts)[0]

    def f_fwd(traced_params, initial_guess, effective_bc, ts):
        sol = differentiable_solve(traced_params, initial_guess, effective_bc, ts)
        return sol, (traced_params, sol, effective_bc, ts)

    def f_bwd(res, v):
        traced_params, sol, effective_bc, ts = res

        # Symmetric BC elimination ⇒ PᵀJP symmetric ⇒ the adjoint reuses the
        # forward operator (assembled at the converged state, never stale).
        _, matvec = _operator(sol, traced_params, effective_bc, ts)
        rhs_reduced = P.T @ v
        op = assemble(matvec) if adj_assembled else matvec
        adjoint_reduced = adj_solve(op, rhs_reduced)
        adjoint_full = P @ adjoint_reduced

        u_total_list = problem.unflatten_fn_sol_list(sol)
        adjoint_list = problem.unflatten_fn_sol_list(adjoint_full)

        def res_fn(tp, bc_arg):
            dofs = jax.flatten_util.ravel_pytree(u_total_list)[0]
            return problem.unflatten_fn_sol_list(
                res_bc_parametric(dofs, tp, bc_arg, ts)
            )

        _, f_vjp = jax.vjp(res_fn, traced_params, effective_bc)
        vjp_iv, vjp_bc = f_vjp(adjoint_list)
        vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
        vjp_bc = jax.tree_util.tree_map(_safe_negate, vjp_bc)

        return (vjp_iv, None, vjp_bc, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    from ..utils import zero_like_initial_guess
    default_initial_guess = zero_like_initial_guess(problem, bc)

    def solver_wrapper(traced_params, initial_guess=None, bc=None, traced_structure=None):
        effective_bc = bc if isinstance(bc, DirichletBC) else default_bc_ref
        ig = default_initial_guess if initial_guess is None else initial_guess
        return differentiable_solve(traced_params, ig, effective_bc, traced_structure)

    default_bc_ref = bc
    return solver_wrapper
