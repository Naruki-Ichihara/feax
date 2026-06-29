"""Reduced (matrix-free) solver path for periodic constraints.

This module contains the P-matrix reduced solve path used by
``feax.solver.create_solver`` when ``P is not None``.
"""

import jax
import jax.numpy as np

from ..assembler import (
    create_matfree_res_J_parametric,
    create_res_bc_parametric,
)
from ..DCboundary import DirichletBC
from .common import _safe_negate, create_iterative_solve_fn
from .options import (
    AMGSolverOptions,
    KrylovSolverOptions,
    MatrixProperty,
    resolve_iterative_solver,
)


def create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options):
    """Create matrix-free reduced solver for periodic boundary conditions.

    The reduced operator ``Pᵀ J_bc P`` is never assembled — it is a Galerkin
    triple product whose sparsity pattern is not known at trace time, so only
    its matrix-free *action* (three matvecs ``Pᵀ(J(P v))``) is available. That
    makes :class:`KrylovSolverOptions` (which needs only a matvec) the only
    supported family here. :class:`AMGSolverOptions` is rejected: AMG needs an
    assembled matrix, which the reduced path does not produce.
    """

    def _reject_amg(opts, role):
        if isinstance(opts, AMGSolverOptions):
            raise TypeError(
                f"AMGSolverOptions is not supported by the reduced (periodic) solver "
                f"for the {role} solve. The reduced operator Pᵀ J P is matrix-free "
                "(never assembled), but AMG requires an assembled matrix. Use "
                "KrylovSolverOptions (e.g. cg) for periodic problems."
            )

    _reject_amg(solver_options, "forward")
    _reject_amg(adjoint_solver_options, "adjoint")

    if not isinstance(solver_options, KrylovSolverOptions):
        raise TypeError(
            "Reduced solver requires KrylovSolverOptions for forward solve, "
            f"got {type(solver_options).__name__}."
        )
    if not isinstance(adjoint_solver_options, KrylovSolverOptions):
        raise TypeError(
            "Reduced solver requires KrylovSolverOptions for adjoint solve, "
            f"got {type(adjoint_solver_options).__name__}."
        )

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
