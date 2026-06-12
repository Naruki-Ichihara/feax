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
from .options import KrylovSolverOptions, MatrixProperty, resolve_iterative_solver


def create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options):
    """Create matrix-free reduced solver for periodic boundary conditions."""

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

    def reduced_solve_fn(internal_vars, initial_guess_full, effective_bc):
        # One matfree pass returns the BC-applied residual and the tangent
        # matvec (J_bc @ w via JVP); the reduced operator is Pᵀ J_bc P.
        res_full, J_matvec = matfree_res_J(
            initial_guess_full, internal_vars, effective_bc
        )
        res_reduced = P.T @ res_full

        def J_reduced_matvec(v_reduced):
            return P.T @ J_matvec(P @ v_reduced)

        x0 = np.zeros(P.shape[1])
        sol_reduced = fwd_linear_solve_fn(J_reduced_matvec, -res_reduced, x0)
        sol_full = initial_guess_full + P @ sol_reduced
        return sol_full, None

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess, effective_bc):
        return reduced_solve_fn(internal_vars, initial_guess, effective_bc)[0]

    def f_fwd(internal_vars, initial_guess, effective_bc):
        sol = differentiable_solve(internal_vars, initial_guess, effective_bc)
        return sol, (internal_vars, sol, initial_guess, effective_bc)

    def f_bwd(res, v):
        internal_vars, sol, initial_guess, effective_bc = res

        # sol already includes initial_guess (total solution). Symmetric BC ⇒
        # J_bc is symmetric ⇒ Jᵀ = J, so the forward matvec serves the adjoint.
        _, J_matvec = matfree_res_J(sol, internal_vars, effective_bc)
        rhs_reduced = P.T @ v

        def adjoint_matvec(adjoint_reduced):
            return P.T @ J_matvec(P @ adjoint_reduced)

        x0_reduced = np.zeros_like(rhs_reduced)
        adjoint_reduced = adj_linear_solve_fn(adjoint_matvec, rhs_reduced, x0_reduced)

        adjoint_full = P @ adjoint_reduced

        # VJP of residual w.r.t. internal_vars and bc
        u_total_list = problem.unflatten_fn_sol_list(sol)
        adjoint_list = problem.unflatten_fn_sol_list(adjoint_full)

        def res_fn(iv, bc_arg):
            dofs = jax.flatten_util.ravel_pytree(u_total_list)[0]
            return problem.unflatten_fn_sol_list(
                res_bc_parametric(dofs, iv, bc_arg)
            )

        _, f_vjp = jax.vjp(res_fn, internal_vars, effective_bc)
        vjp_iv, vjp_bc = f_vjp(adjoint_list)
        vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
        vjp_bc = jax.tree_util.tree_map(_safe_negate, vjp_bc)

        return (vjp_iv, None, vjp_bc)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    from ..utils import zero_like_initial_guess
    default_initial_guess = zero_like_initial_guess(problem, bc)

    def solver_wrapper(internal_vars, initial_guess=None, bc=None):
        effective_bc = bc if isinstance(bc, DirichletBC) else _default_bc
        ig = default_initial_guess if initial_guess is None else initial_guess
        return differentiable_solve(internal_vars, ig, effective_bc)

    return solver_wrapper
