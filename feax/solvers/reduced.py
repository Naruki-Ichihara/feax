"""Reduced (matrix-free) solver path for periodic constraints.

This module contains the P-matrix reduced solve path used by
``feax.solver.create_solver`` when ``P is not None``.
"""

import jax
import jax.numpy as np

from ..assembler import (
    create_J_bc_parametric,
    create_res_bc_parametric,
)
from ..DCboundary import DirichletBC
from .common import _safe_negate, create_iterative_solve_fn
from .options import IterativeSolverOptions, MatrixProperty, resolve_iterative_solver


def create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options):
    """Create matrix-free reduced solver for periodic boundary conditions."""

    if not isinstance(solver_options, IterativeSolverOptions):
        raise TypeError(
            "Reduced solver requires IterativeSolverOptions for forward solve, "
            f"got {type(solver_options).__name__}."
        )
    if not isinstance(adjoint_solver_options, IterativeSolverOptions):
        raise TypeError(
            "Reduced solver requires IterativeSolverOptions for adjoint solve, "
            f"got {type(adjoint_solver_options).__name__}."
        )

    J_bc_parametric = create_J_bc_parametric(problem)
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

    def create_reduced_matvec_parametric(sol_full, internal_vars, effective_bc):
        J_full = J_bc_parametric(sol_full, internal_vars, effective_bc)

        def reduced_matvec(v_reduced):
            v_full = P @ v_reduced
            Jv_full = J_full @ v_full
            Jv_reduced = P.T @ Jv_full
            return Jv_reduced

        return reduced_matvec

    def compute_reduced_residual_parametric(sol_full, internal_vars, effective_bc):
        res_full = res_bc_parametric(sol_full, internal_vars, effective_bc)
        return P.T @ res_full

    def reduced_solve_fn(internal_vars, initial_guess_full, effective_bc):
        res_reduced = compute_reduced_residual_parametric(
            initial_guess_full, internal_vars, effective_bc
        )
        J_reduced_matvec = create_reduced_matvec_parametric(
            initial_guess_full, internal_vars, effective_bc
        )

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

        # sol already includes initial_guess (total solution)
        J_full = J_bc_parametric(sol, internal_vars, effective_bc)
        rhs_reduced = P.T @ v

        def adjoint_matvec(adjoint_reduced):
            adjoint_full = P @ adjoint_reduced
            Jt_adjoint_full = J_full.T @ adjoint_full
            return P.T @ Jt_adjoint_full

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
