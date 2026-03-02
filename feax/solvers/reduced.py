"""Reduced (matrix-free) solver path for periodic constraints.

This module contains the P-matrix reduced solve path used by
``feax.solver.create_solver`` when ``P is not None``.
"""

import jax
import jax.numpy as jnp

from ..assembler import create_J_bc_function, create_res_bc_function
from .options import IterativeSolverOptions, MatrixProperty, resolve_iterative_solver
from .common import _safe_negate, create_iterative_solve_fn


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

    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)

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

    def create_reduced_matvec(sol_full, internal_vars):
        J_full = J_bc_func(sol_full, internal_vars)

        def reduced_matvec(v_reduced):
            v_full = P @ v_reduced
            Jv_full = J_full @ v_full
            Jv_reduced = P.T @ Jv_full
            return Jv_reduced

        return reduced_matvec

    def compute_reduced_residual(sol_full, internal_vars):
        res_full = res_bc_func(sol_full, internal_vars)
        return P.T @ res_full

    def reduced_solve_fn(internal_vars, initial_guess_full):
        res_reduced = compute_reduced_residual(initial_guess_full, internal_vars)
        J_reduced_matvec = create_reduced_matvec(initial_guess_full, internal_vars)

        x0 = jnp.zeros(P.shape[1])
        sol_reduced = fwd_linear_solve_fn(J_reduced_matvec, -res_reduced, x0)
        sol_full = P @ sol_reduced
        return sol_full, None

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        return reduced_solve_fn(internal_vars, initial_guess)[0]

    def f_fwd(internal_vars, initial_guess):
        sol = differentiable_solve(internal_vars, initial_guess)
        # Save initial_guess so f_bwd can reconstruct total displacement:
        # u_total = sol (= P @ u'_red) + initial_guess
        return sol, (internal_vars, sol, initial_guess)

    def f_bwd(res, v):
        internal_vars, sol, initial_guess = res

        # Evaluate Jacobian and VJP at total displacement u_total,
        # not just periodic fluctuation P @ u'_red.
        u_total = sol + initial_guess
        J_full = J_bc_func(u_total, internal_vars)
        rhs_reduced = P.T @ v

        def adjoint_matvec(adjoint_reduced):
            adjoint_full = P @ adjoint_reduced
            Jt_adjoint_full = J_full.T @ adjoint_full
            return P.T @ Jt_adjoint_full

        x0_reduced = jnp.zeros_like(rhs_reduced)
        adjoint_reduced = adj_linear_solve_fn(adjoint_matvec, rhs_reduced, x0_reduced)

        adjoint_full = P @ adjoint_reduced

        def constraint_fn(dofs, internal_vars):
            return res_bc_func(dofs, internal_vars)

        def constraint_fn_sol_to_sol(sol_list, internal_vars):
            dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
            con_vec = constraint_fn(dofs, internal_vars)
            return problem.unflatten_fn_sol_list(con_vec)

        def get_partial_params_c_fn(sol_list):
            def partial_params_c_fn(internal_vars):
                return constraint_fn_sol_to_sol(sol_list, internal_vars)

            return partial_params_c_fn

        def get_vjp_contraint_fn_params(internal_vars, sol_list):
            partial_c_fn = get_partial_params_c_fn(sol_list)

            def vjp_linear_fn(v_list):
                _, f_vjp = jax.vjp(partial_c_fn, internal_vars)
                val, = f_vjp(v_list)
                return val

            return vjp_linear_fn

        u_total_list = problem.unflatten_fn_sol_list(u_total)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, u_total_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_full))
        vjp_result = jax.tree_util.tree_map(_safe_negate, vjp_result)

        return (vjp_result, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve
