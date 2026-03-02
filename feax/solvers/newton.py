"""Newton solver utilities for FEAX.

Contains x0 initialization, Armijo line-search implementations,
and Newton solve variants used by ``feax.solver``.
"""

import logging
from typing import Optional

import jax
import jax.numpy as jnp

from ..assembler import create_J_bc_function, create_res_bc_function
from .linear import linear_solve_adjoint
from .common import (
    _safe_negate,
    create_x0,
    create_linear_solve_fn,
    check_convergence,
)
from ..problem import MatrixView
from .options import NewtonOptions

logger = logging.getLogger(__name__)


def create_newton_solve_fn(
    iter_num,
    J_bc_func,
    res_bc_func,
    bc,
    newton_options,
    linear_solver_options,
    linear_solve_fn,
    x0_fn,
    matrix_view,
):
    """Create a Newton solve callable for adaptive or fixed-iteration solves."""
    if iter_num == 1:
        raise ValueError(
            "create_newton_solve_fn does not handle iter_num==1. "
            "Use linear_solver.create_linear_solver for the linear_once path."
        )

    if iter_num is None:
        armijo_fn = create_armijo_line_search_jax(
            res_bc_func,
            c1=newton_options.line_search_c1,
            rho=newton_options.line_search_rho,
            max_backtracks=newton_options.line_search_max_backtracks,
        )
        return lambda internal_vars, initial_sol: newton_solve(
            J_bc_func,
            res_bc_func,
            initial_sol,
            bc,
            newton_options,
            linear_solver_options,
            internal_vars,
            linear_solve_fn=linear_solve_fn,
            armijo_search_fn=armijo_fn,
            x0_fn=x0_fn,
            matrix_view=matrix_view,
        )

    armijo_fn = create_armijo_line_search_scan(
        res_bc_func,
        c1=newton_options.line_search_c1,
        rho=newton_options.line_search_rho,
        max_backtracks=newton_options.line_search_max_backtracks,
    )
    return lambda internal_vars, initial_sol: newton_solve_fori(
        J_bc_func,
        res_bc_func,
        initial_sol,
        bc,
        newton_options,
        iter_num,
        linear_solver_options=linear_solver_options,
        internal_vars=internal_vars,
        linear_solve_fn=linear_solve_fn,
        armijo_search_fn=armijo_fn,
        x0_fn=x0_fn,
        matrix_view=matrix_view,
    )


def _create_differentiable_newton_solver(
    problem,
    bc,
    J_bc_func,
    res_bc_func,
    solve_fn,
    adjoint_solver_options,
    adjoint_linear_solve_fn,
):
    """Create custom-VJP Newton solver wrapper around a prepared solve_fn."""
    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        return solve_fn(internal_vars, initial_guess)[0]

    def f_fwd(internal_vars, initial_guess):
        sol = differentiable_solve(internal_vars, initial_guess)
        return sol, (internal_vars, sol)

    def f_bwd(res, v):
        internal_vars, sol = res

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

        J = J_bc_func(sol, internal_vars)
        v_vec = jax.flatten_util.ravel_pytree(v)[0]

        use_transpose = True
        if problem.matrix_view in (MatrixView.UPPER, MatrixView.LOWER):
            use_transpose = False
            logger.debug(
                "Using J directly (no transpose) for adjoint solve with problem.matrix_view=%s",
                problem.matrix_view.name,
            )

        J_adjoint = J.transpose() if use_transpose else J
        adjoint_vec = linear_solve_adjoint(
            J_adjoint,
            v_vec,
            adjoint_solver_options,
            problem.matrix_view,
            bc,
            linear_solve_fn=adjoint_linear_solve_fn,
        )
        sol_list = problem.unflatten_fn_sol_list(sol)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
        vjp_result = jax.tree_util.tree_map(_safe_negate, vjp_result)

        return (vjp_result, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve


def create_newton_solver(
    problem,
    bc,
    linear_options,
    adjoint_linear_options,
    iter_num: Optional[int],
    newton_options: Optional[NewtonOptions] = None,
    internal_vars=None,
):
    """Create a differentiable Newton solver (iter_num is None or >1)."""
    if iter_num == 1:
        raise ValueError(
            "create_newton_solver does not support iter_num==1. "
            "Use linear_solver.create_linear_solver for that path."
        )

    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)

    linear_solve_fn = create_linear_solve_fn(linear_options)
    adjoint_fn_shared = adjoint_linear_options is linear_options
    if adjoint_fn_shared:
        adjoint_linear_solve_fn = linear_solve_fn
    else:
        adjoint_linear_solve_fn = create_linear_solve_fn(adjoint_linear_options)

    def _is_cudss(opts):
        from .options import DirectSolverOptions
        return isinstance(opts, DirectSolverOptions) and opts.solver == "cudss"

    sample_J = None
    if internal_vars is not None and (_is_cudss(linear_options) or _is_cudss(adjoint_linear_options)):
        from ..utils import zero_like_initial_guess
        initial_tmp = zero_like_initial_guess(problem, bc)
        sample_J = J_bc_func(initial_tmp, internal_vars)

    if sample_J is not None:
        b_tmp = jnp.zeros(sample_J.shape[0])
        if _is_cudss(linear_options):
            print("[feax] Pre-warming cuDSS solver (forward) with sample Jacobian...")
            linear_solve_fn(sample_J, b_tmp, b_tmp)
            print("[feax] cuDSS solver (forward) initialized.")
        if _is_cudss(adjoint_linear_options) and adjoint_linear_solve_fn is not linear_solve_fn:
            print("[feax] Pre-warming cuDSS solver (adjoint) with sample Jacobian...")
            adjoint_linear_solve_fn(sample_J, b_tmp, b_tmp)
            print("[feax] cuDSS solver (adjoint) initialized.")

    if newton_options is None:
        newton_options = NewtonOptions()

    if newton_options.internal_jit:
        print("[feax] internal_jit=True: JIT-compiling internal linear solver (forward).")
        linear_solve_fn = jax.jit(linear_solve_fn)
        if adjoint_fn_shared:
            adjoint_linear_solve_fn = linear_solve_fn
        else:
            print("[feax] internal_jit=True: JIT-compiling internal linear solver (adjoint).")
            adjoint_linear_solve_fn = jax.jit(adjoint_linear_solve_fn)

    x0_fn_from_opts = None
    from .options import IterativeSolverOptions
    if isinstance(linear_options, IterativeSolverOptions):
        x0_fn_from_opts = linear_options.x0_fn
    x0_fn = x0_fn_from_opts or create_x0(
        bc_rows=bc.bc_rows,
        bc_vals=bc.bc_vals,
    )

    solve_fn = create_newton_solve_fn(
        iter_num=iter_num,
        J_bc_func=J_bc_func,
        res_bc_func=res_bc_func,
        bc=bc,
        newton_options=newton_options,
        linear_solver_options=linear_options,
        linear_solve_fn=linear_solve_fn,
        x0_fn=x0_fn,
        matrix_view=problem.matrix_view,
    )

    return _create_differentiable_newton_solver(
        problem=problem,
        bc=bc,
        J_bc_func=J_bc_func,
        res_bc_func=res_bc_func,
        solve_fn=solve_fn,
        adjoint_solver_options=adjoint_linear_options,
        adjoint_linear_solve_fn=adjoint_linear_solve_fn,
    )


def create_newton_differentiable_solver(
    problem,
    bc,
    solver_options,
    adjoint_solver_options,
    iter_num: Optional[int],
    newton_options: Optional[NewtonOptions] = None,
    internal_vars=None,
):
    """Backward-compatible alias for ``create_newton_solver``."""
    return create_newton_solver(
        problem=problem,
        bc=bc,
        linear_options=solver_options,
        adjoint_linear_options=adjoint_solver_options,
        iter_num=iter_num,
        newton_options=newton_options,
        internal_vars=internal_vars,
    )

def create_armijo_line_search_jax(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX while_loop Armijo line search."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -jnp.dot(res, res)

        init_state = (1.0, 0, False, sol, res_norm)

        def cond_fn(state):
            _, step, found_good, _, _ = state
            return jnp.logical_not(found_good) & (step < max_backtracks)

        def body_fn(state):
            alpha, step, _, best_sol, best_norm = state
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)

            is_valid = jnp.logical_not(jnp.any(jnp.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            is_acceptable = is_valid & armijo_satisfied

            new_sol = jnp.where(is_acceptable, trial_sol, best_sol)
            new_norm = jnp.where(is_acceptable, trial_norm, best_norm)
            new_alpha = jnp.where(is_acceptable, alpha, alpha * rho)

            return (new_alpha, step + 1, is_acceptable, new_sol, new_norm)

        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        final_alpha, _, found_good, new_sol, new_norm = final_state

        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)

        final_sol = jnp.where(found_good, new_sol, fallback_sol)
        final_norm = jnp.where(found_good, new_norm, fallback_norm)
        final_alpha_out = jnp.where(found_good, final_alpha, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search


def create_armijo_line_search_scan(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX scan-based Armijo line search (vmap-friendly)."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -jnp.dot(res, res)

        def scan_fn(carry, _):
            alpha, best_sol, best_norm, found_good = carry
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)

            is_valid = jnp.logical_not(jnp.any(jnp.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            is_acceptable = is_valid & armijo_satisfied

            should_update = is_acceptable & jnp.logical_not(found_good)
            new_sol = jnp.where(should_update, trial_sol, best_sol)
            new_norm = jnp.where(should_update, trial_norm, best_norm)
            new_alpha = alpha * rho
            new_found = found_good | is_acceptable

            return (new_alpha, new_sol, new_norm, new_found), None

        init_carry = (1.0, sol, res_norm, False)
        (final_alpha, best_sol, best_norm, found_good), _ = jax.lax.scan(
            scan_fn, init_carry, None, length=max_backtracks
        )

        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)

        final_sol = jnp.where(found_good, best_sol, fallback_sol)
        final_norm = jnp.where(found_good, best_norm, fallback_norm)
        final_alpha_out = jnp.where(found_good, final_alpha / rho, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search


def create_armijo_line_search_python(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create Python-loop Armijo line search (debug path)."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -jnp.dot(res, res)

        alpha = 1.0
        for _ in range(max_backtracks):
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)

            is_valid = not jnp.any(jnp.isnan(trial_res))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit

            if is_valid and armijo_satisfied:
                return trial_sol, trial_norm, alpha, True

            alpha *= rho

        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        return fallback_sol, fallback_norm, 1e-8, False

    return line_search


def newton_solve(J_bc_applied, res_bc_applied, initial_guess, bc, newton_options, linear_solver_options,
                 internal_vars=None, P_mat=None,
                 linear_solve_fn=None, armijo_search_fn=None, x0_fn=None,
                 matrix_view: MatrixView = MatrixView.FULL):
    """Newton solver using JAX while_loop for JIT compatibility."""
    if x0_fn is None:
        _legacy_x0_fn = getattr(linear_solver_options, "linear_solver_x0_fn", None)
        if _legacy_x0_fn is not None:
            x0_fn = _legacy_x0_fn
        else:
            x0_fn = create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals, P_mat=P_mat)

    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(linear_solver_options)

    armijo_search = armijo_search_fn
    if armijo_search is None:
        armijo_search = create_armijo_line_search_jax(
            res_bc_applied,
            c1=newton_options.line_search_c1,
            rho=newton_options.line_search_rho,
            max_backtracks=newton_options.line_search_max_backtracks,
        )

    def cond_fun(state):
        sol, res_norm, initial_res_norm, iter_count = state
        rel_res_norm = res_norm / (initial_res_norm + 1e-30)
        continue_iter = (res_norm > newton_options.tol) & (rel_res_norm > newton_options.rel_tol) & (iter_count < newton_options.max_iter)
        return continue_iter

    def body_fun(state):
        sol, res_norm, initial_res_norm, iter_count = state

        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        x0 = x0_fn(sol)
        delta_sol = linear_solve_fn(J, -res, x0=x0)
        if linear_solver_options.check_convergence:
            delta_sol = check_convergence(
                A=J,
                x=delta_sol,
                b=-res,
                solver_options=linear_solver_options,
                matrix_view=matrix_view,
                solver_label="Newton linear solve",
            )

        new_sol, new_norm, alpha, success = armijo_search(sol, delta_sol, res, res_norm, internal_vars)

        if linear_solver_options.verbose:
            jax.debug.print(
                "Newton iter {i:3d}: res_norm = {r:.6e}, rel = {rr:.6e}, alpha = {a:.4f}, success = {s}",
                i=iter_count, r=new_norm, rr=new_norm / (initial_res_norm + 1e-30), a=alpha, s=success,
            )

        return (new_sol, new_norm, initial_res_norm, iter_count + 1)

    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_guess, initial_res_norm, initial_res_norm, 0)

    if linear_solver_options.verbose:
        jax.debug.print("Newton solver starting: initial res_norm = {r:.6e}", r=initial_res_norm)

    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    if linear_solver_options.verbose:
        final_sol, final_res_norm, final_initial_res_norm, final_iter = final_state
        jax.debug.print(
            "Newton solver converged: final_iter = {i}, final_res_norm = {r:.6e}, rel = {rr:.6e}",
            i=final_iter, r=final_res_norm, rr=final_res_norm / (final_initial_res_norm + 1e-30),
        )

    return final_state


def newton_solve_fori(J_bc_applied, res_bc_applied, initial_guess, bc, newton_options, num_iters,
                      linear_solver_options,
                      internal_vars=None, P_mat=None,
                      linear_solve_fn=None, armijo_search_fn=None, x0_fn=None,
                      matrix_view: MatrixView = MatrixView.FULL):
    """Newton solver using JAX fori_loop for fixed iterations."""
    if x0_fn is None:
        _legacy_x0_fn = getattr(linear_solver_options, "linear_solver_x0_fn", None)
        if _legacy_x0_fn is not None:
            x0_fn = _legacy_x0_fn
        else:
            x0_fn = create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals, P_mat=P_mat)

    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(linear_solver_options)

    armijo_search = armijo_search_fn
    if armijo_search is None:
        armijo_search = create_armijo_line_search_scan(
            res_bc_applied,
            c1=newton_options.line_search_c1,
            rho=newton_options.line_search_rho,
            max_backtracks=newton_options.line_search_max_backtracks,
        )

    def newton_iteration(i, state):
        sol, res_norm = state

        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        x0 = x0_fn(sol)
        delta_sol = linear_solve_fn(J, -res, x0)
        if linear_solver_options.check_convergence:
            delta_sol = check_convergence(
                A=J,
                x=delta_sol,
                b=-res,
                solver_options=linear_solver_options,
                matrix_view=matrix_view,
                solver_label="Newton linear solve",
            )

        new_sol, new_res_norm, alpha, success = armijo_search(sol, delta_sol, res, res_norm, internal_vars)

        if linear_solver_options.verbose:
            jax.debug.print(
                "Newton iter {iter:3d}: res_norm = {r:.6e}, alpha = {a:.4f}, success = {s}",
                iter=i, r=new_res_norm, a=alpha, s=success,
            )

        return (new_sol, new_res_norm)

    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)

    if linear_solver_options.verbose:
        jax.debug.print("Newton solver (fori) starting: initial res_norm = {r:.6e}", r=initial_res_norm)

    final_state = jax.lax.fori_loop(
        0, num_iters,
        newton_iteration,
        (initial_guess, initial_res_norm),
    )

    final_sol, final_res_norm = final_state
    converged = final_res_norm < newton_options.tol

    if linear_solver_options.verbose:
        jax.debug.print(
            "Newton solver (fori) finished: {n} iterations, final_res_norm = {r:.6e}, converged = {c}",
            n=num_iters, r=final_res_norm, c=converged,
        )

    return final_sol, final_res_norm, converged


def newton_solve_py(J_bc_applied, res_bc_applied, initial_guess, bc, newton_options, linear_solver_options,
                    internal_vars=None, P_mat=None,
                    matrix_view: MatrixView = MatrixView.FULL):
    """Newton solver using Python while loop (non-JIT debug path)."""
    tol = newton_options.tol
    rel_tol = newton_options.rel_tol
    max_iter = newton_options.max_iter
    line_search_c1 = newton_options.line_search_c1
    line_search_rho = newton_options.line_search_rho
    line_search_max_backtracks = newton_options.line_search_max_backtracks

    _legacy_x0_fn = getattr(linear_solver_options, "linear_solver_x0_fn", None)
    if _legacy_x0_fn is not None:
        x0_fn = _legacy_x0_fn
    else:
        x0_fn = create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals, P_mat=P_mat)

    linear_solve_fn = create_linear_solve_fn(linear_solver_options)

    armijo_line_search = create_armijo_line_search_python(
        res_bc_applied,
        c1=line_search_c1,
        rho=line_search_rho,
        max_backtracks=line_search_max_backtracks,
    )

    sol = initial_guess
    if internal_vars is not None:
        initial_res = res_bc_applied(sol, internal_vars)
    else:
        initial_res = res_bc_applied(sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    res_norm = initial_res_norm
    iter_count = 0

    if linear_solver_options.verbose:
        logger.info(f"Newton solver (py) starting: initial res_norm = {initial_res_norm:.6e}")

    while (res_norm > tol and
           res_norm / initial_res_norm > rel_tol and
           iter_count < max_iter):

        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        x0 = x0_fn(sol)
        delta_sol = linear_solve_fn(J, -res, x0)
        if linear_solver_options.check_convergence:
            delta_sol = check_convergence(
                A=J,
                x=delta_sol,
                b=-res,
                solver_options=linear_solver_options,
                matrix_view=matrix_view,
                solver_label="Newton linear solve",
            )

        new_sol, new_res_norm, alpha, success = armijo_line_search(
            sol, delta_sol, res, res_norm, internal_vars
        )

        if linear_solver_options.verbose:
            logger.info(
                f"Newton iter {iter_count:3d}: res_norm = {new_res_norm:.6e}, "
                f"alpha = {alpha:.4f}, success = {success}"
            )

        sol = new_sol
        res_norm = new_res_norm
        iter_count += 1

    converged = (res_norm <= tol or
                 res_norm / initial_res_norm <= rel_tol)

    if linear_solver_options.verbose:
        logger.info(
            f"Newton solver (py) finished: iter_count = {iter_count}, "
            f"final_res_norm = {res_norm:.6e}, converged = {converged}"
        )

    return sol, res_norm, converged, iter_count
