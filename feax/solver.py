"""
Nonlinear and linear solvers for FEAX finite element framework.

This module provides Newton-Raphson solvers and solver configuration
utilities for solving finite element problems. It includes both JAX-based
solvers for performance and Python-based solvers for debugging.

Key Features:
- Newton-Raphson solvers with line search and convergence control
- Multiple solver variants: while loop, fixed iterations, and Python debugging
- Solver configuration via AbstractSolverOptions hierarchy
  (DirectSolverOptions, IterativeSolverOptions, or legacy SolverOptions)
- Support for multipoint constraints via prolongation matrices

Linear solver implementations and preconditioners are provided in
``linear_solver.py``. Solver configuration options are in ``solver_option.py``.
"""

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from typing import Optional, Callable, Any
import logging

from .assembler import create_J_bc_function, create_res_bc_function
from .DCboundary import DirichletBC
from .problem import MatrixView, Problem
from .solver_option import (
    AbstractSolverOptions, SolverOptions, CUDSSOptions, CUDSSMatrixType, CUDSSMatrixView,
    DirectSolverOptions, IterativeSolverOptions,
    detect_matrix_property, resolve_direct_solver, resolve_iterative_solver,
)
from .linear_solver import (
    create_linear_solve_fn,
    check_linear_convergence,
    linear_solve_adjoint,
)

logger = logging.getLogger(__name__)


def _safe_negate(x):
    """Negate array, handling JAX's float0 type for zero gradients.

    When differentiating through computations where some parameters don't
    affect the output, JAX uses a special 'float0' dtype to represent
    zero gradients efficiently. This function handles negation properly
    for both regular arrays and float0 arrays.

    Parameters
    ----------
    x : jax.Array or np.ndarray
        Array to negate. May have float0 dtype.

    Returns
    -------
    jax.Array or np.ndarray
        Negated array, or unchanged if float0 dtype.
    """
    if hasattr(x, 'dtype'):
        dtype_str = str(x.dtype)
        if 'float0' in dtype_str or 'V' in dtype_str:
            return x
    return -x


def create_x0(bc_rows=None, bc_vals=None, P_mat=None):
    """Create initial guess function for linear solver following JAX-FEM approach.

    Parameters
    ----------
    bc_rows : array-like, optional
        Row indices of boundary condition locations
    bc_vals : array-like, optional
        Boundary condition values
    P_mat : BCOO matrix, optional
        Prolongation matrix for reduced problems (maps reduced to full DOFs)

    Returns
    -------
    x0_fn : callable
        Function that takes current solution and returns initial guess for increment

    Notes
    -----
    Implements the exact x0 computation from the row elimination solver:
    x0_1 = assign_bc(zeros, problem) - sets BC values at BC locations, 0 elsewhere
    x0_2 = copy_bc(current_sol, problem) - copies current solution values at BC locations, 0 elsewhere
    x0 = x0_1 - x0_2 - the correct initial guess computation

    For reduced problems (when P_mat is provided):
    x0_2 = copy_bc(P @ current_sol_reduced, problem) - expand reduced sol and copy BC
    x0 = P.T @ (x0_1 - x0_2) - transform back to reduced space

    Examples
    --------
    >>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0])
    >>> solver_options = IterativeSolverOptions(x0_fn=x0_fn)

    >>> # Usage with reduced problem
    >>> x0_fn = create_x0(bc_rows, bc_vals, P_mat=P)
    """

    def x0_fn(current_sol):
        """BC-aware strategy: correct x0 method from row elimination solver."""
        if bc_rows is None or bc_vals is None:
            return jnp.zeros_like(current_sol)

        bc_rows_array = jnp.array(bc_rows) if isinstance(bc_rows, (tuple, list)) else bc_rows
        bc_vals_array = jnp.array(bc_vals) if isinstance(bc_vals, (tuple, list)) else bc_vals

        if P_mat is not None:
            x0_1 = jnp.zeros(P_mat.shape[0])
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)

            current_sol_full = P_mat @ current_sol
            x0_2 = jnp.zeros(P_mat.shape[0])
            x0_2 = x0_2.at[bc_rows_array].set(current_sol_full[bc_rows_array])

            x0 = P_mat.T @ (x0_1 - x0_2)
        else:
            x0_1 = jnp.zeros_like(current_sol)
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)

            x0_2 = jnp.zeros_like(current_sol)
            x0_2 = x0_2.at[bc_rows_array].set(current_sol[bc_rows_array])

            x0 = x0_1 - x0_2

        return x0

    return x0_fn


def create_armijo_line_search_jax(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX-compatible Armijo backtracking line search using jax.lax.scan.

    This function returns a line search function that can be JIT-compiled.

    Parameters
    ----------
    res_bc_applied : callable
        Residual function with boundary conditions applied.
        Signature: res_bc_applied(sol, internal_vars=None) -> residual
    c1 : float, default 1e-4
        Armijo constant for sufficient decrease condition
    rho : float, default 0.5
        Backtracking factor (alpha *= rho each iteration)
    max_backtracks : int, default 30
        Maximum number of backtracking steps

    Returns
    -------
    line_search_fn : callable
        Line search function with signature:
        (sol, delta_sol, initial_res_norm, internal_vars=None) -> (new_sol, new_norm, alpha, success)
    """

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        """Execute Armijo line search.

        Parameters
        ----------
        sol : array
            Current solution
        delta_sol : array
            Search direction (Newton step)
        res : array
            Residual at current solution
        res_norm : float
            Norm of residual
        internal_vars : InternalVars, optional
            Internal variables for residual evaluation

        Returns
        -------
        new_sol : array
            Updated solution
        new_norm : float
            Residual norm at new solution
        alpha : float
            Step size used
        success : bool
            Whether a valid step was found
        """
        grad_merit = -jnp.dot(res, res)

        # State: (alpha, step, found_good, best_sol, best_norm)
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

        # Fallback if no acceptable step found
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
    """Create JAX-compatible Armijo backtracking line search using jax.lax.scan.

    This version uses scan (fixed iterations) and is vmap-compatible.
    Use this for newton_solve_fori where vmap support is required.

    Parameters
    ----------
    res_bc_applied : callable
        Residual function with boundary conditions applied.
    c1 : float, default 1e-4
        Armijo constant for sufficient decrease condition
    rho : float, default 0.5
        Backtracking factor
    max_backtracks : int, default 30
        Number of backtracking steps (all are always evaluated)

    Returns
    -------
    line_search_fn : callable
        Line search function with same signature as while_loop version.
    """

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

            # Only update if acceptable and not already found
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

        # Fallback if no acceptable step found
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
    """Create Python-based Armijo backtracking line search.

    This version uses Python loops and is suitable for debugging or non-JIT contexts.

    Parameters
    ----------
    res_bc_applied : callable
        Residual function with boundary conditions applied
    c1 : float, default 1e-4
        Armijo constant
    rho : float, default 0.5
        Backtracking factor
    max_backtracks : int, default 30
        Maximum backtracking steps

    Returns
    -------
    line_search_fn : callable
        Line search function
    """
    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        """Execute Armijo line search using Python loop."""
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


def newton_solve(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None, P_mat=None,
                 linear_solve_fn=None, armijo_search_fn=None, x0_fn=None):
    """Newton solver using JAX while_loop for JIT compatibility.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC
    linear_solve_fn : callable, optional
        Pre-created linear solve function. If None, created internally.
    armijo_search_fn : callable, optional
        Pre-created Armijo line search function. If None, created internally.
    x0_fn : callable, optional
        Pre-created initial guess function. If None, created internally.

    Returns
    -------
    sol, res_norm, initial_res_norm, iter_count : tuple
        Solution, residual norm, initial residual norm, and iteration count
    """

    if x0_fn is None:
        if solver_options.linear_solver_x0_fn is not None:
            x0_fn = solver_options.linear_solver_x0_fn
        else:
            x0_fn = create_x0(
                bc_rows=bc.bc_rows,
                bc_vals=bc.bc_vals,
                P_mat=P_mat
            )

    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(solver_options)

    armijo_search = armijo_search_fn
    if armijo_search is None:
        armijo_search = create_armijo_line_search_jax(
            res_bc_applied,
            c1=solver_options.line_search_c1,
            rho=solver_options.line_search_rho,
            max_backtracks=solver_options.line_search_max_backtracks
        )

    def cond_fun(state):
        sol, res_norm, initial_res_norm, iter_count = state
        rel_res_norm = res_norm / (initial_res_norm + 1e-30)
        continue_iter = (res_norm > solver_options.tol) & (rel_res_norm > solver_options.rel_tol) & (iter_count < solver_options.max_iter)
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

        new_sol, new_norm, alpha, success = armijo_search(sol, delta_sol, res, res_norm, internal_vars)

        if solver_options.verbose:
            jax.debug.print(
                "Newton iter {i:3d}: res_norm = {r:.6e}, rel = {rr:.6e}, alpha = {a:.4f}, success = {s}",
                i=iter_count, r=new_norm, rr=new_norm / (initial_res_norm + 1e-30), a=alpha, s=success
            )

        return (new_sol, new_norm, initial_res_norm, iter_count + 1)

    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_guess, initial_res_norm, initial_res_norm, 0)

    if solver_options.verbose:
        jax.debug.print("Newton solver starting: initial res_norm = {r:.6e}", r=initial_res_norm)

    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    if solver_options.verbose:
        final_sol, final_res_norm, final_initial_res_norm, final_iter = final_state
        jax.debug.print(
            "Newton solver converged: final_iter = {i}, final_res_norm = {r:.6e}, rel = {rr:.6e}",
            i=final_iter, r=final_res_norm, rr=final_res_norm / (final_initial_res_norm + 1e-30)
        )

    return final_state


def newton_solve_fori(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, num_iters: int, internal_vars=None, P_mat=None,
                      linear_solve_fn=None, armijo_search_fn=None, x0_fn=None):
    """Newton solver using JAX fori_loop for fixed iterations - optimized for vmap.

    Designed for vmap with fixed iterations and consistent computational graph.
    Uses scan-based Armijo line search for vmap compatibility.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    num_iters : int
        Fixed number of iterations
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC
    linear_solve_fn : callable, optional
        Pre-created linear solve function. If None, created internally.
    armijo_search_fn : callable, optional
        Pre-created Armijo line search function. If None, created internally (scan-based).
    x0_fn : callable, optional
        Pre-created initial guess function. If None, created internally.

    Returns
    -------
    sol, final_res_norm, converged : tuple
        Solution, residual norm, and convergence flag
    """

    if x0_fn is None:
        if solver_options.linear_solver_x0_fn is not None:
            x0_fn = solver_options.linear_solver_x0_fn
        else:
            x0_fn = create_x0(
                bc_rows=bc.bc_rows,
                bc_vals=bc.bc_vals,
                P_mat=P_mat
            )

    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(solver_options)

    armijo_search = armijo_search_fn
    if armijo_search is None:
        # Use scan-based line search for vmap compatibility
        armijo_search = create_armijo_line_search_scan(
            res_bc_applied,
            c1=solver_options.line_search_c1,
            rho=solver_options.line_search_rho,
            max_backtracks=solver_options.line_search_max_backtracks
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

        new_sol, new_res_norm, alpha, success = armijo_search(sol, delta_sol, res, res_norm, internal_vars)

        if solver_options.verbose:
            jax.debug.print(
                "Newton iter {iter:3d}: res_norm = {r:.6e}, alpha = {a:.4f}, success = {s}",
                iter=i, r=new_res_norm, a=alpha, s=success
            )

        return (new_sol, new_res_norm)

    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)

    if solver_options.verbose:
        jax.debug.print("Newton solver (fori) starting: initial res_norm = {r:.6e}", r=initial_res_norm)

    final_state = jax.lax.fori_loop(
        0, num_iters,
        newton_iteration,
        (initial_guess, initial_res_norm)
    )

    final_sol, final_res_norm = final_state
    converged = final_res_norm < solver_options.tol

    if solver_options.verbose:
        jax.debug.print(
            "Newton solver (fori) finished: {n} iterations, final_res_norm = {r:.6e}, converged = {c}",
            n=num_iters, r=final_res_norm, c=converged
        )

    return final_sol, final_res_norm, converged


def newton_solve_py(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None, P_mat=None):
    """Newton solver using Python while loop - non-JIT version for debugging.

    Uses Python control flow for easier debugging. Not JIT-compatible.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC

    Returns
    -------
    sol, final_res_norm, converged, num_iters : tuple
        Solution, residual norm, convergence flag, and iteration count
    """

    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )

    linear_solve_fn = create_linear_solve_fn(solver_options)

    armijo_line_search = create_armijo_line_search_python(
        res_bc_applied,
        c1=solver_options.line_search_c1,
        rho=solver_options.line_search_rho,
        max_backtracks=solver_options.line_search_max_backtracks
    )

    sol = initial_guess
    if internal_vars is not None:
        initial_res = res_bc_applied(sol, internal_vars)
    else:
        initial_res = res_bc_applied(sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    res_norm = initial_res_norm
    iter_count = 0

    if solver_options.verbose:
        logger.info(f"Newton solver (py) starting: initial res_norm = {initial_res_norm:.6e}")

    while (res_norm > solver_options.tol and
           res_norm / initial_res_norm > solver_options.rel_tol and
           iter_count < solver_options.max_iter):

        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)

        x0 = x0_fn(sol)
        delta_sol = linear_solve_fn(J, -res, x0)

        new_sol, new_res_norm, alpha, success = armijo_line_search(
            sol, delta_sol, res, res_norm, internal_vars
        )

        if solver_options.verbose:
            logger.info(f"Newton iter {iter_count:3d}: res_norm = {new_res_norm:.6e}, alpha = {alpha:.4f}, success = {success}")

        sol = new_sol
        res_norm = new_res_norm
        iter_count += 1

    converged = (res_norm <= solver_options.tol or
                res_norm / initial_res_norm <= solver_options.rel_tol)

    if solver_options.verbose:
        logger.info(f"Newton solver (py) finished: iter_count = {iter_count}, final_res_norm = {res_norm:.6e}, converged = {converged}")

    return sol, res_norm, converged, iter_count


def linear_solve(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, matrix_view: MatrixView, internal_vars=None, P_mat=None,
                  linear_solve_fn=None, x0_fn=None):
    """Linear solver for problems that converge in one iteration.

    Optimized for linear problems (e.g., linear elasticity). Performs single Newton step.

    Parameters
    ----------
    J_bc_applied : callable
        Jacobian function with BC applied
    res_bc_applied : callable
        Residual function with BC applied
    initial_guess : array
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration
    matrix_view : MatrixView
        Matrix storage format from the problem.
    internal_vars : InternalVars, optional
        Material properties and parameters
    P_mat : BCOO matrix, optional
        Prolongation matrix for MPC/periodic BC
    linear_solve_fn : callable, optional
        Pre-created linear solve function. If None, created internally.
    x0_fn : callable, optional
        Pre-created initial guess function. If None, created internally.

    Returns
    -------
    sol, None : tuple
        Solution and None (for compatibility)

    For linear problems, this single iteration achieves the exact solution.
    """

    if x0_fn is None:
        if solver_options.linear_solver_x0_fn is not None:
            x0_fn = solver_options.linear_solver_x0_fn
        else:
            x0_fn = create_x0(
                bc_rows=bc.bc_rows,
                bc_vals=bc.bc_vals,
                P_mat=P_mat
            )

    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(solver_options)

    if internal_vars is not None:
        res = res_bc_applied(initial_guess, internal_vars)
        J = J_bc_applied(initial_guess, internal_vars)
    else:
        res = res_bc_applied(initial_guess)
        J = J_bc_applied(initial_guess)

    x0 = x0_fn(initial_guess)
    b = -res
    delta_sol = linear_solve_fn(J, b, x0)

    if solver_options.check_convergence:
        delta_sol = check_linear_convergence(
            A=J,
            x=delta_sol,
            b=b,
            solver_options=solver_options,
            matrix_view=matrix_view,
            solver_label="Linear solver",
        )

    sol = initial_guess + delta_sol

    return sol, None


def _get_reduced_solver_params(opts: IterativeSolverOptions):
    """Extract (solver_name, tol, atol, maxiter) from IterativeSolverOptions.

    ``"auto"`` resolves to ``"cg"`` because P^T K P is SPD for well-posed
    linear elastic unit cells.
    """
    name = opts.solver if opts.solver != "auto" else "cg"
    return name, opts.tol, opts.atol, opts.maxiter


def _reduced_iterative_solve(matvec, rhs, x0, solver_name, tol, atol, maxiter):
    """Dispatch to the appropriate JAX sparse iterative solver."""
    if solver_name == "cg":
        return jax.scipy.sparse.linalg.cg(
            matvec, rhs, x0=x0, tol=tol, atol=atol, maxiter=maxiter
        )
    elif solver_name == "bicgstab":
        return jax.scipy.sparse.linalg.bicgstab(
            matvec, rhs, x0=x0, tol=tol, atol=atol, maxiter=maxiter
        )
    elif solver_name == "gmres":
        return jax.scipy.sparse.linalg.gmres(
            matvec, rhs, x0=x0, tol=tol, atol=atol, maxiter=maxiter
        )
    else:
        raise ValueError(
            f"Unsupported iterative solver for reduced problem: {solver_name!r}. "
            "Choose 'cg', 'bicgstab', or 'gmres'."
        )


def _create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options, iter_num):
    """Create matrix-free reduced solver for periodic boundary conditions."""

    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)

    fwd_name, fwd_tol, fwd_atol, fwd_maxiter = _get_reduced_solver_params(solver_options)
    adj_name, adj_tol, adj_atol, adj_maxiter = _get_reduced_solver_params(adjoint_solver_options)
    fwd_verbose = solver_options.verbose
    adj_verbose = adjoint_solver_options.verbose

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
        sol_reduced, info = _reduced_iterative_solve(
            J_reduced_matvec, -res_reduced, x0,
            fwd_name, fwd_tol, fwd_atol, fwd_maxiter,
        )
        if fwd_verbose:
            jax.debug.print(
                "    [{name}] info={info} (0=converged, >0=max_iter reached)",
                name=fwd_name, info=info,
            )
        sol_full = P @ sol_reduced
        return sol_full, None

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        return reduced_solve_fn(internal_vars, initial_guess)[0]

    def f_fwd(internal_vars, initial_guess):
        sol = differentiable_solve(internal_vars, initial_guess)
        # Save initial_guess so f_bwd can reconstruct the total displacement
        # u_total = sol (= P @ u'_red) + initial_guess
        return sol, (internal_vars, sol, initial_guess)

    def f_bwd(res, v):
        internal_vars, sol, initial_guess = res

        # Evaluate Jacobian and VJP at the total displacement u_total,
        # not just the periodic fluctuation P @ u'_red.
        u_total = sol + initial_guess
        J_full = J_bc_func(u_total, internal_vars)
        rhs_reduced = P.T @ v

        def adjoint_matvec(adjoint_reduced):
            adjoint_full = P @ adjoint_reduced
            Jt_adjoint_full = J_full.T @ adjoint_full
            return P.T @ Jt_adjoint_full

        x0_reduced = jnp.zeros_like(rhs_reduced)
        adjoint_reduced, adj_info = _reduced_iterative_solve(
            adjoint_matvec, rhs_reduced, x0_reduced,
            adj_name, adj_tol, adj_atol, adj_maxiter,
        )
        if adj_verbose:
            jax.debug.print(
                "    [{name} adjoint] info={info} (0=converged, >0=max_iter reached)",
                name=adj_name, info=adj_info,
            )

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


def create_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[AbstractSolverOptions] = None,
    adjoint_solver_options: Optional[AbstractSolverOptions] = None,
    iter_num: Optional[int] = None,
    P: Optional[BCOO] = None,
    internal_vars=None,
    internal_jit: bool = False,
) -> Callable:
    """Create a differentiable solver with custom VJP for gradient computation.

    Parameters
    ----------
    problem : Problem
        The feax Problem instance.
    bc : DirichletBC
        Boundary conditions.
    solver_options : AbstractSolverOptions, optional
        Options for the forward linear solve. Accepts any subclass of
        ``AbstractSolverOptions``:

        - ``DirectSolverOptions``: Direct solvers (cudss, spsolve, cholesky, lu, qr).
        - ``IterativeSolverOptions``: Iterative solvers (cg, bicgstab, gmres).
        - ``SolverOptions``: Legacy unified options (backward compatible).

        When ``solver="auto"``, the algorithm is selected automatically by
        assembling the initial Jacobian and calling ``detect_matrix_property``.
        Defaults to ``SolverOptions()`` if not specified.
    adjoint_solver_options : AbstractSolverOptions, optional
        Options for the adjoint solve in the backward pass.
        Defaults to same as ``solver_options``.
    iter_num : int, optional
        Number of Newton iterations:

        - ``None``: Adaptive Newton solve with while loop (NOT vmappable).
        - ``1``: Single linear solve (vmappable). Recommended for linear problems.
        - ``>1``: Fixed-iteration Newton solve (vmappable).
    P : BCOO matrix, optional
        Prolongation matrix for periodic boundary conditions.
    internal_vars : InternalVars, optional
        Sample internal variables for auto solver selection and cuDSS
        pre-warming. Required when ``solver="auto"`` or cuDSS is used.
    internal_jit : bool, optional
        When ``True`` and ``iter_num != 1``, wraps the internal linear solver
        with ``jax.jit`` so that each Newton iteration's linear solve is
        compiled separately from the outer computation graph.  This is most
        effective for iterative solvers (CG, BiCGSTAB, GMRES) called only once;
        for repeated outer calls prefer ``jax.jit(solver)`` instead.
        Ignored (with a warning) when ``iter_num == 1`` because the linear
        solver is invoked only once per solve and internal JIT provides no
        benefit.  Default: ``False``.

    Returns
    -------
    callable
        When ``DirectSolverOptions`` is used:
            ``solver(internal_vars) -> solution``
            (``initial_guess`` is optional and ignored if provided.)
        When ``IterativeSolverOptions`` or ``SolverOptions`` is used:
            ``solver(internal_vars, initial_guess) -> solution``

    Examples
    --------
    >>> # Direct solver (auto-selects cuDSS on GPU, spsolve on CPU)
    >>> solver = create_solver(problem, bc, DirectSolverOptions(),
    ...                        iter_num=1, internal_vars=internal_vars)
    >>> solution = solver(internal_vars)
    >>>
    >>> # Iterative solver with auto selection
    >>> solver = create_solver(problem, bc, IterativeSolverOptions(),
    ...                        iter_num=1, internal_vars=internal_vars)
    >>> solution = solver(internal_vars, initial_guess)
    >>>
    >>> # Explicit solver selection (no internal_vars needed for non-cuDSS)
    >>> solver = create_solver(problem, bc, IterativeSolverOptions(solver="gmres"),
    ...                        iter_num=1)
    >>> solution = solver(internal_vars, initial_guess)
    """

    # Reduced (matrix-free) path: requires IterativeSolverOptions
    if P is not None:
        if solver_options is None:
            solver_options = IterativeSolverOptions()
        elif not isinstance(solver_options, IterativeSolverOptions):
            raise ValueError(
                "solver_options must be IterativeSolverOptions when P (prolongation matrix) "
                f"is provided, got {type(solver_options).__name__}. "
                "The reduced problem is matrix-free and only supports iterative solvers "
                "(cg, bicgstab, gmres)."
            )
        if adjoint_solver_options is None:
            adjoint_solver_options = solver_options
        elif not isinstance(adjoint_solver_options, IterativeSolverOptions):
            raise ValueError(
                "adjoint_solver_options must be IterativeSolverOptions when P is provided, "
                f"got {type(adjoint_solver_options).__name__}."
            )
        return _create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options, iter_num)

    if solver_options is None:
        solver_options = SolverOptions()
    if adjoint_solver_options is None:
        adjoint_solver_options = solver_options

    def _validate_cudss_options(opts: CUDSSOptions, role: str) -> None:
        if opts.matrix_view not in (CUDSSMatrixView.FULL, CUDSSMatrixView.UPPER, CUDSSMatrixView.LOWER):
            raise ValueError(
                f"cudss_options.matrix_view must be FULL, UPPER, or LOWER for {role} solver. "
                f"Got: {opts.matrix_view.name}"
            )

        if opts.matrix_view in (CUDSSMatrixView.UPPER, CUDSSMatrixView.LOWER):
            if opts.matrix_type not in (CUDSSMatrixType.SYMMETRIC, CUDSSMatrixType.SPD):
                logger.warning(
                    f"{role} solver: Using matrix_view={opts.matrix_view.name} with matrix_type={opts.matrix_type.name}. "
                    f"For best performance, use matrix_type=SYMMETRIC or matrix_type=SPD with triangular storage."
                )

        if opts.matrix_type not in (CUDSSMatrixType.GENERAL, CUDSSMatrixType.SYMMETRIC, CUDSSMatrixType.SPD):
            raise ValueError(
                f"cudss_options.matrix_type must be GENERAL, SYMMETRIC, or SPD for {role} solver. "
                f"Got: {opts.matrix_type.name}."
            )

    # Resolve "auto" for new option types using detect_matrix_property
    _new_option_types = (DirectSolverOptions, IterativeSolverOptions)
    _shared_opts = adjoint_solver_options is solver_options
    _needs_auto = (
        (isinstance(solver_options, _new_option_types) and solver_options.solver == "auto") or
        (isinstance(adjoint_solver_options, _new_option_types) and adjoint_solver_options.solver == "auto")
    )

    # Validate cuDSS options
    def _is_cudss(opts):
        return (
            (isinstance(opts, DirectSolverOptions) and opts.solver == "cudss") or
            (isinstance(opts, SolverOptions) and opts.linear_solver == "cudss")
        )

    if _needs_auto:
        if internal_vars is None:
            raise ValueError(
                "internal_vars is required when solver_options has solver='auto'. "
                "Pass a sample InternalVars to enable automatic matrix property detection, "
                "or specify the solver explicitly (e.g. IterativeSolverOptions(solver='cg'))."
            )

    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)

    # Assemble sample Jacobian if needed (auto detection or cuDSS pre-warming)
    _sample_J = None
    if internal_vars is not None and (_needs_auto or _is_cudss(solver_options) or _is_cudss(adjoint_solver_options)):
        from .utils import zero_like_initial_guess
        _initial_tmp = zero_like_initial_guess(problem, bc)
        _sample_J = J_bc_func(_initial_tmp, internal_vars)

    # Resolve "auto" using detected matrix property
    if _needs_auto:
        _mp = detect_matrix_property(_sample_J, matrix_view=problem.matrix_view)
        from .solver_option import detect_backend
        _backend = detect_backend()

        if isinstance(solver_options, _new_option_types) and solver_options.solver == "auto":
            _category = "direct" if isinstance(solver_options, DirectSolverOptions) else "iterative"
            if isinstance(solver_options, DirectSolverOptions):
                solver_options = resolve_direct_solver(solver_options, _mp, matrix_view=problem.matrix_view)
            else:
                solver_options = resolve_iterative_solver(solver_options, _mp)
            print(f"[feax] Auto solver ({_category}): backend={_backend.name}, matrix_property={_mp.name} -> {solver_options.solver}")

        if _shared_opts:
            adjoint_solver_options = solver_options
        elif isinstance(adjoint_solver_options, _new_option_types) and adjoint_solver_options.solver == "auto":
            _category = "direct" if isinstance(adjoint_solver_options, DirectSolverOptions) else "iterative"
            if isinstance(adjoint_solver_options, DirectSolverOptions):
                adjoint_solver_options = resolve_direct_solver(adjoint_solver_options, _mp, matrix_view=problem.matrix_view)
            else:
                adjoint_solver_options = resolve_iterative_solver(adjoint_solver_options, _mp)
            print(f"[feax] Auto adjoint solver ({_category}): backend={_backend.name}, matrix_property={_mp.name} -> {adjoint_solver_options.solver}")

    # Validate cuDSS options
    if _is_cudss(solver_options):
        opts = solver_options.cudss_options if isinstance(solver_options, DirectSolverOptions) else solver_options.cudss_options
        _validate_cudss_options(opts, "primary")
    if _is_cudss(adjoint_solver_options):
        opts = adjoint_solver_options.cudss_options if isinstance(adjoint_solver_options, DirectSolverOptions) else adjoint_solver_options.cudss_options
        _validate_cudss_options(opts, "adjoint")

    # Pre-create linear solve functions.
    # For cuDSS, also pre-warm with sample Jacobian so that CuDSSSolver is
    # initialized with concrete values. This avoids tracer leaks when
    # jax.grad traces through custom_vjp (the nonlocal cudss_solver is
    # already set, so the initialization branch is skipped during tracing).
    linear_solve_fn = create_linear_solve_fn(solver_options)
    _adjoint_fn_shared = adjoint_solver_options is solver_options
    if _adjoint_fn_shared:
        adjoint_linear_solve_fn = linear_solve_fn
    else:
        adjoint_linear_solve_fn = create_linear_solve_fn(adjoint_solver_options)

    if _sample_J is not None:
        _b_tmp = jnp.zeros(_sample_J.shape[0])
        if _is_cudss(solver_options):
            print("[feax] Pre-warming cuDSS solver (forward) with sample Jacobian...")
            linear_solve_fn(_sample_J, _b_tmp, _b_tmp)
            print("[feax] cuDSS solver (forward) initialized.")
        if _is_cudss(adjoint_solver_options) and adjoint_linear_solve_fn is not linear_solve_fn:
            print("[feax] Pre-warming cuDSS solver (adjoint) with sample Jacobian...")
            adjoint_linear_solve_fn(_sample_J, _b_tmp, _b_tmp)
            print("[feax] cuDSS solver (adjoint) initialized.")

    # internal_jit: wrap linear solve functions with jax.jit so that each
    # Newton iteration's linear solve is compiled independently.  Must be
    # applied AFTER cuDSS pre-warming to avoid tracer leaks.
    if internal_jit:
        import warnings as _warnings
        if iter_num == 1:
            _warnings.warn(
                "[feax] internal_jit=True is ignored when iter_num=1: "
                "the linear solver is called only once per solve and "
                "internal JIT provides no benefit. "
                "Use jax.jit(solver) to JIT the entire solve instead.",
                UserWarning,
                stacklevel=2,
            )
        else:
            print("[feax] internal_jit=True: JIT-compiling internal linear solver (forward).")
            linear_solve_fn = jax.jit(linear_solve_fn)
            if _adjoint_fn_shared:
                adjoint_linear_solve_fn = linear_solve_fn
            else:
                print("[feax] internal_jit=True: JIT-compiling internal linear solver (adjoint).")
                adjoint_linear_solve_fn = jax.jit(adjoint_linear_solve_fn)

    # Extract x0_fn from the appropriate option type
    _x0_fn_from_opts = None
    if isinstance(solver_options, IterativeSolverOptions):
        _x0_fn_from_opts = solver_options.x0_fn
    elif isinstance(solver_options, SolverOptions):
        _x0_fn_from_opts = solver_options.linear_solver_x0_fn
    x0_fn = _x0_fn_from_opts or create_x0(
        bc_rows=bc.bc_rows,
        bc_vals=bc.bc_vals,
    )

    # For new option types, Newton-level parameters use defaults from SolverOptions.
    # newton_solve/linear_solve still receive a SolverOptions for Newton control.
    if isinstance(solver_options, _new_option_types):
        _newton_opts = SolverOptions(
            verbose=solver_options.verbose,
            check_convergence=solver_options.check_convergence,
            convergence_threshold=solver_options.convergence_threshold,
        )
    else:
        _newton_opts = solver_options

    if iter_num is None:
        armijo_fn = create_armijo_line_search_jax(
            res_bc_func,
            c1=_newton_opts.line_search_c1,
            rho=_newton_opts.line_search_rho,
            max_backtracks=_newton_opts.line_search_max_backtracks
        )
        solve_fn = lambda internal_vars, initial_sol: newton_solve(
            J_bc_func, res_bc_func, initial_sol, bc, _newton_opts, internal_vars,
            linear_solve_fn=linear_solve_fn, armijo_search_fn=armijo_fn, x0_fn=x0_fn
        )
    elif iter_num == 1:
        solve_fn = lambda internal_vars, initial_sol: linear_solve(
            J_bc_func, res_bc_func, initial_sol, bc, _newton_opts, problem.matrix_view, internal_vars,
            linear_solve_fn=linear_solve_fn, x0_fn=x0_fn
        )
    else:
        armijo_fn = create_armijo_line_search_scan(
            res_bc_func,
            c1=_newton_opts.line_search_c1,
            rho=_newton_opts.line_search_rho,
            max_backtracks=_newton_opts.line_search_max_backtracks
        )
        solve_fn = lambda internal_vars, initial_sol: newton_solve_fori(
            J_bc_func, res_bc_func, initial_sol, bc, _newton_opts, iter_num, internal_vars,
            linear_solve_fn=linear_solve_fn, armijo_search_fn=armijo_fn, x0_fn=x0_fn
        )

    # Determine if the solver can omit initial_guess.
    # Direct solvers on linear problems (iter_num == 1) don't need an
    # initial guess because there is no Newton iteration.  Non-linear
    # problems always need one because the Newton loop updates it.
    def _solver_can_omit_x0(opts, iters):
        """Return True when the caller may omit initial_guess."""
        if iters != 1:
            return False          # Newton solver always needs initial_guess
        if isinstance(opts, DirectSolverOptions):
            return True
        if isinstance(opts, IterativeSolverOptions):
            return False
        if isinstance(opts, SolverOptions):
            return opts.linear_solver not in {"cg", "bicgstab", "gmres"}
        return False

    _can_omit_x0 = _solver_can_omit_x0(solver_options, iter_num)
    if _can_omit_x0:
        from .utils import zero_like_initial_guess as _zero_init
        _default_initial_guess = _zero_init(problem, bc)

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
            J_adjoint, v_vec, adjoint_solver_options, problem.matrix_view, bc,
            linear_solve_fn=adjoint_linear_solve_fn
        )
        sol_list = problem.unflatten_fn_sol_list(sol)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
        vjp_result = jax.tree_util.tree_map(_safe_negate, vjp_result)

        return (vjp_result, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    if not _can_omit_x0:
        return differentiable_solve

    # Linear + direct solver wrapper: initial_guess is optional
    import warnings

    def solver_wrapper(internal_vars, initial_guess=None):
        if initial_guess is not None:
            warnings.warn(
                "initial_guess is ignored for direct solvers. "
                "You can omit it: solver(internal_vars)",
                stacklevel=2,
            )
        return differentiable_solve(internal_vars, _default_initial_guess)

    return solver_wrapper
