"""Newton solver utilities for FEAX.

Contains x0 initialization, Armijo line-search implementations,
and Newton solve variants used by ``feax.solver``.
"""

import logging
from typing import Optional

import jax
import jax.numpy as np

from ..assembler import create_J_bc_function, create_res_bc_function
from ..problem import MatrixView
from .common import (
    _safe_negate,
    check_convergence,
    create_jacobi_preconditioner,
    create_linear_solve_fn,
    prewarm_cudss_solvers,
    create_x0,
)
from .linear import linear_solve_adjoint
from .options import NewtonOptions

logger = logging.getLogger(__name__)


def _ensure_verbose_logging():
    """Ensure the logger can emit INFO messages when verbose=True."""
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
        if not logger.handlers and not logging.root.handlers:
            logger.addHandler(logging.StreamHandler())


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
    J_bc_parametric=None,
    res_bc_parametric=None,
    x0_fn_parametric=None,
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
        return lambda internal_vars, initial_sol, bc_arg=None: newton_solve(
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

    # Build parametric Armijo for fori_loop vmap path
    armijo_fn_parametric = None
    if res_bc_parametric is not None:
        armijo_fn_parametric = create_armijo_line_search_scan_parametric(
            res_bc_parametric,
            c1=newton_options.line_search_c1,
            rho=newton_options.line_search_rho,
            max_backtracks=newton_options.line_search_max_backtracks,
        )

    def _solve(internal_vars, initial_sol, bc_arg=None):
        if bc_arg is not None and J_bc_parametric is not None:
            return newton_solve_fori_parametric(
                J_bc_parametric, res_bc_parametric,
                initial_sol, bc_arg,
                newton_options, iter_num,
                linear_solver_options=linear_solver_options,
                internal_vars=internal_vars,
                linear_solve_fn=linear_solve_fn,
                armijo_search_fn=armijo_fn_parametric,
                x0_fn_parametric=x0_fn_parametric,
                matrix_view=matrix_view,
            )
        return newton_solve_fori(
            J_bc_func, res_bc_func,
            initial_sol, bc,
            newton_options, iter_num,
            linear_solver_options=linear_solver_options,
            internal_vars=internal_vars,
            linear_solve_fn=linear_solve_fn,
            armijo_search_fn=armijo_fn,
            x0_fn=x0_fn,
            matrix_view=matrix_view,
        )

    return _solve


def _create_differentiable_newton_solver(
    problem,
    bc,
    J_bc_func,
    res_bc_func,
    solve_fn,
    adjoint_solver_options,
    adjoint_linear_solve_fn,
    extra_res_bc_fn=None,
    bulk_J_bc_func=None,
    J_bc_parametric=None,
    res_bc_parametric=None,
    symmetric_bc=True,
):
    """Create custom-VJP Newton solver wrapper around a prepared solve_fn.

    Parameters
    ----------
    extra_res_bc_fn : callable, optional
        BC-zeroed extra residual function (for hybrid adjoint solve).
    bulk_J_bc_func : callable, optional
        Original (unwrapped) bulk Jacobian function.  Required when
        ``extra_res_bc_fn`` is provided so the adjoint can construct
        ``J_adjoint_matvec(v) = J_bulk^T @ v + VJP(extra_res_bc, v)``.
    J_bc_parametric : callable, optional
        Parametric Jacobian ``(sol, iv, bc) -> BCOO`` for vmap over bc_vals.
    res_bc_parametric : callable, optional
        Parametric residual ``(sol, iv, bc) -> array`` for vmap over bc_vals.
    symmetric_bc : bool, default True
        Whether symmetric BC elimination is used in the forward solve.
        When True, a correction is applied in the backward pass to account
        for the zeroed BC coupling columns (K_fb) in the Jacobian, which
        are needed for correct gradients w.r.t. bc_vals.
    """
    _has_parametric = J_bc_parametric is not None and res_bc_parametric is not None
    # When symmetric BC elimination is used and we need bc_vals gradients,
    # the adjoint at BC DOFs must be corrected.  The symmetric Jacobian
    # zeros out K_fb columns, but the true residual Jacobian keeps them.
    # The correction is: λ_bc = v_bc - K_fb^T @ λ_free, computed via
    # the bulk (un-eliminated) Jacobian.
    _needs_adjoint_bc_correction = symmetric_bc and _has_parametric

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess, bc_arg):
        return solve_fn(internal_vars, initial_guess, bc_arg)[0]

    def f_fwd(internal_vars, initial_guess, bc_arg):
        sol = differentiable_solve(internal_vars, initial_guess, bc_arg)
        return sol, (internal_vars, sol, bc_arg)

    def f_bwd(res, v):
        internal_vars, sol, effective_bc = res

        # Choose parametric or closure-based functions for adjoint
        if _has_parametric:
            _res_fn = lambda dofs, iv: res_bc_parametric(dofs, iv, effective_bc)
            _J_fn = lambda s, iv: J_bc_parametric(s, iv, effective_bc)
        else:
            _res_fn = res_bc_func
            _J_fn = J_bc_func

        v_vec = jax.flatten_util.ravel_pytree(v)[0]

        if extra_res_bc_fn is not None and bulk_J_bc_func is not None:
            # Hybrid adjoint: J_adjoint_matvec(v) = J_bulk^T @ v + VJP(extra_res_bc, v)
            J_bulk = bulk_J_bc_func(sol, internal_vars)
            _, vjp_extra = jax.vjp(extra_res_bc_fn, sol)

            def adjoint_matvec(w):
                Jtw_bulk = J_bulk.T @ w
                Jtw_extra, = vjp_extra(w)
                return Jtw_bulk + Jtw_extra

            adjoint_vec = linear_solve_adjoint(
                adjoint_matvec,
                v_vec,
                adjoint_solver_options,
                problem.matrix_view,
                effective_bc,
                linear_solve_fn=adjoint_linear_solve_fn,
            )
        else:
            J = _J_fn(sol, internal_vars)

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
                effective_bc,
                linear_solve_fn=adjoint_linear_solve_fn,
            )

        # Correct adjoint at BC DOFs for symmetric BC elimination.
        # The symmetric Jacobian J_sym has zeroed BC columns, so the adjoint
        # solve J_sym^T λ = v gives correct λ_free but wrong λ_bc = v_bc.
        # The true adjoint satisfies J_true^T λ = v where J_true keeps BC
        # columns (K_fb).  Since J_true^T is block lower-triangular:
        #   λ_free = K_ff^{-T} v_free  (same as symmetric, already correct)
        #   λ_bc = v_bc - K_fb^T λ_free  (correction needed)
        if _needs_adjoint_bc_correction:
            from ..assembler import _get_J
            sol_list_for_correction = problem.unflatten_fn_sol_list(sol)
            K_bulk = _get_J(problem, sol_list_for_correction, internal_vars)
            # Isolate λ_free by zeroing BC entries
            lambda_free_padded = adjoint_vec.at[effective_bc.bc_rows].set(0.0)
            # K_fb^T @ λ_free = (K_bulk^T @ λ_free_padded)[bc_rows]
            correction = (K_bulk.T @ lambda_free_padded)[effective_bc.bc_rows]
            adjoint_vec = adjoint_vec.at[effective_bc.bc_rows].set(
                v_vec[effective_bc.bc_rows] - correction
            )

        # VJP of residual w.r.t. internal_vars and bc
        adjoint_list = problem.unflatten_fn_sol_list(adjoint_vec)
        sol_list = problem.unflatten_fn_sol_list(sol)

        if _has_parametric:
            def res_fn_params(iv, bc_arg):
                dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
                return problem.unflatten_fn_sol_list(
                    res_bc_parametric(dofs, iv, bc_arg)
                )

            _, f_vjp = jax.vjp(res_fn_params, internal_vars, effective_bc)
            vjp_iv, vjp_bc = f_vjp(adjoint_list)
            vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
            vjp_bc = jax.tree_util.tree_map(_safe_negate, vjp_bc)
        else:
            def res_fn_iv(iv):
                dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
                return problem.unflatten_fn_sol_list(
                    res_bc_func(dofs, iv)
                )

            _, f_vjp = jax.vjp(res_fn_iv, internal_vars)
            vjp_iv, = f_vjp(adjoint_list)
            vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
            vjp_bc = None

        return (vjp_iv, None, vjp_bc)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    def solver_with_bc(internal_vars, initial_guess, bc=None):
        """Differentiable Newton solver with optional BC override.

        Parameters
        ----------
        internal_vars : InternalVars
            Internal variables (material parameters, loads, etc.).
        initial_guess : jnp.ndarray
            Initial guess for the Newton solve.
        bc : DirichletBC, optional
            Override boundary conditions.  When provided, the forward
            solve uses this BC instead of the one captured at construction.
            The ``bc_rows`` / ``bc_mask`` must match the original BC
            (same constrained DOFs); only ``bc_vals`` should differ.
        """
        from ..DCboundary import DirichletBC
        effective_bc = bc if isinstance(bc, DirichletBC) else _default_bc
        return differentiable_solve(internal_vars, initial_guess, effective_bc)

    _default_bc = bc
    return solver_with_bc


def create_newton_solver(
    problem,
    bc,
    linear_options,
    adjoint_linear_options,
    iter_num: Optional[int],
    newton_options: Optional[NewtonOptions] = None,
    internal_vars=None,
    extra_residual_fn=None,
    symmetric_bc: bool = True,
):
    """Create a differentiable Newton solver (iter_num is None or >1).

    Parameters
    ----------
    extra_residual_fn : callable, optional
        Additional residual: ``extra_residual_fn(sol_flat) -> residual_flat``.
        When provided, uses hybrid matrix-free Newton-Krylov: feax assembles
        the bulk Jacobian (sparse), and the extra contribution's JVP is
        computed via ``jax.jvp``.  The combined matvec is:
        ``J_total @ v = J_bulk @ v + jvp(extra_res_bc, sol, v)``.
        Dirichlet BC rows of the extra residual are zeroed automatically.
    """
    if iter_num == 1:
        raise ValueError(
            "create_newton_solver does not support iter_num==1. "
            "Use linear_solver.create_linear_solver for that path."
        )

    J_bc_func = create_J_bc_function(problem, bc, symmetric=symmetric_bc)
    res_bc_func = create_res_bc_function(problem, bc)

    # --- Hybrid matrix-free path ---
    # When extra_residual_fn is provided, JAX loop primitives (while_loop,
    # fori_loop) cannot be used because the hybrid Jacobian is a Python
    # callable (matvec), not a JAX array.  Use a Python Newton loop instead.
    if extra_residual_fn is not None:
        bc_rows = bc.bc_rows

        def extra_res_bc(sol):
            """Extra residual with Dirichlet BC rows zeroed."""
            return extra_residual_fn(sol).at[bc_rows].set(0.0)

        if newton_options is None:
            newton_options = NewtonOptions()

        from .options import IterativeSolverOptions
        if not isinstance(linear_options, IterativeSolverOptions):
            raise TypeError(
                "extra_residual_fn requires IterativeSolverOptions "
                f"(got {type(linear_options).__name__})."
            )

        iterative_solve_fn = create_linear_solve_fn(linear_options)
        adjoint_fn_shared = adjoint_linear_options is linear_options
        adjoint_linear_solve_fn = (
            iterative_solve_fn if adjoint_fn_shared
            else create_linear_solve_fn(adjoint_linear_options)
        )

        if newton_options.internal_jit:
            logger.info(
                "extra_residual_fn: JIT-compiling res_bc_func and J_bc_func. "
                "The hybrid Newton loop itself runs as a Python loop (not JAX-traced)."
            )
            J_bc_func = jax.jit(J_bc_func)
            res_bc_func = jax.jit(res_bc_func)

        _verbose = linear_options.verbose

        def _hybrid_total_res(sol, internal_vars):
            """Total residual = bulk + extra (with BCs zeroed)."""
            return res_bc_func(sol, internal_vars) + extra_res_bc(sol)

        _iterative_solver = getattr(
            jax.scipy.sparse.linalg, linear_options.solver
        )

        def hybrid_solve_fn(internal_vars, initial_sol, bc_arg=None):
            """Python-loop Newton solver with hybrid matrix-free Jacobian."""
            sol = initial_sol
            res_total = _hybrid_total_res(sol, internal_vars)
            res_norm = float(np.linalg.norm(res_total))
            initial_res_norm = res_norm

            if _verbose:
                _ensure_verbose_logging()
                logger.info(f"[hybrid Newton] initial res_norm = {initial_res_norm:.6e}")

            for it in range(newton_options.max_iter):
                if res_norm < newton_options.tol:
                    break
                if it > 0 and res_norm / (initial_res_norm + 1e-30) < newton_options.rel_tol:
                    break

                # Bulk Jacobian (sparse) + hybrid matvec
                J_bulk = J_bc_func(sol, internal_vars)

                def matvec(v, _sol=sol):
                    _, Jv_extra = jax.jvp(extra_res_bc, (_sol,), (v,))
                    return J_bulk @ v + Jv_extra

                # Jacobi preconditioner from bulk Jacobian diagonal
                M = None
                if linear_options.use_jacobi_preconditioner and linear_options.preconditioner is None:
                    M = create_jacobi_preconditioner(J_bulk, linear_options.jacobi_shift)
                else:
                    M = linear_options.preconditioner

                x0 = np.zeros_like(sol)
                _kw = dict(
                    x0=x0, M=M,
                    tol=linear_options.tol,
                    atol=linear_options.atol,
                    maxiter=linear_options.maxiter,
                )
                if linear_options.solver == 'gmres':
                    _kw['restart'] = linear_options.restart or min(200, sol.shape[0])
                du, _ = _iterative_solver(matvec, -res_total, **_kw)
                du_norm = float(np.linalg.norm(du))

                # Early termination if linear solver stagnated
                if du_norm < 1e-30:
                    if _verbose:
                        logger.info(
                            f"[hybrid Newton] iter {it:3d}: "
                            f"linear solver returned du=0, stopping"
                        )
                    break

                # Simple backtracking line search (accept any decrease)
                alpha = 1.0
                _ls_success = False
                for _bt in range(newton_options.line_search_max_backtracks):
                    trial_sol = sol + alpha * du
                    trial_res = _hybrid_total_res(trial_sol, internal_vars)
                    trial_norm = float(np.linalg.norm(trial_res))
                    if trial_norm < res_norm:
                        _ls_success = True
                        break
                    alpha *= newton_options.line_search_rho

                if not _ls_success:
                    if _verbose:
                        logger.info(
                            f"[hybrid Newton] iter {it:3d}: "
                            f"line search failed, stopping"
                        )
                    break

                sol = trial_sol
                res_total = trial_res
                res_norm = trial_norm

                if _verbose:
                    logger.info(
                        f"[hybrid Newton] iter {it:3d}: "
                        f"res_norm = {res_norm:.6e}, "
                        f"rel = {res_norm / (initial_res_norm + 1e-30):.6e}, "
                        f"du_norm = {du_norm:.6e}, alpha = {alpha:.4f}"
                    )

            rel = res_norm / (initial_res_norm + 1e-30)
            converged = (res_norm < newton_options.tol) or (rel < newton_options.rel_tol)
            if _verbose:
                status = "converged" if converged else "NOT converged"
                logger.info(
                    f"[hybrid Newton] {status} in {it + 1} iterations, "
                    f"res_norm = {res_norm:.6e}, rel = {rel:.6e}"
                )
            if not converged:
                logger.warning(
                    f"[hybrid Newton] did NOT converge after {it + 1} iterations "
                    f"(res_norm = {res_norm:.6e}, tol = {newton_options.tol:.1e}, "
                    f"rel = {rel:.6e}, rel_tol = {newton_options.rel_tol:.1e})"
                )

            return sol, res_norm, converged

        return _create_differentiable_newton_solver(
            problem=problem,
            bc=bc,
            J_bc_func=J_bc_func,
            res_bc_func=res_bc_func,
            solve_fn=hybrid_solve_fn,
            adjoint_solver_options=adjoint_linear_options,
            adjoint_linear_solve_fn=adjoint_linear_solve_fn,
            extra_res_bc_fn=extra_res_bc,
            bulk_J_bc_func=J_bc_func,
            symmetric_bc=symmetric_bc,
        )

    # --- Standard path (no extra residual) ---
    linear_solve_fn = create_linear_solve_fn(linear_options)
    adjoint_fn_shared = adjoint_linear_options is linear_options
    if adjoint_fn_shared:
        adjoint_linear_solve_fn = linear_solve_fn
    else:
        adjoint_linear_solve_fn = create_linear_solve_fn(adjoint_linear_options)

    if newton_options is None:
        newton_options = NewtonOptions()

    # ------------------------------------------------------------------ #
    # Python-loop path  (make_jittable=False, the default)                #
    #                                                                      #
    # Each component (residual, Jacobian, linear solve) is compiled        #
    # separately on first call instead of being fused into one giant XLA   #
    # program.  This avoids the multi-minute JIT compilation that          #
    # jax.lax.fori_loop / while_loop incurs for large 3-D problems.        #
    #                                                                      #
    # When internal_jit=True the three functions are explicitly wrapped     #
    # with jax.jit so they are always dispatched as compiled kernels.       #
    # When internal_jit=False JAX still compiles them on first call (eager  #
    # tracing), but the Python-level loop keeps the overall graph small.    #
    # ------------------------------------------------------------------ #
    if not newton_options.make_jittable:
        _res_fn = J_bc_func_for_adjoint = J_bc_func
        _J_fn = res_bc_func

        # Parametric functions: take bc as an explicit argument
        # so that a single JIT-compiled function can serve different bc_vals,
        # and so the backward pass can compute gradients w.r.t. bc_vals.
        from ..assembler import create_J_bc_parametric, create_res_bc_parametric
        from ..DCboundary import apply_boundary_to_res as _apply_res_bc
        _res_bc_parametric = create_res_bc_parametric(problem)
        _J_bc_parametric = create_J_bc_parametric(problem, symmetric=symmetric_bc)

        if newton_options.internal_jit:
            logger.info(
                "Python-loop Newton: JIT-compiling res_bc_func, J_bc_func, "
                "and linear_solve_fn individually."
            )
            # Prewarm cuDSS BEFORE JIT-wrapping so that _warmup() runs with
            # concrete (non-traced) matrix indices. Without this, the first
            # call to jax.jit(linear_solve_fn) would trace through _warmup()
            # and hit TracerArrayConversionError on onp.asarray(A.indices).
            from .common import prewarm_cudss_solvers
            prewarm_cudss_solvers(
                problem=problem,
                bc=bc,
                internal_vars=internal_vars,
                J_bc_func=J_bc_func,
                forward_options=linear_options,
                adjoint_options=adjoint_linear_options,
                forward_solve_fn=linear_solve_fn,
                adjoint_solve_fn=adjoint_linear_solve_fn,
            )
            J_bc_func = jax.jit(J_bc_func)
            res_bc_func = jax.jit(res_bc_func)
            _res_bc_parametric = jax.jit(_res_bc_parametric)
            linear_solve_fn = jax.jit(linear_solve_fn)
            if adjoint_fn_shared:
                adjoint_linear_solve_fn = linear_solve_fn
            else:
                adjoint_linear_solve_fn = jax.jit(adjoint_linear_solve_fn)
        else:
            logger.info(
                "Python-loop Newton: running with eager JAX execution "
                "(set internal_jit=True for explicit JIT on each component)."
            )

        from .options import IterativeSolverOptions
        x0_fn_from_opts = None
        if isinstance(linear_options, IterativeSolverOptions):
            x0_fn_from_opts = linear_options.x0_fn
        x0_fn = x0_fn_from_opts or create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals)

        def python_loop_solve_fn(internal_vars, initial_sol, bc_arg=None):
            if bc_arg is not None and bc_arg is not bc:
                # Use parametric residual that takes bc as argument
                _res_bc = lambda sol, iv: _res_bc_parametric(sol, iv, bc_arg)
                _x0 = create_x0(bc_rows=bc_arg.bc_rows, bc_vals=bc_arg.bc_vals)
                _effective_bc = bc_arg
            else:
                _res_bc = res_bc_func
                _x0 = x0_fn
                _effective_bc = bc
            return newton_solve_py(
                J_bc_applied=J_bc_func,
                res_bc_applied=_res_bc,
                initial_guess=initial_sol,
                bc=_effective_bc,
                newton_options=newton_options,
                linear_solver_options=linear_options,
                internal_vars=internal_vars,
                linear_solve_fn=linear_solve_fn,
                x0_fn=_x0,
                matrix_view=problem.matrix_view,
            )

        return _create_differentiable_newton_solver(
            problem=problem,
            bc=bc,
            J_bc_func=J_bc_func,
            res_bc_func=res_bc_func,
            solve_fn=python_loop_solve_fn,
            adjoint_solver_options=adjoint_linear_options,
            adjoint_linear_solve_fn=adjoint_linear_solve_fn,
            J_bc_parametric=_J_bc_parametric,
            res_bc_parametric=_res_bc_parametric,
            symmetric_bc=symmetric_bc,
        )

    # ------------------------------------------------------------------ #
    # JAX-traceable path  (make_jittable=True)                            #
    # Uses fori_loop / while_loop – the entire Newton loop is compiled     #
    # into one XLA program.  iter_num must be provided.                   #
    # ------------------------------------------------------------------ #
    if iter_num is None:
        raise ValueError(
            "make_jittable=True requires iter_num to be specified. "
            "Pass an integer iter_num (e.g. iter_num=4) or set "
            "make_jittable=False (default) to use the Python-loop path."
        )

    prewarm_cudss_solvers(
        problem=problem,
        bc=bc,
        internal_vars=internal_vars,
        J_bc_func=J_bc_func,
        forward_options=linear_options,
        adjoint_options=adjoint_linear_options,
        forward_solve_fn=linear_solve_fn,
        adjoint_solve_fn=adjoint_linear_solve_fn,
    )

    if newton_options.internal_jit:
        logger.info("make_jittable + internal_jit=True: JIT-compiling linear solver (forward).")
        linear_solve_fn = jax.jit(linear_solve_fn)
        if adjoint_fn_shared:
            adjoint_linear_solve_fn = linear_solve_fn
        else:
            logger.info("make_jittable + internal_jit=True: JIT-compiling linear solver (adjoint).")
            adjoint_linear_solve_fn = jax.jit(adjoint_linear_solve_fn)

    from .options import IterativeSolverOptions
    x0_fn_from_opts = None
    if isinstance(linear_options, IterativeSolverOptions):
        x0_fn_from_opts = linear_options.x0_fn
    x0_fn = x0_fn_from_opts or create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals)

    # Parametric functions for vmap over bc_vals (fori_loop path only)
    from ..assembler import create_J_bc_parametric, create_res_bc_parametric
    from .common import create_x0_parametric
    J_bc_parametric_fn = create_J_bc_parametric(problem, symmetric=symmetric_bc)
    res_bc_parametric_fn = create_res_bc_parametric(problem)
    x0_fn_parametric = create_x0_parametric()

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
        J_bc_parametric=J_bc_parametric_fn,
        res_bc_parametric=res_bc_parametric_fn,
        x0_fn_parametric=x0_fn_parametric,
    )

    return _create_differentiable_newton_solver(
        problem=problem,
        bc=bc,
        J_bc_func=J_bc_func,
        res_bc_func=res_bc_func,
        solve_fn=solve_fn,
        adjoint_solver_options=adjoint_linear_options,
        adjoint_linear_solve_fn=adjoint_linear_solve_fn,
        J_bc_parametric=J_bc_parametric_fn,
        res_bc_parametric=res_bc_parametric_fn,
        symmetric_bc=symmetric_bc,
    )

def create_armijo_line_search_jax(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX while_loop Armijo line search."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -np.dot(res, res)

        init_state = (1.0, 0, False, sol, res_norm)

        def cond_fn(state):
            _, step, found_good, _, _ = state
            return np.logical_not(found_good) & (step < max_backtracks)

        def body_fn(state):
            alpha, step, _, best_sol, best_norm = state
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = np.linalg.norm(trial_res)

            is_valid = np.logical_not(np.any(np.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            is_acceptable = is_valid & armijo_satisfied

            new_sol = np.where(is_acceptable, trial_sol, best_sol)
            new_norm = np.where(is_acceptable, trial_norm, best_norm)
            new_alpha = np.where(is_acceptable, alpha, alpha * rho)

            return (new_alpha, step + 1, is_acceptable, new_sol, new_norm)

        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        final_alpha, _, found_good, new_sol, new_norm = final_state

        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = np.linalg.norm(fallback_res)

        final_sol = np.where(found_good, new_sol, fallback_sol)
        final_norm = np.where(found_good, new_norm, fallback_norm)
        final_alpha_out = np.where(found_good, final_alpha, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search


def create_armijo_line_search_scan(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX scan-based Armijo line search (vmap-friendly)."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -np.dot(res, res)

        def scan_fn(carry, _):
            alpha, best_sol, best_norm, found_good = carry
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = np.linalg.norm(trial_res)

            is_valid = np.logical_not(np.any(np.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            is_acceptable = is_valid & armijo_satisfied

            should_update = is_acceptable & np.logical_not(found_good)
            new_sol = np.where(should_update, trial_sol, best_sol)
            new_norm = np.where(should_update, trial_norm, best_norm)
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
        fallback_norm = np.linalg.norm(fallback_res)

        final_sol = np.where(found_good, best_sol, fallback_sol)
        final_norm = np.where(found_good, best_norm, fallback_norm)
        final_alpha_out = np.where(found_good, final_alpha / rho, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search


def create_armijo_line_search_scan_parametric(res_bc_parametric, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create scan-based Armijo line search with bc as explicit argument (vmap-friendly)."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars, bc):
        grad_merit = -np.dot(res, res)

        def scan_fn(carry, _):
            alpha, best_sol, best_norm, found_good = carry
            trial_sol = sol + alpha * delta_sol
            trial_res = res_bc_parametric(trial_sol, internal_vars, bc)
            trial_norm = np.linalg.norm(trial_res)

            is_valid = np.logical_not(np.any(np.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            is_acceptable = is_valid & armijo_satisfied

            should_update = is_acceptable & np.logical_not(found_good)
            new_sol = np.where(should_update, trial_sol, best_sol)
            new_norm = np.where(should_update, trial_norm, best_norm)
            new_alpha = alpha * rho
            new_found = found_good | is_acceptable

            return (new_alpha, new_sol, new_norm, new_found), None

        init_carry = (1.0, sol, res_norm, False)
        (final_alpha, best_sol, best_norm, found_good), _ = jax.lax.scan(
            scan_fn, init_carry, None, length=max_backtracks
        )

        fallback_sol = sol + 1e-8 * delta_sol
        fallback_res = res_bc_parametric(fallback_sol, internal_vars, bc)
        fallback_norm = np.linalg.norm(fallback_res)

        final_sol = np.where(found_good, best_sol, fallback_sol)
        final_norm = np.where(found_good, best_norm, fallback_norm)
        final_alpha_out = np.where(found_good, final_alpha / rho, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search


def create_armijo_line_search_python(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create Python-loop Armijo line search (debug path)."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -np.dot(res, res)

        alpha = 1.0
        for _ in range(max_backtracks):
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = np.linalg.norm(trial_res)

            is_valid = not np.any(np.isnan(trial_res))
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
        fallback_norm = np.linalg.norm(fallback_res)
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
    initial_res_norm = np.linalg.norm(initial_res)
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
    initial_res_norm = np.linalg.norm(initial_res)

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


def newton_solve_fori_parametric(
    J_bc_parametric, res_bc_parametric, initial_guess, bc,
    newton_options, num_iters,
    linear_solver_options,
    internal_vars,
    linear_solve_fn, armijo_search_fn, x0_fn_parametric,
    matrix_view: MatrixView = MatrixView.FULL,
):
    """Newton solver using fori_loop with bc as traced argument (vmap-compatible)."""

    def newton_iteration(i, state):
        sol, res_norm = state

        res = res_bc_parametric(sol, internal_vars, bc)
        J = J_bc_parametric(sol, internal_vars, bc)

        x0 = x0_fn_parametric(sol, bc)
        delta_sol = linear_solve_fn(J, -res, x0)
        if linear_solver_options.check_convergence:
            delta_sol = check_convergence(
                A=J, x=delta_sol, b=-res,
                solver_options=linear_solver_options,
                matrix_view=matrix_view,
                solver_label="Newton linear solve",
            )

        new_sol, new_res_norm, alpha, success = armijo_search_fn(
            sol, delta_sol, res, res_norm, internal_vars, bc
        )

        if linear_solver_options.verbose:
            jax.debug.print(
                "Newton iter {iter:3d}: res_norm = {r:.6e}, alpha = {a:.4f}, success = {s}",
                iter=i, r=new_res_norm, a=alpha, s=success,
            )

        return (new_sol, new_res_norm)

    initial_res = res_bc_parametric(initial_guess, internal_vars, bc)
    initial_res_norm = np.linalg.norm(initial_res)

    final_state = jax.lax.fori_loop(
        0, num_iters,
        newton_iteration,
        (initial_guess, initial_res_norm),
    )

    final_sol, final_res_norm = final_state
    converged = final_res_norm < newton_options.tol
    return final_sol, final_res_norm, converged


def newton_solve_py(J_bc_applied, res_bc_applied, initial_guess, bc, newton_options, linear_solver_options,
                    internal_vars=None, P_mat=None,
                    linear_solve_fn=None, x0_fn=None,
                    matrix_view: MatrixView = MatrixView.FULL):
    """Newton solver using Python while loop.

    ``linear_solve_fn`` and ``x0_fn`` can be passed in pre-built (and
    optionally pre-JIT-compiled) by the caller, avoiding redundant
    reconstruction on every time step.
    """
    tol = newton_options.tol
    rel_tol = newton_options.rel_tol
    max_iter = newton_options.max_iter
    line_search_c1 = newton_options.line_search_c1
    line_search_rho = newton_options.line_search_rho
    line_search_max_backtracks = newton_options.line_search_max_backtracks

    if x0_fn is None:
        _legacy_x0_fn = getattr(linear_solver_options, "linear_solver_x0_fn", None)
        if _legacy_x0_fn is not None:
            x0_fn = _legacy_x0_fn
        else:
            x0_fn = create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals, P_mat=P_mat)

    if linear_solve_fn is None:
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
    initial_res_norm = np.linalg.norm(initial_res)
    res_norm = initial_res_norm
    iter_count = 0

    if linear_solver_options.verbose:
        _ensure_verbose_logging()
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

        # Guard: if the linear solver returned NaN/Inf (e.g. bicgstab
        # exceeded maxiter and diverged), do not corrupt the solution.
        if bool(np.any(np.isnan(delta_sol) | np.isinf(delta_sol))):
            logger.warning(
                f"Newton iter {iter_count}: linear solver returned NaN/Inf. "
                "Stopping Newton iterations to prevent solution corruption."
            )
            break

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
