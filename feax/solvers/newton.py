"""Newton solver for FEAX.

Provides:

* :func:`create_newton_solver` — the entry point used by ``feax.solver``. With
  ``extra_residual_fn`` it runs the hybrid matrix-free Newton-Krylov path;
  otherwise it delegates to the unified traced solver below.
* :func:`create_callback_newton_solver` — the standard nonlinear solver. The
  iteration is a traced ``jax.lax.while_loop`` (one Newton step per loop body;
  the graph does not grow with ``max_iter``), so it composes natively with
  ``jax.jit`` / ``jax.vmap`` / ``jax.grad``, with a CSR-direct fused
  residual+Jacobian forward and a single traced adjoint solve.
* :func:`create_armijo_line_search_scan` — the (vmap-friendly) line search.
"""

import logging
from typing import Optional

import jax
import jax.numpy as np

from ..assembler import (
    _get_J_csr,
    create_J_bc_csr_function,
    create_J_bc_csr_parametric,
    create_matfree_Kt_parametric,
    create_matfree_res_J_parametric,
    create_res_bc_parametric,
    create_res_J_bc_csr_parametric,
)
from ..csr import CSRMatrix, transpose_with_maps
from ..problem import MatrixView
from .common import (
    _safe_negate,
    check_convergence,
    create_jacobi_preconditioner,
    create_linear_solve_fn,
    create_x0,
    create_x0_parametric,
    prewarm_direct_solvers,
)
from .linear import linear_solve_adjoint
from .options import DirectSolverOptions, NewtonOptions

logger = logging.getLogger(__name__)


class NewtonLineSearchError(RuntimeError):
    """Raised when Armijo line search exhausts backtracks without a descent step.

    A failed line search means the proposed Newton direction is not a
    descent direction for the residual merit function ½‖r‖².  This is
    almost always a symptom of an inconsistent Jacobian, a bad linear
    solve, or a degenerate state, rather than a problem the line search
    itself can recover from — so the Python-loop Newton path raises this
    by default rather than silently truncating the iteration.
    """


def _ensure_verbose_logging():
    """Ensure the logger can emit INFO messages when verbose=True."""
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
        if not logger.handlers and not logging.root.handlers:
            logger.addHandler(logging.StreamHandler())


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
    def differentiable_solve(internal_vars, initial_guess, bc_arg, sv):
        return solve_fn(internal_vars, initial_guess, bc_arg, sv)[0]

    def f_fwd(internal_vars, initial_guess, bc_arg, sv):
        sol = differentiable_solve(internal_vars, initial_guess, bc_arg, sv)
        return sol, (internal_vars, sol, bc_arg, sv)

    def f_bwd(res, v):
        internal_vars, sol, effective_bc, sv = res

        # Choose parametric or closure-based functions for adjoint
        if _has_parametric:
            _res_fn = lambda dofs, iv: res_bc_parametric(dofs, iv, effective_bc, sv)
            _J_fn = lambda s, iv: J_bc_parametric(s, iv, effective_bc, sv)
        else:
            _res_fn = res_bc_func
            _J_fn = J_bc_func

        v_vec = jax.flatten_util.ravel_pytree(v)[0]

        if extra_res_bc_fn is not None and bulk_J_bc_func is not None:
            # Hybrid adjoint: J_adjoint_matvec(v) = J_bulk^T @ v + VJP(extra_res_bc, v)
            J_bulk = bulk_J_bc_func(sol, internal_vars, effective_bc, sv)
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
            sol_list_for_correction = problem.unflatten_fn_sol_list(sol)
            kdata, kindptr, kindices = _get_J_csr(
                problem, sol_list_for_correction, internal_vars, sv)
            K_bulk = CSRMatrix(kdata, kindptr, kindices,
                               (problem.num_total_dofs_all_vars,) * 2)
            # Isolate λ_free by zeroing BC entries
            lambda_free_padded = adjoint_vec.at[effective_bc.bc_rows].set(0.0)
            # K_fb^T @ λ_free = (K_bulk^T @ λ_free_padded)[bc_rows]
            correction = K_bulk.rmatvec(lambda_free_padded)[effective_bc.bc_rows]
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
                    res_bc_parametric(dofs, iv, bc_arg, sv)
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

        return (vjp_iv, None, vjp_bc, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    def solver_with_bc(internal_vars, initial_guess, bc=None, static_vars=None):
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
        static_vars : StaticVars, optional
            Structural arrays passed as runtime arguments to the jitted
            per-iteration kernels (and the traced adjoint), so nothing
            mesh-sized is baked into their compilation caches.
        """
        from ..DCboundary import DirichletBC
        effective_bc = bc if isinstance(bc, DirichletBC) else _default_bc
        return differentiable_solve(internal_vars, initial_guess, effective_bc, static_vars)

    _default_bc = bc
    return solver_with_bc


def create_newton_solver(
    problem,
    bc,
    linear_options,
    adjoint_linear_options,
    newton_options: Optional[NewtonOptions] = None,
    internal_vars=None,
    extra_residual_fn=None,
    symmetric_bc: bool = True,
):
    """Create a differentiable nonlinear Newton solver.

    The Newton iteration always runs adaptively (to ``newton_options.tol`` /
    ``rel_tol``, capped at ``max_iter``). For a single linear solve use
    :func:`feax.solvers.linear.create_linear_solver` instead.

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

        from .options import KrylovSolverOptions
        if not isinstance(linear_options, KrylovSolverOptions):
            raise TypeError(
                "extra_residual_fn requires KrylovSolverOptions "
                f"(got {type(linear_options).__name__})."
            )

        iterative_solve_fn = create_linear_solve_fn(linear_options)
        adjoint_fn_shared = adjoint_linear_options is linear_options
        adjoint_linear_solve_fn = (
            iterative_solve_fn if adjoint_fn_shared
            else create_linear_solve_fn(adjoint_linear_options)
        )

        # Bulk Jacobian / residual as PARAMETRIC functions: (sol, iv, bc, sv).
        # bc and StaticVars are runtime arguments of the jit boundary, so the
        # per-iteration compiled kernels bake no mesh-sized constants and the
        # BC override (``solver(..., bc=...)``) reaches the forward loop. The
        # hybrid Newton loop runs as a Python loop (the matrix-free Jacobian is
        # a callable, not a JAX array); jitting the per-call kernels makes each
        # dispatch one compiled program.
        J_bc_param = create_J_bc_csr_parametric(problem, symmetric=symmetric_bc)
        res_bc_param = create_res_bc_parametric(problem)
        J_bc_param_jit = jax.jit(J_bc_param)
        res_bc_param_jit = jax.jit(res_bc_param)

        _verbose = linear_options.verbose

        def _hybrid_total_res(sol, internal_vars, bc_arg, sv):
            """Total residual = bulk + extra (with BCs zeroed)."""
            return res_bc_param_jit(sol, internal_vars, bc_arg, sv) + extra_res_bc(sol)

        _iterative_solver = getattr(
            jax.scipy.sparse.linalg, linear_options.solver
        )

        def hybrid_solve_fn(internal_vars, initial_sol, bc_arg, sv=None):
            """Python-loop Newton solver with hybrid matrix-free Jacobian."""
            sol = initial_sol
            res_total = _hybrid_total_res(sol, internal_vars, bc_arg, sv)
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
                J_bulk = J_bc_param_jit(sol, internal_vars, bc_arg, sv)

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
                    trial_res = _hybrid_total_res(trial_sol, internal_vars, bc_arg, sv)
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
            J_bc_func=None,
            res_bc_func=None,
            solve_fn=hybrid_solve_fn,
            adjoint_solver_options=adjoint_linear_options,
            adjoint_linear_solve_fn=adjoint_linear_solve_fn,
            extra_res_bc_fn=extra_res_bc,
            bulk_J_bc_func=J_bc_param,
            J_bc_parametric=J_bc_param,
            res_bc_parametric=res_bc_param,
            symmetric_bc=symmetric_bc,
        )

    # --- Standard path (no extra residual): unified callback solver ---
    return create_callback_newton_solver(
        problem=problem,
        bc=bc,
        linear_options=linear_options,
        adjoint_linear_options=adjoint_linear_options,
        newton_options=newton_options,
        internal_vars=internal_vars,
        symmetric_bc=symmetric_bc,
    )


def create_armijo_line_search_scan(res_bc_applied, c1=1e-4, rho=0.5, max_backtracks=30):
    """Create JAX scan-based Armijo line search (vmap-friendly)."""

    def line_search(sol, delta_sol, res, res_norm, internal_vars=None):
        grad_merit = -np.dot(res, res)

        def scan_fn(carry, _):
            alpha, best_sol, best_norm, best_alpha, found_good = carry
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
            new_alpha_accepted = np.where(should_update, alpha, best_alpha)
            new_alpha = alpha * rho
            new_found = found_good | is_acceptable

            return (new_alpha, new_sol, new_norm, new_alpha_accepted, new_found), None

        # ``best_alpha`` records the *actual* alpha at which the Armijo
        # condition was first satisfied — this is what callers care about.
        # Initialised to NaN so it's obvious if the line search fails.
        init_carry = (1.0, sol, res_norm, np.nan, False)
        (_, best_sol, best_norm, best_alpha, found_good), _ = jax.lax.scan(
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
        final_alpha_out = np.where(found_good, best_alpha, 1e-8)

        return final_sol, final_norm, final_alpha_out, found_good

    return line_search



def create_callback_newton_solver(
    problem,
    bc,
    linear_options,
    adjoint_linear_options,
    newton_options: NewtonOptions = None,
    internal_vars=None,
    symmetric_bc: bool = True,
):
    """Create a differentiable, jit/vmap-safe Newton solver (traced loop).

    The Newton iteration (inherently data-dependent) is a
    :func:`jax.lax.while_loop` whose body is one Newton step (fused CSR
    residual+Jacobian assembly + linear solve + Armijo line search). The body
    is traced once, so the compiled graph holds a single step regardless of
    ``max_iter``, and the solver composes natively with ``jax.jit`` /
    ``jax.vmap`` / ``jax.grad``. The backward pass is a single adjoint linear
    solve plus the residual VJP (custom_vjp — nothing differentiates through
    the loop).

    ``raise_on_line_search_failure`` is honored for eager calls only; under
    jit/vmap a failed line search stops the iteration (no exception can be
    raised from traced values).

    Returns ``solver(internal_vars, initial_guess, bc=None, static_vars=None)
    -> solution``.
    """
    if newton_options is None:
        newton_options = NewtonOptions()

    # Operator representation follows the solve method: direct factorizes the
    # assembled CSR matrix; Krylov (cg/bicgstab/gmres) needs only a matvec, so
    # it uses the matrix-free operator (a residual JVP) — no assembly.
    _fwd_direct = isinstance(linear_options, DirectSolverOptions)
    _adj_direct = isinstance(adjoint_linear_options, DirectSolverOptions)

    res_param = create_res_bc_parametric(problem)
    # Forward: fused residual+CSR Jacobian (direct) or residual + matrix-free
    # matvec (Krylov).
    res_J_param = create_res_J_bc_csr_parametric(problem, symmetric=symmetric_bc)
    matfree_res_J_param = create_matfree_res_J_parametric(problem, symmetric=symmetric_bc)
    # Adjoint operators: assembled CSR (direct) or matrix-free (Krylov), plus the
    # un-eliminated K_bulk^T for the symmetric-BC bc_vals-gradient correction.
    J_csr_param = create_J_bc_csr_parametric(problem, symmetric=symmetric_bc)
    matfree_Kt_param = create_matfree_Kt_parametric(problem)

    linear_solve_fn = create_linear_solve_fn(linear_options, cache_namespace="forward")
    if adjoint_linear_options is linear_options:
        adjoint_linear_solve_fn = create_linear_solve_fn(
            adjoint_linear_options, cache_namespace="adjoint")
    else:
        adjoint_linear_solve_fn = create_linear_solve_fn(
            adjoint_linear_options, cache_namespace="adjoint")

    x0_param = create_x0_parametric()

    # Pre-warm direct solve closures with concrete CSR structure (outside any
    # trace). Required for the StaticVars path: the traced adjoint sees
    # tracer-valued CSR structure, so the cuDSS closure cannot initialize
    # lazily there. No-op when internal_vars is None or no direct solver is used.
    if _fwd_direct or _adj_direct:
        prewarm_direct_solvers(
            problem=problem,
            bc=bc,
            internal_vars=internal_vars,
            J_bc_func=create_J_bc_csr_function(problem, bc, symmetric=symmetric_bc),
            forward_options=linear_options,
            adjoint_options=adjoint_linear_options,
            forward_solve_fn=linear_solve_fn,
            adjoint_solve_fn=adjoint_linear_solve_fn,
        )

    tol = newton_options.tol
    rel_tol = newton_options.rel_tol
    max_iter = newton_options.max_iter

    # Vectorizable (scan-based) Armijo line search keyed on (iv, bc, sv).
    _res_for_armijo = lambda s, ivbc: res_param(s, ivbc[0], ivbc[1], ivbc[2])
    armijo = create_armijo_line_search_scan(
        _res_for_armijo,
        c1=newton_options.line_search_c1,
        rho=newton_options.line_search_rho,
        max_backtracks=newton_options.line_search_max_backtracks,
    )

    _raise_on_ls_fail = newton_options.raise_on_line_search_failure

    _fwd_res_J = res_J_param if _fwd_direct else matfree_res_J_param

    def one_step(sol, iv, bc_, sv):
        """One Newton step (fully traceable → vmaps natively).

        Returns ``(new_sol, new_res_norm, line_search_found)``.
        """
        # Residual + operator: assembled CSRMatrix (direct) or matrix-free
        # matvec (Krylov). ``J`` is whatever the linear solver consumes.
        res, J = _fwd_res_J(sol, iv, bc_, sv)
        res_norm = np.linalg.norm(res)
        x0 = x0_param(sol, bc_)
        du = linear_solve_fn(J, -res, x0)
        new_sol, new_norm, _alpha, found = armijo(
            sol, du, res, res_norm, internal_vars=(iv, bc_, sv))
        return new_sol, new_norm, found

    def _res_norm_single(sol, iv, bc_, sv):
        return np.linalg.norm(res_param(sol, iv, bc_, sv))

    # ---- forward: traced Newton loop (lax.while_loop) ---------------------
    # The loop body is traced ONCE — the compiled graph holds a single Newton
    # step regardless of ``max_iter`` (a while_loop does not unroll). The whole
    # solve therefore:
    #   * composes with jax.jit natively, with no pure_callback node and no
    #     host round-trip per iteration (each step is one fused program, not
    #     hundreds of eagerly dispatched kernels);
    #   * vmaps natively (batched while_loop: all lanes iterate until every
    #     lane's condition is false, converged lanes are masked) — the CPU
    #     direct backends keep their block-diagonal custom_vmap rules;
    #   * works with GPU FFI backends (cuDSS): inside a pure_callback the
    #     computation is placed on the Host platform, where the CUDA FFI
    #     handler does not exist. CPU direct backends (spsolve/cholmod/
    #     umfpack) are pure_callback-based and run inside the loop body fine.
    # Gradients are unaffected — custom_vjp wraps the solve, nothing
    # differentiates through the loop.
    #
    # Host-side behavior: ``raise_on_line_search_failure`` is honored for
    # eager (non-traced, non-vmapped) calls, where the final line-search flag
    # is concrete. Under jit/vmap an exception cannot be raised from traced
    # values; a failed line search simply stops the iteration there.
    def _forward(iv, sol0, bc_, sv):
        res00 = _res_norm_single(sol0, iv, bc_, sv)

        def cond(state):
            sol, rn, it, found = state
            not_converged = (rn > tol) & ((rn / np.maximum(res00, 1e-30)) > rel_tol)
            return (it < max_iter) & not_converged & found

        def body(state):
            sol, rn, it, found = state
            new_sol, new_norm, f = one_step(sol, iv, bc_, sv)
            return (new_sol, new_norm, it + 1, f)

        init = (sol0, res00, np.array(0, dtype=np.int32), np.array(True))
        sol, _, it, found = jax.lax.while_loop(cond, body, init)

        if _raise_on_ls_fail and not isinstance(found, jax.core.Tracer):
            if not bool(found):
                raise NewtonLineSearchError(
                    f"Newton iter {int(it)}: Armijo line search failed after "
                    f"{newton_options.line_search_max_backtracks} backtracks. "
                    "The Newton direction is not a descent direction for "
                    "0.5*||r||^2 — usually a sign of an inconsistent Jacobian "
                    "or a bad linear solve.")
        return sol

    # ---- adjoint (single traced linear solve; vmaps natively) ------------
    _needs_bc_correction = symmetric_bc

    def _adjoint(internal_vars, sol, effective_bc, v, sv):
        v_vec = v
        src = sv if sv is not None else problem
        use_transpose = problem.matrix_view not in (MatrixView.UPPER, MatrixView.LOWER)

        # Adjoint operator Jᵀ.
        if _adj_direct:
            J = J_csr_param(sol, internal_vars, effective_bc, sv)
            J_adjoint = transpose_with_maps(
                J, src.csr_T_perm, src.csr_T_indptr, src.csr_T_indices
            ) if use_transpose else J
        elif symmetric_bc:
            # Symmetric BC ⇒ J is symmetric ⇒ Jᵀ = J (matrix-free matvec).
            _, J_adjoint = matfree_res_J_param(sol, internal_vars, effective_bc, sv)
        else:
            # Non-symmetric BC: Jᵀ is the VJP of the BC-applied residual.
            _, _vjp = jax.vjp(
                lambda s: res_param(s, internal_vars, effective_bc, sv), sol)
            J_adjoint = lambda w: _vjp(w)[0]

        adjoint_vec = linear_solve_adjoint(
            J_adjoint, v_vec, adjoint_linear_options, problem.matrix_view,
            effective_bc, linear_solve_fn=adjoint_linear_solve_fn,
        )

        # Symmetric-elimination BC correction so bc_vals gradients are correct:
        #   λ_bc = v_bc - K_fb^T λ_free,  with K_bulk the un-eliminated Jacobian
        #   (assembled transpose for direct, matrix-free VJP for Krylov).
        if _needs_bc_correction:
            lambda_free = adjoint_vec.at[effective_bc.bc_rows].set(0.0)
            if _adj_direct:
                sol_list = problem.unflatten_fn_sol_list(sol)
                kdata, kindptr, kindices = _get_J_csr(problem, sol_list, internal_vars, sv)
                K_bulk = CSRMatrix(kdata, kindptr, kindices,
                                   (problem.num_total_dofs_all_vars,) * 2)
                ktw = K_bulk.rmatvec(lambda_free)
            else:
                ktw = matfree_Kt_param(sol, internal_vars, sv)(lambda_free)
            correction = ktw[effective_bc.bc_rows]
            adjoint_vec = adjoint_vec.at[effective_bc.bc_rows].set(
                v_vec[effective_bc.bc_rows] - correction)

        return adjoint_vec

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess, effective_bc, sv):
        return _forward(internal_vars, initial_guess, effective_bc, sv)

    def f_fwd(internal_vars, initial_guess, effective_bc, sv):
        sol = differentiable_solve(internal_vars, initial_guess, effective_bc, sv)
        return sol, (internal_vars, sol, effective_bc, sv)

    def f_bwd(res, v):
        internal_vars, sol, effective_bc, sv = res
        adjoint_vec = _adjoint(internal_vars, sol, effective_bc, v, sv)

        def res_fn(iv, bc_arg):
            return problem.unflatten_fn_sol_list(
                res_param(sol, iv, bc_arg, sv))

        adjoint_list = problem.unflatten_fn_sol_list(adjoint_vec)
        _, f_vjp = jax.vjp(res_fn, internal_vars, effective_bc)
        vjp_iv, vjp_bc = f_vjp(adjoint_list)
        vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
        vjp_bc = jax.tree_util.tree_map(_safe_negate, vjp_bc)
        return (vjp_iv, None, vjp_bc, None)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    from ..utils import zero_like_initial_guess
    default_initial_guess = zero_like_initial_guess(problem, bc)
    _default_bc = bc

    def solver(internal_vars, initial_guess=None, bc=None, static_vars=None):
        from ..DCboundary import DirichletBC
        effective_bc = bc if isinstance(bc, DirichletBC) else _default_bc
        ig = default_initial_guess if initial_guess is None else initial_guess
        return differentiable_solve(internal_vars, ig, effective_bc, static_vars)

    return solver
