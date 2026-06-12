"""
Linear solver implementations for FEAX finite element framework.

This module provides low-level linear algebra utilities and solver
selection logic for solving systems of the form A x = b arising
in finite element analysis.

Key Features:
- Jacobi (diagonal) preconditioner
- Solver selection: cg, bicgstab, gmres, spsolve, umfpack, cholmod, cudss
- Convergence checking for ill-conditioned problems
- Adjoint linear solve for gradient computation
- create_linear_solver: high-level differentiable solver for linear problems
"""

import logging
from typing import Any, Callable, Optional

import jax
import jax.numpy as np

from ..assembler import (
    create_J_bc_csr_function,
    create_J_bc_csr_parametric,
    create_matfree_res_J_parametric,
    create_res_bc_parametric,
)
from ..csr import transpose_with_maps
from ..DCboundary import DirichletBC
from ..problem import MatrixView, Problem
from .common import (
    _safe_negate,
    check_convergence,
    create_linear_solve_fn,
    prewarm_direct_solvers,
    create_x0,
)
from .options import (
    AbstractSolverOptions,
    DirectSolverOptions,
    KrylovSolverOptions,
    MatrixProperty,
    SolverOptions,
    resolve_direct_solver,
    resolve_iterative_solver,
)

logger = logging.getLogger(__name__)


def linear_solve_adjoint(
    A,
    b,
    solver_options: AbstractSolverOptions,
    matrix_view: MatrixView,
    bc=None,
    linear_solve_fn: Optional[Callable] = None,
):
    """Solve linear system for adjoint problem.

    Parameters
    ----------
    A : BCOO sparse matrix
        The transposed Jacobian matrix (J^T)
    b : jax.numpy.ndarray
        Right-hand side vector (cotangent vector from VJP)
    solver_options : SolverOptions or DirectSolverOptions or KrylovSolverOptions
        Solver configuration for adjoint solve
    matrix_view : MatrixView
        Matrix storage format from the problem
    bc : DirichletBC, optional
        Boundary conditions for computing initial guess
    linear_solve_fn : callable, optional
        Pre-created linear solve function. If None, one is created from
        solver_options internally.

    Returns
    -------
    sol : jax.numpy.ndarray
        Solution to the adjoint system: A @ sol = b

    Notes
    -----
    For the adjoint problem, boundary conditions are already incorporated into
    the transposed Jacobian matrix. The initial guess uses BC-aware computation
    when bc is provided, which can improve convergence for problems with
    Dirichlet boundary conditions.
    """
    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(solver_options)

    x0 = None
    if solver_options.uses_x0():
        if bc is not None and hasattr(bc, 'bc_rows') and hasattr(bc, 'bc_vals'):
            x0 = np.zeros_like(b)
            bc_rows_array = np.array(bc.bc_rows) if not isinstance(bc.bc_rows, np.ndarray) else bc.bc_rows
            x0 = x0.at[bc_rows_array].set(b[bc_rows_array])
        else:
            x0 = np.zeros_like(b)

    sol = linear_solve_fn(A, b, x0)

    if solver_options.check_convergence:
        sol = check_convergence(
            A=A,
            x=sol,
            b=b,
            solver_options=solver_options,
            matrix_view=matrix_view,
            solver_label="Adjoint solver",
        )

    return sol


def linear_solve(
    J_bc_applied,
    res_bc_applied,
    initial_guess,
    bc: DirichletBC,
    solver_options: AbstractSolverOptions,
    matrix_view: MatrixView,
    internal_vars=None,
    P_mat=None,
    linear_solve_fn: Optional[Callable] = None,
    x0_fn: Optional[Callable] = None,
):
    """Single-step linear solve used by create_solver(linear=True)."""

    if linear_solve_fn is None:
        linear_solve_fn = create_linear_solve_fn(solver_options)

    if internal_vars is not None:
        res = res_bc_applied(initial_guess, internal_vars)
        J = J_bc_applied(initial_guess, internal_vars)
    else:
        res = res_bc_applied(initial_guess)
        J = J_bc_applied(initial_guess)

    b = -res
    x0 = None
    if solver_options.uses_x0():
        if x0_fn is None:
            x0_fn = create_x0(bc_rows=bc.bc_rows, bc_vals=bc.bc_vals, P_mat=P_mat)
        x0 = x0_fn(initial_guess)

    delta_sol = linear_solve_fn(J, b, x0)

    if solver_options.check_convergence:
        delta_sol = check_convergence(
            A=J,
            x=delta_sol,
            b=b,
            solver_options=solver_options,
            matrix_view=matrix_view,
            solver_label="Linear solver",
        )

    sol = initial_guess + delta_sol
    return sol, None


def create_linear_solver(
    problem: Problem,
    bc: DirichletBC,
    solver_options: Optional[AbstractSolverOptions] = None,
    adjoint_solver_options: Optional[AbstractSolverOptions] = None,
    internal_vars=None,
    symmetric_bc: bool = True,
) -> Callable[[Any, np.ndarray], np.ndarray]:
    """Create a differentiable solver for linear FE problems.

    Simpler and more focused alternative to ``create_solver(linear=True)``
    when the problem is known to be linear (e.g. linear elasticity).
    The returned function supports ``jax.grad`` via a custom VJP based on
    the adjoint method.

    Parameters
    ----------
    problem : Problem
        The feax Problem instance.
    bc : DirichletBC
        Boundary conditions.
    solver_options : DirectSolverOptions or KrylovSolverOptions, optional
        Options for the forward linear solve (defaults to KrylovSolverOptions()).
    adjoint_solver_options : DirectSolverOptions or KrylovSolverOptions, optional
        Options for the adjoint solve used in the backward pass.
        Defaults to the same options as the forward solve.
    internal_vars : InternalVars, optional
        Sample internal variables used to pre-warm cuDSS with concrete CSR
        structure before any JAX tracing. Recommended when using cuDSS and
        composing ``jax.jit`` with ``jax.grad``.
    symmetric_bc : bool, default True
        Use symmetric Dirichlet elimination (zero BC rows *and* columns). Linear
        FE operators are symmetric after symmetric elimination, so the Krylov
        adjoint reuses the forward matvec (``Jᵀ = J``).

    Returns
    -------
    differentiable_solve : callable
        A function with signature ``(internal_vars, initial_guess) -> solution``
        that is differentiable w.r.t. ``internal_vars`` via ``jax.grad``.

    Notes
    -----
    The operator representation follows the solve method: ``DirectSolverOptions``
    assembles the Jacobian straight into CSR (for factorization), while
    ``KrylovSolverOptions`` (cg/bicgstab/gmres) is fully **matrix-free** — the
    forward and adjoint matvecs are residual JVPs, so the element Jacobian is
    never materialized.

    Forward pass performs a single linear solve:

    ```python
    J * delta_sol = -res
    sol = initial_guess + delta_sol
    ```

    Backward pass solves the adjoint system:

    ```python
    J^T * adjoint = v
    ```

    and returns the VJP of the residual w.r.t. ``internal_vars``.

    Examples
    --------
    >>> solver = create_linear_solver(problem, bc)
    >>> initial = fe.zero_like_initial_guess(problem, bc)
    >>> sol = solver(internal_vars, initial)
    >>>
    >>> # Gradient w.r.t. internal_vars
    >>> def loss(internal_vars):
    ...     sol = solver(internal_vars, initial)
    ...     return np.sum(sol ** 2)
    >>> grad = jax.grad(loss)(internal_vars)
    """
    # Linear FE problems are typically SPD after Dirichlet elimination.
    # Resolve "auto" eagerly so low-level factories never receive unresolved options.
    default_matrix_property = MatrixProperty.SPD

    if solver_options is None:
        solver_options = KrylovSolverOptions()
    if adjoint_solver_options is None:
        adjoint_solver_options = solver_options

    shared_opts = adjoint_solver_options is solver_options

    if isinstance(solver_options, SolverOptions) or isinstance(adjoint_solver_options, SolverOptions):
        raise RuntimeError(
            "SolverOptions has been removed. "
            "Use DirectSolverOptions or KrylovSolverOptions."
        )
    if not isinstance(solver_options, (DirectSolverOptions, KrylovSolverOptions)):
        raise TypeError(
            "Unsupported solver_options type. "
            f"Expected DirectSolverOptions or KrylovSolverOptions, got {type(solver_options).__name__}."
        )
    if not isinstance(adjoint_solver_options, (DirectSolverOptions, KrylovSolverOptions)):
        raise TypeError(
            "Unsupported adjoint_solver_options type. "
            f"Expected DirectSolverOptions or KrylovSolverOptions, got {type(adjoint_solver_options).__name__}."
        )

    if isinstance(solver_options, KrylovSolverOptions):
        solver_options = resolve_iterative_solver(solver_options, default_matrix_property)
    elif isinstance(solver_options, DirectSolverOptions):
        solver_options = resolve_direct_solver(
            solver_options,
            default_matrix_property,
            matrix_view=problem.matrix_view,
        )

    if shared_opts:
        adjoint_solver_options = solver_options
    else:
        if isinstance(adjoint_solver_options, KrylovSolverOptions):
            adjoint_solver_options = resolve_iterative_solver(adjoint_solver_options, default_matrix_property)
        elif isinstance(adjoint_solver_options, DirectSolverOptions):
            adjoint_solver_options = resolve_direct_solver(
                adjoint_solver_options,
                default_matrix_property,
                matrix_view=problem.matrix_view,
            )

    _default_bc = bc

    # Operator representation follows the solve method: direct factorizes the
    # assembled CSR matrix; Krylov (cg/bicgstab/gmres) needs only a matvec, so
    # it uses the matrix-free operator (a residual JVP) — no assembly at all.
    _fwd_direct = isinstance(solver_options, DirectSolverOptions)
    _adj_direct = isinstance(adjoint_solver_options, DirectSolverOptions)

    # CSR-direct operators (only built when a direct solver is on either path).
    J_bc_csr_func = create_J_bc_csr_function(problem, bc, symmetric=symmetric_bc) if (_fwd_direct or _adj_direct) else None
    J_bc_csr_parametric = create_J_bc_csr_parametric(problem, symmetric=symmetric_bc) if (_fwd_direct or _adj_direct) else None
    # Matrix-free operator (fused BC-applied residual + JVP matvec) for Krylov.
    matfree_res_J_param = create_matfree_res_J_parametric(problem, symmetric=symmetric_bc)
    res_bc_parametric = create_res_bc_parametric(problem)
    linear_solve_fn = create_linear_solve_fn(
        solver_options,
        cache_namespace="forward",
    )
    if adjoint_solver_options is solver_options:
        adjoint_linear_solve_fn = linear_solve_fn
    else:
        adjoint_linear_solve_fn = create_linear_solve_fn(
            adjoint_solver_options,
            cache_namespace="adjoint",
        )
    x0_fn = solver_options.x0_fn if isinstance(solver_options, KrylovSolverOptions) else None

    # Pre-warm with the CSR Jacobian when a direct solver is used so the cuDSS
    # symbolic structure matches the canonical CSR fed at solve time. (FEM
    # patterns are structurally symmetric, so J and J^T share structure and the
    # adjoint reuses the same warmup.)
    prewarm_direct_solvers(
        problem=problem,
        bc=bc,
        internal_vars=internal_vars,
        J_bc_func=J_bc_csr_func,
        forward_options=solver_options,
        adjoint_options=adjoint_solver_options,
        forward_solve_fn=linear_solve_fn,
        adjoint_solve_fn=adjoint_linear_solve_fn,
    )

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess, effective_bc):
        # Operator follows the solve method: direct assembles a CSR matrix;
        # Krylov uses the matrix-free (residual JVP) matvec — no assembly.
        if _fwd_direct:
            res = res_bc_parametric(initial_guess, internal_vars, effective_bc)
            op = J_bc_csr_parametric(initial_guess, internal_vars, effective_bc)
        else:
            res, op = matfree_res_J_param(initial_guess, internal_vars, effective_bc)

        b = -res
        x0 = None
        if solver_options.uses_x0():
            _x0_fn = x0_fn if x0_fn is not None else create_x0(
                bc_rows=effective_bc.bc_rows, bc_vals=effective_bc.bc_vals)
            x0 = _x0_fn(initial_guess)

        delta_sol = linear_solve_fn(op, b, x0)

        if solver_options.check_convergence:
            delta_sol = check_convergence(
                A=op, x=delta_sol, b=b,
                solver_options=solver_options,
                matrix_view=problem.matrix_view,
                solver_label="Linear solver",
            )

        return initial_guess + delta_sol

    def f_fwd(internal_vars, initial_guess, effective_bc):
        sol = differentiable_solve(internal_vars, initial_guess, effective_bc)
        return sol, (internal_vars, sol, effective_bc)

    def f_bwd(res, v):
        internal_vars, sol, effective_bc = res

        # Adjoint solve: J^T @ adjoint = v
        use_transpose = problem.matrix_view not in (MatrixView.UPPER, MatrixView.LOWER)
        if _adj_direct:
            # CSR-direct adjoint: assemble J, then form J^T via precomputed
            # transpose maps (a pure gather, no sort). UPPER/LOWER store a
            # symmetric view so no transpose is applied.
            J = J_bc_csr_parametric(sol, internal_vars, effective_bc)
            J_adjoint = transpose_with_maps(
                J, problem.csr_T_perm, problem.csr_T_indptr, problem.csr_T_indices
            ) if use_transpose else J
        elif symmetric_bc:
            # Symmetric BC ⇒ J is symmetric ⇒ Jᵀ = J (matrix-free matvec).
            _, J_adjoint = matfree_res_J_param(sol, internal_vars, effective_bc)
        else:
            # Non-symmetric BC: Jᵀ is the VJP of the BC-applied residual.
            _, _vjp = jax.vjp(
                lambda s: res_bc_parametric(s, internal_vars, effective_bc), sol)
            J_adjoint = lambda w: _vjp(w)[0]
        adjoint_vec = linear_solve_adjoint(
            J_adjoint, v, adjoint_solver_options, problem.matrix_view,
            effective_bc,
            linear_solve_fn=adjoint_linear_solve_fn
        )

        # VJP of residual w.r.t. internal_vars and bc
        def res_fn(iv, bc_arg):
            return problem.unflatten_fn_sol_list(
                res_bc_parametric(sol, iv, bc_arg)
            )

        adjoint_list = problem.unflatten_fn_sol_list(adjoint_vec)
        _, f_vjp = jax.vjp(res_fn, internal_vars, effective_bc)
        vjp_iv, vjp_bc = f_vjp(adjoint_list)
        vjp_iv = jax.tree_util.tree_map(_safe_negate, vjp_iv)
        vjp_bc = jax.tree_util.tree_map(_safe_negate, vjp_bc)

        return (vjp_iv, None, vjp_bc)

    differentiable_solve.defvjp(f_fwd, f_bwd)

    can_omit_initial_guess = not solver_options.uses_x0()

    import warnings
    from ..utils import zero_like_initial_guess

    default_initial_guess = zero_like_initial_guess(problem, bc)

    def solver_wrapper(internal_vars, initial_guess=None, bc=None):
        effective_bc = bc if isinstance(bc, DirichletBC) else _default_bc
        ig = default_initial_guess if (can_omit_initial_guess and initial_guess is None) else initial_guess
        if ig is None:
            ig = default_initial_guess
        return differentiable_solve(internal_vars, ig, effective_bc)

    return solver_wrapper
