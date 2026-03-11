"""Fully matrix-free Newton solver for nonlinear problems.

This module provides a matrix-free Newton solver where the tangent
operator is computed via JVP (Jacobian-vector product) of the residual,
eliminating the need for sparse matrix assembly.

The inner linear system at each Newton step is solved by an iterative
solver (CG, BiCGSTAB, or GMRES) applied to the JVP-based matvec.

The solver is suitable for problems with custom energy contributions
(e.g., cohesive zones, phase-field fracture) that are difficult to
express within the standard FE assembly framework.

Usage pattern::

    from feax.solvers.matrix_free import MatrixFreeOptions, newton_solve

    # Define total energy as a pure JAX function
    def total_energy(u_flat, *args):
        return elastic_energy(u_flat) + custom_energy(u_flat, *args)

    # Solve
    u, info = newton_solve(total_energy, u0, fixed_dofs, args=(delta_max,))
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Options
# ============================================================================

@dataclass(frozen=True)
class LinearSolverOptions:
    """Options for the inner iterative linear solver.

    Parameters
    ----------
    solver : str, default "cg"
        Iterative solver algorithm: "cg", "bicgstab", or "gmres".
    tol : float, default 1e-10
        Relative tolerance for the iterative solver.
    atol : float, default 1e-8
        Absolute tolerance for the iterative solver.
    maxiter : int, default 200
        Maximum number of iterations.
    restart : int, default 200
        Restart parameter for GMRES. Only used when solver="gmres".
    """
    solver: str = "cg"
    tol: float = 1e-10
    atol: float = 1e-8
    maxiter: int = 200
    restart: int = 200

    def __post_init__(self):
        valid = ("cg", "bicgstab", "gmres")
        if self.solver not in valid:
            raise ValueError(
                f"Invalid solver: {self.solver}. Choose from {valid}"
            )


@dataclass(frozen=True)
class MatrixFreeOptions:
    """Options for the matrix-free Newton solver.

    Parameters
    ----------
    newton_tol : float, default 1e-8
        Absolute tolerance on the Newton residual norm.
    newton_max_iter : int, default 200
        Maximum number of Newton iterations.
    linear_solver : LinearSolverOptions, default LinearSolverOptions()
        Options for the inner iterative linear solver.
    verbose : bool, default False
        Log Newton iteration info via jax.debug.print (visible inside JIT).
    """
    newton_tol: float = 1e-8
    newton_max_iter: int = 200
    linear_solver: LinearSolverOptions = field(default_factory=LinearSolverOptions)
    verbose: bool = False


# ============================================================================
# Info
# ============================================================================

@dataclass
class NewtonInfo:
    """Convergence information returned by newton_solve.

    Attributes
    ----------
    converged : bool
        Whether the solver converged within tolerance.
    res_norm : float
        Final residual norm.
    n_iter : int
        Number of Newton iterations performed.
    """
    converged: bool
    res_norm: float
    n_iter: int


# ============================================================================
# Inner linear solver
# ============================================================================

def _create_linear_solve(options: LinearSolverOptions, project_fn, verbose=False):
    """Create a projected iterative linear solve function.

    The returned function solves A @ x = b with fixed DOFs projected out,
    using jax.scipy.sparse.linalg solvers.

    Parameters
    ----------
    options : LinearSolverOptions
        Solver configuration.
    project_fn : callable
        Projection function to zero out fixed DOFs.
    verbose : bool
        If True, print linear solver residual via jax.debug.print.

    Returns
    -------
    solve : callable
        Function ``solve(matvec, rhs, newton_iter) -> x``.
    """
    def projected_matvec(matvec):
        def _matvec(v):
            return project_fn(matvec(v))
        return _matvec

    def _report(newton_iter, x, A_fn, b):
        if verbose:
            residual = A_fn(x) - b
            res_norm = np.linalg.norm(residual)
            du_norm = np.linalg.norm(x)
            jax.debug.print(
                "  Newton iter {k}: linear solver ({s}): "
                "du_norm = {d:.6e}, linear_res = {r:.6e}",
                k=newton_iter, s=options.solver, d=du_norm, r=res_norm,
            )

    if options.solver == "cg":
        def solve(matvec, rhs, newton_iter):
            A_fn = projected_matvec(matvec)
            b = project_fn(rhs)
            x0 = np.zeros_like(rhs)
            x, _ = jax.scipy.sparse.linalg.cg(
                A_fn, b, x0=x0, tol=options.tol, atol=options.atol,
                maxiter=options.maxiter,
            )
            x = project_fn(x)
            _report(newton_iter, x, A_fn, b)
            return x
        return solve

    if options.solver == "bicgstab":
        def solve(matvec, rhs, newton_iter):
            A_fn = projected_matvec(matvec)
            b = project_fn(rhs)
            x0 = np.zeros_like(rhs)
            x, _ = jax.scipy.sparse.linalg.bicgstab(
                A_fn, b, x0=x0, tol=options.tol, atol=options.atol,
                maxiter=options.maxiter,
            )
            x = project_fn(x)
            _report(newton_iter, x, A_fn, b)
            return x
        return solve

    if options.solver == "gmres":
        def solve(matvec, rhs, newton_iter):
            A_fn = projected_matvec(matvec)
            b = project_fn(rhs)
            x0 = np.zeros_like(rhs)
            x, _ = jax.scipy.sparse.linalg.gmres(
                A_fn, b, x0=x0, tol=options.tol, atol=options.atol,
                maxiter=options.maxiter, restart=options.restart,
            )
            x = project_fn(x)
            _report(newton_iter, x, A_fn, b)
            return x
        return solve

    raise ValueError(f"Unknown solver: {options.solver}")


# ============================================================================
# Core solver
# ============================================================================

def newton_solve(
    energy_fn: Callable,
    u0: jax.Array,
    fixed_dofs: jax.Array,
    args: tuple = (),
    options: Optional[MatrixFreeOptions] = None,
) -> Tuple[jax.Array, NewtonInfo]:
    """Fully matrix-free Newton solver.

    Solves the nonlinear problem ``grad(energy_fn)(u, *args) = 0``
    subject to Dirichlet boundary conditions (fixed DOFs).

    The tangent operator is computed via JVP of the gradient,
    requiring no sparse matrix assembly. The inner linear system
    is solved by an iterative solver (CG, BiCGSTAB, or GMRES).

    Parameters
    ----------
    energy_fn : callable
        Total energy function ``energy_fn(u_flat, *args) -> scalar``.
        Must be differentiable with respect to ``u_flat``.
    u0 : jax.Array
        Initial guess (should already have BC values set).
    fixed_dofs : jax.Array
        Indices of DOFs with Dirichlet boundary conditions.
        These DOFs are zeroed in the residual and search direction.
    args : tuple, optional
        Extra arguments passed to ``energy_fn``.
    options : MatrixFreeOptions, optional
        Solver options. Uses defaults if not provided.

    Returns
    -------
    u : jax.Array
        Solution vector.
    info : NewtonInfo
        Convergence information.

    Examples
    --------
    >>> def energy(u, delta_max):
    ...     return elastic_energy(u) + cohesive_energy(u, delta_max)
    >>> u_sol, info = newton_solve(energy, u0, fixed_dofs, args=(delta_max,))
    >>> if not info.converged:
    ...     print(f"Warning: did not converge, res={info.res_norm:.2e}")
    """
    if options is None:
        options = MatrixFreeOptions()

    gradient_fn = jax.grad(energy_fn)

    def project(v):
        return v.at[fixed_dofs].set(0.0)

    def residual_fn(u):
        return gradient_fn(u, *args)

    def tangent_matvec(u, v):
        _, Jv = jax.jvp(residual_fn, (u,), (v,))
        return Jv

    linear_solve = _create_linear_solve(
        options.linear_solver, project, verbose=options.verbose,
    )

    def cond_fn(state):
        _, res_norm, k = state
        not_converged = ~(res_norm <= options.newton_tol)  # True for nan
        return not_converged & (k < options.newton_max_iter)

    def body_fn(state):
        u, _, k = state
        res = project(residual_fn(u))
        res_norm = np.linalg.norm(res)

        matvec = lambda v: tangent_matvec(u, v)
        du = linear_solve(matvec, -res, k)

        u = u + du
        new_res = project(residual_fn(u))
        new_res_norm = np.linalg.norm(new_res)

        if options.verbose:
            jax.debug.print(
                "  Newton iter {k}: res_norm = {r:.6e} -> {nr:.6e}",
                k=k, r=res_norm, nr=new_res_norm,
            )

        return u, new_res_norm, k + 1

    res0 = project(residual_fn(u0))
    res0_norm = np.linalg.norm(res0)
    state0 = (u0, res0_norm, 0)

    u_sol, final_res, n_iter = jax.lax.while_loop(cond_fn, body_fn, state0)

    info = NewtonInfo(
        converged=bool(final_res <= options.newton_tol),
        res_norm=float(final_res),
        n_iter=int(n_iter),
    )

    if not info.converged:
        logger.warning(
            f"Newton solver did NOT converge in {info.n_iter} iterations "
            f"(res_norm={info.res_norm:.2e}, tol={options.newton_tol:.1e})"
        )

    return u_sol, info


# ============================================================================
# Energy integration helper
# ============================================================================

def create_energy_fn(problem) -> Callable:
    """Create an energy integration function from a feax Problem.

    Builds a pure JAX function that computes the total energy by
    integrating the problem's energy density over the domain::

        E(u) = ∫ ψ(∇u) dΩ

    The energy density is obtained from ``problem.get_energy_density()``.

    Parameters
    ----------
    problem : feax.Problem
        A feax problem. Must define ``get_energy_density()``
        returning a non-None callable.

    Returns
    -------
    energy : callable
        Function ``energy(u_flat) -> scalar`` that computes
        the total energy ∫ ψ(∇u) dΩ.

    Raises
    ------
    ValueError
        If ``problem.get_energy_density()`` returns None.
    """
    psi_fn = problem.get_energy_density()
    if psi_fn is None:
        raise ValueError(
            f"{type(problem).__name__}.get_energy_density() returned None. "
            "Define get_energy_density() to use create_energy_fn()."
        )

    fe0 = problem.fes[0]
    cells = problem.cells_list[0]
    sg = fe0.shape_grads      # (num_cells, num_quads, num_nodes, dim)
    jxw = fe0.JxW              # (num_cells, num_quads)
    vec = fe0.vec

    def energy(u_flat):
        u = u_flat.reshape(-1, vec)
        cell_u = u[cells]  # (num_cells, num_nodes_per_cell, vec)

        def cell_energy(cell_sol, cell_sg, cell_jxw):
            u_grads = np.sum(
                cell_sol[None, :, :, None] * cell_sg[:, :, None, :],
                axis=1,
            )  # (num_quads, vec, dim)
            return np.sum(jax.vmap(lambda ug, w: psi_fn(ug) * w)(u_grads, cell_jxw))

        return np.sum(jax.vmap(cell_energy)(cell_u, sg, jxw))

    return energy
