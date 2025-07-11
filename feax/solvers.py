"""
JAX-native linear solvers for FEAX.

This module provides efficient linear solvers using JAX's built-in sparse linear algebra
routines. Solvers support configuration through JAX-native dataclasses.
"""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg, bicgstab, gmres
from jax.experimental import sparse
from dataclasses import dataclass
from typing import Optional, Callable, Union, Tuple
import functools

from feax import logger


@dataclass
class SolverOptions:
    """Options for JAX linear solvers.
    
    Parameters
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    atol : float
        Absolute tolerance (optional)
    M : Optional[Callable]
        Preconditioner function
    x0 : Optional[jax.Array]
        Initial guess
    restart : Optional[int]
        Number of iterations between restarts (for GMRES)
    """
    max_iter: int = 1000
    tol: float = 1e-6
    atol: float = 0.0
    M: Optional[Callable] = None
    x0: Optional[jax.Array] = None
    restart: Optional[int] = 20  # For GMRES


def _solve_cg(A_fn: Callable, b: jax.Array, options: SolverOptions) -> Tuple[jax.Array, dict]:
    """Internal CG solver that takes a matrix-vector product function."""
    x, info = cg(A_fn, b, 
                 x0=options.x0, 
                 tol=options.tol,
                 atol=options.atol,
                 maxiter=options.max_iter,
                 M=options.M)
    
    # JAX solvers return None on success
    converged = info is None or info == 0
    iterations = 0 if info is None else info
    return x, {"iterations": iterations, "converged": converged}


def _solve_bicgstab(A_fn: Callable, b: jax.Array, options: SolverOptions) -> Tuple[jax.Array, dict]:
    """Internal BiCGSTAB solver that takes a matrix-vector product function."""
    x, info = bicgstab(A_fn, b,
                        x0=options.x0,
                        tol=options.tol, 
                        atol=options.atol,
                        maxiter=options.max_iter,
                        M=options.M)
    
    # JAX solvers return None on success
    converged = info is None or info == 0
    iterations = 0 if info is None else info
    return x, {"iterations": iterations, "converged": converged}


def _solve_gmres(A_fn: Callable, b: jax.Array, options: SolverOptions) -> Tuple[jax.Array, dict]:
    """Internal GMRES solver that takes a matrix-vector product function."""
    x, info = gmres(A_fn, b,
                    x0=options.x0,
                    tol=options.tol,
                    atol=options.atol,
                    maxiter=options.max_iter,
                    M=options.M,
                    restart=options.restart)
    
    # JAX solvers return None on success
    converged = info is None or info == 0
    iterations = 0 if info is None else info
    return x, {"iterations": iterations, "converged": converged}


# JAX-compatible solver functions
def solve_cg(A: sparse.BCOO, b: jax.Array, options: SolverOptions) -> Tuple[jax.Array, dict]:
    """Solve linear system using Conjugate Gradient method.
    
    Parameters
    ----------
    A : sparse.BCOO
        Symmetric positive definite sparse matrix
    b : jax.Array
        Right-hand side vector
    options : SolverOptions
        Solver options
        
    Returns
    -------
    x : jax.Array
        Solution vector
    info : dict
        Solver information (iterations, converged)
    """
    logger.debug(f"Solving with CG: max_iter={options.max_iter}, tol={options.tol}")
    
    # CG expects a function that computes A @ x
    A_fn = lambda x: A @ x
    
    x, info = _solve_cg(A_fn, b, options)
    
    logger.debug(f"CG converged: {info['converged']}, iterations: {info['iterations']}")
    
    return x, info


def solve_bicgstab(A: sparse.BCOO, b: jax.Array, options: SolverOptions) -> Tuple[jax.Array, dict]:
    """Solve linear system using BiCGSTAB method.
    
    Parameters
    ----------
    A : sparse.BCOO
        General sparse matrix
    b : jax.Array
        Right-hand side vector
    options : SolverOptions
        Solver options
        
    Returns
    -------
    x : jax.Array
        Solution vector
    info : dict
        Solver information (iterations, converged)
    """
    logger.debug(f"Solving with BiCGSTAB: max_iter={options.max_iter}, tol={options.tol}")
    
    # BiCGSTAB expects a function that computes A @ x
    A_fn = lambda x: A @ x
    
    x, info = _solve_bicgstab(A_fn, b, options)
    
    logger.debug(f"BiCGSTAB converged: {info['converged']}, iterations: {info['iterations']}")
    
    return x, info


def solve_gmres(A: sparse.BCOO, b: jax.Array, options: SolverOptions) -> Tuple[jax.Array, dict]:
    """Solve linear system using GMRES method.
    
    Parameters
    ----------
    A : sparse.BCOO
        General sparse matrix
    b : jax.Array
        Right-hand side vector
    options : SolverOptions
        Solver options
        
    Returns
    -------
    x : jax.Array
        Solution vector
    info : dict
        Solver information (iterations, converged)
    """
    logger.debug(f"Solving with GMRES: max_iter={options.max_iter}, tol={options.tol}, restart={options.restart}")
    
    # GMRES expects a function that computes A @ x
    A_fn = lambda x: A @ x
    
    x, info = _solve_gmres(A_fn, b, options)
    
    logger.debug(f"GMRES converged: {info['converged']}, iterations: {info['iterations']}")
    
    return x, info


# Main solver interface
def solve(A: sparse.BCOO, b: jax.Array, 
          method: str = "bicgstab",
          options: Optional[SolverOptions] = None) -> Tuple[jax.Array, dict]:
    """Solve sparse linear system Ax = b using JAX-native solvers.
    
    Parameters
    ----------
    A : sparse.BCOO
        Sparse matrix in BCOO format
    b : jax.Array
        Right-hand side vector
    method : str
        Solver method: 'cg', 'bicgstab', or 'gmres'
    options : SolverOptions
        Solver options (uses defaults if None)
        
    Returns
    -------
    x : jax.Array
        Solution vector
    info : dict
        Solver information
        
    Examples
    --------
    >>> from feax.solvers import solve, SolverOptions
    >>> options = SolverOptions(max_iter=500, tol=1e-8)
    >>> x, info = solve(A, b, method='bicgstab', options=options)
    """
    if options is None:
        options = SolverOptions()
    
    logger.info(f"Solving linear system with {method.upper()}")
    
    # Select solver
    solver_map = {
        'cg': solve_cg,
        'bicgstab': solve_bicgstab,
        'gmres': solve_gmres
    }
    
    if method not in solver_map:
        raise ValueError(f"Unknown solver method: {method}. Choose from {list(solver_map.keys())}")
    
    solver_fn = solver_map[method]
    
    # Solve system
    x, info = solver_fn(A, b, options)
    
    # Add method to info
    info['method'] = method
    
    return x, info


# Preconditioner helpers
def jacobi_preconditioner(A: sparse.BCOO) -> Callable:
    """Create Jacobi (diagonal) preconditioner for sparse matrix.
    
    Parameters
    ----------
    A : sparse.BCOO
        Sparse matrix
        
    Returns
    -------
    M : Callable
        Preconditioner function M(x) = D^{-1} x
    """
    # Extract diagonal
    diag_indices = jnp.arange(A.shape[0])
    diag_mask = (A.indices[:, 0] == A.indices[:, 1])
    diag_vals = jnp.where(diag_mask, A.data, 0.0)
    
    # Sum duplicates (if any) and get diagonal
    diag = sparse.BCOO((diag_vals, A.indices), shape=A.shape).todense().diagonal()
    
    # Avoid division by zero
    diag_inv = jnp.where(jnp.abs(diag) > 1e-10, 1.0 / diag, 1.0)
    
    return lambda x: diag_inv * x


def ilu_preconditioner(A: sparse.BCOO, fill_factor: float = 1.0) -> Callable:
    """Create incomplete LU preconditioner (simplified version).
    
    Note: This is a placeholder for a more sophisticated ILU implementation.
    For now, falls back to Jacobi preconditioning.
    
    Parameters
    ----------
    A : sparse.BCOO
        Sparse matrix
    fill_factor : float
        Fill factor (not used in current implementation)
        
    Returns
    -------
    M : Callable
        Preconditioner function
    """
    logger.warning("ILU preconditioner not fully implemented, using Jacobi instead")
    return jacobi_preconditioner(A)


# JIT-compiled versions for better performance
solve_cg_jit = jax.jit(solve_cg, static_argnames=['options'])
solve_bicgstab_jit = jax.jit(solve_bicgstab, static_argnames=['options'])
solve_gmres_jit = jax.jit(solve_gmres, static_argnames=['options'])