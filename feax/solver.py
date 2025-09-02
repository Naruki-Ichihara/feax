"""
Nonlinear and linear solvers for FEAX finite element framework.

This module provides Newton-Raphson solvers, linear solvers, and solver configuration
utilities for solving finite element problems. It includes both JAX-based solvers
for performance and Python-based solvers for debugging.

Key Features:
- Newton-Raphson solvers with line search and convergence control
- Multiple solver variants: while loop, fixed iterations, and Python debugging
- Jacobi preconditioning for improved convergence
- Comprehensive solver configuration through SolverOptions dataclass
- Support for multipoint constraints via prolongation matrices
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Callable, Union, Tuple, Any, TYPE_CHECKING
from .assembler import create_J_bc_function, create_res_bc_function
from .DCboundary import DirichletBC

if TYPE_CHECKING:
    from feax.problem import Problem
    from feax.internal_vars import InternalVars


def create_jacobi_preconditioner(A: jax.experimental.sparse.BCOO, shift: float = 1e-12) -> jax.experimental.sparse.BCOO:
    """Create Jacobi (diagonal) preconditioner from sparse matrix.
    
    Parameters
    ----------
    A : BCOO sparse matrix
        The system matrix to precondition
    shift : float, default 1e-12
        Small value added to diagonal for numerical stability
        
    Returns
    -------
    M : LinearOperator
        Jacobi preconditioner as diagonal inverse matrix
        
    Notes
    -----
    This creates a diagonal preconditioner M = diag(A)^{-1} with regularization.
    The preconditioner is JAX-compatible and avoids dynamic indexing.
    For elasticity problems with extreme material contrasts, this helps
    condition number significantly.
    """
    
    def extract_diagonal(A):
        """Extract diagonal from BCOO sparse matrix avoiding dynamic indexing."""
        # Get matrix dimensions
        n = A.shape[0]
        
        # Find diagonal entries by checking where row == col
        diagonal_mask = A.indices[:, 0] == A.indices[:, 1]
        
        # Extract diagonal values - use scatter_add to handle duplicates
        diag = jnp.zeros(n)
        diagonal_indices = jnp.where(diagonal_mask, A.indices[:, 0], n)  # Use n as dummy index
        diagonal_values = jnp.where(diagonal_mask, A.data, 0.0)
        diag = diag.at[diagonal_indices].add(diagonal_values)  # Handles out-of-bounds gracefully
        
        return diag
    
    def jacobi_matvec(diag_inv, x):
        """Apply Jacobi preconditioner: M @ x = diag_inv * x"""
        return diag_inv * x
    
    # Extract diagonal and compute inverse with regularization
    diagonal = extract_diagonal(A)
    diagonal_regularized = diagonal + shift
    diagonal_inv = 1.0 / diagonal_regularized
    
    # Create LinearOperator-like function
    def M_matvec(x):
        return jacobi_matvec(diagonal_inv, x)
    
    return M_matvec


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
    >>> # Usage with BC information
    >>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0]) 
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    
    >>> # Usage with reduced problem
    >>> x0_fn = create_x0(bc_rows, bc_vals, P_mat=P)
    """
    
    def x0_fn(current_sol):
        """BC-aware strategy: correct x0 method from row elimination solver."""
        if bc_rows is None or bc_vals is None:
            # Fallback to zeros if BC info not provided
            return jnp.zeros_like(current_sol)
            
        # Convert to JAX arrays if needed (for JIT compatibility)
        bc_rows_array = jnp.array(bc_rows) if isinstance(bc_rows, (tuple, list)) else bc_rows
        bc_vals_array = jnp.array(bc_vals) if isinstance(bc_vals, (tuple, list)) else bc_vals
        
        if P_mat is not None:
            # Reduced problem case - following ref.py logic
            # x0_1 = assign_bc(zeros_full, problem)
            x0_1 = jnp.zeros(P_mat.shape[0])  # Full size
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)
            
            # x0_2 = copy_bc(P @ current_sol_reduced, problem)
            current_sol_full = P_mat @ current_sol  # Expand reduced to full
            x0_2 = jnp.zeros(P_mat.shape[0])
            x0_2 = x0_2.at[bc_rows_array].set(current_sol_full[bc_rows_array])
            
            # x0 = P.T @ (x0_1 - x0_2) - transform to reduced space
            x0 = P_mat.T @ (x0_1 - x0_2)
        else:
            # Standard (non-reduced) problem case
            # x0_1 = assign_bc(zeros, problem) - sets BC values at BC locations, 0 elsewhere
            x0_1 = jnp.zeros_like(current_sol)
            x0_1 = x0_1.at[bc_rows_array].set(bc_vals_array)
            
            # x0_2 = copy_bc(current_sol, problem) - copies current solution values at BC locations, 0 elsewhere
            x0_2 = jnp.zeros_like(current_sol)
            x0_2 = x0_2.at[bc_rows_array].set(current_sol[bc_rows_array])
            
            # x0 = x0_1 - x0_2 (the original correct implementation)
            x0 = x0_1 - x0_2
        
        return x0
        
    return x0_fn


@dataclass(frozen=True)
class SolverOptions:
    """Configuration options for the Newton solver.
    
    Parameters
    ----------
    tol : float, default 1e-6
        Absolute tolerance for residual vector (l2 norm)
    rel_tol : float, default 1e-8
        Relative tolerance for residual vector (l2 norm)
    max_iter : int, default 100
        Maximum number of Newton iterations
    linear_solver : str, default "cg"
        Linear solver type. Options: "cg", "bicgstab", "gmres"
    preconditioner : callable, optional
        Preconditioner function for linear solver
    use_jacobi_preconditioner : bool, default False
        Whether to use Jacobi (diagonal) preconditioner automatically
    jacobi_shift : float, default 1e-12
        Regularization parameter for Jacobi preconditioner
    linear_solver_tol : float, default 1e-10
        Tolerance for linear solver
    linear_solver_atol : float, default 1e-10
        Absolute tolerance for linear solver
    linear_solver_maxiter : int, default 10000
        Maximum iterations for linear solver
    linear_solver_x0_fn : callable, optional
        Custom function to compute initial guess: f(current_sol) -> x0
    """
    
    tol: float = 1e-6
    rel_tol: float = 1e-8
    max_iter: int = 100
    linear_solver: str = "cg"  # Options: "cg", "bicgstab", "gmres"
    preconditioner: Optional[Callable] = None
    use_jacobi_preconditioner: bool = False
    jacobi_shift: float = 1e-12
    linear_solver_tol: float = 1e-10
    linear_solver_atol: float = 1e-10
    linear_solver_maxiter: int = 10000
    linear_solver_x0_fn: Optional[Callable] = None  # Function to compute initial guess: f(current_sol) -> x0


def newton_solve(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None, P_mat=None):
    """Newton solver using JAX while_loop for JIT compatibility.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and optionally internal_vars.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and optionally internal_vars.
    initial_guess : jax.numpy.ndarray
        Initial solution guess for the Newton solver. For time-dependent problems, this can be
        updated with the solution from the previous time step.
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options. For advanced usage, set linear_solver_x0_fn to provide
        custom initial guess for linear solver following jax-fem approach.
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
    P_mat : BCOO matrix, optional
        Prolongation matrix for reduced problems (maps reduced to full DOFs)
        
    Returns
    -------
    tuple of (sol, res_norm, rel_res_norm, iter_count)
        sol : jax.numpy.ndarray - Solution vector
        res_norm : float - Final residual norm
        rel_res_norm : float - Relative residual norm
        iter_count : int - Number of iterations performed
    """
    
    # Resolve x0 function based on options (at function definition time, not JAX-traced)
    if solver_options.linear_solver_x0_fn is not None:
        # User provided custom function
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        # Create bc_aware x0 function
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver functions for JAX compatibility (no conditionals inside JAX-traced code)
    def solve_cg(A, b, x0):
        # Determine preconditioner
        if solver_options.use_jacobi_preconditioner and solver_options.preconditioner is None:
            M = create_jacobi_preconditioner(A, solver_options.jacobi_shift)
        else:
            M = solver_options.preconditioner
            
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=M,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        # Determine preconditioner
        if solver_options.use_jacobi_preconditioner and solver_options.preconditioner is None:
            M = create_jacobi_preconditioner(A, solver_options.jacobi_shift)
        else:
            M = solver_options.preconditioner
            
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=M,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        # Determine preconditioner
        if solver_options.use_jacobi_preconditioner and solver_options.preconditioner is None:
            M = create_jacobi_preconditioner(A, solver_options.jacobi_shift)
        else:
            M = solver_options.preconditioner
            
        x, _ = jax.scipy.sparse.linalg.gmres(
            A, b, x0=x0,
            M=M,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    # Validate solver choice first (at function definition time)
    valid_solvers = {"cg", "bicgstab", "gmres"}
    if solver_options.linear_solver not in valid_solvers:
        raise ValueError(f"Unknown linear solver: {solver_options.linear_solver}. Choose from {valid_solvers}")
    
    # Select solver function - since we validated, we can use conditionals here
    # This happens at function definition time, not inside JAX-traced code
    if solver_options.linear_solver == "cg":
        linear_solve_fn = solve_cg
    elif solver_options.linear_solver == "bicgstab":
        linear_solve_fn = solve_bicgstab
    else:  # Must be gmres since we validated above
        linear_solve_fn = solve_gmres
    
    def linear_solve_jit(A, b, x0=None):
        """Solve linear system Ax = b using JAX sparse solvers."""
        # Assume A is already in BCOO format (which it should be from the assembler)
        # Use pre-selected solver function (no conditionals in JAX-traced code)
        return linear_solve_fn(A, b, x0)
    
    def cond_fun(state):
        """Condition function for while loop."""
        sol, res_norm, rel_res_norm, iter_count = state
        continue_iter = (res_norm > solver_options.tol) & (rel_res_norm > solver_options.rel_tol) & (iter_count < solver_options.max_iter)
        return continue_iter
    
    def body_fun(state):
        """Body function for while loop - performs one Newton iteration."""
        sol, res_norm, rel_res_norm, iter_count = state
        
        # Compute residual and Jacobian
        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)
        
        # Compute initial guess for increment
        x0 = x0_fn(sol)
        
        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_jit(J, -res, x0=x0)
        
        # Efficient Armijo backtracking line search
        # More efficient than vectorized evaluation for large problems
        
        initial_res_norm = jnp.linalg.norm(res)
        grad_merit = -jnp.dot(res, res)  # Directional derivative
        c1 = 1e-4  # Armijo constant
        
        # Define backtracking parameters
        rho = 0.5  # Backtracking factor
        max_backtracks = 30  # Allow many backtracks for hard problems
        
        def armijo_line_search_body(carry, _):
            """Body function for line search loop."""
            alpha, found_good, best_sol, best_norm = carry
            
            # Try current alpha
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)
            
            # Check if valid (no NaN) and satisfies Armijo condition
            is_valid = jnp.logical_not(jnp.any(jnp.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - initial_res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            
            is_acceptable = is_valid & armijo_satisfied
            
            # Update carry: if acceptable and not found yet, use this
            new_found = found_good | is_acceptable
            new_sol = jnp.where(jnp.logical_not(found_good) & is_acceptable, trial_sol, best_sol)
            new_norm = jnp.where(jnp.logical_not(found_good) & is_acceptable, trial_norm, best_norm)
            new_alpha = jnp.where(is_acceptable, alpha, alpha * rho)
            
            return (new_alpha, new_found, new_sol, new_norm), None
        
        # Initialize with full Newton step
        init_carry = (1.0, False, sol + delta_sol, jnp.inf)
        
        # Run line search
        final_carry, _ = jax.lax.scan(
            armijo_line_search_body, 
            init_carry, 
            jnp.arange(max_backtracks)
        )
        
        _, found_good, new_sol, new_norm = final_carry
        
        # If no good step found, use very small step as fallback
        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        
        sol = jnp.where(found_good, new_sol, fallback_sol)
        res_norm = jnp.where(found_good, new_norm, fallback_norm)
        
        # Update iteration count
        iter_count = iter_count + 1
        
        return (sol, res_norm, rel_res_norm, iter_count)
    
    # Initial state
    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_guess, initial_res_norm, 1.0, 0)
    
    # Run Newton iterations using while_loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    return final_state


def newton_solve_fori(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, num_iters: int, internal_vars=None, P_mat=None):
    """Newton solver using JAX fori_loop for fixed iterations - optimized for vmap.
    
    This solver is specifically designed for use with vmap where we need:
    1. Fixed number of iterations (no early termination)
    2. No print statements or side effects
    3. Consistent computational graph across all vmapped instances
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and optionally internal_vars.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and optionally internal_vars.
    initial_guess : jax.numpy.ndarray
        Initial solution guess for the Newton solver. For time-dependent problems, this can be
        updated with the solution from the previous time step.
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options
    num_iters : int
        Fixed number of Newton iterations to perform
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
    P_mat : BCOO matrix, optional
        Prolongation matrix for reduced problems (maps reduced to full DOFs)
        
    Returns
    -------
    tuple of (sol, final_res_norm, converged)
        sol : jax.numpy.ndarray - Solution vector after num_iters iterations
        final_res_norm : float - Final residual norm
        converged : bool - Whether solution converged (res_norm < tol)
        
    Example
    -------
    >>> # Solve multiple problems in parallel
    >>> def solve_single(params, init_guess):
    >>>     J_fn = lambda sol: create_jacobian(sol, params)
    >>>     res_fn = lambda sol: create_residual(sol, params)
    >>>     sol, norm, converged = newton_solve_fori(J_fn, res_fn, init_guess, bc, options, 10)
    >>>     return sol
    >>> 
    >>> # Vectorize over multiple parameter sets
    >>> all_solutions = jax.vmap(solve_single)(parameter_array, initial_guesses)
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.gmres(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    # Select solver
    valid_solvers = {"cg", "bicgstab", "gmres"}
    if solver_options.linear_solver not in valid_solvers:
        raise ValueError(f"Unknown linear solver: {solver_options.linear_solver}. Choose from {valid_solvers}")
    
    if solver_options.linear_solver == "cg":
        linear_solve_fn = solve_cg
    elif solver_options.linear_solver == "bicgstab":
        linear_solve_fn = solve_bicgstab
    else:
        linear_solve_fn = solve_gmres
    
    def newton_iteration(_, state):
        """Single Newton iteration for fori_loop."""
        sol, res_norm = state
        
        # Compute residual and Jacobian
        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)
        
        # Compute initial guess for increment
        x0 = x0_fn(sol)
        
        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_fn(J, -res, x0)
        
        # Efficient Armijo backtracking line search
        # More efficient than vectorized evaluation for large problems
        
        initial_res_norm = jnp.linalg.norm(res)
        grad_merit = -jnp.dot(res, res)  # Directional derivative
        c1 = 1e-4  # Armijo constant
        
        # Define backtracking parameters
        rho = 0.5  # Backtracking factor
        max_backtracks = 30  # Allow many backtracks for hard problems
        
        def armijo_line_search_body(carry, _):
            """Body function for line search loop."""
            alpha, found_good, best_sol, best_norm = carry
            
            # Try current alpha
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)
            
            # Check if valid (no NaN) and satisfies Armijo condition
            is_valid = jnp.logical_not(jnp.any(jnp.isnan(trial_res)))
            merit_decrease = 0.5 * (trial_norm**2 - initial_res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            
            is_acceptable = is_valid & armijo_satisfied
            
            # Update carry: if acceptable and not found yet, use this
            new_found = found_good | is_acceptable
            new_sol = jnp.where(jnp.logical_not(found_good) & is_acceptable, trial_sol, best_sol)
            new_norm = jnp.where(jnp.logical_not(found_good) & is_acceptable, trial_norm, best_norm)
            new_alpha = jnp.where(is_acceptable, alpha, alpha * rho)
            
            return (new_alpha, new_found, new_sol, new_norm), None
        
        # Initialize with full Newton step
        init_carry = (1.0, False, sol + delta_sol, jnp.inf)
        
        # Run line search
        final_carry, _ = jax.lax.scan(
            armijo_line_search_body, 
            init_carry, 
            jnp.arange(max_backtracks)
        )
        
        _, found_good, new_sol, new_norm = final_carry
        
        # If no good step found, use very small step as fallback
        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        
        sol = jnp.where(found_good, new_sol, fallback_sol)
        res_norm = jnp.where(found_good, new_norm, fallback_norm)
        
        return (sol, res_norm)
    
    # Initial residual norm
    if internal_vars is not None:
        initial_res = res_bc_applied(initial_guess, internal_vars)
    else:
        initial_res = res_bc_applied(initial_guess)
    initial_res_norm = jnp.linalg.norm(initial_res)
    
    # Run fixed number of iterations using fori_loop
    final_state = jax.lax.fori_loop(
        0, num_iters,
        newton_iteration,
        (initial_guess, initial_res_norm)
    )
    
    final_sol, final_res_norm = final_state
    
    # Check convergence
    converged = final_res_norm < solver_options.tol
    
    return final_sol, final_res_norm, converged


def newton_solve_py(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None):
    """Newton solver using Python while loop - non-JIT version for debugging.
    
    This solver uses regular Python control flow instead of JAX control flow,
    making it easier to debug and understand. It cannot be JIT compiled but
    provides the same functionality as newton_solve with detailed introspection.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and optionally internal_vars.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and optionally internal_vars.
    initial_guess : jax.numpy.ndarray
        Initial solution guess for the Newton solver. For time-dependent problems, this can be
        updated with the solution from the previous time step.
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
    P_mat : BCOO matrix, optional
        Prolongation matrix for reduced problems (maps reduced to full DOFs)
        
    Returns
    -------
    tuple of (sol, final_res_norm, converged, num_iters)
        sol : jax.numpy.ndarray - Solution vector
        final_res_norm : float - Final residual norm
        converged : bool - Whether solution converged
        num_iters : int - Number of iterations performed
        
    Example
    -------
    >>> sol, res_norm, converged, iters = newton_solve_py(J_fn, res_fn, init_guess, bc, options)
    >>> print(f"Converged: {converged} in {iters} iterations, residual: {res_norm:.2e}")
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.gmres(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    # Select solver
    valid_solvers = {"cg", "bicgstab", "gmres"}
    if solver_options.linear_solver not in valid_solvers:
        raise ValueError(f"Unknown linear solver: {solver_options.linear_solver}. Choose from {valid_solvers}")
    
    if solver_options.linear_solver == "cg":
        linear_solve_fn = solve_cg
    elif solver_options.linear_solver == "bicgstab":
        linear_solve_fn = solve_bicgstab
    else:
        linear_solve_fn = solve_gmres
    
    def armijo_line_search(sol, delta_sol, res, res_norm):
        """Python version of Armijo backtracking line search."""
        grad_merit = -jnp.dot(res, res)  # Directional derivative
        c1 = 1e-4  # Armijo constant
        rho = 0.5  # Backtracking factor
        max_backtracks = 30
        
        alpha = 1.0
        for _ in range(max_backtracks):
            # Try current alpha
            trial_sol = sol + alpha * delta_sol
            if internal_vars is not None:
                trial_res = res_bc_applied(trial_sol, internal_vars)
            else:
                trial_res = res_bc_applied(trial_sol)
            trial_norm = jnp.linalg.norm(trial_res)
            
            # Check if valid (no NaN) and satisfies Armijo condition
            is_valid = not jnp.any(jnp.isnan(trial_res))
            merit_decrease = 0.5 * (trial_norm**2 - res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            
            if is_valid and armijo_satisfied:
                return trial_sol, trial_norm, alpha, True
            
            alpha *= rho
        
        # If no good step found, use very small step as fallback
        fallback_sol = sol + 1e-8 * delta_sol
        if internal_vars is not None:
            fallback_res = res_bc_applied(fallback_sol, internal_vars)
        else:
            fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        return fallback_sol, fallback_norm, 1e-8, False
    
    # Initialize
    sol = initial_guess
    if internal_vars is not None:
        initial_res = res_bc_applied(sol, internal_vars)
    else:
        initial_res = res_bc_applied(sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    res_norm = initial_res_norm
    iter_count = 0
    
    # Main Newton loop
    while (res_norm > solver_options.tol and 
           res_norm / initial_res_norm > solver_options.rel_tol and 
           iter_count < solver_options.max_iter):
        
        # Compute residual and Jacobian
        if internal_vars is not None:
            res = res_bc_applied(sol, internal_vars)
            J = J_bc_applied(sol, internal_vars)
        else:
            res = res_bc_applied(sol)
            J = J_bc_applied(sol)
        
        # Compute initial guess for increment
        x0 = x0_fn(sol)
        
        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_fn(J, -res, x0)
        
        # Line search
        new_sol, new_res_norm, _, _ = armijo_line_search(
            sol, delta_sol, res, res_norm
        )
        
        # Update solution
        sol = new_sol
        res_norm = new_res_norm
        iter_count += 1
        
    
    # Check convergence
    converged = (res_norm <= solver_options.tol or 
                res_norm / initial_res_norm <= solver_options.rel_tol)
    
    return sol, res_norm, converged, iter_count


def linear_solve(J_bc_applied, res_bc_applied, initial_guess, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None, P_mat=None):
    """Linear solver for problems that converge in one iteration (no while loop).
    
    This solver is optimized for linear elasticity problems where the solution
    can be found in a single Newton iteration. By eliminating the while loop,
    this solver is much more efficient for vmap operations.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and optionally internal_vars,
        returns sparse matrix.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and optionally internal_vars,
        returns residual vector.
    initial_guess : jax.numpy.ndarray
        Initial solution guess (typically zeros with BC values set). For time-dependent problems,
        this can be the solution from the previous time step.
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
    P_mat : BCOO matrix, optional
        Prolongation matrix for reduced problems (maps reduced to full DOFs)
        
    Returns
    -------
    tuple of (sol, None)
        sol : jax.numpy.ndarray - Solution vector
        None : placeholder for compatibility with other solver returns
        
    Notes
    -----
    This solver performs exactly one Newton iteration:
    1. Compute residual: res = res_bc_applied(initial_guess)
    2. Compute Jacobian: J = J_bc_applied(initial_guess)
    3. Solve: J * delta_sol = -res
    4. Update: sol = initial_guess + delta_sol
    
    For linear problems, this single iteration achieves the exact solution.
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals,
            P_mat=P_mat
        )
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.gmres(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    # Validate and select solver
    valid_solvers = {"cg", "bicgstab", "gmres"}
    if solver_options.linear_solver not in valid_solvers:
        raise ValueError(f"Unknown linear solver: {solver_options.linear_solver}. Choose from {valid_solvers}")
    
    if solver_options.linear_solver == "cg":
        linear_solve_fn = solve_cg
    elif solver_options.linear_solver == "bicgstab":
        linear_solve_fn = solve_bicgstab
    else:
        linear_solve_fn = solve_gmres
    
    # Single Newton iteration (no while loop)
    # Step 1: Compute residual and Jacobian
    if internal_vars is not None:
        res = res_bc_applied(initial_guess, internal_vars)
        J = J_bc_applied(initial_guess, internal_vars)
    else:
        res = res_bc_applied(initial_guess)
        J = J_bc_applied(initial_guess)
    
    # Step 2: Compute initial guess for increment
    x0 = x0_fn(initial_guess)
    
    # Step 3: Solve linear system: J * delta_sol = -res
    delta_sol = linear_solve_fn(J, -res, x0)
    
    # Step 4: Update solution
    sol = initial_guess + delta_sol
    
    return sol, None

def __linear_solve_adjoint(A, b, solver_options: SolverOptions):
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, _ = jax.scipy.sparse.linalg.gmres(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    # Validate and select solver
    valid_solvers = {"cg", "bicgstab", "gmres"}
    if solver_options.linear_solver not in valid_solvers:
        raise ValueError(f"Unknown linear solver: {solver_options.linear_solver}. Choose from {valid_solvers}")
    
    if solver_options.linear_solver == "cg":
        linear_solve_fn = solve_cg
    elif solver_options.linear_solver == "bicgstab":
        linear_solve_fn = solve_bicgstab
    else:
        linear_solve_fn = solve_gmres
    
    x0 = None
    sol = linear_solve_fn(A, b, x0)
    
    return sol


def _create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options, iter_num):
    """Create matrix-free reduced solver for periodic boundary conditions."""
    
    # Create full space functions
    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)
    
    # Matrix-free reduced operations
    def create_reduced_matvec(sol_full, internal_vars):
        """Create matrix-vector product function for reduced Jacobian."""
        J_full = J_bc_func(sol_full, internal_vars)
        
        def reduced_matvec(v_reduced):
            v_full = P @ v_reduced          # Expand to full space
            Jv_full = J_full @ v_full       # Apply full Jacobian
            Jv_reduced = P.T @ Jv_full      # Reduce back
            return Jv_reduced
        return reduced_matvec
    
    def compute_reduced_residual(sol_full, internal_vars):
        """Compute residual in reduced space."""
        res_full = res_bc_func(sol_full, internal_vars)
        return P.T @ res_full
    
    # Matrix-free solver function
    def reduced_solve_fn(internal_vars, initial_guess_full):
        """Solve in reduced space using matrix-free CG."""
        # Compute reduced residual
        res_reduced = compute_reduced_residual(initial_guess_full, internal_vars)
        
        # Create reduced Jacobian matvec
        J_reduced_matvec = create_reduced_matvec(initial_guess_full, internal_vars)
        
        # Solve reduced system: J_reduced @ sol_reduced = -res_reduced
        sol_reduced, _ = jax.scipy.sparse.linalg.cg(J_reduced_matvec, -res_reduced, 
                                                   tol=solver_options.tol,
                                                   maxiter=solver_options.linear_solver_maxiter)
        
        # Map back to full space
        sol_full = P @ sol_reduced
        return sol_full, None
    
    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        """Matrix-free reduced solver with automatic differentiation."""
        return reduced_solve_fn(internal_vars, initial_guess)[0]
    
    def f_fwd(internal_vars, initial_guess):
        """Forward function for custom VJP."""
        sol = differentiable_solve(internal_vars, initial_guess)
        return sol, (internal_vars, sol)
    
    def f_bwd(res, v):
        """Backward function using matrix-free adjoint."""
        internal_vars, sol = res
        
        # Create adjoint matvec operator: (P.T @ J.T @ P) @ adjoint = P.T @ v
        J_full = J_bc_func(sol, internal_vars)
        rhs_reduced = P.T @ v
        
        def adjoint_matvec(adjoint_reduced):
            adjoint_full = P @ adjoint_reduced    # Expand to full space
            Jt_adjoint_full = J_full.T @ adjoint_full  # Apply transpose Jacobian
            return P.T @ Jt_adjoint_full          # Reduce back
        
        # Solve adjoint system: J_reduced.T @ adjoint_reduced = rhs_reduced
        adjoint_reduced, _ = jax.scipy.sparse.linalg.cg(adjoint_matvec, rhs_reduced,
                                                        tol=adjoint_solver_options.tol)
        
        # Compute VJP for internal variables
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
        
        sol_list = problem.unflatten_fn_sol_list(sol)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_full))
        vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)

        return (vjp_result, None)  # No gradient w.r.t. initial_guess
    
    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve


def create_solver(problem, bc, solver_options=None, adjoint_solver_options=None, iter_num=None, P=None):
    """Create a differentiable solver that returns gradients w.r.t. internal_vars using custom VJP.
    
    This solver uses the self-adjoint approach for efficient gradient computation:
    - Forward mode: standard Newton solve
    - Backward mode: solve adjoint system to compute gradients
    
    Parameters
    ----------
    problem : Problem
        The feax Problem instance (modular API - no internal_vars in constructor)
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions, optional
        Options for forward solve (defaults to SolverOptions())
    adjoint_solver_options : dict, optional
        Options for adjoint solve (defaults to same as forward solve)
    iter_num : int, optional
        Number of iterations to perform. Controls which solver is used:
        - None: Use while loop newton_solve (adaptive iterations, NOT vmappable)
        - 1: Use linear_solve (single iteration for linear problems, vmappable)
        - >1: Use newton_solve_fori with fixed number of iterations (vmappable)
        Note: When iter_num is not None, the solver is vmappable since it uses fixed iterations.
        Recommended: Use iter_num=1 for linear problems for optimal performance.
    P : BCOO matrix, optional
        Prolongation matrix for periodic boundary conditions (maps reduced to full DOFs).
        If provided, solver works in reduced space using matrix-free operations for memory efficiency.
        
    Returns
    -------
    differentiable_solve : callable
        Function that takes (internal_vars, initial_guess) and returns solution with gradient support
        
    Notes
    -----
    The returned function has signature: differentiable_solve(internal_vars, initial_guess) -> solution
    where gradients flow through internal_vars (material properties, loadings, etc.)
    
    The initial_guess parameter is required to avoid conditionals that slow down JAX compilation.
    For the first solve, you can pass zeros with BC values set:
        initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
        initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    
    Based on the self-adjoint approach from the reference implementation in ref_solver.py.
    The adjoint method is more efficient than forward-mode AD for optimization problems
    where we need gradients w.r.t. many parameters but few outputs.
    
    When iter_num is specified (not None), the solver becomes vmappable as it uses fixed
    iterations without dynamic control flow. This is essential for parallel solving of
    multiple parameter sets using jax.vmap.
    
    Examples
    --------
    >>> # Create differentiable solver
    >>> diff_solve = create_solver(problem, bc)
    >>> 
    >>> # Prepare initial guess
    >>> initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
    >>> initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    >>> 
    >>> # First solve
    >>> solution = diff_solve(internal_vars, initial_guess)
    >>> 
    >>> # For time-dependent problems, update initial guess each timestep
    >>> for t in timesteps:
    >>>     solution = diff_solve(internal_vars_at_t, solution)  # Use previous solution
    >>> 
    >>> # For linear problems (e.g., linear elasticity), use single iteration for best performance
    >>> # This is both faster and vmappable
    >>> diff_solve_linear = create_solver(problem, bc, iter_num=1)
    >>> 
    >>> # For fixed iteration count (e.g., for vmap)
    >>> diff_solve_fixed = create_solver(problem, bc, iter_num=10)
    >>> 
    >>> # Define loss function
    >>> def loss_fn(internal_vars):
    ...     initial_guess = jnp.zeros(problem.num_total_dofs_all_vars)
    ...     initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    ...     sol = diff_solve(internal_vars, initial_guess)
    ...     return jnp.sum(sol**2)  # Example loss
    >>> 
    >>> # Compute gradients w.r.t. internal_vars
    >>> grad_fn = jax.grad(loss_fn)
    >>> gradients = grad_fn(internal_vars)
    """
    
    # Set default options
    if solver_options is None:
        solver_options = SolverOptions()
    if adjoint_solver_options is None:
        adjoint_solver_options = SolverOptions(
            linear_solver="bicgstab",  # More robust than CG
            tol=1e-10,
            use_jacobi_preconditioner=True
        )

    # Branch between standard and reduced solver
    if P is not None:
        return _create_reduced_solver(problem, bc, P, solver_options, adjoint_solver_options, iter_num)
    
    # Standard solver (original implementation)
    J_bc_func = create_J_bc_function(problem, bc)
    res_bc_func = create_res_bc_function(problem, bc)
    
    # Standard case - no special handling
    if iter_num is None:
        solve_fn = lambda internal_vars, initial_sol: newton_solve(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, internal_vars
        )
    elif iter_num == 1:
        solve_fn = lambda internal_vars, initial_sol: linear_solve(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, internal_vars
        )
    else:
        solve_fn = lambda internal_vars, initial_sol: newton_solve_fori(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, iter_num, internal_vars
        )

    @jax.custom_vjp
    def differentiable_solve(internal_vars, initial_guess):
        """Forward solve: standard Newton iteration.
        
        Parameters
        ----------
        internal_vars : InternalVars
            Material properties, loadings, etc.
        initial_guess : jax.numpy.ndarray
            Initial guess for the solution vector. For time-dependent problems,
            this should be the solution from the previous time step.
            
        Returns
        -------
        sol : jax.numpy.ndarray
            Solution vector
        """
        return solve_fn(internal_vars, initial_guess)[0]
    
    def f_fwd(internal_vars, initial_guess):
        """Forward function for custom VJP.
        
        Returns solution and residuals needed for backward pass.
        """
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
        adjoint_vec = __linear_solve_adjoint(J.transpose(), v_vec, adjoint_solver_options)
        sol_list = problem.unflatten_fn_sol_list(sol)
        vjp_linear_fn = get_vjp_contraint_fn_params(internal_vars, sol_list)
        vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
        vjp_result = jax.tree_util.tree_map(lambda x: -x, vjp_result)

        return (vjp_result, None)  # No gradient w.r.t. initial_guess
    
    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve