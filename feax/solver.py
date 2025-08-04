import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Callable
from jax.experimental.sparse import BCOO, eye, bcoo_multiply_sparse
from dataclasses import dataclass
from .assembler import get_J, get_res
from .assembler import create_J_bc_function, create_res_bc_function
from .DCboundary import DirichletBC


def create_x0(bc_rows=None, bc_vals=None):
    """Create initial guess function for linear solver following JAX-FEM approach.
    
    Parameters
    ----------
    bc_rows : array-like, optional
        Row indices of boundary condition locations
    bc_vals : array-like, optional  
        Boundary condition values
        
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
    
    Examples
    --------
    >>> # Usage with BC information
    >>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0]) 
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    """
    
    def x0_fn(current_sol):
        """BC-aware strategy: correct x0 method from row elimination solver."""
        if bc_rows is None or bc_vals is None:
            # Fallback to zeros if BC info not provided
            return jnp.zeros_like(current_sol)
            
        # Convert to JAX arrays if needed (for JIT compatibility)
        bc_rows_array = jnp.array(bc_rows) if isinstance(bc_rows, (tuple, list)) else bc_rows
        bc_vals_array = jnp.array(bc_vals) if isinstance(bc_vals, (tuple, list)) else bc_vals
        
        # Implement the exact logic from the old row elimination solver:
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
    linear_solver_tol : float, default 1e-10
        Tolerance for linear solver
    linear_solver_atol : float, default 1e-10
        Absolute tolerance for linear solver
    linear_solver_maxiter : int, default 10000
        Maximum iterations for linear solver
    linear_solver_x0_fn : callable, optional
        Custom function to compute initial guess: f(current_sol) -> x0
    initial_guess : array-like, optional
        Initial guess for the solution vector in newton solver. If None, uses zeros with BC values set.
    """
    
    tol: float = 1e-6
    rel_tol: float = 1e-8
    max_iter: int = 100
    linear_solver: str = "cg"  # Options: "cg", "bicgstab", "gmres"
    preconditioner: Optional[Callable] = None
    linear_solver_tol: float = 1e-10
    linear_solver_atol: float = 1e-10
    linear_solver_maxiter: int = 10000
    linear_solver_x0_fn: Optional[Callable] = None  # Function to compute initial guess: f(current_sol) -> x0
    initial_guess: Optional[object] = None  # Initial guess for the solution vector in newton solver


def newton_solve(J_bc_applied, res_bc_applied, initial_sol, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None):
    """Newton solver using JAX while_loop for JIT compatibility.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and optionally internal_vars.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and optionally internal_vars.
    initial_sol : jax.numpy.ndarray
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options. For advanced usage, set linear_solver_x0_fn to provide
        custom initial guess for linear solver following jax-fem approach.
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
        
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
            bc_vals=bc.bc_vals
        )
    
    # Define solver functions for JAX compatibility (no conditionals inside JAX-traced code)
    def solve_cg(A, b, x0):
        x, info = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, info = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, info = jax.scipy.sparse.linalg.gmres(
            A, b, x0=x0,
            M=solver_options.preconditioner,
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
    
    @jax.jit
    def linear_solve(A, b, x0=None):
        """Solve linear system Ax = b using JAX sparse solvers - JIT compiled."""
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
        delta_sol = linear_solve(J, -res, x0=x0)
        
        # Efficient Armijo backtracking line search
        # More efficient than vectorized evaluation for large problems
        
        initial_res_norm = jnp.linalg.norm(res)
        grad_merit = -jnp.dot(res, res)  # Directional derivative
        c1 = 1e-4  # Armijo constant
        
        # Define backtracking parameters
        rho = 0.5  # Backtracking factor
        max_backtracks = 30  # Allow many backtracks for hard problems
        
        def armijo_line_search_body(carry, i):
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
            is_valid = ~jnp.any(jnp.isnan(trial_res))
            merit_decrease = 0.5 * (trial_norm**2 - initial_res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            
            is_acceptable = is_valid & armijo_satisfied
            
            # Update carry: if acceptable and not found yet, use this
            new_found = found_good | is_acceptable
            new_sol = jnp.where(~found_good & is_acceptable, trial_sol, best_sol)
            new_norm = jnp.where(~found_good & is_acceptable, trial_norm, best_norm)
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
        initial_res = res_bc_applied(initial_sol, internal_vars)
    else:
        initial_res = res_bc_applied(initial_sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_sol, initial_res_norm, 1.0, 0)
    
    # Run Newton iterations using while_loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    return final_state


def newton_solve_fori(J_bc_applied, res_bc_applied, initial_sol, bc: DirichletBC, solver_options: SolverOptions, num_iters: int, internal_vars=None):
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
    initial_sol : jax.numpy.ndarray
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options
    num_iters : int
        Fixed number of Newton iterations to perform
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
        
    Returns
    -------
    tuple of (sol, final_res_norm, converged)
        sol : jax.numpy.ndarray - Solution vector after num_iters iterations
        final_res_norm : float - Final residual norm
        converged : bool - Whether solution converged (res_norm < tol)
        
    Example
    -------
    >>> # Solve multiple problems in parallel
    >>> def solve_single(params):
    >>>     J_fn = lambda sol: create_jacobian(sol, params)
    >>>     res_fn = lambda sol: create_residual(sol, params)
    >>>     sol, norm, converged = newton_solve_fori(J_fn, res_fn, init_sol, options, 10)
    >>>     return sol
    >>> 
    >>> # Vectorize over multiple parameter sets
    >>> all_solutions = jax.vmap(solve_single)(parameter_array)
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals
        )
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, info = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, info = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, info = jax.scipy.sparse.linalg.gmres(
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
    
    def newton_iteration(i, state):
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
        
        def armijo_line_search_body(carry, i):
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
            is_valid = ~jnp.any(jnp.isnan(trial_res))
            merit_decrease = 0.5 * (trial_norm**2 - initial_res_norm**2)
            armijo_satisfied = merit_decrease <= c1 * alpha * grad_merit
            
            is_acceptable = is_valid & armijo_satisfied
            
            # Update carry: if acceptable and not found yet, use this
            new_found = found_good | is_acceptable
            new_sol = jnp.where(~found_good & is_acceptable, trial_sol, best_sol)
            new_norm = jnp.where(~found_good & is_acceptable, trial_norm, best_norm)
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
        initial_res = res_bc_applied(initial_sol, internal_vars)
    else:
        initial_res = res_bc_applied(initial_sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    
    # Run fixed number of iterations using fori_loop
    final_state = jax.lax.fori_loop(
        0, num_iters,
        newton_iteration,
        (initial_sol, initial_res_norm)
    )
    
    final_sol, final_res_norm = final_state
    
    # Check convergence
    converged = final_res_norm < solver_options.tol
    
    return final_sol, final_res_norm, converged


def newton_solve_py(J_bc_applied, res_bc_applied, initial_sol, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None):
    """Newton solver using Python while loop - non-JIT version for debugging.
    
    This solver uses regular Python control flow instead of JAX control flow,
    making it easier to debug and understand. It cannot be JIT compiled but
    provides the same functionality as newton_solve with better introspection.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and optionally internal_vars.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and optionally internal_vars.
    initial_sol : jax.numpy.ndarray
        Initial solution guess
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
        
    Returns
    -------
    tuple of (sol, final_res_norm, converged, num_iters)
        sol : jax.numpy.ndarray - Solution vector
        final_res_norm : float - Final residual norm
        converged : bool - Whether solution converged
        num_iters : int - Number of iterations performed
        
    Example
    -------
    >>> sol, res_norm, converged, iters = newton_solve_py(J_fn, res_fn, init_sol, options)
    >>> print(f"Converged: {converged} in {iters} iterations, residual: {res_norm:.2e}")
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals
        )
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, info = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, info = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, info = jax.scipy.sparse.linalg.gmres(
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
        for i in range(max_backtracks):
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
    sol = initial_sol
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
        new_sol, new_res_norm, alpha, line_search_success = armijo_line_search(
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


def linear_solve(J_bc_applied, res_bc_applied, initial_sol, bc: DirichletBC, solver_options: SolverOptions, internal_vars=None):
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
    initial_sol : jax.numpy.ndarray
        Initial solution guess (typically zeros with BC values set)
    bc : DirichletBC
        Boundary conditions
    solver_options : SolverOptions
        Solver configuration options
    internal_vars : InternalVars, optional
        Internal variables (e.g., material properties) to pass to J_bc_applied and res_bc_applied.
        If provided, these functions will be called with (sol, internal_vars).
        
    Returns
    -------
    sol : jax.numpy.ndarray
        Solution vector
        
    Notes
    -----
    This solver performs exactly one Newton iteration:
    1. Compute residual: res = res_bc_applied(initial_sol)
    2. Compute Jacobian: J = J_bc_applied(initial_sol)
    3. Solve: J * delta_sol = -res
    4. Update: sol = initial_sol + delta_sol
    
    For linear problems, this single iteration achieves the exact solution.
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=bc.bc_rows,
            bc_vals=bc.bc_vals
        )
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, info = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, info = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, info = jax.scipy.sparse.linalg.gmres(
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
        res = res_bc_applied(initial_sol, internal_vars)
        J = J_bc_applied(initial_sol, internal_vars)
    else:
        res = res_bc_applied(initial_sol)
        J = J_bc_applied(initial_sol)
    
    # Step 2: Compute initial guess for increment
    x0 = x0_fn(initial_sol)
    
    # Step 3: Solve linear system: J * delta_sol = -res
    delta_sol = linear_solve_fn(J, -res, x0)
    
    # Step 4: Update solution
    sol = initial_sol + delta_sol
    
    return sol, None

def __linear_solve_adjoint(A, b, solver_options: SolverOptions):
    
    # Define solver functions
    def solve_cg(A, b, x0):
        x, info = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, 
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_bicgstab(A, b, x0):
        x, info = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0,
            M=solver_options.preconditioner,
            tol=solver_options.linear_solver_tol,
            atol=solver_options.linear_solver_atol,
            maxiter=solver_options.linear_solver_maxiter
        )
        return x
    
    def solve_gmres(A, b, x0):
        x, info = jax.scipy.sparse.linalg.gmres(
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

def create_solver(problem, bc, solver_options=None, adjoint_solver_options=None, iter_num=None):
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
        Options for forward solve (defaults to SolverOptions()). Can include initial_guess
        to provide a custom initial solution vector instead of zeros with BC values
    adjoint_solver_options : dict, optional
        Options for adjoint solve (defaults to same as forward solve)
    iter_num : int, optional
        Number of iterations to perform. Controls which solver is used:
        - None: Use while loop newton_solve (adaptive iterations, NOT vmappable)
        - 1: Use linear_solve (single iteration for linear problems, vmappable)
        - >1: Use newton_solve_fori with fixed number of iterations (vmappable)
        Note: When iter_num is not None, the solver is vmappable since it uses fixed iterations.
        Recommended: Use iter_num=1 for linear problems for optimal performance.
        
    Returns
    -------
    differentiable_solve : callable
        Function that takes internal_vars and returns solution with gradient support
        
    Notes
    -----
    The returned function has signature: differentiable_solve(internal_vars) -> solution
    where gradients flow through internal_vars (material properties, loadings, etc.)
    
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
    >>> # Create differentiable solver with custom initial guess
    >>> solver_opts = SolverOptions(initial_guess=my_initial_solution)
    >>> diff_solve = create_solver(problem, bc, solver_opts)
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
    ...     sol = diff_solve(internal_vars)
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
            tol=1e-10
        )

    J_bc_func = jax.jit(create_J_bc_function(problem, bc))
    res_bc_func = jax.jit(create_res_bc_function(problem, bc))

    # Initial solution
    if solver_options.initial_guess is not None:
        initial_sol = solver_options.initial_guess
    else:
        initial_sol = jnp.zeros(problem.num_total_dofs_all_vars)
        initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)
        
    # Solve using appropriate method based on iter_num
    if iter_num is None:
        solve_fn = lambda internal_vars: newton_solve(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, internal_vars
        )
    elif iter_num == 1:
        solve_fn = lambda internal_vars: linear_solve(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, internal_vars
        )
    else:
        solve_fn = lambda internal_vars: newton_solve_fori(
            J_bc_func, res_bc_func, initial_sol, bc, solver_options, iter_num, internal_vars
        )

    @jax.custom_vjp
    def differentiable_solve(internal_vars):
        """Forward solve: standard Newton iteration.
        
        Parameters
        ----------
        internal_vars : InternalVars
            Material properties, loadings, etc.
            
        Returns
        -------
        sol : jax.numpy.ndarray
            Solution vector
        """
        return solve_fn(internal_vars)[0]
    
    def f_fwd(internal_vars):
        """Forward function for custom VJP.
        
        Returns solution and residuals needed for backward pass.
        """
        sol = differentiable_solve(internal_vars)
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
                primals_output, f_vjp = jax.vjp(partial_c_fn, internal_vars)
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

        return (vjp_result,)
    
    differentiable_solve.defvjp(f_fwd, f_bwd)
    return differentiable_solve