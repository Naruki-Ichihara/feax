import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Callable
from jax.experimental.sparse import BCOO, eye, bcoo_multiply_sparse
from dataclasses import dataclass


def create_diagonals_fn():
    """Create a function that returns vector of diagonals of BCOO matrix.
        
    Returns
    -------
    diagonals_fn : callable
        Function that takes a BCOO matrix and returns the diagonal elements as a vector
        
    Notes
    -----
    For a BCOO matrix, the diagonal elements are extracted by finding
    entries where row index equals column index.
    """
    def diagonals_fn(A):
        """Extract diagonal elements from BCOO matrix.
        
        Parameters
        ----------
        A : jax.experimental.sparse.BCOO
            BCOO sparse matrix
        """
        # Get indices and data from BCOO matrix
        indices = A.indices  # Shape: (nnz, 2) - [row, col] pairs
        data = A.data       # Shape: (nnz,) - values
        
        # Find diagonal entries where row == col
        diagonal_mask = indices[:, 0] == indices[:, 1]
        diagonal_indices = indices[diagonal_mask, 0]  # Row indices of diagonal elements
        diagonal_values = data[diagonal_mask]         # Values of diagonal elements
        
        # Create full diagonal vector (zeros for missing diagonal entries)
        n = A.shape[0]  # Matrix size
        diagonals = jnp.zeros(n)
        diagonals = diagonals.at[diagonal_indices].set(diagonal_values)
        
        return diagonals
    
    return diagonals_fn


def create_x0(bc_rows=None, bc_vals=None, strategy="zeros"):
    """Create initial guess function for linear solver following JAX-FEM approach.
    
    Parameters
    ----------
    bc_rows : array-like, optional
        Row indices of boundary condition locations
    bc_vals : array-like, optional  
        Boundary condition values, required for x0_strategy="bc_aware"
    strategy : str, default "zeros"
        Strategy for creating initial guess:
        - "zeros": Return zeros (good default for increments)
        - "bc_aware": Correct x0 method from row elimination solver (x0_1 - x0_2)
        
    Returns
    -------
    x0_fn : callable
        Function that takes current solution and returns initial guess for increment
        
    Notes
    -----
    The bc_aware strategy implements the exact x0 computation from the row elimination solver:
    x0_1 = assign_bc(zeros, problem) - sets BC values at BC locations, 0 elsewhere
    x0_2 = copy_bc(current_sol, problem) - copies current solution values at BC locations, 0 elsewhere  
    x0 = x0_1 - x0_2 - the correct initial guess computation
    
    Examples
    --------
    >>> # Basic usage (zeros everywhere)
    >>> x0_fn = create_x0()
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    
    >>> # Advanced usage with BC information
    >>> x0_fn = create_x0(bc_rows=[0, 1, 2], bc_vals=[1.0, 0.0, 2.0], strategy="bc_aware") 
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    """
    
    if strategy == "zeros":
        def x0_fn(current_sol):
            """Default strategy: zeros everywhere (good for increments)."""
            return jnp.zeros_like(current_sol)
            
    elif strategy == "bc_aware":
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
            
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'zeros' or 'bc_aware'.")
        
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
        If provided, overrides x0_strategy
    x0_strategy : str, default "zeros"
        Strategy for initial guess. Options: "zeros", "bc_aware"
        Used only if linear_solver_x0_fn is None
    bc_rows : array-like, optional
        Boundary condition row indices, required for x0_strategy="bc_aware"
    bc_vals : array-like, optional
        Boundary condition values, required for x0_strategy="bc_aware"
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
    x0_strategy: str = "zeros"  # Options: "zeros", "bc_aware" - used if linear_solver_x0_fn is None
    bc_rows: Optional[object] = None  # Boundary condition row indices for x0_strategy="bc_aware"
    bc_vals: Optional[object] = None  # Boundary condition values for x0_strategy="bc_aware"


def newton_solve(J_bc_applied, res_bc_applied, initial_sol, solver_options: SolverOptions):
    """Newton solver using JAX while_loop for JIT compatibility.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and returns sparse matrix.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and returns residual vector.
    initial_sol : jax.numpy.ndarray
        Initial solution guess
    solver_options : SolverOptions
        Solver configuration options. For advanced usage, set linear_solver_x0_fn to provide
        custom initial guess for linear solver following jax-fem approach.
        
    Returns
    -------
    sol : jax.numpy.ndarray
        Solution vector
    """
    
    # Resolve x0 function based on options (at function definition time, not JAX-traced)
    if solver_options.linear_solver_x0_fn is not None:
        # User provided custom function
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        # Use x0_strategy to create function automatically
        x0_fn = create_x0(
            bc_rows=solver_options.bc_rows,
            bc_vals=solver_options.bc_vals,
            strategy=solver_options.x0_strategy
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
    
    # Select solver function - since we validated, we can use simple conditionals here
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
        fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        
        sol = jnp.where(found_good, new_sol, fallback_sol)
        res_norm = jnp.where(found_good, new_norm, fallback_norm)
        
        # Update iteration count
        iter_count = iter_count + 1
        
        return (sol, res_norm, rel_res_norm, iter_count)
    
    # Initial state
    initial_res = res_bc_applied(initial_sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_sol, initial_res_norm, 1.0, 0)
    
    # Run Newton iterations using while_loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    return final_state


def newton_solve_fori(J_bc_applied, res_bc_applied, initial_sol, solver_options: SolverOptions, num_iters: int):
    """Newton solver using JAX fori_loop for fixed iterations - optimized for vmap.
    
    This solver is specifically designed for use with vmap where we need:
    1. Fixed number of iterations (no early termination)
    2. No print statements or side effects
    3. Consistent computational graph across all vmapped instances
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied
    res_bc_applied : callable
        Function that computes the residual with BC applied
    initial_sol : jax.numpy.ndarray
        Initial solution guess
    solver_options : SolverOptions
        Solver configuration options
    num_iters : int
        Fixed number of Newton iterations to perform
        
    Returns
    -------
    sol : jax.numpy.ndarray
        Solution vector after num_iters iterations
    final_res_norm : float
        Final residual norm
    converged : bool
        Whether solution converged (res_norm < tol)
        
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
            bc_rows=solver_options.bc_rows,
            bc_vals=solver_options.bc_vals,
            strategy=solver_options.x0_strategy
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
        fallback_res = res_bc_applied(fallback_sol)
        fallback_norm = jnp.linalg.norm(fallback_res)
        
        sol = jnp.where(found_good, new_sol, fallback_sol)
        res_norm = jnp.where(found_good, new_norm, fallback_norm)
        
        return (sol, res_norm)
    
    # Initial residual norm
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


def linear_solve(J_bc_applied, res_bc_applied, initial_sol, solver_options: SolverOptions):
    """Linear solver for problems that converge in one iteration (no while loop).
    
    This solver is optimized for linear elasticity problems where the solution
    can be found in a single Newton iteration. By eliminating the while loop,
    this solver is much more efficient for vmap operations.
    
    Parameters
    ----------
    J_bc_applied : callable
        Function that computes the Jacobian with BC applied. Takes solution vector and returns sparse matrix.
    res_bc_applied : callable
        Function that computes the residual with BC applied. Takes solution vector and returns residual vector.
    initial_sol : jax.numpy.ndarray
        Initial solution guess (typically zeros with BC values set)
    solver_options : SolverOptions
        Solver configuration options
        
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
            bc_rows=solver_options.bc_rows,
            bc_vals=solver_options.bc_vals,
            strategy=solver_options.x0_strategy
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
    res = res_bc_applied(initial_sol)
    J = J_bc_applied(initial_sol)
    
    # Step 2: Compute initial guess for increment
    x0 = x0_fn(initial_sol)
    
    # Step 3: Solve linear system: J * delta_sol = -res
    delta_sol = linear_solve_fn(J, -res, x0)
    
    # Step 4: Update solution
    sol = initial_sol + delta_sol
    
    return sol