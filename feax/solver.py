import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Callable


def create_x0(bc_rows=None, bc_vals=None, strategy="zeros"):
    """Create initial guess function for linear solver following JAX-FEM approach.
    
    Parameters
    ----------
    bc_rows : array-like, optional
        Row indices of boundary condition locations
    bc_vals : array-like, optional  
        Boundary condition values (not used in current implementation)
    strategy : str, default "zeros"
        Strategy for creating initial guess:
        - "zeros": Return zeros (good default for increments)
        - "bc_aware": Explicitly ensure BC locations are zero (future enhancement)
        
    Returns
    -------
    x0_fn : callable
        Function that takes current solution and returns initial guess for increment
        
    Notes
    -----
    The key insight from JAX-FEM is that when solving J * delta_sol = -res,
    the increment delta_sol should be zero at boundary condition locations
    since the solution is already correct there.
    
    Examples
    --------
    >>> # Basic usage (zeros everywhere)
    >>> x0_fn = create_x0()
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    
    >>> # Advanced usage with BC information
    >>> x0_fn = create_x0(bc_rows=[0, 1, 2], strategy="bc_aware") 
    >>> solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)
    """
    
    if strategy == "zeros":
        def x0_fn(current_sol):
            """Default strategy: zeros everywhere (good for increments)."""
            return jnp.zeros_like(current_sol)
            
    elif strategy == "bc_aware":
        def x0_fn(current_sol):
            """BC-aware strategy: explicitly ensure BC locations are zero."""
            x0 = jnp.zeros_like(current_sol)
            if bc_rows is not None:
                # Convert to JAX array if it's a tuple (for JIT compatibility)
                bc_rows_array = jnp.array(bc_rows) if isinstance(bc_rows, (tuple, list)) else bc_rows
                # Explicitly set BC locations to zero (though they already are)
                # This is more for clarity and potential future enhancements
                x0 = x0.at[bc_rows_array].set(0.0)
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
        # Uses either custom function or default zeros
        x0 = x0_fn(sol)
        
        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve(J, -res, x0=x0)
        
        # Update solution
        sol = sol + delta_sol
        
        # Compute new residual and norms
        res = res_bc_applied(sol)
        res_norm = jnp.linalg.norm(res)
        
        # Update iteration count
        iter_count = iter_count + 1
        
        return (sol, res_norm, rel_res_norm, iter_count)
    
    # Initial state
    initial_res = res_bc_applied(initial_sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_state = (initial_sol, initial_res_norm, 1.0, 0)
    
    # Run Newton iterations using while_loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    sol, final_res_norm, final_rel_res_norm, final_iter = final_state
    
    # For the first iteration, compute relative residual norm
    def compute_relative_norm(state):
        sol, res_norm, _, iter_count = state
        rel_res_norm = res_norm / initial_res_norm
        return (sol, res_norm, rel_res_norm, iter_count)
    
    # Apply relative norm computation after first iteration
    final_state = jax.lax.cond(
        initial_res_norm > 0,
        lambda s: compute_relative_norm(s),
        lambda s: s,
        final_state
    )
    
    sol, _, _, _ = final_state
    
    return sol


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


def newton_solve_scan(J_bc_applied, res_bc_applied, initial_sol, solver_options: SolverOptions, num_iter: int):
    """Newton solver using lax.scan for fixed number of iterations.
    
    This solver performs exactly `num_iter` Newton iterations using lax.scan,
    which can be more efficient than while_loop for fixed iteration counts
    and is better suited for vmap operations.
    
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
    num_iter : int
        Number of Newton iterations to perform
        
    Returns
    -------
    sol : jax.numpy.ndarray
        Solution vector after num_iter iterations
    res_norm : float
        Final residual norm
        
    Notes
    -----
    Unlike newton_solve, this function:
    - Does NOT check convergence
    - Always performs exactly num_iter iterations
    - Returns both solution and final residual norm
    - Is more suitable for vmap when iteration count is known
    """
    
    # Resolve x0 function based on options
    if solver_options.linear_solver_x0_fn is not None:
        x0_fn = solver_options.linear_solver_x0_fn
    else:
        x0_fn = create_x0(
            bc_rows=solver_options.bc_rows,
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
    
    def scan_body(carry, _):
        """Body function for lax.scan - performs one Newton iteration.
        
        carry: (sol, res_norm)
        _: unused (iteration index provided by scan)
        """
        sol, _ = carry
        
        # Compute residual and Jacobian
        res = res_bc_applied(sol)
        J = J_bc_applied(sol)
        
        # Compute initial guess for increment
        x0 = x0_fn(sol)
        
        # Solve linear system: J * delta_sol = -res
        delta_sol = linear_solve_fn(J, -res, x0)
        
        # Update solution
        sol = sol + delta_sol
        
        # Compute new residual norm for monitoring
        res = res_bc_applied(sol)
        res_norm = jnp.linalg.norm(res)
        
        return (sol, res_norm), res_norm
    
    # Initial state
    initial_res = res_bc_applied(initial_sol)
    initial_res_norm = jnp.linalg.norm(initial_res)
    initial_carry = (initial_sol, initial_res_norm)
    
    # Run fixed number of Newton iterations using scan
    # Use None for xs and length parameter for dynamic iteration count
    final_carry, res_norms = jax.lax.scan(scan_body, initial_carry, None, length=num_iter)
    final_sol, final_res_norm = final_carry
    
    return final_sol, final_res_norm


# Documentation for advanced usage:
"""
JAX-FEM style initial guess strategies - three ways to specify:

# Method 1: Most convenient - use x0_strategy in SolverOptions
solver_options = SolverOptions(
    x0_strategy="bc_aware",
    bc_rows=bc_rows  # boundary condition row indices
)

# Method 2: Using create_x0 function
x0_fn = create_x0(bc_rows=bc_rows, strategy="bc_aware")
solver_options = SolverOptions(linear_solver_x0_fn=x0_fn)

# Method 3: Custom function
def my_x0_fn(current_sol):
    return jnp.zeros_like(current_sol)
solver_options = SolverOptions(linear_solver_x0_fn=my_x0_fn)

The key idea from JAX-FEM is that the increment should be zero at boundary locations
since the solution is already correct there. This can improve convergence.

Available x0_strategy options:
- "zeros": Return zeros everywhere (default, good for increments)
- "bc_aware": Explicitly ensure BC locations are zero
"""