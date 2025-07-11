"""
Test suite for Newton solver in feax.solvers module.

Tests JAX-compatible Newton solver functionality.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from feax.solvers import newton_solve, NewtonOptions, SolverOptions


class TestNewtonSolver:
    """Test Newton solver functionality."""
    
    def test_newton_options(self):
        """Test NewtonOptions dataclass."""
        # Default options
        opts = NewtonOptions()
        assert opts.max_iter == 10
        assert opts.tol == 1e-6
        assert opts.linear_solver == 'bicgstab'
        assert opts.linear_solver_options is not None
        assert opts.linear_solver_options.max_iter == 1000
        
        # Custom options
        lin_opts = SolverOptions(max_iter=500, tol=1e-10)
        opts = NewtonOptions(max_iter=20, tol=1e-8, linear_solver='cg', 
                           linear_solver_options=lin_opts)
        assert opts.max_iter == 20
        assert opts.tol == 1e-8
        assert opts.linear_solver == 'cg'
        assert opts.linear_solver_options.max_iter == 500
    
    def test_scalar_nonlinear_equation(self):
        """Test Newton solver on scalar equation: x^2 - 2 = 0."""
        def residual_fn(x):
            r = x**2 - 2
            J = 2*x
            # Convert to proper matrix format for solver
            J_matrix = sparse.BCOO((jnp.array([J[0]]), jnp.array([[0, 0]])), shape=(1, 1))
            return jnp.array([r[0]]), J_matrix
        
        options = NewtonOptions(max_iter=10, tol=1e-10)
        x0 = jnp.array([1.0])
        x, info = newton_solve(residual_fn, x0, options)
        
        assert info['converged']
        assert info['iterations'] < 10
        assert jnp.allclose(x[0]**2, 2.0, rtol=1e-10)
        assert jnp.isclose(x[0], jnp.sqrt(2), rtol=1e-10)
    
    def test_vector_nonlinear_system(self):
        """Test Newton solver on 2D nonlinear system."""
        # System: x^2 + y^2 - 4 = 0
        #         x - y = 0
        # Solution: x = y = sqrt(2)
        
        def residual_fn(u):
            x, y = u[0], u[1]
            r1 = x**2 + y**2 - 4
            r2 = x - y
            r = jnp.array([r1, r2])
            
            # Jacobian
            J_dense = jnp.array([[2*x, 2*y],
                                [1.0, -1.0]])
            # Convert to sparse
            indices = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            data = jnp.array([J_dense[0, 0], J_dense[0, 1], 
                             J_dense[1, 0], J_dense[1, 1]])
            J = sparse.BCOO((data, indices), shape=(2, 2))
            
            return r, J
        
        options = NewtonOptions(max_iter=15, tol=1e-8)
        u0 = jnp.array([1.0, 1.5])
        u, info = newton_solve(residual_fn, u0, options)
        
        assert info['converged']
        assert info['iterations'] < 10
        assert jnp.allclose(u[0], jnp.sqrt(2), rtol=1e-7)
        assert jnp.allclose(u[1], jnp.sqrt(2), rtol=1e-7)
    
    def test_jit_compilation(self):
        """Test that Newton solver can be JIT compiled."""
        def residual_fn(x):
            r = x**2 - 2
            J = sparse.BCOO((jnp.array([2*x[0]]), jnp.array([[0, 0]])), shape=(1, 1))
            return r, J
        
        options = NewtonOptions(max_iter=5, tol=1e-8)
        
        # JIT compile the solver
        newton_solve_jit = jax.jit(newton_solve, static_argnames=['options'])
        
        x0 = jnp.array([1.0])
        x1, info1 = newton_solve_jit(residual_fn, x0, options)
        x2, info2 = newton_solve_jit(residual_fn, x0, options)
        
        # Results should be identical
        assert jnp.allclose(x1, x2)
        assert info1['converged'] == info2['converged']
        assert info1['iterations'] == info2['iterations']
    
    def test_vmap_compatibility(self):
        """Test that Newton solver can be vmapped."""
        # Solve x^2 - c = 0 for different values of c
        def make_residual_fn(c):
            def residual_fn(x):
                r = x**2 - c
                J = sparse.BCOO((jnp.array([2*x[0]]), jnp.array([[0, 0]])), shape=(1, 1))
                return jnp.array([r[0]]), J
            return residual_fn
        
        # Multiple target values
        c_values = jnp.array([1.0, 2.0, 3.0, 4.0])
        x0_values = jnp.ones((4, 1))
        
        options = NewtonOptions(max_iter=10, tol=1e-8)
        
        # Vmap over different problems
        def solve_single(c, x0):
            return newton_solve(make_residual_fn(c), x0, options)
        
        solutions, infos = jax.vmap(solve_single)(c_values, x0_values)
        
        # Check all converged
        assert jnp.all(infos['converged'])
        
        # Check solutions
        for i, c in enumerate(c_values):
            assert jnp.isclose(solutions[i, 0]**2, c, rtol=1e-7)
    
    def test_convergence_history(self):
        """Test that convergence history is properly recorded."""
        def residual_fn(x):
            r = x**3 - 8  # x^3 = 8, solution x = 2
            J = sparse.BCOO((jnp.array([3*x[0]**2]), jnp.array([[0, 0]])), shape=(1, 1))
            return jnp.array([r[0]]), J
        
        options = NewtonOptions(max_iter=20, tol=1e-10)
        x0 = jnp.array([1.0])
        x, info = newton_solve(residual_fn, x0, options)
        
        assert info['converged']
        
        # Check residual norms decrease
        norms = info['residual_norms']
        n_iter = info['iterations']
        
        # Check that the final residual is much smaller than the initial
        initial_norm = norms[0]
        final_norm = info['final_residual_norm']
        assert final_norm < initial_norm * 1e-6
        
        # Final residual should be small
        assert final_norm < options.tol
    
    def test_no_convergence(self):
        """Test behavior when Newton solver doesn't converge."""
        # Difficult problem that won't converge easily
        def residual_fn(x):
            r = jnp.exp(x[0]) - 1e10  # Very large exponential
            J = sparse.BCOO((jnp.array([jnp.exp(x[0])]), jnp.array([[0, 0]])), shape=(1, 1))
            return jnp.array([r]), J
        
        options = NewtonOptions(max_iter=5, tol=1e-10)
        x0 = jnp.array([0.0])
        x, info = newton_solve(residual_fn, x0, options)
        
        assert not info['converged']
        assert info['iterations'] == 5
    
    def test_multiple_initial_guesses(self):
        """Test Newton solver with different initial guesses."""
        def residual_fn(x):
            # x^2 - 4 = 0, solutions at x = ±2
            r = x**2 - 4
            J = sparse.BCOO((jnp.array([2*x[0]]), jnp.array([[0, 0]])), shape=(1, 1))
            return jnp.array([r[0]]), J
        
        options = NewtonOptions(max_iter=10, tol=1e-8)
        
        # Test positive initial guess
        x0_pos = jnp.array([1.0])
        x_pos, info_pos = newton_solve(residual_fn, x0_pos, options)
        assert info_pos['converged']
        assert jnp.isclose(x_pos[0], 2.0, rtol=1e-7)
        
        # Test negative initial guess
        x0_neg = jnp.array([-1.0])
        x_neg, info_neg = newton_solve(residual_fn, x0_neg, options)
        assert info_neg['converged']
        assert jnp.isclose(x_neg[0], -2.0, rtol=1e-7)
    
    def test_linear_solver_options(self):
        """Test that linear solver options are properly used."""
        def residual_fn(x):
            r = x**2 - 2
            J = sparse.BCOO((jnp.array([2*x[0]]), jnp.array([[0, 0]])), shape=(1, 1))
            return jnp.array([r[0]]), J
        
        # Custom linear solver options
        lin_opts = SolverOptions(max_iter=5, tol=1e-12)
        options = NewtonOptions(
            max_iter=10, 
            tol=1e-8,
            linear_solver='cg',
            linear_solver_options=lin_opts
        )
        
        x0 = jnp.array([1.0])
        x, info = newton_solve(residual_fn, x0, options)
        
        assert info['converged']
        assert jnp.all(info['linear_solver_converged'][:info['iterations']])