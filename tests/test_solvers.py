"""
Test suite for feax.solvers module.

Tests JAX-native linear solvers functionality.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

from feax.solvers import (
    SolverOptions, solve, solve_cg, solve_bicgstab, solve_gmres,
    jacobi_preconditioner
)


class TestSolvers:
    """Test solvers module functionality."""
    
    def create_test_system(self, n=10):
        """Create a simple test linear system."""
        # Create a symmetric positive definite matrix for CG testing
        A_dense = np.random.randn(n, n)
        A_dense = A_dense @ A_dense.T + n * np.eye(n)  # Make SPD
        
        # Convert to sparse BCOO format
        indices = np.array([[i, j] for i in range(n) for j in range(n)])
        data = A_dense.flatten()
        A = sparse.BCOO((data, indices), shape=(n, n))
        
        # Create RHS vector
        x_true = np.random.randn(n)
        b = A_dense @ x_true
        
        return A, jnp.array(b), jnp.array(x_true)
    
    def create_nonsymmetric_system(self, n=10):
        """Create a nonsymmetric test linear system."""
        # Create a nonsymmetric matrix
        A_dense = np.random.randn(n, n) + n * np.eye(n)
        
        # Convert to sparse BCOO format
        indices = np.array([[i, j] for i in range(n) for j in range(n)])
        data = A_dense.flatten()
        A = sparse.BCOO((data, indices), shape=(n, n))
        
        # Create RHS vector
        x_true = np.random.randn(n)
        b = A_dense @ x_true
        
        return A, jnp.array(b), jnp.array(x_true)
    
    def test_solver_options(self):
        """Test SolverOptions dataclass."""
        # Default options
        opts = SolverOptions()
        assert opts.max_iter == 1000
        assert opts.tol == 1e-6
        assert opts.atol == 0.0
        assert opts.M is None
        assert opts.x0 is None
        assert opts.restart == 20
        
        # Custom options
        opts = SolverOptions(max_iter=500, tol=1e-8, atol=1e-10)
        assert opts.max_iter == 500
        assert opts.tol == 1e-8
        assert opts.atol == 1e-10
    
    def test_solve_cg(self):
        """Test Conjugate Gradient solver."""
        A, b, x_true = self.create_test_system(20)
        options = SolverOptions(max_iter=100, tol=1e-8)
        
        x, info = solve_cg(A, b, options)
        
        assert info['converged']
        assert jnp.allclose(x, x_true, rtol=1e-6)
    
    def test_solve_bicgstab(self):
        """Test BiCGSTAB solver."""
        A, b, x_true = self.create_nonsymmetric_system(20)
        options = SolverOptions(max_iter=100, tol=1e-8)
        
        x, info = solve_bicgstab(A, b, options)
        
        assert info['converged']
        assert jnp.allclose(x, x_true, rtol=1e-6)
    
    def test_solve_gmres(self):
        """Test GMRES solver."""
        A, b, x_true = self.create_nonsymmetric_system(20)
        options = SolverOptions(max_iter=100, tol=1e-8, restart=10)
        
        x, info = solve_gmres(A, b, options)
        
        assert info['converged']
        assert jnp.allclose(x, x_true, rtol=1e-6)
    
    def test_solve_interface(self):
        """Test main solve interface."""
        A, b, x_true = self.create_test_system(20)
        
        # Test CG
        x, info = solve(A, b, method='cg')
        assert info['converged']
        assert info['method'] == 'cg'
        assert jnp.allclose(x, x_true, rtol=1e-5)
        
        # Test BiCGSTAB
        x, info = solve(A, b, method='bicgstab')
        assert info['converged']
        assert info['method'] == 'bicgstab'
        assert jnp.allclose(x, x_true, rtol=1e-5)
        
        # Test GMRES
        x, info = solve(A, b, method='gmres')
        assert info['converged']
        assert info['method'] == 'gmres'
        assert jnp.allclose(x, x_true, rtol=1e-5)
    
    def test_solve_with_custom_options(self):
        """Test solve with custom options."""
        A, b, x_true = self.create_test_system(50)
        options = SolverOptions(max_iter=200, tol=1e-10)
        
        x, info = solve(A, b, method='bicgstab', options=options)
        
        assert info['converged']
        assert jnp.allclose(x, x_true, rtol=1e-8)
    
    def test_solve_with_initial_guess(self):
        """Test solve with initial guess."""
        A, b, x_true = self.create_test_system(20)
        
        # Use a close initial guess
        x0 = x_true + 0.1 * jnp.ones_like(x_true)
        options = SolverOptions(x0=x0, tol=1e-8)
        
        x, info = solve(A, b, method='cg', options=options)
        
        assert info['converged']
        assert jnp.allclose(x, x_true, rtol=1e-6)
    
    def test_jacobi_preconditioner(self):
        """Test Jacobi preconditioner."""
        A, b, x_true = self.create_test_system(20)
        
        # Create preconditioner
        M = jacobi_preconditioner(A)
        
        # Test preconditioner
        options = SolverOptions(M=M, tol=1e-8)
        x, info = solve(A, b, method='cg', options=options)
        
        assert info['converged']
        assert jnp.allclose(x, x_true, rtol=1e-6)
    
    def test_invalid_method(self):
        """Test invalid solver method."""
        A, b, _ = self.create_test_system(10)
        
        with pytest.raises(ValueError, match="Unknown solver method"):
            solve(A, b, method='invalid_method')
    
    def test_solver_jit_compilation(self):
        """Test that solvers are JIT-compiled."""
        A, b, x_true = self.create_test_system(20)
        options = SolverOptions(tol=1e-8)
        
        # First call - compilation
        x1, info1 = solve_bicgstab(A, b, options)
        
        # Second call - should be faster (already compiled)
        x2, info2 = solve_bicgstab(A, b, options)
        
        # Results should be identical
        assert jnp.allclose(x1, x2)
        assert info1['converged'] == info2['converged']
    
    def test_large_system(self):
        """Test solver on larger system."""
        n = 100
        A, b, x_true = self.create_test_system(n)
        options = SolverOptions(max_iter=200, tol=1e-6)
        
        x, info = solve(A, b, method='bicgstab', options=options)
        
        assert info['converged']
        # Relax tolerance for larger system
        assert jnp.allclose(x, x_true, rtol=1e-4)
    
    def test_solver_convergence_failure(self):
        """Test solver behavior when convergence fails."""
        A, b, _ = self.create_test_system(50)
        
        # Set very strict tolerance and low iteration limit
        options = SolverOptions(max_iter=2, tol=1e-12)
        
        x, info = solve(A, b, method='cg', options=options)
        
        # Should not converge with such strict settings
        assert not info['converged']
        assert info['iterations'] > 0