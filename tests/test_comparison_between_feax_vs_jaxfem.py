"""
Test suite for boundary condition handling and assembly accuracy.

This test validates that the boundary condition application methods
produce correct results and maintain numerical accuracy.
"""

import pytest
import jax
import jax.numpy as np
from jax.experimental import sparse
import scipy

from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.assembler import get_J, get_res
from feax.DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res, update_J
from feax.solver import newton_solve, SolverOptions

from jax_fem.problem import Problem as JaxFemProblem
from jax_fem.solver import solver, get_A, apply_bc_vec


# Material properties
E = 70e3
nu = 0.3


class TestElasticityProblemFeax(FeaxProblem):
    """Test elasticity problem for feax implementation."""
    
    def get_tensor_map(self):
        def stress(u_grad, internal_vars=None):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, internal_vars=None):
            return np.array([0., 0., 1.])
        return [surface_map]


class TestElasticityProblemJaxFem(JaxFemProblem):
    """Test elasticity problem for jaxfem implementation."""
    
    def get_tensor_map(self):
        def stress(u_grad, internal_vars=None):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress
    
    def get_surface_maps(self):
        def surface_map(u, x, internal_vars=None):
            return np.array([0., 0., 1.])
        return [surface_map]


@pytest.fixture
def test_problems():
    """Create test problems for comparison."""
    # Create small mesh for testing
    meshio_mesh = box_mesh_gmsh(2, 2, 2, 1., 1., 1., 
                               data_dir='/tmp', ele_type='HEX8')
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    # Define boundary conditions
    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def one_dirichlet_val(point):
        return 2.

    def dirichlet_val_x2(point):
        return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 6.) -
                (point[2] - 0.5) * np.sin(np.pi / 6.) - point[1]) / 2.

    def dirichlet_val_x3(point):
        return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 6.) +
                (point[2] - 0.5) * np.cos(np.pi / 6.) - point[2]) / 2.

    dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                         [one_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                         [zero_dirichlet_val] * 3]

    feax_problem = TestElasticityProblemFeax(
        mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
    )

    jaxfem_problem = TestElasticityProblemJaxFem(
        mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2, 
        dirichlet_bc_info=dirichlet_bc_info, location_fns=[right]
    )

    return feax_problem, jaxfem_problem


class TestBoundaryConditions:
    """Test suite for boundary condition handling and assembly accuracy."""

    def test_jacobian_assembly_accuracy(self, test_problems):
        """Test that Feax Jacobian assembly produces correct results."""
        feax_problem, jaxfem_problem = test_problems
        
        sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)

        # Feax Jacobian
        J_feax = get_J(feax_problem, sol_unflat)
        J_feax_dense = J_feax.todense()

        # JaxFem Jacobian (reference)
        jaxfem_problem.newton_update(sol_unflat)
        indices = np.stack([jaxfem_problem.I, jaxfem_problem.J], axis=1)
        shape = (jaxfem_problem.num_total_dofs_all_vars, jaxfem_problem.num_total_dofs_all_vars)
        J_jaxfem = sparse.BCOO((jaxfem_problem.V, indices), shape=shape)
        J_jaxfem_dense = J_jaxfem.todense()

        # Compare
        assert np.allclose(J_feax_dense, J_jaxfem_dense, atol=1e-8), \
            "Feax and JaxFem Jacobians do not match"

    def test_residual_assembly_accuracy(self, test_problems):
        """Test that Feax residual assembly produces correct results."""
        feax_problem, jaxfem_problem = test_problems
        
        sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)

        # Feax residual
        res_feax = get_res(feax_problem, sol_unflat)
        res_feax_flat = jax.flatten_util.ravel_pytree(res_feax)[0]

        # JaxFem residual
        res_jaxfem = jaxfem_problem.compute_residual(sol_unflat)
        res_jaxfem_flat = jax.flatten_util.ravel_pytree(res_jaxfem)[0]

        # Compare
        assert np.allclose(res_feax_flat, res_jaxfem_flat, atol=1e-8), \
            "Feax and JaxFem residuals do not match"

    def test_update_J_boundary_condition_handling(self, test_problems):
        """Test that update_J correctly applies boundary conditions."""
        feax_problem, _ = test_problems
        
        sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
        
        # Get original Jacobian
        J_original = get_J(feax_problem, sol_unflat)
        bc = DirichletBC.from_problem(feax_problem)
        
        # Apply optimized update_J
        J_optimized = update_J(bc, J_original)
        
        # Should have same shape
        assert J_optimized.shape == J_original.shape, \
            "update_J changed matrix shape"
        
        # Should preserve sparsity structure
        assert J_optimized.indices.shape == J_original.indices.shape, \
            "update_J changed sparsity structure"
        
        # Check that BC rows are properly handled
        J_opt_dense = J_optimized.todense()
        
        # Verify BC rows have zeros in off-diagonal positions
        for bc_row in bc.bc_rows:
            row_vals = J_opt_dense[bc_row, :]
            non_diagonal_vals = np.concatenate([row_vals[:bc_row], row_vals[bc_row+1:]])
            assert np.allclose(non_diagonal_vals, 0.0, atol=1e-10), \
                f"BC row {bc_row} has non-zero off-diagonal entries"
            
            # Diagonal should be non-zero (may be > 1 due to structure preservation)
            assert not np.isclose(J_opt_dense[bc_row, bc_row], 0.0, atol=1e-10), \
                f"BC diagonal at row {bc_row} should not be zero"

    def test_update_J_sparsity_preservation(self, test_problems):
        """Test that update_J preserves sparse matrix structure for optimization."""
        feax_problem, _ = test_problems
        
        sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
        
        J_original = get_J(feax_problem, sol_unflat)
        bc = DirichletBC.from_problem(feax_problem)
        
        J_updated = update_J(bc, J_original)
        
        # Structure should be preserved (same shape)
        assert J_updated.shape == J_original.shape, \
            "update_J changed matrix shape"
        
        # Indices structure should be preserved
        assert J_updated.indices.shape == J_original.indices.shape, \
            "update_J changed indices structure"

    def test_update_J_jit_compatibility(self, test_problems):
        """Test that update_J is compatible with JAX JIT compilation."""
        feax_problem, _ = test_problems
        
        sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
        
        J_original = get_J(feax_problem, sol_unflat)
        bc = DirichletBC.from_problem(feax_problem)
        
        # Create JIT-compiled version
        @jax.jit
        def update_J_jit(bc, J):
            return update_J(bc, J)
        
        # Test JIT compilation
        J_jit = update_J_jit(bc, J_original)
        J_regular = update_J(bc, J_original)
        
        # Results should be identical
        assert np.allclose(J_jit.data, J_regular.data, atol=1e-15), \
            "JIT-compiled update_J produces different results"

    def test_update_J_solution_accuracy(self, test_problems):
        """Test that complete solution using update_J produces correct results."""
        feax_problem, jaxfem_problem = test_problems
        
        # Solve with Feax using update_J
        bc = DirichletBC.from_problem(feax_problem)
        initial_sol = np.zeros(feax_problem.num_total_dofs_all_vars)
        initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)

        @jax.jit
        def solve_feax(initial_sol):
            def J_func(sol_flat):
                sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
                J = get_J(feax_problem, sol_unflat)
                return update_J(bc, J)  # Use optimized update_J
            
            def res_func(sol_flat):
                sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
                res = get_res(feax_problem, sol_unflat)
                res_flat = jax.flatten_util.ravel_pytree(res)[0]
                return apply_boundary_to_res(bc, res_flat, sol_flat)
            
            return newton_solve(J_func, res_func, initial_sol, 
                               SolverOptions(tol=1e-8, linear_solver="cg"))

        # Solve with JaxFem (reference)
        def solve_jaxfem():
            solution_list = solver(jaxfem_problem)
            return jax.flatten_util.ravel_pytree(solution_list)[0]

        solution_feax = solve_feax(initial_sol)
        solution_jaxfem = solve_jaxfem()

        # Compare solutions
        max_diff = np.max(np.abs(solution_feax - solution_jaxfem))
        assert np.allclose(solution_feax, solution_jaxfem, atol=1e-8), \
            f"Solutions do not match, max difference: {max_diff}"

    def test_update_J_matrix_properties(self, test_problems):
        """Test that update_J maintains essential matrix properties."""
        feax_problem, _ = test_problems
        
        sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
        
        # Get Jacobians and apply BC
        J_feax = get_J(feax_problem, sol_unflat)
        bc = DirichletBC.from_problem(feax_problem)
        J_bc_feax = update_J(bc, J_feax)
        
        # Matrix should still be square and proper size
        assert J_bc_feax.shape[0] == J_bc_feax.shape[1], "Matrix should be square"
        assert J_bc_feax.shape[0] == feax_problem.num_total_dofs_all_vars, \
            "Matrix size should match DOFs"
        
        # Should preserve sparsity structure
        assert J_bc_feax.indices.shape == J_feax.indices.shape, \
            "Sparsity structure changed"
        
        # Verify that BC rows are properly modified
        J_bc_dense = J_bc_feax.todense()
        for bc_row in bc.bc_rows:
            # Off-diagonals should be 0.0
            row_vals = J_bc_dense[bc_row, :]
            for col in range(J_bc_dense.shape[1]):
                if col != bc_row:
                    assert np.isclose(row_vals[col], 0.0, atol=1e-10), \
                        f"BC row {bc_row}, col {col} should be 0.0"