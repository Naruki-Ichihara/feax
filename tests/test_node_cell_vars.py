"""
Simple test to verify node-based and cell-based internal variables work correctly.
"""

import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh

jax.config.update("jax_enable_x64", True)

# Simple elasticity problem
E0 = 70e3
nu = 0.3

class ElasticityProblem(Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            strain = 0.5 * (u_grad + u_grad.T)
            sigma = lam * np.trace(strain) * np.eye(self.dim) + 2.0 * mu * strain
            return sigma
        return stress

# Create small mesh
mesh = box_mesh(4, 2, 2, 1.0, 0.5, 0.5)
print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

# Boundary conditions
def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right(point):
    return np.isclose(point[0], 1.0, atol=1e-5)

bc_config = DirichletBCConfig([
    DirichletBCSpec(location=left, component='all', value=0.0),
    DirichletBCSpec(location=right, component=0, value=0.01)
])

# Create problem
problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, ele_type='HEX8', gauss_order=2)
bc = bc_config.create_bc(problem)
solver_options = SolverOptions(tol=1e-8, linear_solver="cg")
solver = create_solver(problem, bc, solver_options, iter_num=1)

print(f"Problem: {problem.num_total_dofs_all_vars} DOFs")
print(f"Number of cells: {problem.num_cells}")
print(f"Number of nodes: {problem.fes[0].num_total_nodes}")

# Create initial guess
initial_guess = zero_like_initial_guess(problem, bc)

# Test 1: Node-based variable
print("\n=== Test 1: Node-based variable ===")
E_node = InternalVars.create_node_var(problem, E0)
print(f"E_node shape: {E_node.shape}")
print(f"Expected: ({problem.fes[0].num_total_nodes},)")
assert E_node.shape == (problem.fes[0].num_total_nodes,), "Node-based variable has wrong shape!"

internal_vars_node = InternalVars(volume_vars=(E_node,))
sol_node = solver(internal_vars_node, initial_guess)
print(f"Solution computed successfully with node-based vars!")
print(f"Max displacement: {np.max(np.abs(sol_node)):.6e}")

# Test 2: Cell-based variable
print("\n=== Test 2: Cell-based variable ===")
E_cell = InternalVars.create_cell_var(problem, E0)
print(f"E_cell shape: {E_cell.shape}")
print(f"Expected: ({problem.num_cells},)")
assert E_cell.shape == (problem.num_cells,), "Cell-based variable has wrong shape!"

internal_vars_cell = InternalVars(volume_vars=(E_cell,))
sol_cell = solver(internal_vars_cell, initial_guess)
print(f"Solution computed successfully with cell-based vars!")
print(f"Max displacement: {np.max(np.abs(sol_cell)):.6e}")

# Test 3: Quad-based variable (legacy)
print("\n=== Test 3: Quad-based variable (legacy) ===")
E_quad = InternalVars.create_uniform_volume_var(problem, E0)
print(f"E_quad shape: {E_quad.shape}")
print(f"Expected: ({problem.num_cells}, {problem.fes[0].num_quads})")
assert E_quad.shape == (problem.num_cells, problem.fes[0].num_quads), "Quad-based variable has wrong shape!"

internal_vars_quad = InternalVars(volume_vars=(E_quad,))
sol_quad = solver(internal_vars_quad, initial_guess)
print(f"Solution computed successfully with quad-based vars!")
print(f"Max displacement: {np.max(np.abs(sol_quad)):.6e}")

# Test 4: Compare solutions - they should all be very similar
print("\n=== Test 4: Compare solutions ===")
diff_node_cell = np.max(np.abs(sol_node - sol_cell))
diff_node_quad = np.max(np.abs(sol_node - sol_quad))
diff_cell_quad = np.max(np.abs(sol_cell - sol_quad))

print(f"Max diff (node vs cell): {diff_node_cell:.6e}")
print(f"Max diff (node vs quad): {diff_node_quad:.6e}")
print(f"Max diff (cell vs quad): {diff_cell_quad:.6e}")

# All should be nearly identical for uniform material
tolerance = 1e-10
assert diff_node_cell < tolerance, f"Node and cell solutions differ by {diff_node_cell}"
assert diff_node_quad < tolerance, f"Node and quad solutions differ by {diff_node_quad}"
assert diff_cell_quad < tolerance, f"Cell and quad solutions differ by {diff_cell_quad}"

print("\n✅ All tests passed!")
print("\n=== Memory comparison ===")
print(f"Node-based: {E_node.nbytes} bytes ({E_node.size} values)")
print(f"Cell-based: {E_cell.nbytes} bytes ({E_cell.size} values)")
print(f"Quad-based: {E_quad.nbytes} bytes ({E_quad.size} values)")
print(f"Node-based is {E_quad.size / E_node.size:.1f}x more memory efficient than quad-based")
print(f"Cell-based is {E_quad.size / E_cell.size:.1f}x more memory efficient than quad-based")
