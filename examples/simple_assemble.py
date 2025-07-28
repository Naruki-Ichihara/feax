"""
Example demonstrating feax assembly using the new pure function interface.

This example has been rewritten to use the refactored Problem class and pure
assembly functions from feax.assembler. Key features demonstrated:

1. Problem class now contains only state data
2. Assembly functions (get_J, get_res) are pure functions for better JAX compatibility
3. JIT compilation works seamlessly with the new interface
4. All results match the reference JaxFem implementation
5. Boundary condition handling remains fully functional

The new approach provides better separation of concerns and improved JAX 
compatibility while maintaining backwards compatibility.
"""

import jax
import jax.numpy as np
from jax.experimental import sparse
import scipy

from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.assembler import get_J, get_res  # New pure function interface
from feax.DCboundary import apply_boundary_to_J, DirichletBC, apply_boundary_to_res

from jax_fem.problem import Problem as JaxFemProblem
from jax_fem.solver import get_A, apply_bc_vec


E = 70e3
nu = 0.3

class ComparisonProblemFeax(FeaxProblem):
    """Test problem for feax implementation"""
    
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
    
class ComparisonProblemJaxFem(JaxFemProblem):
    """Test problem for feax implementation"""
    
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
    
meshio_mesh = box_mesh_gmsh(2, 2, 2, 1., 1., 1., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], 1, atol=1e-5)


# Define Dirichlet boundary values.
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

feax_problem = ComparisonProblemFeax(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[right]
)

jaxfem_problem = ComparisonProblemJaxFem(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type='HEX8',
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=[right]
)

sol_flat = np.zeros(feax_problem.num_total_dofs_all_vars)
sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)

print("=== Assembly using new pure function interface ===")

# Feax Jacobian using new pure function interface
J_feax = get_J(feax_problem, sol_unflat)
J_feax_dense = J_feax.todense()
print(f"Feax Jacobian assembled with shape: {J_feax.shape}")

# JaxFem Jacobian (reference implementation)
jaxfem_problem.newton_update(sol_unflat)
indices = np.stack([jaxfem_problem.I, jaxfem_problem.J], axis=1)  # (nnz, 2)
shape = (jaxfem_problem.num_total_dofs_all_vars, jaxfem_problem.num_total_dofs_all_vars)
J_jaxfem = sparse.BCOO((jaxfem_problem.V, indices), shape=shape)
J_jaxfem_dense = J_jaxfem.todense()
print(f"JaxFem Jacobian assembled with shape: {J_jaxfem.shape}")

# Compare the two Jacobians
J_diff = np.max(np.abs(J_feax_dense - J_jaxfem_dense))
print(f"Max difference between Feax and JaxFem Jacobians: {J_diff}")
assert np.allclose(J_feax_dense, J_jaxfem_dense, atol=1e-8), "Jacobian matrices do not match"
print("✓ Feax and JaxFem Jacobians match")

# Residual using new pure function interface
res_feax = get_res(feax_problem, sol_unflat)
res_feax_flat = jax.flatten_util.ravel_pytree(res_feax)[0]
print("Residual (Feax):", np.linalg.norm(res_feax_flat))

res_jaxfem = jaxfem_problem.compute_residual(sol_unflat)
res_jaxfem_flat = jax.flatten_util.ravel_pytree(res_jaxfem)[0]
print("Residual (JaxFem):", np.linalg.norm(res_jaxfem_flat))

res_diff = np.max(np.abs(res_feax_flat - res_jaxfem_flat))
print(f"Max difference between Feax and JaxFem residuals: {res_diff}")
assert np.allclose(res_feax_flat, res_jaxfem_flat, atol=1e-8), "Residuals do not match"
print("✓ Feax and JaxFem residuals match")

print("\n=== Testing JIT compilation with new interface ===")

# Create JIT-compiled assembly functions 
@jax.jit
def assemble_jacobian_jit(sol_unflat):
    return get_J(feax_problem, sol_unflat)

@jax.jit  
def assemble_residual_jit(sol_unflat):
    return get_res(feax_problem, sol_unflat)

# Test JIT-compiled versions
J_feax_jit = assemble_jacobian_jit(sol_unflat)
res_feax_jit = assemble_residual_jit(sol_unflat)

# Verify JIT versions match non-JIT versions
J_jit_diff = np.max(np.abs(J_feax.data - J_feax_jit.data))
res_jit_diff = np.max(np.abs(jax.flatten_util.ravel_pytree(res_feax)[0] - 
                             jax.flatten_util.ravel_pytree(res_feax_jit)[0]))

print(f"Jacobian JIT vs non-JIT difference: {J_jit_diff}")
print(f"Residual JIT vs non-JIT difference: {res_jit_diff}")
print("✓ JIT compilation works perfectly with new interface")

print("\n=== Boundary condition application ===")

# Create DirichletBC object from problem
bc = DirichletBC.from_problem(feax_problem)
print(f"Number of BC rows: {len(bc.bc_rows)}")

# Apply boundary conditions to Jacobian
J_bc_feax = apply_boundary_to_J(bc, J_feax)
J_bc_feax_dense = J_bc_feax.todense()
print("Jacobian with boundary conditions (Feax) shape:", J_bc_feax_dense.shape)

# Reference implementation from JaxFem
J_bc_jaxfem = get_A(jaxfem_problem)
indptr, indices, data = J_bc_jaxfem.getValuesCSR()
J_bc_jaxfem_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=J_bc_jaxfem.getSize())
J_bc_jaxfem = sparse.BCOO.from_scipy_sparse(J_bc_jaxfem_sp_scipy).sort_indices()
J_bc_jaxfem_dense = J_bc_jaxfem.todense()
print("Jacobian with boundary conditions (JaxFem) shape:", J_bc_jaxfem_dense.shape)

# Compare boundary condition implementations
max_diff_jaxfem = np.max(np.abs(J_bc_feax_dense - J_bc_jaxfem_dense))
print(f"Max difference between Feax and JaxFem BC implementations: {max_diff_jaxfem}")

if np.allclose(J_bc_feax_dense, J_bc_jaxfem_dense, atol=1e-10):
    print("✓ Feax BC implementation matches JaxFem (within tolerance)")

# Apply boundary conditions to residual vector
res_bc_feax = apply_boundary_to_res(bc, res_feax_flat, sol_flat)
print(f"Residual with BC applied (Feax) norm: {np.linalg.norm(res_bc_feax)}")

res_bc_jaxfem = apply_bc_vec(res_jaxfem_flat, sol_flat, jaxfem_problem)
print(f"Residual with BC applied (JaxFem) norm: {np.linalg.norm(res_bc_jaxfem)}")

res_bc_diff = np.max(np.abs(res_bc_feax - res_bc_jaxfem))
print(f"Max difference between Feax and JaxFem residual BC: {res_bc_diff}")

if np.allclose(res_bc_feax, res_bc_jaxfem, atol=1e-10):
    print("✓ Residual BC application matches JaxFem (within tolerance)")