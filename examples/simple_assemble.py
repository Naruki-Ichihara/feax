import jax
import jax.numpy as np
from jax.experimental import sparse
import scipy

from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.DCboundary import apply_boundary_to_J

from jax_fem.problem import Problem as JaxFemProblem
from jax_fem.solver import get_A


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
    return 1.

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

# Feax Jacobian
J_feax = feax_problem.get_J(sol_unflat)
J_feax_dense = J_feax.todense()

# JaxFem Jacobian
jaxfem_problem.newton_update(sol_unflat)
indices = np.stack([jaxfem_problem.I, jaxfem_problem.J], axis=1)  # (nnz, 2)
shape = (jaxfem_problem.num_total_dofs_all_vars, jaxfem_problem.num_total_dofs_all_vars)
J_jaxfem = sparse.BCOO((jaxfem_problem.V, indices), shape=shape)
J_jaxfem_dense = J_jaxfem.todense()

# Compare the two Jacobians
assert np.allclose(J_feax_dense, J_jaxfem_dense, atol=1e-8), "Jacobian matrices do not match"

# Residual
res_feax = feax_problem.get_res(sol_unflat)
res_feax_flat = jax.flatten_util.ravel_pytree(res_feax)[0]
print("Residual (Feax):", np.linalg.norm(res_feax_flat))

res_jaxfem = jaxfem_problem.compute_residual(sol_unflat)
res_jaxfem_flat = jax.flatten_util.ravel_pytree(res_jaxfem)[0]
print("Residual (JaxFem):", np.linalg.norm(res_jaxfem_flat))
assert np.allclose(res_feax_flat, res_jaxfem_flat, atol=1e-8), "Residuals do not match"

# J applied boundary conditions
J_bc_feax = apply_boundary_to_J(feax_problem, J_feax)
J_bc_feax_dense = J_bc_feax.todense()
print("Jacobian with boundary conditions (Feax) shape:", J_bc_feax_dense.shape)
J_bc_jaxfem = get_A(jaxfem_problem)
indptr, indices, data = J_bc_jaxfem.getValuesCSR()
J_bc_jaxfem_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=J_bc_jaxfem.getSize())
J_bc_jaxfem = sparse.BCOO.from_scipy_sparse(J_bc_jaxfem_sp_scipy).sort_indices()
J_bc_jaxfem_dense = J_bc_jaxfem.todense()
print("Jacobian with boundary conditions (JaxFem) shape:", J_bc_jaxfem_dense.shape)
assert np.allclose(J_bc_feax_dense, J_bc_jaxfem_dense, atol=1e-8), "Jacobian with boundary conditions does not match"