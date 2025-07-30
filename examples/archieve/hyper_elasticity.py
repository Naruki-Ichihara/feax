import jax
import jax.numpy as np
from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.assembler import get_J, get_res
from feax.DCboundary import DirichletBC, apply_boundary_to_J, apply_boundary_to_res, create_J_bc_updater
from feax.solver import newton_solve, SolverOptions, diagonals
from jax_fem.problem import Problem as JaxFemProblem
from jax_fem.solver import get_A, apply_bc_vec
from jax.experimental import sparse
import scipy
import time

class HyperElasticityFeax(FeaxProblem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):
        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad, internal_vars=None):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
class HyperElasticityJaxfem(JaxFemProblem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
meshio_mesh = box_mesh_gmsh(5, 5, 5, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], 1, atol=1e-5)


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                     [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                     [zero_dirichlet_val] * 3]

feax_problem = HyperElasticityFeax(mesh,
                          vec=3,
                          dim=3,
                          dirichlet_bc_info=dirichlet_bc_info)

jaxfem_problem = HyperElasticityJaxfem(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info
)

bc = DirichletBC.from_bc_info(feax_problem, dirichlet_bc_info)

initial_sol = np.zeros(feax_problem.num_total_dofs_all_vars)
initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)
initial_sol_unflatten = feax_problem.unflatten_fn_sol_list(initial_sol)

# Feax Jacobian using new pure function interface
J_feax = get_J(feax_problem, initial_sol_unflatten)
J_feax_dense = J_feax.todense()
print(f"Feax Jacobian assembled with shape: {J_feax.shape}")
# JaxFem Jacobian (reference implementation)
jaxfem_problem.newton_update(initial_sol_unflatten)
indices = np.stack([jaxfem_problem.I, jaxfem_problem.J], axis=1)  # (nnz, 2)
shape = (jaxfem_problem.num_total_dofs_all_vars, jaxfem_problem.num_total_dofs_all_vars)
J_jaxfem = sparse.BCOO((jaxfem_problem.V, indices), shape=shape)
J_jaxfem_dense = J_jaxfem.todense()
print(f"JaxFem Jacobian assembled with shape: {J_jaxfem.shape}")
assert np.allclose(J_feax_dense, J_jaxfem_dense, atol=1e-8), "Jacobian matrices do not match"
print("✓ Feax and JaxFem Jacobians match")

res = get_res(feax_problem, initial_sol_unflatten)
res_flat = jax.flatten_util.ravel_pytree(res)[0]
res_val_before_bc = np.linalg.norm(res_flat)
print(f"Residual before bc applied (Feax): {res_val_before_bc}")

res_jaxfem = jaxfem_problem.compute_residual(initial_sol_unflatten)
res_jaxfem_flat = jax.flatten_util.ravel_pytree(res_jaxfem)[0]
print("Residual before bc applied (JaxFem):", np.linalg.norm(res_jaxfem_flat))

# Method 1
start = time.time()
J_bc_feax_1 = apply_boundary_to_J(bc, J_feax)
end = time.time()
J_bc_feax_dense_1 = J_bc_feax_1.todense()
diff = end - start
print(f"{diff} sec. to apply bc")

# Method 2
J_fn = jax.jit(create_J_bc_updater(feax_problem, bc))
start = time.time()
J_bc_feax_2 = J_fn(initial_sol)
end = time.time()
diff = end - start
print(f"{diff} sec. to apply bc")
J_bc_feax_dense_2 = J_bc_feax_2.todense()

assert np.allclose(J_bc_feax_dense_1, J_bc_feax_dense_2, atol=1e-8), "Jacobian matrices do not match"

# Recall J_fn
new_sol = np.ones(feax_problem.num_total_dofs_all_vars)*0.01
new_sol = new_sol.at[bc.bc_rows].set(bc.bc_vals)
start = time.time()
J_bc_feax_3 = J_fn(new_sol)
end = time.time()
J_bc_feax_dense_3 = J_bc_feax_3.todense()
diff = end - start
print(f"{diff} sec. to apply bc")

# Reference implementation from JaxFem
J_bc_jaxfem = get_A(jaxfem_problem)
indptr, indices, data = J_bc_jaxfem.getValuesCSR()
J_bc_jaxfem_sp_scipy = scipy.sparse.csr_array((data, indices, indptr), shape=J_bc_jaxfem.getSize())
J_bc_jaxfem = sparse.BCOO.from_scipy_sparse(J_bc_jaxfem_sp_scipy).sort_indices()
J_bc_jaxfem_dense = J_bc_jaxfem.todense()

assert np.allclose(J_bc_feax_dense_1, J_bc_jaxfem_dense, atol=1e-10)
assert np.allclose(J_bc_feax_dense_2, J_bc_jaxfem_dense, atol=1e-10)

new_sol_unflatten = feax_problem.unflatten_fn_sol_list(new_sol)
jaxfem_problem.newton_update(new_sol_unflatten)
J_bc_jaxfem_2 = get_A(jaxfem_problem)
indptr, indices, data = J_bc_jaxfem_2.getValuesCSR()
J_bc_jaxfem_sp_scipy_2 = scipy.sparse.csr_array((data, indices, indptr), shape=J_bc_jaxfem_2.getSize())
J_bc_jaxfem_2 = sparse.BCOO.from_scipy_sparse(J_bc_jaxfem_sp_scipy_2).sort_indices()
J_bc_jaxfem_dense_2 = J_bc_jaxfem_2.todense()

assert np.allclose(J_bc_feax_dense_3, J_bc_jaxfem_dense_2, atol=1e-10)

feax_J_bc = J_bc_feax_2