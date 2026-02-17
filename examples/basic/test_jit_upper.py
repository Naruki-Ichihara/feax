"""Test: JIT compatibility with UPPER triangular matrix"""
import feax as fe
from feax.problem import MatrixView
import jax
import jax.numpy as jnp
import time

# Setup
E, nu, traction, tol = 70e3, 0.3, 1e-3, 1e-5
L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)
left = lambda p: jnp.isclose(p[0], 0., tol)
right = lambda p: jnp.isclose(p[0], L, tol)

class LinearElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu, lmbda = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
            eps = 0.5*(u_grad + u_grad.T)
            return lmbda*jnp.trace(eps)*jnp.eye(self.dim) + 2*mu*eps
        return stress
    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0., 0., t])]

# UPPER triangular matrix
problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right], matrix_view=MatrixView.UPPER)
bc = fe.DirichletBCConfig([fe.DirichletBCSpec(left, "all", 0.)]).create_bc(problem)
solver_opts = fe.SolverOptions.from_problem(problem)
solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
initial = fe.zero_like_initial_guess(problem, bc)
internal_vars = fe.InternalVars((), [(fe.InternalVars.create_uniform_surface_var(problem, traction),)])

print(f"Matrix view: {problem.matrix_view.name}")
print(f"Nodes: {mesh.points.shape[0]}, DOFs: {problem.num_total_dofs_all_vars}")
print(f"Solver: {solver_opts.cudss_options.matrix_type.name} + {solver_opts.cudss_options.matrix_view.name}")

# JIT compile solver
jit_solver = jax.jit(solver)

# Test
start = time.time()
sol = jit_solver(internal_vars, initial)
elapsed = time.time() - start

J_info = fe.get_jacobian_info(problem, problem.unflatten_fn_sol_list(sol), internal_vars)
print(f"Jacobian NNZ: {J_info['nnz']:,}, Time: {elapsed:.4f}s, Max |u|: {jnp.max(jnp.abs(sol)):.6e}")

# re-run
start = time.time()
sol = jit_solver(internal_vars, initial)
elapsed = time.time() - start
J_info = fe.get_jacobian_info(problem, problem.unflatten_fn_sol_list(sol), internal_vars)
print(f"Jacobian NNZ: {J_info['nnz']:,}, Time: {elapsed:.4f}s, Max |u|: {jnp.max(jnp.abs(sol)):.6e}")