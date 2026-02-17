"""Investigate: Why does JAX cache conflict with cuDSS?"""
import feax as fe
from feax.problem import MatrixView
import jax
import jax.numpy as jnp
import gc

E, nu, traction, tol = 70e3, 0.3, 1e-3, 1e-5
L, W, H = 100, 10, 10
mesh = fe.mesh.box_mesh((L, W, H), mesh_size=2)
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

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right], matrix_view=MatrixView.UPPER)
bc = fe.DirichletBCConfig([fe.DirichletBCSpec(left, "all", 0.)]).create_bc(problem)
solver_opts = fe.SolverOptions.from_problem(problem)
solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1)
initial = fe.zero_like_initial_guess(problem, bc)
internal_vars = fe.InternalVars((), [(fe.InternalVars.create_uniform_surface_var(problem, traction),)])

print("=== Test 1: Call get_J ONLY from outside (never use JIT solver) ===")
# Never use JIT, just call solver normally
sol1 = solver(internal_vars, initial)
print(f"Solve 1: Max |u|: {jnp.max(jnp.abs(sol1)):.6e}")

# Call get_J from outside
J1 = fe.get_J(problem, problem.unflatten_fn_sol_list(sol1), internal_vars)
print(f"get_J called, NNZ: {J1.nse:,}")

# Call solver again (no JIT)
sol2 = solver(internal_vars, initial)
print(f"Solve 2: Max |u|: {jnp.max(jnp.abs(sol2)):.6e}")
print(f"Diff: {jnp.linalg.norm(sol1 - sol2):.6e}")
print("✓ Works fine without JIT\n")

# Clear everything
jax.clear_caches()
gc.collect()

print("=== Test 2: Use JIT solver but NEVER call get_J from outside ===")
jit_solver = jax.jit(solver)

sol3 = jit_solver(internal_vars, initial)
print(f"JIT Solve 1: Max |u|: {jnp.max(jnp.abs(sol3)):.6e}")

# Do NOT call get_J from outside
sol4 = jit_solver(internal_vars, initial)
print(f"JIT Solve 2: Max |u|: {jnp.max(jnp.abs(sol4)):.6e}")
print(f"Diff: {jnp.linalg.norm(sol3 - sol4):.6e}")
print("✓ JIT works fine when get_J is NOT called from outside\n")

# Clear everything
jax.clear_caches()
gc.collect()

print("=== Test 3: Use JIT solver AND call get_J from outside ===")
jit_solver2 = jax.jit(solver)

sol5 = jit_solver2(internal_vars, initial)
print(f"JIT Solve 1: Max |u|: {jnp.max(jnp.abs(sol5)):.6e}")

# Call get_J from OUTSIDE after JIT solver used it INSIDE
print("Calling get_J from outside...")
J2 = fe.get_J(problem, problem.unflatten_fn_sol_list(sol5), internal_vars)
print(f"get_J called, NNZ: {J2.nse:,}")

try:
    sol6 = jit_solver2(internal_vars, initial)
    print(f"JIT Solve 2: Max |u|: {jnp.max(jnp.abs(sol6)):.6e}")
    print(f"Diff: {jnp.linalg.norm(sol5 - sol6):.6e}")
    if jnp.linalg.norm(sol5 - sol6) > 1e-6:
        print("✗ Results differ - BUG!")
    else:
        print("✓ Works (bug is fixed with automatic cache clear)")
except Exception as e:
    print(f"✗ CRASH: {e}")

print("\n=== Conclusion ===")
print("Bug occurs when:")
print("1. JIT-compiled solver calls get_J internally (inside JIT boundary)")
print("2. User calls get_J from outside (outside JIT boundary)")
print("3. JIT-compiled solver is called again")
print("\nRoot cause: JAX cache from get_J conflicts with cuDSS GPU memory")
