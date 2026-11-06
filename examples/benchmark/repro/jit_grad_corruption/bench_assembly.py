import os, sys, time, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, feax as fe
n = int(sys.argv[1])
mesh = fe.mesh.box_mesh((1.0, 1.0, 1.0), mesh_size=1.0 / n, element_type="HEX8")
class Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(g, E):
            eps = 0.5 * (g + g.T); return E * (0.3 / (1.3 * 0.4) * jnp.trace(eps) * jnp.eye(3) + eps / 1.3)
        return stress
problem = Elast(mesh, vec=3, dim=3, ele_type="HEX8")
bc = fe.DirichletBCConfig([fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
ts = fe.TracedStructure.from_problem(problem)
E = fe.TracedParams.create_cell_var(problem, 1.0); tp = fe.TracedParams(volume_vars=(E,))
u = jnp.zeros(problem.num_total_dofs_all_vars) + 1e-3
res = jax.jit(lambda u, tp, ts: fe.get_res(problem, problem.unflatten_fn_sol_list(u), tp, ts))
jac = jax.jit(lambda u, tp, ts: fe.get_jacobian(problem, problem.unflatten_fn_sol_list(u), tp, ts).data)
for name, fn in (("residual", res), ("jacobian (CSR)", jac)):
    out = fn(u, tp, ts); jax.block_until_ready(out)
    t0 = time.perf_counter(); 
    for _ in range(5): out = fn(u, tp, ts)
    jax.block_until_ready(out); dt = (time.perf_counter() - t0) / 5
    print(f"{os.environ.get('FEAX_ASSEMBLY_SCATTER', 'scan'):6s} {name:15s} {problem.num_total_dofs_all_vars:8d} DOF: {dt * 1e3:8.2f} ms/call", flush=True)
