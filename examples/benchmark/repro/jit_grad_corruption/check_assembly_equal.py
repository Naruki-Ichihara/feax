import os, sys, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as onp, feax as fe
ele = sys.argv[1]
mesh = fe.mesh.box_mesh((1.0, 0.7, 0.5), mesh_size=0.1, element_type=ele)
class Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(g, E):
            eps = 0.5 * (g + g.T); return E * (0.3 / (1.3 * 0.4) * jnp.trace(eps) * jnp.eye(3) + eps / 1.3)
        return stress
    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, -t]) * (1 + 0.1 * u[0])]   # u-dependent (face Jacobian non-zero)
problem = Elast(mesh, vec=3, dim=3, ele_type=ele, location_fns=[lambda p: jnp.isclose(p[0], 1.0, atol=1e-6)])
ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
E = jnp.asarray(1.0 + onp.random.default_rng(0).random(problem.num_cells)); t = fe.TracedParams.create_uniform_surface_var(problem, 1e-2)
tp = fe.TracedParams(volume_vars=(E,), surface_vars=[(t,)])
u = jnp.asarray(1e-2 * onp.random.default_rng(1).standard_normal(problem.num_total_dofs_all_vars))
sol = problem.unflatten_fn_sol_list(u)
r_ts = jnp.concatenate([x.reshape(-1) for x in fe.get_res(problem, sol, tp, ts)]); r_no = jnp.concatenate([x.reshape(-1) for x in fe.get_res(problem, sol, tp, None)])
J_ts = fe.get_jacobian(problem, sol, tp, ts).data; J_no = fe.get_jacobian(problem, sol, tp, None).data
from feax.assembler import _get_res_J_csr
rl, Jf, _, _ = _get_res_J_csr(problem, sol, tp, ts)
rf = jnp.concatenate([x.reshape(-1) for x in rl])
mode = os.environ.get("FEAX_ASSEMBLY_SCATTER", "fanin"); out_dir = os.environ.get("OUT_DIR", ".")
arrays = dict(r_ts=onp.asarray(r_ts), r_no=onp.asarray(r_no), J_ts=onp.asarray(J_ts), J_no=onp.asarray(J_no), rf=onp.asarray(rf), Jf=onp.asarray(Jf))
onp.savez(os.path.join(out_dir, f"asm_{mode}_{ele}.npz"), **arrays)
print(ele, "saved; |r|", float(jnp.linalg.norm(r_ts)), "|J|", float(jnp.linalg.norm(J_ts)), "ts-vs-closure", float(jnp.abs(r_ts-r_no).max()), float(jnp.abs(J_ts-J_no).max()), "fused-vs-split", float(jnp.abs(rf-r_ts).max()), float(jnp.abs(Jf-J_ts).max()))
for other in ("atomic", "fanin", "scan"):
    path = os.path.join(out_dir, f"asm_{other}_{ele}.npz")
    if other != mode and os.path.exists(path):
        ref = onp.load(path)
        print(ele, f"{other} vs {mode}:", {k: float(onp.abs(ref[k] - arrays[k]).max()) for k in arrays})
