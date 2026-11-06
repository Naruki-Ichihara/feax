"""Variants of the reduced-operator assembly (scatter vs gather/mask) in the failing jitted gradient."""
import sys, os, jax; jax.config.update("jax_enable_x64", True)
if os.environ.get("INV_SIMPLIFIED_CONSTS"): jax.config.update("jax_use_simplified_jaxpr_constants", True); print("simplified constants ON", flush=True)
import numpy as onp, jax.numpy as jnp, feax as fe, feax.flat as flat, feax.asd as asd, asdex
from feax.csr import CSRMatrix
from feax.vmap import LinearSystemBatch, _lhs_engine
from feax.flat.solver import _reduced_system, _translation_pin_dofs, macro_displacement, average_stress, UNIT_STRAINS_3D
from feax.assembler import create_matfree_res_J_parametric
from feax.asd import connectivity_pattern, reduced_operator_pattern
VARIANT = sys.argv[1]            # slots: scatter|gather ; pins: scatter|mask   e.g. "scatter-scatter"
SLOTS, PINS = VARIANT.split("-")
if SLOTS == "gather":
    def producer_gather(coloring, compressed_fn):
        pat = asd._to_scipy_pattern(coloring.sparsity)
        slots = onp.asarray(asd._csr_slots(pat, coloring.sparsity.rows, coloring.sparsity.cols)); nnz = int(pat.nnz)
        inv = onp.full(nnz, -1, dtype=onp.int64); inv[slots] = onp.arange(len(slots))
        hit = jnp.asarray(inv >= 0); invc = jnp.asarray(onp.maximum(inv, 0))
        indptr = jnp.asarray(pat.indptr.astype(onp.int32)); indices = jnp.asarray(pat.indices.astype(onp.int32))
        def fn(x):
            data_pat = asdex.decompress_data(compressed_fn(x), coloring)
            return CSRMatrix(jnp.where(hit, data_pat[invc], 0.0), indptr, indices, tuple(int(s) for s in pat.shape))
        return fn, pat
    asd._csr_producer = producer_gather
E0, NU = 100.0, 0.3
class Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E/(2*(1+nu)); lam = E*nu/((1+nu)*(1-2*nu)); eps = 0.5*(u_grad+u_grad.T)
            return lam*jnp.trace(eps)*jnp.eye(3) + 2*mu*eps
        return stress
class Box(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size=0.25): return fe.mesh.box_mesh((1.0, 1.0, 1.0), mesh_size=mesh_size, element_type="HEX8")
cell = Box(mesh_size=0.25); mesh = cell.mesh
problem = Elast(mesh, vec=3, dim=3, ele_type="HEX8")
P = flat.pbc.prolongation_matrix(flat.pbc.periodic_bc_3D(cell, vec=3, dim=3), mesh, vec=3)
bc = fe.DirichletBCConfig([]).create_bc(problem)
B = 3; rho = 0.2 + 0.8*onp.random.default_rng(0).random((B, len(mesh.cells)))
E = jnp.asarray(E0*(1e-3 + rho**3)); nu = fe.TracedParams.create_cell_var(problem, NU)
rs = _reduced_system(problem, bc, P, mesh, 3); strains = UNIT_STRAINS_3D; K = 6
# re-implement operator_data with the chosen pin variant (same pattern / assembler as rs)
ndof = problem.num_total_dofs_all_vars
matfree_res_J = create_matfree_res_J_parametric(problem, symmetric=True)
R_pat = reduced_operator_pattern(P, connectivity_pattern(problem)); assemble = asd.operator_assembler(R_pat)
pins = _translation_pin_dofs(P, 3, 3); ip = onp.asarray(R_pat.indptr); jc = onp.asarray(R_pat.indices)
rows_np = onp.repeat(onp.arange(len(ip) - 1), onp.diff(ip))
keep = jnp.asarray(~(onp.isin(rows_np, pins) | onp.isin(jc, pins)))
pin_diag = jnp.asarray(onp.isin(rows_np, pins) & (rows_np == jc))
diag_slots = jnp.asarray(onp.flatnonzero(onp.asarray(pin_diag)))
def operator_data(tp):
    _, J_matvec = matfree_res_J(jnp.zeros(ndof), tp, bc, None)
    Kd = assemble(lambda v: P.T @ J_matvec(P @ v)).data
    if PINS == "scatter": return jnp.where(keep, Kd, 0.0).at[diag_slots].set(1.0)
    return jnp.where(pin_diag, 1.0, jnp.where(keep, Kd, 0.0))
front = fe.TracedParams(volume_vars=(0, None))
values = lambda tp, u0, ts: jax.vmap(operator_data, in_axes=(front,))(tp)
rhs = lambda tp, u0, ts: jax.vmap(lambda t: jax.vmap(lambda e: rs.case_rhs(e, t)[0])(strains), in_axes=(front,))(tp)
rows = jnp.asarray(rs.rows)
def cot(tp, X, lam, u0, ts):
    V, vjp_v = jax.vjp(lambda t: values(t, u0, ts), tp)
    Vb = -sum(lam[:, k][:, rows] * X[:, k][:, rs.indices] for k in range(K))
    _, vjp_b = jax.vjp(lambda t: rhs(t, u0, ts), tp)
    return jax.tree_util.tree_map(lambda a, b: a + b, vjp_v(Vb)[0], vjp_b(lam)[0])
spec = LinearSystemBatch(values=values, rhs=rhs, pattern=lambda ts: (rs.indptr, rs.indices), diag_slots=lambda ts: rs.diag_slots, n_rhs=K, cotangents=cot)
eng = _lhs_engine(spec, front, fe.shared_factor(), None, None, fe.KrylovSolverOptions(solver="cg", tol=1e-12, maxiter=2000))
u_mac = jax.vmap(lambda s_: macro_displacement(mesh, s_))(strains)
def C_of(X, Ea):
    def finish(x_b, Eb):
        t = fe.TracedParams(volume_vars=(Eb, nu))
        return jax.vmap(lambda x, um: average_stress(problem, um + rs.zero_mean(P @ x), t, 3))(x_b, u_mac).T
    return jax.vmap(finish)(X, Ea)
W = jnp.asarray(onp.random.default_rng(1).standard_normal((B, 6, 6)))
def f(Ea):
    X, info = eng.solve(fe.TracedParams(volume_vars=(Ea, nu)), None, None, None)
    return jnp.sum(W * C_of(X, Ea)), info.iterations.max()
g_e, it_e = jax.grad(f, has_aux=True)(E)
import os
if os.environ.get("INV_AOT"):
    comp = jax.jit(jax.grad(f, has_aux=True)).lower(E).compile(); print("AOT compiled", flush=True)
    for r in range(3):
        g_j, it_j = comp(E); print(f"   AOT run {r}: grad rel err {float(jnp.abs(g_j-g_e).max()/jnp.abs(g_e).max()):.0e} fwd its {int(it_j)}", flush=True)
    sys.exit(0)
fj = jax.jit(jax.grad(f, has_aux=True))
res = []
for r in range(int(os.environ.get("NRUNS", "4"))):
    g_j, it_j = fj(E); res.append(f"{float(jnp.abs(g_j-g_e).max()/jnp.abs(g_e).max()):.0e}/{int(it_j)}")
print(f"{VARIANT:16s} eager its {int(it_e)} | jit (grad rel err / fwd its): " + "  ".join(res), flush=True)
