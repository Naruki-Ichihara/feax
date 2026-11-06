"""Gradient program with the prolongation products as gathers (no BCOO scatter-add): first-run test."""
import sys, os, jax; jax.config.update("jax_enable_x64", True)
import numpy as onp, jax.numpy as jnp
VAR = sys.argv[1]   # "gatherP" or "bcoo"
sys.argv = ["inside_assembly.py", "scatter-scatter"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inside_assembly.py")).read().split("g_e, it_e = jax.grad(f, has_aux=True)(E)")[0]
exec(src)
import scipy.sparse as sp
Pc = sp.csr_matrix((onp.asarray(P.data), (onp.asarray(P.indices)[:, 0], onp.asarray(P.indices)[:, 1])), shape=P.shape)
assert onp.allclose(Pc.data, 1.0) and onp.all(onp.diff(Pc.indptr) == 1), "P is not a 0/1 map with one entry per row"
red_of_full = jnp.asarray(Pc.indices.astype(onp.int32))                       # (ndof,)
fan = onp.bincount(Pc.indices, minlength=P.shape[1]); F = int(fan.max())
table = onp.full((P.shape[1], F), -1, dtype=onp.int64)
cnt = onp.zeros(P.shape[1], dtype=onp.int64)
for full, red in enumerate(Pc.indices):
    table[red, cnt[red]] = full; cnt[red] += 1
tab_j = jnp.asarray(onp.maximum(table, 0)); tab_mask = jnp.asarray(table >= 0)
def P_apply(v):   return v[red_of_full]                                       # P @ v
def PT_apply(y):  return jnp.sum(jnp.where(tab_mask, y[tab_j], 0.0), axis=1)  # P^T @ y
if VAR in ("gatherP", "gatherAll"):
    if VAR == "gatherAll":
        import asdex
        from feax.csr import CSRMatrix
        def producer_gather(coloring, compressed_fn):
            pat = asd._to_scipy_pattern(coloring.sparsity)
            slots = onp.asarray(asd._csr_slots(pat, coloring.sparsity.rows, coloring.sparsity.cols)); nnz_ = int(pat.nnz)
            inv = onp.full(nnz_, -1, dtype=onp.int64); inv[slots] = onp.arange(len(slots))
            hit = jnp.asarray(inv >= 0); invc = jnp.asarray(onp.maximum(inv, 0))
            ip_ = jnp.asarray(pat.indptr.astype(onp.int32)); ix_ = jnp.asarray(pat.indices.astype(onp.int32))
            def fn(x):
                data_pat = asdex.decompress_data(compressed_fn(x), coloring)
                return CSRMatrix(jnp.where(hit, data_pat[invc], 0.0), ip_, ix_, tuple(int(s) for s in pat.shape))
            return fn, pat
        asd._csr_producer = producer_gather
        assemble = asd.operator_assembler(R_pat)
    def operator_data_g(tp):
        _, J_matvec = matfree_res_J(jnp.zeros(ndof), tp, bc, None)
        Kd = assemble(lambda v: PT_apply(J_matvec(P_apply(v)))).data
        if VAR == "gatherAll": return jnp.where(pin_diag, 1.0, jnp.where(keep, Kd, 0.0))
        return jnp.where(keep, Kd, 0.0).at[diag_slots].set(1.0)
    values_g = lambda tp, u0, ts: jax.vmap(operator_data_g, in_axes=(front,))(tp)
    from feax.flat.solver import macro_displacement as _md
    from feax.assembler import create_res_bc_parametric
    res_bc = create_res_bc_parametric(problem)
    pins_j = jnp.asarray(pins)
    def case_rhs_g(e, t):
        um = _md(mesh, e); return -(PT_apply(res_bc(um, t, bc, None))).at[pins_j].set(0.0)
    rhs_g = lambda tp, u0, ts: jax.vmap(lambda t: jax.vmap(lambda e: case_rhs_g(e, t))(strains), in_axes=(front,))(tp)
    def cot_g(tp, X, lam, u0, ts):
        V, vjp_v = jax.vjp(lambda t: values_g(t, u0, ts), tp)
        Vb = -sum(lam[:, k][:, rows] * X[:, k][:, rs.indices] for k in range(K))
        _, vjp_b = jax.vjp(lambda t: rhs_g(t, u0, ts), tp)
        return jax.tree_util.tree_map(lambda a, b: a + b, vjp_v(Vb)[0], vjp_b(lam)[0])
    spec_g = LinearSystemBatch(values=values_g, rhs=rhs_g, pattern=lambda ts: (rs.indptr, rs.indices), diag_slots=lambda ts: rs.diag_slots, n_rhs=K, cotangents=cot_g)
    eng_g = _lhs_engine(spec_g, front, fe.shared_factor(), None, None, fe.KrylovSolverOptions(solver="cg", tol=1e-12, maxiter=2000))
    def C_of_g(X, Ea):
        def finish(x_b, Eb):
            t = fe.TracedParams(volume_vars=(Eb, nu))
            return jax.vmap(lambda x, um: average_stress(problem, um + rs.zero_mean(P_apply(x)), t, 3))(x_b, u_mac).T
        return jax.vmap(finish)(X, Ea)
    def f(Ea):
        X, info = eng_g.solve(fe.TracedParams(volume_vars=(Ea, nu)), None, None, None)
        return jnp.sum(W * C_of_g(X, Ea)), info.iterations.max()
g_e, it_e = jax.grad(f, has_aux=True)(E)
comp = jax.jit(jax.grad(f, has_aux=True)).lower(E).compile()
res = []
for r in range(3):
    g_j, it_j = comp(E); res.append(f"{float(jnp.abs(g_j-g_e).max()/jnp.abs(g_e).max()):.0e}/{int(it_j)}")
print(f"{VAR}: eager its {int(it_e)} | runs: " + "  ".join(res), flush=True)
