"""Many topology optimizations at once with GCMMA: shared-factor PCG + ``BatchedGCMMA``.

The GCMMA counterpart of ``batched_topology_optimization_shared_pcg.py``: B
independent compliance-minimization problems of the same cantilever, one per
load direction, optimized together. The B forward (and adjoint) solves of every
evaluation go through ``fe.vmap`` -- batched CG preconditioned by ONE cuDSS
factorization of the mean matrix (``fe.shared_factor``) per group of ``SHARE``
designs, at first the whole batch (the designs start identical), assembled
``CHUNK`` designs at a time -- and the update is ``BatchedGCMMA`` from ``feax.experimental.mma``.

``jax.vmap`` of ``GCMMA`` would evaluate every design on its own, B separate
solves per evaluation, which is exactly what the shared factorization avoids.
``BatchedGCMMA`` receives the whole batch ``X (B, n)`` in its value-and-gradient
function and returns ``(B,)``, ``(B, n)``, ``(B, m)``, ``(B, m, n)`` from ONE
batched evaluation, while the asymptotes, the convex subproblems and the
conservativity (inner) loop are per design. Each inner iteration costs one more
batched evaluation for all designs, so ``MAX_INNER`` is kept small; watch
``state.inner_iterations`` (one count per design) next to the CG counts.

As in the OC script, the group factorizations (``batched.factorize``) are made
once and reused by every evaluation (``batched(..., factors=)``); whenever the
mean CG count exceeds ``CG_THRESHOLD`` the batch is regrouped by operator
similarity (``batched.operator_signature`` -> ``gene.similarity_order``), the
group size ``SHARE`` is halved and the new groups are factorized. The GCMMA
state is unaffected (the designs keep their original order, only the solve is
permuted).

What to look at in the output: the CG iteration count per design
(``info.iterations``). The mean-matrix preconditioner is good while the designs
are similar (early iterations) and degrades as they separate into different
layouts; grouping designs with neighbouring load angles (``SHARE``) keeps it
effective. The final designs are written to one VTU, one point field each.

Requires cuDSS (``spineax``); without it ``fe.vmap`` falls back to a Jacobi
preconditioner (per chunk: ``SHARE`` is then ``CHUNK``) and the iteration
counts are much higher.

Run:
    python batched_topology_optimization_shared_pcg_gcmma.py
"""

import os
from time import perf_counter

# spineax keeps its last SPINEAX_FACTOR_CACHE factorizations (default 8) on the
# GPU and cannot free them under jit; every group and adjoint solve mints one,
# and only one is in use at a time. Must be set before spineax is imported.
os.environ.setdefault("SPINEAX_FACTOR_CACHE", "1")

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
from feax import gene
from feax.experimental.mma import BatchedGCMMA

# ── Problem ───────────────────────────────────────────────────────────────
# Problem size and batch (overridable from the environment for larger runs, e.g.
# NX=80 NY=20 NZ=20 B=100 ITERS=10 for 100 designs at ~100k DOF with one
# factorization shared by the whole batch).
_env = lambda name, default: type(default)(os.environ.get(name, default))
L, W, H = 4.0, 1.0, 1.0
NX, NY, NZ = _env("NX", 48), _env("NY", 12), _env("NZ", 12)
E0, NU = 1.0, 0.3
PENAL, EPS = 3.0, 1e-3
VOLFRAC = 0.4
B = _env("B", 8)           # designs: load directions 0 .. 180 (1 - 1/B) deg from -z in the y-z plane
SHARE = _env("SHARE", 0) or None   # designs per shared factorization (0: the whole batch -- the designs start identical, so one mean factorization is exact; CG_THRESHOLD splits the batch as they separate)
CHUNK = _env("CHUNK", 4) or None   # designs assembled at a time (memory: CHUNK stiffness matrices; 0: all at once)
ITERS = _env("ITERS", 40)
FILTER_RADIUS = 2.5 * (L / NX)
BETA = _env("BETA", 2.0)       # Heaviside projection sharpness, fixed for the whole run (no continuation)
MOVE = 0.2                 # GCMMA move limit (fraction of xmax - xmin)
MAX_INNER = 3              # GCMMA inner (conservativity) iterations: each one is a batched evaluation
TRACTION = 1e-2
REORDER_EVERY = _env("REORDER_EVERY", 0)   # also re-sort (and re-factorize) every this many iterations (0: only when regrouping)
CG_THRESHOLD = _env("CG_THRESHOLD", 200)   # when the mean CG count exceeds this: regroup by similarity, halve SHARE, re-factorize (0: never)
MAX_GROUPS = _env("MAX_GROUPS", 0)         # cap on the number of groups the halving may create (0: down to one design per group)
REPORT_EVERY = _env("REPORT_EVERY", 5)     # print a progress line every this many iterations
METHOD = "gcmma"
OUT_DIR = os.environ.get("OUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_batched_topopt_shared_pcg_gcmma")
os.makedirs(OUT_DIR, exist_ok=True)


class SIMPElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            mu = E / (2.0 * (1.0 + NU))
            lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2.0 * mu * eps
        return stress

    def get_surface_maps(self):
        # feax adds the surface map with a + sign: the map returns minus the traction
        def traction(u, x, ty, tz):
            return jnp.array([0.0, ty, tz])
        return [traction]


mesh = fe.StructuredGrid((NX, NY, NZ), spacing=(L / NX, W / NY, H / NZ)).to_mesh()
patch = lambda p: (jnp.isclose(p[0], L, atol=1e-6) & (jnp.abs(p[1] - 0.5 * W) <= 0.2 * W + 1e-6)
                   & (jnp.abs(p[2] - 0.5 * H) <= 0.2 * H + 1e-6))
problem = SIMPElasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[patch])
bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0),
]).create_bc(problem)
ts = fe.TracedStructure.from_problem(problem)
u0 = fe.zero_like_initial_guess(problem, bc)
n_nodes = mesh.points.shape[0]
print(f"cantilever {NX}x{NY}x{NZ} HEX8: {n_nodes:,} nodes, {u0.shape[0]:,} DOF, {B} designs")

# ── Loads: one direction per design, batched surface variables ─────────────
theta = onp.pi * onp.arange(B) / B                                   # angle from -z toward +y
unit = fe.TracedParams.create_uniform_surface_var(problem, 1.0)
lead = (B,) + (1,) * unit.ndim
TY = jnp.asarray(-TRACTION * onp.sin(theta)).reshape(lead) * unit[None]
TZ = jnp.asarray(TRACTION * onp.cos(theta)).reshape(lead) * unit[None]
loads = (TY, TZ)

# ── The solver, batched by fe.vmap ────────────────────────────────────────
tp0 = fe.TracedParams(volume_vars=(gene.simp(jnp.full(n_nodes, VOLFRAC), E0, PENAL, EPS),),
                      surface_vars=[(TY[0], TZ[0])])
# The single-case solver is only the template fe.vmap batches. Built with the
# batch's own CG options it holds nothing: a DirectSolverOptions template would
# keep a pre-warmed cuDSS factorization the batch never uses (7 GB at 500k DOF).
cg = fe.KrylovSolverOptions(solver="cg", tol=1e-7, atol=1e-12, maxiter=3000)
solver = fe.create_solver(problem, bc, cg, linear=True, traced_params=tp0, traced_structure=ts)
in_axes = fe.TracedParams(volume_vars=(0,), surface_vars=[(0, 0)])


def make_batched(share):
    # share=None is the whole batch here; Jacobi is per design, so without
    # cuDSS there is nothing to share and the batch just runs CHUNK at a time.
    cudss = fe.solvers.options.has_cudss()
    return fe.vmap(
        solver, in_axes=in_axes,
        options=cg,
        preconditioner=fe.shared_factor() if cudss else fe.jacobi(),
        chunk=CHUNK, share=(share or B) if cudss else None, return_info=True)


share = SHARE
batched = make_batched(share)
print(batched)

# ── gene: filter, responses, volume weights (all take a batch (B, n)) ───────
filt = gene.create_density_filter(mesh, FILTER_RADIUS)
volume = gene.create_volume_fn(problem)
compliance = gene.create_compliance_fn(
    problem, surface_load_params=[(jnp.asarray(-TRACTION * onp.sin(theta)),
                                   jnp.asarray(TRACTION * onp.cos(theta)))])
vol_w = jax.grad(volume)(jnp.zeros(n_nodes))                        # nodal volume weights
# The volume constraint is imposed on the filtered density: the filter is
# linear, so its weight on the design variables is the filter's transpose
# applied to the nodal weights (one vector-Jacobian product).
dv = jax.vjp(filt, jnp.zeros(n_nodes))[1](vol_w)[0]


# The TracedStructure, the initial guess and the filter are mesh-sized: they
# go in as arguments of every jitted function (``mesh``, threaded through the
# GCMMA step as one of its args). Closed over, they would be compiled in as
# constants, and at ~10^5 DOF and up XLA spends minutes (on one thread)
# folding them.
mesh_args = (ts, u0, filt)


def make_evaluate(batched):
    @jax.jit
    def evaluate(X, beta, order, factors, mesh):
        """Compliance of every design, its sensitivity w.r.t. the design field,
        the volume fractions of the filtered densities and the CG iteration
        counts. The batch is solved in the order ``order``, preconditioned by
        ``factors`` (one per group of SHARE consecutive designs, from
        ``factorize``); results come back in the original order."""
        ts, u0, filt = mesh
        back = jnp.argsort(order)

        def objective(X):
            rho = gene.heaviside_projection(filt(X), beta=beta)
            E = gene.simp(rho, E0, PENAL, EPS)
            tp = fe.TracedParams(volume_vars=(E[order],), surface_vars=[(TY[order], TZ[order])])
            U, info = batched(tp, u0, ts, factors=factors)
            c = compliance(U.dofs[back])                           # (B,)
            return jnp.sum(c), (c, volume(filt(X)), info.iterations[back])
        (_, (c, v, its)), dc = jax.value_and_grad(objective, has_aux=True)(X)
        return c, dc, v, its
    return evaluate


def similarity_order(X, beta):
    """Order the designs so that consecutive ones have similar operators."""
    rho = gene.heaviside_projection(filt(X), beta=beta)
    sig = batched.operator_signature(
        fe.TracedParams(volume_vars=(gene.simp(rho, E0, PENAL, EPS),), surface_vars=[loads]))
    return gene.similarity_order(sig)


def factorize(X, beta, order):
    """Factorize the mean stiffness matrix of every group at the designs X
    (in the solve order); the factors are reused until the next regrouping.
    Without cuDSS (Jacobi) there is nothing to factorize: None."""
    if not fe.solvers.options.has_cudss():
        return None
    n_groups = -(-B // (share or B))
    # every group's cuDSS entry stays resident while reused (read per entry)
    os.environ["SPINEAX_FACTOR_CACHE"] = str(n_groups)
    rho = gene.heaviside_projection(filt(X), beta=beta)
    tp = fe.TracedParams(volume_vars=(gene.simp(rho, E0, PENAL, EPS)[order],),
                         surface_vars=[(TY[order], TZ[order])])
    return batched.factorize(tp, u0, ts)


evaluate = make_evaluate(batched)


# ── Optimization loop ──────────────────────────────────────────────────────
X = jnp.full((B, n_nodes), VOLFRAC)
beta = BETA
order = jnp.arange(B)
factors = factorize(X, beta, order)
c, dc, v, its = evaluate(X, beta, order, factors, mesh_args)
c0 = c
beta_at = lambda k: BETA


def report(k, beta, c, v, its, t_start):
    ratio = c / c0
    print(f"{k:4d}  {beta:4.0f}  {float(ratio.mean()):9.4f}  {float(ratio.max()):8.4f}  "
          f"[{float(v.min()):.3f}, {float(v.max()):.3f}]  "
          f"{float(its.mean()):8.1f} / {int(its.max()):4d}  {(share or B):5d}  "
          f"{perf_counter() - t_start:6.1f}s")


# ── CSV logs: one row per iteration (history.csv) and per design per iteration
#    (designs.csv), flushed as they are written ─────────────────────────────
import csv
from datetime import datetime, timezone

CSV_DIR = os.environ.get("CSV_DIR") or OUT_DIR
os.makedirs(CSV_DIR, exist_ok=True)
RUN_TAG = os.environ.get("RUN_TAG") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
HISTORY_COLUMNS = [
    "run", "method", "iter", "beta", "share", "chunk", "n_groups", "dof", "designs",
    "c_ratio_mean", "c_ratio_min", "c_ratio_max", "vol_min", "vol_max",
    "cg_mean", "cg_min", "cg_max", "cg_sum", "inner_mean", "inner_max",
    "t_update_s", "t_reorder_s", "t_factorize_s", "t_eval_s", "t_regroup_s", "t_iter_s", "elapsed_s",
    "reordered", "regrouped", "order",
]
DESIGN_COLUMNS = ["run", "method", "iter", "design", "angle_deg", "group", "position",
                  "c_ratio", "cg_iters", "volume", "inner_iters"]


class CsvLog:
    def __init__(self, path, columns):
        self.path, self.columns = path, columns
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=columns)
        if new:
            self.w.writeheader()

    def row(self, **kw):
        self.w.writerow({k: kw.get(k, "") for k in self.columns})
        self.fh.flush()


history_log = CsvLog(os.path.join(CSV_DIR, "history.csv"), HISTORY_COLUMNS)
design_log = CsvLog(os.path.join(CSV_DIR, "designs.csv"), DESIGN_COLUMNS)


def log_iteration(k, beta, c, v, its, *, times, reordered, regrouped, inner=None, elapsed):
    """Write the iteration row and the B design rows."""
    ratio = onp.asarray(c / c0)
    its_ = onp.asarray(its)
    v_ = onp.asarray(v)
    ordr = onp.asarray(order)
    size = share or B
    position = onp.empty(B, int)
    position[ordr] = onp.arange(B)
    inner_ = None if inner is None else onp.asarray(inner)
    history_log.row(
        run=RUN_TAG, method=METHOD, iter=k, beta=beta, share=size, chunk=CHUNK, n_groups=-(-B // size),
        dof=int(u0.shape[0]), designs=B,
        c_ratio_mean=float(ratio.mean()), c_ratio_min=float(ratio.min()), c_ratio_max=float(ratio.max()),
        vol_min=float(v_.min()), vol_max=float(v_.max()),
        cg_mean=float(its_.mean()), cg_min=int(its_.min()), cg_max=int(its_.max()), cg_sum=int(its_.sum()),
        inner_mean="" if inner_ is None else float(inner_.mean()),
        inner_max="" if inner_ is None else int(inner_.max()),
        t_update_s=times.get("update", 0.0), t_reorder_s=times.get("reorder", 0.0),
        t_factorize_s=times.get("factorize", 0.0),
        t_eval_s=times.get("eval", 0.0), t_regroup_s=times.get("regroup", 0.0),
        t_iter_s=sum(times.values()), elapsed_s=elapsed,
        reordered=int(reordered), regrouped=int(regrouped), order=" ".join(map(str, ordr.tolist())))
    for i in range(B):
        design_log.row(
            run=RUN_TAG, method=METHOD, iter=k, design=i, angle_deg=float(onp.degrees(theta[i])),
            group=int(position[i] // size), position=int(position[i]), c_ratio=float(ratio[i]),
            cg_iters=int(its_[i]), volume=float(v_[i]),
            inner_iters="" if inner_ is None else int(inner_[i]))


def timed(fn, *args):
    t0 = perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, perf_counter() - t0


print(f"\n{'iter':>4}  {'beta':>4}  {'c/c0 mean':>9}  {'c/c0 max':>8}  {'vol':>11}  "
      f"{'CG iters mean/max':>17}  {'share':>5}  {'time':>6}  inner iters (per design)")
t_start = perf_counter()


# GCMMA over the batch. The objective is c / c0 per design, the constraint
# V / VOLFRAC - 1 <= 0 on the filtered density; ``beta`` and the solve order
# ride along as arguments of the evaluation (a changed group size means a new
# batched solver, so the step is re-jitted then; the GCMMA state is unchanged).
def value_and_grad(X, beta, order, factors, mesh):
    c, dc, v, its = evaluate(X, beta, order, factors, mesh)      # dc: own design only
    g = v / VOLFRAC - 1.0
    dg = jnp.broadcast_to(dv / (VOLFRAC * vol_w.sum()), X.shape)
    return c / c0, dc / c0[:, None], g[:, None], dg[:, None, :], (c, v, its)


opt = BatchedGCMMA(value_and_grad, 0.0, 1.0, n_constraints=1, max_inner=MAX_INNER,
                   move=MOVE, has_aux=True)
step = jax.jit(opt.step)
_t0 = perf_counter()
state = jax.block_until_ready(opt.init(X, beta, order, factors, mesh_args))
log_iteration(0, beta, state.aux[0], state.aux[1], state.aux[2],
              times={"eval": perf_counter() - _t0}, reordered=False, regrouped=False,
              inner=state.inner_iterations, elapsed=0.0)
refactor = False
for k in range(1, ITERS + 1):
    times, reordered, regrouped = {}, False, False
    beta = beta_at(k)
    if REORDER_EVERY and k % REORDER_EVERY == 1:
        order, times["reorder"] = timed(similarity_order, state.x, beta)
        reordered = refactor = True
    if refactor:                                 # new groups: factorize them at the current designs
        factors, times["factorize"] = timed(factorize, state.x, beta, order)
        refactor = False
    # one GCMMA step = subproblems + (1 + inner) batched evaluations
    state, times["eval"] = timed(step, state, beta, order, factors, mesh_args)
    c, v, its = state.aux
    log_share, log_order = share, order
    if CG_THRESHOLD and float(its.mean()) > CG_THRESHOLD:
        # the preconditioner is losing its grip: regroup the designs by operator
        # similarity and, down to MAX_GROUPS groups, halve the group size (more,
        # closer factorizations beat hundreds of CG iterations).
        new_order, times["regroup"] = timed(similarity_order, state.x, beta)
        min_share = -(-B // MAX_GROUPS) if MAX_GROUPS else 1
        new_share = max(min_share, (share or B) // 2)
        regrouped = True
    share, order = log_share, log_order
    log_iteration(k, beta, c, v, its, times=times, reordered=reordered, regrouped=regrouped,
                  inner=state.inner_iterations, elapsed=perf_counter() - t_start)
    if k % REPORT_EVERY == 0 or k == 1:
        report(k, beta, c, v, its, t_start)
        print(f"{'':>82}  {onp.asarray(state.inner_iterations)}")
    if regrouped:
        order, refactor = new_order, True
        if new_share != (share or B):
            share = new_share
            batched = make_batched(share)
            evaluate = make_evaluate(batched)
            step = jax.jit(opt.step)
X = state.x
jax.block_until_ready(X)
history_log.fh.close(); design_log.fh.close()
print(f"history -> {history_log.path}\ndesigns -> {design_log.path}")
print(f"\n{ITERS} iterations of {B} designs in {perf_counter() - t_start:.1f} s "
      f"({(perf_counter() - t_start) / ITERS / B:.3f} s per design per iteration)")
print("final c/c0 per design:", onp.round(onp.asarray(c / c0), 4))
print("CG iterations per design (last):", onp.asarray(its), " share:", share)

# ── Save ──────────────────────────────────────────────────────────────────
out = OUT_DIR
rho = gene.heaviside_projection(filt(X), beta=beta)
fe.utils.save_sol(
    mesh=mesh, sol_file=os.path.join(out, "designs.vtu"),
    point_infos=[(f"rho_{i:02d}_{onp.degrees(theta[i]):05.1f}deg", rho[i]) for i in range(B)])
print(f"wrote {out}/designs.vtu")
