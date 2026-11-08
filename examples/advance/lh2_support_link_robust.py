"""Inner-tank support links of an aircraft LH2 double-wall tank.

Eight links (front/rear x top/bottom/left/right) hold the inner tank; links
with identical load sets share a design (here: top, bottom, side). All designs
are optimised together:

    minimise    sum_d m_d (mu_d + KAPPA sigma_d) / sum_d m_d
    subject to  sum_d m_d Q_d <= Q_ALLOW

mu_d, sigma_d: weighted mean / std over load cases of U_dk / U_ref_dk (stored
energy over that of the solid block). Q_d: heat leak of one link. m_d: number of
links per design. Thermal strain is coupled one way (T -> u). Elastic cases are
superposed from unit-force and thermal-only solves, batched with fe.vmap.

Run:
    python lh2_support_link_robust.py [--loads loads.csv] [--mass 5000] [--qallow 10]

loads.csv: link,case,Fx,Fy,Fz [N], link-local frame (z: outer shell -> tank).
"""
import argparse
import collections
import csv
import os
import time

os.environ.setdefault("SPINEAX_FACTOR_CACHE", "1")

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
import feax.gene as gene
from feax.experimental.mma import GCMMA
from feax.utils import save_sol

jax.config.update("jax_enable_x64", True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--loads", default=None)
_ap.add_argument("--mass", type=float, default=5000.0)    # kg
_ap.add_argument("--qallow", type=float, default=10.0)    # W, all 8 links
ARGS, _ = _ap.parse_known_args()

# ── Parameters (mm, N, MPa, K, W) ──────────────────────────────────────────
LX = LY = 50.0
LZ = 100.0
ELEM_SIZE = 2.5
PASSIVE_THICKNESS = 5.0

# G-10CR, isotropic
E_SOLID, NU = 28.0e3, 0.15
K_MEAN = 0.40e-3                   # int k dT / dT, 20-293 K
T_HOT, T_COLD = 293.0, 20.0
T_REF = T_HOT
ALPHA = 0.0024 / (T_HOT - T_COLD)
E_MIN, K_MIN = 1e-6, 1e-4
RAMP_Q_E, RAMP_Q_K = 8.0, -0.8
HEAT_TO_WATT = 2.0 / (T_HOT - T_COLD)
LAM = E_SOLID * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_SOLID / (2.0 * (1.0 + NU))

INNER_MASS = ARGS.mass
G0 = 9.80665
LINKS = ("front_top", "front_bottom", "front_left", "front_right",
         "rear_top", "rear_bottom", "rear_left", "rear_right")
N_LINKS = len(LINKS)
LOAD_FACTOR_TABLE = (               # name, n_x, n_y, n_z, +-n_y
    ("manoeuvre +2.5",  0.0, 0.0,  2.5, False),   # CS 25.337
    ("manoeuvre -1.0",  0.0, 0.0, -1.0, False),   # CS 25.337
    ("gust lateral",    0.0, 1.33, 1.0, True),    # CS 25.363
    ("ground turning",  0.0, 0.5,  1.0, True),    # CS 25.495
    ("braking",        -0.8, 0.0,  1.2, False),   # CS 25.493
    ("landing drag",   -0.6, 0.0,  2.5, False),   # CS 25.479
    ("crosswind / yaw", 0.0, 1.0,  2.0, True),
)
CASE_NAMES, CASE_N, _w = [], [], []
for _name, _nx, _ny, _nz, _pm in LOAD_FACTOR_TABLE:
    for _s in ((1.0, -1.0) if _pm else (1.0,)):
        CASE_NAMES.append(_name + (("+y" if _s > 0 else "-y") if _pm else ""))
        CASE_N.append((_nx, _s * _ny, _nz))
        _w.append(0.5 if _pm else 1.0)
CASE_N = onp.asarray(CASE_N)
N_CASES = len(CASE_N)
CASE_WEIGHTS = onp.asarray(_w) / onp.sum(_w)

KAPPA = 1.0
SIGMA_EPS = 1e-8
Q_ALLOW = ARGS.qallow
REF_FLOOR = 1e-3
LOAD_IGNORE = 1e-3                 # cases with ~no force on a link get w = 0

MIN_WIDTH = 6.0
FILTER_RADIUS = gene.radius_for_minimum_width(MIN_WIDTH, 0.7)
HEAVISIDE_BETAS = (2, 4, 8, 12, 16)
ITER_PER_STAGE = (6, 6, 10, 20)
ITER_PER_STAGE_FINAL = 50
FINAL_WINDOW = 10
FINAL_FTOL_REL = 1e-3
FINAL_FEAS_TOL = 1e-3
MMA_MAX_INNER = 5
MMA_C_CONSTRAINT = 1e7
X_INIT = 0.2

SHARE_TH = 1                       # one factor per design: exact
SHARE_ME = 4                       # 4 sub-cases of a design share its operator
CHUNK_ME = 4
CG_TOL, CG_MAXITER = 1e-9, 2000

OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output_lh2_support_link_robust")
os.makedirs(OUTROOT, exist_ok=True)


# ── Loads ──────────────────────────────────────────────────────────────────
def link_frame(name):
    """Rows: link-local x, y, z in aircraft axes (X fwd, Y left, Z up)."""
    ez = onp.asarray({"top": (0, 0, -1), "bottom": (0, 0, 1),
                      "left": (0, -1, 0), "right": (0, 1, 0)}[name.split("_")[1]],
                     float)
    ex = onp.array([1.0, 0.0, 0.0])
    return onp.stack([ex, onp.cross(ez, ex), ez])


def factor_loads():
    """Determinate split of F = -m g n: Z over top/bottom, Y over left/right
    (axial, /4), X over all 8 (shear, /8)."""
    F = onp.zeros((N_LINKS, N_CASES, 3))
    for li, name in enumerate(LINKS):
        side = name.split("_")[1]
        R = link_frame(name)
        for k, n in enumerate(CASE_N):
            Fg = -INNER_MASS * G0 * n
            F[li, k] = R @ onp.array([
                Fg[0] / 8.0,
                Fg[1] / 4.0 if side in ("left", "right") else 0.0,
                Fg[2] / 4.0 if side in ("top", "bottom") else 0.0])
    return F


def read_loads(path):
    F = onp.full((N_LINKS, N_CASES, 3), onp.nan)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            F[int(row["link"]), int(row["case"])] = (
                float(row["Fx"]), float(row["Fy"]), float(row["Fz"]))
    if onp.isnan(F).any():
        raise ValueError(f"{path}: missing (link, case) rows")
    return F


LOADS = read_loads(ARGS.loads) if ARGS.loads else factor_loads()

_Fn = onp.linalg.norm(LOADS, axis=2)
LINK_WEIGHTS = CASE_WEIGHTS[None, :] * (_Fn >= LOAD_IGNORE * _Fn.max(axis=1, keepdims=True))
LINK_WEIGHTS /= LINK_WEIGHTS.sum(axis=1, keepdims=True)


def _signature(i):
    scale = max(_Fn.max(), 1e-30)
    return tuple(sorted(tuple(onp.round(onp.r_[w, f / scale], 9))
                        for f, w in zip(LOADS[i], LINK_WEIGHTS[i]) if w > 0))


_groups = {}
for _i in range(N_LINKS):
    _groups.setdefault(_signature(_i), []).append(_i)
GROUPS = list(_groups.values())
REP = onp.array([g[0] for g in GROUPS])
MULT = onp.array([len(g) for g in GROUPS], float)
N_DES = len(GROUPS)
DESIGN_NAMES = ["+".join(LINKS[i] for i in g) for g in GROUPS]
LOADS_D = LOADS[REP]
WEIGHTS_D = LINK_WEIGHTS[REP]
DESIGN_OF = onp.empty(N_LINKS, int)
for _d, _g in enumerate(GROUPS):
    DESIGN_OF[_g] = _d

# ── Mesh ───────────────────────────────────────────────────────────────────
TOL = 1e-6 * LZ
fixed_face = lambda p: jnp.isclose(p[2], 0.0, atol=TOL)
loaded_face = lambda p: jnp.isclose(p[2], LZ, atol=TOL)

mesh = fe.mesh.box_mesh(size=(LX, LY, LZ), mesh_size=ELEM_SIZE,
                        element_type="HEX8")
pts = onp.asarray(mesh.points)
cells = onp.asarray(mesh.cells)
n_nodes, n_cells = pts.shape[0], cells.shape[0]
centroids = pts[cells].mean(axis=1)
A_FACE = LX * LY

passive_cell = jnp.asarray((centroids[:, 2] <= PASSIVE_THICKNESS + TOL)
                           | (centroids[:, 2] >= LZ - PASSIVE_THICKNESS - TOL))
design_cell = ~passive_cell
n_design = int(design_cell.sum())
n_total = N_DES * n_cells


# ── Problems ───────────────────────────────────────────────────────────────
def w_E(rho):
    return E_MIN + (1.0 - E_MIN) * rho / (1.0 + RAMP_Q_E * (1.0 - rho))


def w_k(rho):
    return K_MIN + (1.0 - K_MIN) * rho / (1.0 + RAMP_Q_K * (1.0 - rho))


def elastic_strain(u_grad, T):
    return 0.5 * (u_grad + u_grad.T) - ALPHA * (T - T_REF) * jnp.eye(3)


def solid_stress(eps):
    return LAM * jnp.trace(eps) * jnp.eye(3) + 2.0 * MU * eps


class Conduction(fe.Problem):
    def get_energy_density(self):
        def psi(grad_T, rho):
            return 0.5 * K_MEAN * w_k(rho) * jnp.sum(grad_T ** 2)
        return psi


class ThermoElastic(fe.Problem):
    def get_energy_density(self):
        def psi(u_grad, rho, T):
            eps = elastic_strain(u_grad, T)
            return 0.5 * w_E(rho) * jnp.sum(eps * solid_stress(eps))
        return psi

    def get_surface_maps(self):
        def traction(u, x, tx_, ty_, tz_):
            return jnp.array([tx_, ty_, tz_])
        return [traction]


def create_vonmises_fn(problem):
    fe0 = problem.fes[0]
    cells_ = problem.cells_list[0]
    sg, sval, nq = fe0.shape_grads, fe0.shape_vals, fe0.num_quads

    def vm(sigma):
        dev = sigma - jnp.eye(3) * (jnp.trace(sigma) / 3.0)
        return jnp.sqrt(jnp.maximum(1.5 * jnp.sum(dev * dev), 1e-30))

    def vonmises(u_flat, rho, T):
        cell_u = jnp.asarray(u_flat).reshape(-1, 3)[cells_]
        T_q = jnp.einsum("qn,cn->cq", sval, T[cells_])
        rho_q = jnp.tile(rho[:, None], (1, nq))

        def per_cell(cu, csg, rq, tq):
            ug = jnp.sum(cu[None, :, :, None] * csg[:, :, None, :], axis=1)
            return jax.vmap(lambda g, r, t: vm(solid_stress(elastic_strain(g, t))))(
                ug, rq, tq)

        return jax.vmap(per_cell)(cell_u, sg, rho_q, T_q)

    return vonmises


_o = jnp.ones((n_cells,))
T_REF_NODES = jnp.full((n_nodes,), T_REF)
cg = fe.KrylovSolverOptions(solver="cg", tol=CG_TOL, atol=1e-14,
                            maxiter=CG_MAXITER)
cudss = fe.solvers.options.has_cudss()
precond = fe.shared_factor() if cudss else fe.jacobi()

thermal = Conduction(mesh, vec=1, dim=3, ele_type="HEX8")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(fixed_face, "all", T_HOT),
    fe.DirichletBCSpec(loaded_face, "all", T_COLD),
]).create_bc(thermal)
ts_th = fe.TracedStructure.from_problem(thermal)
_solve_T = fe.create_solver(thermal, bc_th, cg, linear=True,
                            traced_structure=ts_th,
                            traced_params=fe.TracedParams(volume_vars=(_o,)))
solve_T_batch = fe.vmap(
    _solve_T, in_axes=fe.TracedParams(volume_vars=(0,)), options=cg,
    preconditioner=precond, chunk=SHARE_TH,
    share=SHARE_TH if cudss else None, return_info=True)
guess_T = fe.zero_like_initial_guess(thermal, bc_th)
energy_T = fe.create_energy_fn(thermal)

mech = ThermoElastic(mesh, vec=3, dim=3, ele_type="HEX8",
                     location_fns=(loaded_face,))
bc_me = fe.DirichletBCConfig([
    fe.DirichletBCSpec(fixed_face, "all", 0.0)]).create_bc(mech)
ts_me = fe.TracedStructure.from_problem(mech)
# Sub-cases per design: unit force x, y, z (T = T_REF), thermal only.
N_SUB = 4
N_ME = N_DES * N_SUB
_unit_sv = fe.TracedParams.create_uniform_surface_var(mech, 1.0)
_t_sub = onp.zeros((N_SUB, 3))
_t_sub[:3] = -onp.eye(3) / A_FACE
_t_b = onp.tile(_t_sub, (N_DES, 1))
_lead = (N_ME,) + (1,) * _unit_sv.ndim
SURF_B = tuple(jnp.asarray(_t_b[:, j]).reshape(_lead) * _unit_sv[None]
               for j in range(3))
_solve_u = fe.create_solver(
    mech, bc_me, cg, linear=True, traced_structure=ts_me,
    traced_params=fe.TracedParams(volume_vars=(_o, T_REF_NODES),
                                  surface_vars=[tuple(s[0] for s in SURF_B)]))
solve_u_batch = fe.vmap(
    _solve_u, in_axes=fe.TracedParams(volume_vars=(0, 0),
                                      surface_vars=[(0, 0, 0)]),
    options=cg, preconditioner=precond, chunk=CHUNK_ME,
    share=SHARE_ME if cudss else None, return_info=True)
guess_u = fe.zero_like_initial_guess(mech, bc_me)
energy_fn = fe.create_energy_fn(mech)
vonmises = create_vonmises_fn(mech)

STRUCTURES = (ts_th, ts_me, guess_T, guess_u)
FILTERS = (gene.create_density_filter(mesh, radius=FILTER_RADIUS,
                                      weight_type="cone", points=centroids),)


# ── Responses ──────────────────────────────────────────────────────────────
def process(xs, beta, filters):
    rho = gene.heaviside_projection(filters[0](xs), beta=beta, threshold=0.5)
    return jnp.where(passive_cell[None, :], 1.0, rho)


def solve_states(rho, F, structures):
    ts_th, ts_me, gT, gu = structures
    sol_T, info_T = solve_T_batch(fe.TracedParams(volume_vars=(rho,)), gT, ts_th)
    T = sol_T.dofs
    Q = jax.vmap(lambda Ti, ri: energy_T(
        Ti, fe.TracedParams(volume_vars=(ri,))))(T, rho) * HEAT_TO_WATT
    T_b = jnp.concatenate([jnp.broadcast_to(T_REF_NODES, (N_DES, 3, n_nodes)),
                           T[:, None, :]], axis=1).reshape(N_ME, n_nodes)
    tp = fe.TracedParams(volume_vars=(jnp.repeat(rho, N_SUB, axis=0), T_b),
                         surface_vars=[SURF_B])
    sol_u, info_u = solve_u_batch(tp, gu, ts_me)
    U = sol_u.dofs.reshape(N_DES, N_SUB, -1)
    u = jnp.einsum("ikj,ijn->ikn", F, U[:, :3]) + U[:, 3:4]
    return Q, T, u, (info_T.iterations, info_u.iterations)


def case_energies(rho, T, u):
    def per_design(r, Ti, ui):
        tp = fe.TracedParams(volume_vars=(r, Ti))
        return jax.vmap(lambda uk: energy_fn(uk, tp))(ui)
    return jax.vmap(per_design)(rho, T, u)


@jax.jit
def reference(F, structures):
    rho = jnp.ones((N_DES, n_cells))
    Q, T, u, _ = solve_states(rho, F, structures)
    return Q, case_energies(rho, T, u)


def responses(x, beta, filters, structures, cdata):
    F, U_ref, w, mult, q_allow = cdata
    rho = process(x.reshape(N_DES, n_cells), beta, filters)
    Q, T, u, its = solve_states(rho, F, structures)
    U = case_energies(rho, T, u) / U_ref
    mu = jnp.sum(w * U, axis=1)
    sig = jnp.sqrt(jnp.sum(w * (U - mu[:, None]) ** 2, axis=1) + SIGMA_EPS)
    J = jnp.sum(mult * (mu + KAPPA * sig)) / jnp.sum(mult)
    c = jnp.sum(mult * Q) / q_allow - 1.0
    aux = dict(Q=Q, mu=mu, sigma=sig, U=U,
               volume=(rho * design_cell).sum(axis=1) / n_design,
               cg_T=its[0], cg_u=its[1])
    return J, c, aux


@jax.jit
def value_and_grad(x, beta, filters, structures, cdata):
    def f(z):
        J, c, aux = responses(z, beta, filters, structures, cdata)
        return (J, c), aux

    (J, c), vjp, aux = jax.vjp(f, x, has_aux=True)
    gJ, = vjp((jnp.ones_like(J), jnp.zeros_like(c)))
    gc, = vjp((jnp.zeros_like(J), jnp.ones_like(c)))
    return J, gJ, c[None], gc[None, :], aux


@jax.jit
def values_only(x, beta, filters, structures, cdata):
    J, c, _ = responses(x, beta, filters, structures, cdata)
    return J, c[None]


@jax.jit
def final_fields(x, beta, filters, structures, cdata):
    rho = process(x.reshape(N_DES, n_cells), beta, filters)
    _, T, u, _ = solve_states(rho, cdata[0], structures)
    vm = jax.vmap(lambda r, Ti, ui: jax.vmap(
        lambda uk: vonmises(uk, r, Ti).mean(axis=1))(ui))(rho, T, u)
    return rho, T, u, vm


# ── Setup ──────────────────────────────────────────────────────────────────
Q_ref, U_raw = (onp.asarray(a) for a in reference(jnp.asarray(LOADS_D), STRUCTURES))
U_REF = onp.maximum(U_raw, REF_FLOOR * U_raw.max(axis=1, keepdims=True))
HEAT_REF_W = float(Q_ref[0])
CDATA = (jnp.asarray(LOADS_D), jnp.asarray(U_REF), jnp.asarray(WEIGHTS_D),
         jnp.asarray(MULT), jnp.float64(Q_ALLOW))

print(f"{N_LINKS} links -> {N_DES} designs, {n_cells:,} cells each; "
      f"solid link {HEAT_REF_W:.3f} W, budget {Q_ALLOW:g} W; cuDSS {cudss}")
print("axial / shear force [kN] per case: " + ", ".join(CASE_NAMES))
for i, name in enumerate(LINKS):
    print(f"  {name:13s} d{DESIGN_OF[i]} "
          + " ".join(f"{f[2]/1e3:+6.1f}/{onp.hypot(f[0], f[1])/1e3:4.1f}"
                     for f in LOADS[i]))

opt = GCMMA(value_and_grad, onp.zeros(n_total), onp.ones(n_total),
            n_constraints=1, value_fn=values_only, has_aux=True,
            c=onp.array([MMA_C_CONSTRAINT]), max_inner=MMA_MAX_INNER)
step = jax.jit(opt.step)

hist = collections.defaultdict(list)
it = 0


def record(state, beta):
    global it
    it += 1
    a = state.aux
    Q = onp.asarray(a["Q"])
    mu, sig = onp.asarray(a["mu"]), onp.asarray(a["sigma"])
    q8 = float(MULT @ Q)
    row = dict(iteration=it, beta=beta, objective=float(state.f0), Q_total=q8)
    for d in range(N_DES):
        row[f"Q{d}"], row[f"mu{d}"], row[f"sigma{d}"] = Q[d], mu[d], sig[d]
        row[f"V{d}"] = float(a["volume"][d])
    for key, val in row.items():
        hist[key].append(float(val))
    print(f"  it {it:4d} b={beta:4.1f} J={row['objective']:.4f} "
          f"Q8={q8:.3f}/{Q_ALLOW:g} W "
          f"mu=[" + " ".join(f"{m:5.2f}" for m in mu) + "] "
          f"sig=[" + " ".join(f"{s:4.2f}" for s in sig) + "] "
          f"cg[T {int(onp.max(a['cg_T']))} u {int(onp.max(a['cg_u']))}]")
    return row["objective"], q8 / Q_ALLOW - 1.0


# ── Optimisation ───────────────────────────────────────────────────────────
x = jnp.full((n_total,), X_INIT)
t0 = time.perf_counter()
for stage, beta in enumerate(HEAVISIDE_BETAS):
    beta_t = jnp.float64(beta)
    final = stage == len(HEAVISIDE_BETAS) - 1
    cap = ITER_PER_STAGE_FINAL if final else ITER_PER_STAGE[stage]
    window = collections.deque(maxlen=FINAL_WINDOW)
    state = opt.init(x, beta_t, FILTERS, STRUCTURES, CDATA)
    record(state, beta)
    for _ in range(cap - 1):
        state = step(state, beta_t, FILTERS, STRUCTURES, CDATA)
        j, margin = record(state, beta)
        if final:
            window.append(j)
            if len(window) == window.maxlen:
                spread = (max(window) - min(window)) / max(
                    abs(sum(window) / len(window)), 1e-12)
                if spread < FINAL_FTOL_REL and margin <= FINAL_FEAS_TOL:
                    break
    x = state.x
dt = time.perf_counter() - t0

# ── Results ────────────────────────────────────────────────────────────────
a = state.aux
Q, mu, sig, U, V = (onp.asarray(a[k]) for k in ("Q", "mu", "sigma", "U", "volume"))
rho, T, u, vm = (onp.asarray(v) for v in final_fields(
    x, jnp.float64(HEAVISIDE_BETAS[-1]), FILTERS, STRUCTURES, CDATA))
vmax = onp.array([float((vm[d] * rho[d][None, :]).max()) for d in range(N_DES)])
Q8 = float(MULT @ Q)

for d, name in enumerate(DESIGN_NAMES):
    print(f"  {name:32s} x{MULT[d]:g} Q={Q[d]:.4f} W V={V[d]:.3f} "
          f"mu={mu[d]:.2f} sigma={sig[d]:.2f} vm_max={vmax[d]:.1f} MPa")
print(f"  total heat leak {Q8:.4f} W (budget {Q_ALLOW:g} W), "
      f"{it} evaluations, {dt:.0f} s")

with open(os.path.join(OUTROOT, "history.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(list(hist))
    wr.writerows(zip(*hist.values()))
with open(os.path.join(OUTROOT, "summary.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["link", "name", "design", "Q_W", "volume", "mu", "sigma",
                 "vm_max_MPa"] + [f"U{k}" for k in range(N_CASES)])
    for i, name in enumerate(LINKS):
        d = DESIGN_OF[i]
        wr.writerow([i, name, d, Q[d], V[d], mu[d], sig[d], vmax[d]]
                    + [U[d, k] if WEIGHTS_D[d, k] > 0 else ""
                       for k in range(N_CASES)])
    wr.writerow(["total", "", "", Q8])

_hits = onp.maximum(onp.bincount(cells.ravel(), minlength=n_nodes), 1)
for d in range(N_DES):
    rho_n = onp.bincount(cells.ravel(), weights=onp.repeat(rho[d], cells.shape[1]),
                         minlength=n_nodes) / _hits
    save_sol(mesh, os.path.join(OUTROOT, f"design{d}_{LINKS[REP[d]].split('_')[1]}.vtu"),
             point_infos=[("density", rho_n), ("temperature", T[d])]
             + [(f"u_case{k}", u[d, k].reshape(-1, 3) * rho_n[:, None])
                for k in range(N_CASES)],
             cell_infos=[("rho", rho[d])]
             + [(f"vm_case{k}", vm[d, k] * rho[d]) for k in range(N_CASES)])
print(f"wrote {OUTROOT}")
