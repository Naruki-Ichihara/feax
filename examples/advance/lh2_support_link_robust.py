"""Inner-tank support links of an aircraft LH2 double-wall tank.

Eight links, four per station at LINK_ANGLES (45 deg diagonals) about the tank
axis, hold the inner tank; links with identical load sets share a design. All designs
are optimised together, density and (ANISOTROPY = True) the fibre orientation
tensor a2 per element:

    minimise    sum_d m_d (mu_d + KAPPA sigma_d) / sum_d m_d
    subject to  sum_d m_d Q_d <= Q_ALLOW

mu_d, sigma_d: weighted mean / std over load cases of U_dk / U_ref_dk (stored
energy over that of the solid block). Q_d: heat leak of one link. m_d: number of
links per design. Glass/epoxy UD lamina, orientation-averaged: K and alpha are
linear in a2, C uses the hybrid closure; a2 = I/3 is the isotropic state used
for the reference and for ANISOTROPY = False. THERMAL_NONLINEAR = True uses
k(T) (NIST G-10CR fits) and solves the thermal problem by Newton; False uses the
20-293 K mean of each k and one linear solve. Q is the heat flow either way. Thermal strain is coupled one way
(T -> u). Elastic cases are superposed from unit-force and thermal-only solves,
batched with fe.vmap.

Requires cuDSS.

Run:
    python lh2_support_link_robust.py
"""
import collections
import contextlib
import csv
import os
import time

os.environ.setdefault("SPINEAX_FACTOR_CACHE", "1")

import jax
import jax.numpy as jnp
import numpy as onp
import pyvista as pv

import feax as fe
import feax.gene as gene
from feax.experimental.mma import GCMMA
from feax.mechanics.orientation import (box_to_triangle, hybrid_closure,
                                        orientation_averaged_stiffness_3d,
                                        principal_direction, smooth_sgn)
from feax.utils import XDMFWriter, save_sol

jax.config.update("jax_enable_x64", True)

# ── Parameters (mm, N, MPa, K, W) ──────────────────────────────────────────
LX = LY = 50.0
LZ = 100.0
ELEM_SIZE = 3
PASSIVE_THICKNESS = 5.0

ANISOTROPY = True
THERMAL_NONLINEAR = True

# E-glass/epoxy UD lamina (approximate)
E1, E2, G12, NU12, NU23 = 45.0e3, 12.0e3, 5.5e3, 0.28, 0.40
# k(T) of G-10CR, NIST fits (4-300 K): log10 k = sum c_i (log10 T)^i, W/(m K).
# Warp (in-plane) for the fibre direction, normal for the transverse one.
K_PAR_FIT = (-2.64827, 8.80228, -24.8998, 41.1625, -39.8754, 23.1778,
             -7.95635, 1.48806, -0.11701)
K_PERP_FIT = (-4.1236, 13.788, -26.068, 26.272, -14.663, 4.4954, -0.6905,
              0.0397)
T_HOT, T_COLD = 293.0, 20.0
T_REF = T_HOT
ALPHA_PAR = 0.0015 / (T_HOT - T_COLD)    # secant, 293 -> 20 K
ALPHA_PERP = 0.0060 / (T_HOT - T_COLD)
E_MIN, K_MIN = 1e-6, 1e-4
RAMP_Q_E, RAMP_Q_K = 8.0, -0.8
SGN_BETA = 10.0
T_ISO = -0.1547008478              # t1 = t2 = T_ISO -> a2 = I/3

INNER_MASS = 5000.0                # kg
G0 = 9.80665
# Links per station at these angles about the tank axis, from +Z (up) towards
# +Y (left): 45 / 135 / 225 / 315 = upper-left, lower-left, lower-right,
# upper-right.
LINK_ANGLES = (45.0, 135.0, 225.0, 315.0)
STATIONS = ("front", "rear")
LINKS = tuple(f"{st}_{int(a)}" for st in STATIONS for a in LINK_ANGLES)
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
Q_ALLOW = 10.0                     # W, all 8 links
REF_FLOOR = 1e-3
LOAD_IGNORE = 1e-3                 # cases with ~no force on a link get w = 0

MIN_WIDTH = 6.0
FILTER_RADIUS = gene.radius_for_minimum_width(MIN_WIDTH, 0.7)
FILTER_RADIUS_ORI = 2.0 * ELEM_SIZE
HEAVISIDE_BETAS = (2, 4, 8, 12, 16)
ITER_PER_STAGE = (6, 6, 10, 20)
ITER_PER_STAGE_FINAL = 50
FINAL_WINDOW = 10
FINAL_FTOL_REL = 1e-3
FINAL_FEAS_TOL = 1e-3
MMA_MAX_INNER = 5
MMA_C_CONSTRAINT = 1e7
X_INIT = 0.2

NEWTON_TOL, NEWTON_RTOL, NEWTON_MAXITER = 1e-10, 1e-10, 30
NEWTON_VERBOSE = False             # print every Newton step of every solve
SHARE_ME = 4                       # 4 sub-cases of a design share its operator
CHUNK_ME = 4
CG_TOL, CG_MAXITER = 1e-9, 2000

OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output_lh2_support_link_robust")
os.makedirs(OUTROOT, exist_ok=True)


# ── Loads ──────────────────────────────────────────────────────────────────
def radial(name):
    """Unit vector from the tank axis to the link, aircraft axes (X fwd, Y left, Z up)."""
    t = onp.radians(float(name.split("_")[1]))
    return onp.array([0.0, onp.sin(t), onp.cos(t)])


def link_frame(name):
    """Rows: link-local x, y, z in aircraft axes; local z points inwards."""
    ez = -radial(name)
    ex = onp.array([1.0, 0.0, 0.0])
    return onp.stack([ex, onp.cross(ez, ex), ez])


def factor_loads():
    """Split of F = -m g n: half per station; in the Y-Z plane the links of a
    station act as equal axial springs (f_i = e_i . (sum e e^T)^-1 F/2); X is
    taken in shear by all links equally."""
    E = onp.stack([radial(n)[1:] for n in LINKS[:len(LINK_ANGLES)]])
    Minv = onp.linalg.inv(E.T @ E)
    F = onp.zeros((N_LINKS, N_CASES, 3))
    for li, name in enumerate(LINKS):
        e = radial(name)
        R = link_frame(name)
        for k, n in enumerate(CASE_N):
            Fg = -INNER_MASS * G0 * n
            f_ax = e[1:] @ Minv @ (Fg[1:] / len(STATIONS))
            F[li, k] = R @ (onp.array([Fg[0] / N_LINKS, 0.0, 0.0]) + f_ax * e)
    return F


LOADS = factor_loads()

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
MID_NODE = int(onp.argmin(onp.linalg.norm(pts - [0.5 * LX, 0.5 * LY, 0.5 * LZ], axis=1)))

passive_cell = jnp.asarray((centroids[:, 2] <= PASSIVE_THICKNESS + TOL)
                           | (centroids[:, 2] >= LZ - PASSIVE_THICKNESS - TOL))
design_cell = ~passive_cell
n_design = int(design_cell.sum())
N_FIELDS = 5 if ANISOTROPY else 1    # rho, t1, t2, t12, t13
n_total = N_DES * N_FIELDS * n_cells


# ── Problems ───────────────────────────────────────────────────────────────
def w_E(rho):
    return E_MIN + (1.0 - E_MIN) * rho / (1.0 + RAMP_Q_E * (1.0 - rho))


def w_k(rho):
    return K_MIN + (1.0 - K_MIN) * rho / (1.0 + RAMP_Q_K * (1.0 - rho))


def unit_trace_a2(t1, t2, t12, t13):
    """Trace-1 PSD a2 from four box variables (s23 = s12 s13)."""
    a11, a22 = box_to_triangle(t1, t2)
    a33 = 1.0 - a11 - a22
    s12, s13 = smooth_sgn(t12, SGN_BETA), smooth_sgn(t13, SGN_BETA)
    a12 = jnp.sqrt(jnp.maximum(a11 * a22, 0.0)) * s12
    a13 = jnp.sqrt(jnp.maximum(a11 * a33, 0.0)) * s13
    a23 = jnp.sqrt(jnp.maximum(a22 * a33, 0.0)) * s12 * s13
    return jnp.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])


def stiffness(a2):
    return orientation_averaged_stiffness_3d(a2, hybrid_closure(a2), E1=E1, E2=E2,
                                             G12=G12, nu12=NU12, nu23=NU23)


def elastic_strain(u_grad, a2, T):
    alpha = ALPHA_PERP * jnp.eye(3) + (ALPHA_PAR - ALPHA_PERP) * a2
    return 0.5 * (u_grad + u_grad.T) - alpha * (T - T_REF)


def k_fit(c, T):
    """W/(mm K)."""
    lt = jnp.log10(jnp.clip(T, 4.0, 300.0))
    return 1e-3 * 10.0 ** sum(ci * lt ** i for i, ci in enumerate(c))


def k_mean(c):
    Ts = onp.linspace(T_COLD, T_HOT, 2001)
    return float(onp.trapezoid(onp.asarray(k_fit(c, Ts)), Ts) / (T_HOT - T_COLD))


K_PAR_MEAN, K_PERP_MEAN = k_mean(K_PAR_FIT), k_mean(K_PERP_FIT)


def conductivities(T):
    """(k_par, k_perp), W/(mm K): k(T), or its 20-293 K mean."""
    if THERMAL_NONLINEAR:
        return k_fit(K_PAR_FIT, T), k_fit(K_PERP_FIT, T)
    return K_PAR_MEAN, K_PERP_MEAN


def heat_flux(grad_T, T, rho, a2):
    """q = -w_k(rho) K(T, a2) grad T."""
    kp, kt = conductivities(T)
    return -w_k(rho) * (kt * jnp.eye(3) + (kp - kt) * a2) @ grad_T


class Conduction(fe.Problem):
    """K may depend on T, so the element residual is written directly."""

    def get_universal_kernel(self):
        sval = self.fes[0].shape_vals
        nn = sval.shape[1]

        def kernel(cell_sol_flat, x, cell_shape_grads, cell_JxW,
                   cell_v_grads_JxW, *cell_vars):
            rho, t1, t2, t12, t13 = (jnp.reshape(v, -1)[0] for v in cell_vars)
            a2 = unit_trace_a2(t1, t2, t12, t13)
            Tn = cell_sol_flat.reshape(nn)
            Tq = sval @ Tn
            gq = jnp.einsum("n,qnd->qd", Tn, cell_shape_grads[:, :nn])
            q = jax.vmap(lambda g, t: heat_flux(g, t, rho, a2))(gq, Tq)
            return -jnp.sum(q[:, None, :] * cell_v_grads_JxW[:, :nn, 0, :],
                            axis=(0, 2))
        return kernel


def create_heat_flow_fn(problem):
    """Heat flow hot -> cold face, Q = (1/LZ) int q_z dV (the reaction of the
    discrete equations for the test function 1 - z/LZ)."""
    fe0 = problem.fes[0]
    cells_ = problem.cells_list[0]
    sg, sval, jxw = fe0.shape_grads, fe0.shape_vals, fe0.JxW

    def heat_flow(T, rho, a2):
        Tc = T[cells_]
        Tq = jnp.einsum("qn,cn->cq", sval, Tc)
        gq = jnp.einsum("cn,cqnd->cqd", Tc, sg)
        qz = jax.vmap(lambda g, t, r, a: jax.vmap(
            lambda gg, tt: heat_flux(gg, tt, r, a)[2])(g, t))(gq, Tq, rho, a2)
        return jnp.sum(qz * jxw) / LZ
    return heat_flow


class ThermoElastic(fe.Problem):
    def get_energy_density(self):
        def psi(u_grad, rho, t1, t2, t12, t13, T):
            a2 = unit_trace_a2(t1, t2, t12, t13)
            eps = elastic_strain(u_grad, a2, T)
            return 0.5 * w_E(rho) * jnp.einsum("ij,ijkl,kl->", eps, stiffness(a2), eps)
        return psi

    def get_surface_maps(self):
        def traction(u, x, tx_, ty_, tz_):
            return jnp.array([tx_, ty_, tz_])
        return [traction]


def create_vonmises_fn(problem):
    fe0 = problem.fes[0]
    cells_ = problem.cells_list[0]
    sg, sval = fe0.shape_grads, fe0.shape_vals

    def vm(sigma):
        dev = sigma - jnp.eye(3) * (jnp.trace(sigma) / 3.0)
        return jnp.sqrt(jnp.maximum(1.5 * jnp.sum(dev * dev), 1e-30))

    def vonmises(u_flat, a2, T):
        """Material stress von Mises per quad point; a2 (n_cells, 3, 3)."""
        cell_u = jnp.asarray(u_flat).reshape(-1, 3)[cells_]
        T_q = jnp.einsum("qn,cn->cq", sval, T[cells_])

        def per_cell(cu, csg, a, tq):
            ug = jnp.sum(cu[None, :, :, None] * csg[:, :, None, :], axis=1)
            C = stiffness(a)
            return jax.vmap(lambda g, t: vm(jnp.einsum(
                "ijkl,kl->ij", C, elastic_strain(g, a, t))))(ug, tq)

        return jax.vmap(per_cell)(cell_u, sg, a2, T_q)

    return vonmises


_o, _z = jnp.ones((n_cells,)), jnp.zeros((n_cells,))
_ti = jnp.full((n_cells,), T_ISO)
ISO = (_ti, _ti, _z, _z)
T_REF_NODES = jnp.full((n_nodes,), T_REF)
cg = fe.KrylovSolverOptions(solver="cg", tol=CG_TOL, atol=1e-14,
                            maxiter=CG_MAXITER)
precond = fe.shared_factor()

thermal = Conduction(mesh, vec=1, dim=3, ele_type="HEX8")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(fixed_face, "all", T_HOT),
    fe.DirichletBCSpec(loaded_face, "all", T_COLD),
]).create_bc(thermal)
ts_th = fe.TracedStructure.from_problem(thermal)
# With k(T) the Newton tangent is non-symmetric: LU, not Cholesky/CG.
_lu = fe.DirectSolverOptions(solver="cudss",
                             cudss_options=fe.CUDSSOptions(matrix_type="GENERAL"))
solve_T = fe.create_solver(
    thermal, bc_th, _lu, adjoint_solver_options=_lu,
    linear=not THERMAL_NONLINEAR,
    traced_structure=ts_th,
    newton_options=fe.NewtonOptions(tol=NEWTON_TOL, rel_tol=NEWTON_RTOL,
                                    max_iter=NEWTON_MAXITER,
                                    verbose=NEWTON_VERBOSE),
    traced_params=fe.TracedParams(volume_vars=(_o,) + ISO))
guess_T = jnp.asarray(T_HOT + (T_COLD - T_HOT) * pts[:, 2] / LZ)
heat_flow = create_heat_flow_fn(thermal)
res_T = fe.create_res_bc_function(thermal, bc_th)

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
    traced_params=fe.TracedParams(volume_vars=(_o,) + ISO + (T_REF_NODES,),
                                  surface_vars=[tuple(s[0] for s in SURF_B)]))
solve_u_batch = fe.vmap(
    _solve_u, in_axes=fe.TracedParams(volume_vars=(0,) * 6,
                                      surface_vars=[(0, 0, 0)]),
    options=cg, preconditioner=precond, chunk=CHUNK_ME, share=SHARE_ME,
    return_info=True)
guess_u = fe.zero_like_initial_guess(mech, bc_me)
energy_fn = fe.create_energy_fn(mech)
vonmises = create_vonmises_fn(mech)

STRUCTURES = (ts_th, ts_me, guess_T, guess_u)
FILTERS = tuple(gene.create_density_filter(mesh, radius=r, weight_type="cone",
                                           points=centroids)
                for r in (FILTER_RADIUS, FILTER_RADIUS_ORI))


# ── Responses ──────────────────────────────────────────────────────────────
def process(x, beta, filters):
    """x (N_DES * N_FIELDS * n_cells,) -> fields (rho, t1, t2, t12, t13), each
    (N_DES, n_cells)."""
    xs = x.reshape(N_DES, N_FIELDS, n_cells)
    rho = gene.heaviside_projection(filters[0](xs[:, 0]), beta=beta, threshold=0.5)
    rho = jnp.where(passive_cell[None, :], 1.0, rho)
    if ANISOTROPY:
        return (rho,) + tuple(filters[1](xs[:, i]) for i in range(1, 5))
    return (rho,) + tuple(jnp.broadcast_to(v, rho.shape) for v in ISO)


def cell_a2(fields):
    """(N_DES, n_cells, 3, 3)."""
    return jax.vmap(jax.vmap(unit_trace_a2))(*fields[1:])


def solve_states(fields, F, structures):
    ts_th, ts_me, gT, gu = structures
    tps = [fe.TracedParams(volume_vars=tuple(f[d] for f in fields))
           for d in range(N_DES)]
    T = jnp.stack([jnp.asarray(getattr(s, "dofs", s)) for s in (
        solve_T(tp_d, gT, traced_structure=ts_th) for tp_d in tps)])
    Q = jax.vmap(heat_flow)(T, fields[0], cell_a2(fields))
    # Newton check: ||R(T)|| at the solution, and relative to ||R(guess)||.
    sg = jax.lax.stop_gradient
    r = jnp.stack([jnp.linalg.norm(res_T(sg(T[d]), sg(tps[d])))
                   for d in range(N_DES)])
    r0 = jnp.stack([jnp.linalg.norm(res_T(gT, sg(tps[d])))
                    for d in range(N_DES)])
    T_b = jnp.concatenate([jnp.broadcast_to(T_REF_NODES, (N_DES, 3, n_nodes)),
                           T[:, None, :]], axis=1).reshape(N_ME, n_nodes)
    tp = fe.TracedParams(
        volume_vars=tuple(jnp.repeat(f, N_SUB, axis=0) for f in fields) + (T_b,),
        surface_vars=[SURF_B])
    sol_u, info_u = solve_u_batch(tp, gu, ts_me)
    U = sol_u.dofs.reshape(N_DES, N_SUB, -1)
    u = jnp.einsum("ikj,ijn->ikn", F, U[:, :3]) + U[:, 3:4]
    diag = dict(cg_u=info_u.iterations, res_T=r, rel_T=r / jnp.maximum(r0, 1e-30),
                T_min=jnp.min(T, axis=1), T_mid=T[:, MID_NODE])
    return Q, T, u, diag


def case_energies(fields, T, u):
    def per_design(Ti, ui, *f):
        tp = fe.TracedParams(volume_vars=f + (Ti,))
        return jax.vmap(lambda uk: energy_fn(uk, tp))(ui)
    return jax.vmap(per_design)(T, u, *fields)


@jax.jit
def reference(F, structures):
    fields = (jnp.ones((N_DES, n_cells)),) + tuple(
        jnp.broadcast_to(v, (N_DES, n_cells)) for v in ISO)
    Q, T, u, _ = solve_states(fields, F, structures)
    return Q, case_energies(fields, T, u)


def responses(x, beta, filters, structures, cdata):
    F, U_ref, w, mult, q_allow = cdata
    fields = process(x, beta, filters)
    rho = fields[0]
    Q, T, u, diag = solve_states(fields, F, structures)
    U = case_energies(fields, T, u) / U_ref
    mu = jnp.sum(w * U, axis=1)
    sig = jnp.sqrt(jnp.sum(w * (U - mu[:, None]) ** 2, axis=1) + SIGMA_EPS)
    J = jnp.sum(mult * (mu + KAPPA * sig)) / jnp.sum(mult)
    c = jnp.sum(mult * Q) / q_allow - 1.0
    aux = dict(Q=Q, mu=mu, sigma=sig, U=U,
               volume=(rho * design_cell).sum(axis=1) / n_design,
               azz=(cell_a2(fields)[..., 2, 2] * rho * design_cell).sum(axis=1)
               / jnp.maximum((rho * design_cell).sum(axis=1), 1e-12),
               rho=rho, a2_diag=jnp.diagonal(cell_a2(fields), axis1=-2, axis2=-1),
               T=T, **diag)
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
    fields = process(x, beta, filters)
    _, T, u, _ = solve_states(fields, cdata[0], structures)
    a2 = cell_a2(fields)
    vm = jax.vmap(lambda a, Ti, ui: jax.vmap(
        lambda uk: vonmises(uk, a, Ti).mean(axis=1))(ui))(a2, T, u)
    return fields[0], a2, T, u, vm


# ── Setup ──────────────────────────────────────────────────────────────────
Q_ref, U_raw = (onp.asarray(a) for a in reference(jnp.asarray(LOADS_D), STRUCTURES))
U_REF = onp.maximum(U_raw, REF_FLOOR * U_raw.max(axis=1, keepdims=True))
HEAT_REF_W = float(Q_ref[0])
CDATA = (jnp.asarray(LOADS_D), jnp.asarray(U_REF), jnp.asarray(WEIGHTS_D),
         jnp.asarray(MULT), jnp.float64(Q_ALLOW))

print(f"{N_LINKS} links -> {N_DES} designs, {n_cells:,} cells each; "
      f"solid link {HEAT_REF_W:.3f} W, budget {Q_ALLOW:g} W")
print(f"ANISOTROPY {ANISOTROPY}, THERMAL_NONLINEAR {THERMAL_NONLINEAR} "
      f"(mean k_par {K_PAR_MEAN*1e3:.3f}, k_perp {K_PERP_MEAN*1e3:.3f} W/mK)")
print("axial / shear force [kN] per case: " + ", ".join(CASE_NAMES))
for i, name in enumerate(LINKS):
    print(f"  {name:13s} d{DESIGN_OF[i]} "
          + " ".join(f"{f[2]/1e3:+6.1f}/{onp.hypot(f[0], f[1])/1e3:4.1f}"
                     for f in LOADS[i]))

_lower = onp.full((N_DES, N_FIELDS, n_cells), -1.0)
_lower[:, 0] = 0.0
_x0 = onp.zeros((N_DES, N_FIELDS, n_cells))
_x0[:, 0] = X_INIT
_x0[:, 1:3] = T_ISO
opt = GCMMA(value_and_grad, _lower.ravel(), onp.ones(n_total),
            n_constraints=1, value_fn=values_only, has_aux=True,
            c=onp.array([MMA_C_CONSTRAINT]), max_inner=MMA_MAX_INNER)
step = jax.jit(opt.step)

# Logs and per-design XDMF time series, written every iteration.
_logs = contextlib.ExitStack()
_hist_fh = _logs.enter_context(open(os.path.join(OUTROOT, "history.csv"), "w",
                                    newline=""))
_cases_fh = _logs.enter_context(open(os.path.join(OUTROOT, "history_cases.csv"),
                                     "w", newline=""))
hist_csv, cases_csv = csv.writer(_hist_fh), csv.writer(_cases_fh)
cases_csv.writerow(["iteration", "beta", "design", "design_links", "case",
                    "case_name", "U_norm", "weight"])
xdmf = [_logs.enter_context(XDMFWriter(mesh, os.path.join(
            OUTROOT, f"design{d}_{LINKS[REP[d]].split('_')[1]}deg_history.xdmf")))
        for d in range(N_DES)]
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
        row[f"azz{d}"] = float(a["azz"][d])
        row[f"newton_res{d}"] = float(a["res_T"][d])
        row[f"T_mid{d}"] = float(a["T_mid"][d])
    if it == 1:
        hist_csv.writerow(list(row))
    hist_csv.writerow([float(v) for v in row.values()])
    U = onp.asarray(a["U"])
    for d in range(N_DES):
        for k in range(N_CASES):
            cases_csv.writerow([it, beta, d, DESIGN_NAMES[d], k, CASE_NAMES[k],
                                U[d, k], WEIGHTS_D[d, k]])
    _hist_fh.flush()
    _cases_fh.flush()
    rho_it, a2d, T_it = (onp.asarray(a[k]) for k in ("rho", "a2_diag", "T"))
    for d, w in enumerate(xdmf):
        w.write_iteration(it, cell_infos=[
            ("rho", rho_it[d]), ("a11", a2d[d, :, 0]), ("a22", a2d[d, :, 1]),
            ("a33", a2d[d, :, 2])], point_infos=[("temperature", T_it[d])])
        w._h5.flush()                      # readable while the run goes on
        w._write_xdmf_xml()
    print(f"  it {it:4d} b={beta:4.1f} J={row['objective']:.4f} "
          f"Q8={q8:.3f}/{Q_ALLOW:g} W "
          f"mu=[" + " ".join(f"{m:5.2f}" for m in mu) + "] "
          f"sig=[" + " ".join(f"{s:4.2f}" for s in sig) + "] "
          f"azz=[" + " ".join(f"{z:4.2f}" for z in onp.asarray(a["azz"])) + "] "
          f"cg {int(onp.max(a['cg_u']))}")
    res, rel = onp.asarray(a["res_T"]), onp.asarray(a["rel_T"])
    ok = (res <= NEWTON_TOL) | (rel <= NEWTON_RTOL)
    print(f"           newton |R|=[" + " ".join(f"{v:.1e}" for v in res) + "] "
          f"rel=[" + " ".join(f"{v:.1e}" for v in rel) + "] "
          f"{'ok' if ok.all() else 'NOT CONVERGED ' + str(onp.flatnonzero(~ok).tolist())}  "
          f"T(z=L/2)=[" + " ".join(f"{v:5.1f}" for v in onp.asarray(a['T_mid'])) + "] K")
    return row["objective"], q8 / Q_ALLOW - 1.0


# ── Optimisation ───────────────────────────────────────────────────────────
x = jnp.asarray(_x0.ravel())
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
_logs.close()

# ── Results ────────────────────────────────────────────────────────────────
a = state.aux
Q, mu, sig, U, V, AZZ = (onp.asarray(a[k]) for k in
                         ("Q", "mu", "sigma", "U", "volume", "azz"))
rho, a2, T, u, vm = (onp.asarray(v) for v in final_fields(
    x, jnp.float64(HEAVISIDE_BETAS[-1]), FILTERS, STRUCTURES, CDATA))
vmax = onp.array([float((vm[d] * rho[d][None, :]).max()) for d in range(N_DES)])
Q8 = float(MULT @ Q)

for d, name in enumerate(DESIGN_NAMES):
    print(f"  {name:32s} x{MULT[d]:g} Q={Q[d]:.4f} W V={V[d]:.3f} "
          f"mu={mu[d]:.2f} sigma={sig[d]:.2f} a_zz={AZZ[d]:.2f} "
          f"vm_max={vmax[d]:.1f} MPa")
print(f"  total heat leak {Q8:.4f} W (budget {Q_ALLOW:g} W), "
      f"{it} evaluations, {dt:.0f} s")

with open(os.path.join(OUTROOT, "summary.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["link", "name", "design", "Q_W", "volume", "mu", "sigma",
                 "a_zz", "vm_max_MPa"] + [f"U{k}" for k in range(N_CASES)])
    for i, name in enumerate(LINKS):
        d = DESIGN_OF[i]
        wr.writerow([i, name, d, Q[d], V[d], mu[d], sig[d], AZZ[d], vmax[d]]
                    + [U[d, k] if WEIGHTS_D[d, k] > 0 else ""
                       for k in range(N_CASES)])
    wr.writerow(["total", "", "", Q8])

_hits = onp.maximum(onp.bincount(cells.ravel(), minlength=n_nodes), 1)
for d in range(N_DES):
    rho_n = onp.bincount(cells.ravel(), weights=onp.repeat(rho[d], cells.shape[1]),
                         minlength=n_nodes) / _hits
    save_sol(mesh, os.path.join(OUTROOT, f"design{d}_{LINKS[REP[d]].split('_')[1]}deg.vtu"),
             point_infos=[("density", rho_n), ("temperature", T[d])]
             + [(f"u_case{k}", u[d, k].reshape(-1, 3) * rho_n[:, None])
                for k in range(N_CASES)],
             cell_infos=[("rho", rho[d]), ("a11", a2[d, :, 0, 0]),
                         ("a22", a2[d, :, 1, 1]), ("a33", a2[d, :, 2, 2]),
                         ("director", onp.asarray(principal_direction(
                             jnp.asarray(a2[d]))) * rho[d][:, None])]
             + [(f"vm_case{k}", vm[d, k] * rho[d]) for k in range(N_CASES)])

# ── Assembly view: the 8 links on a virtual tank (visualisation only) ──────
VIS_R_INNER = 1000.0               # mm, inner tank radius
VIS_LENGTH = 4000.0                # mm, cylindrical part
VIS_STATION = 1200.0               # mm, front / rear station from the centre
R_OUTER = VIS_R_INNER + LZ         # the links span the vacuum gap

_hex = onp.hstack([onp.full((n_cells, 1), 8), cells]).ravel()
_local = pts - onp.array([0.5 * LX, 0.5 * LY, 0.0])
_blocks = []
for i, name in enumerate(LINKS):
    d, R = DESIGN_OF[i], link_frame(name)
    x0 = onp.array([VIS_STATION if name.startswith("front") else -VIS_STATION, 0, 0])
    g = pv.UnstructuredGrid(_hex, onp.full(n_cells, pv.CellType.HEXAHEDRON),
                            x0 - R[2] * R_OUTER + _local @ R)
    g.cell_data["rho"] = rho[d]
    g.cell_data["link"] = onp.full(n_cells, i)
    g.cell_data["design"] = onp.full(n_cells, d)
    g.cell_data["a_axial"] = a2[d, :, 2, 2]
    g.cell_data["director"] = (onp.asarray(principal_direction(jnp.asarray(a2[d])))
                               @ R) * rho[d][:, None]
    g.point_data["temperature"] = T[d]
    _blocks.append(g)
links_vis = pv.merge(_blocks, merge_points=False)


def _tank(radius):
    body = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=radius,
                       height=VIS_LENGTH, resolution=120, capping=False)
    caps = [pv.Sphere(radius=radius, center=(s * VIS_LENGTH / 2, 0, 0),
                      direction=(s, 0, 0), end_phi=90, theta_resolution=120)
            for s in (1, -1)]
    return body.merge(caps)


assembly = pv.MultiBlock({"links": links_vis, "inner_tank": _tank(VIS_R_INNER),
                          "outer_shell": _tank(R_OUTER)})
assembly.save(os.path.join(OUTROOT, "assembly.vtm"))
try:
    pl = pv.Plotter(off_screen=True, window_size=(1600, 900))
    pl.add_mesh(assembly["inner_tank"], color="lightsteelblue", smooth_shading=True)
    pl.add_mesh(assembly["outer_shell"], color="white", opacity=0.12)
    pl.add_mesh(links_vis.threshold(0.5, scalars="rho"), scalars="temperature",
                cmap="coolwarm", scalar_bar_args=dict(title="T [K]"))
    pl.camera_position = "iso"
    pl.screenshot(os.path.join(OUTROOT, "assembly.png"))
    pl.close()
    del pl
    # Cross-section at the front station, seen along the tank axis.
    pl = pv.Plotter(off_screen=True, window_size=(1200, 1200))
    cut = dict(normal=(1, 0, 0), origin=(VIS_STATION, 0, 0))
    pl.add_mesh(assembly["inner_tank"].clip(**cut), color="lightsteelblue")
    pl.add_mesh(assembly["outer_shell"].clip(**cut), color="lightgray", opacity=0.4)
    pl.add_mesh(links_vis.threshold(0.5, scalars="rho"), scalars="temperature",
                cmap="coolwarm", scalar_bar_args=dict(
                    title="T [K]", vertical=True, position_x=0.88, position_y=0.3))
    pl.camera_position = [(VIS_STATION + 4500.0, 0, 0), (VIS_STATION, 0, 0), (0, 0, 1)]
    pl.screenshot(os.path.join(OUTROOT, "assembly_station.png"))
    pl.close()
    del pl
except Exception as e:                            # no off-screen rendering
    print(f"(no assembly.png: {e})")
print(f"wrote {OUTROOT}")
