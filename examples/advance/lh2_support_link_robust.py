"""Inner-tank support links of an aircraft LH2 double-wall tank: eight links
designed together under one heat budget, robust over load cases.

    z = 0     outer shell  T = T_HOT,  clamped (u = 0),       passive band
    z = LZ    inner tank   T = T_COLD, link force applied,   passive band
    sides     free and insulated (vacuum jacket)

    design             rho_d, per ELEMENT, one per symmetry group d of links
    minimise           J = sum_d m_d ( mu_d + KAPPA sigma_d ) / sum_d m_d
    subject to         sum_d m_d Q_d <= Q_ALLOW     (all 8 links)  [W]

    m_d = number of links design d stands for (front = rear, left = right)

    U~_ik   = U_ik / U_ref_ik        stored energy of link i, case k, over solid
    mu_i    = sum_k w_k U~_ik        weighted mean over the load cases
    sigma_i = sqrt(sum_k w_k (U~_ik - mu_i)^2)   spread over the load cases

Eight links -- front and rear station, each with top / bottom / left / right --
hold a 5000 kg inner tank (``INNER_MASS``) at the middle of the fuselage. Every
link sees the same load cases (LOAD_FACTOR_TABLE: 7 rows, 10 cases), and the force a link gets in a given case depends on
where it sits, so each link has its own load matrix ``F[i] (7, 3)`` in its local
frame.

SYMMETRY. Links with the same set of (weight, force) cases get the same
design, detected from the loads themselves (``_signature``), so it holds for
--loads too. With the load-factor table: front = rear, and left = right because
every lateral case is run both ways, giving 3 designs -- top (x2), bottom (x2),
side (x4). Top and bottom are NOT merged: their forces are opposite, and the
thermal strain adds to one and subtracts from the other.

WHY ALL DESIGNS AT ONCE. Each link's heat leak depends only on its own design, but
the budget is on the TOTAL, so the links compete for it: the optimiser hands
more material (and more heat) to the heavily loaded links and starves the
lightly loaded ones. Designing them one by one cannot do that allocation.

THE OBJECTIVE IS ROBUST STIFFNESS. ``U~_ik`` is 1 for the solid block in every
case, so ``mu_i`` measures how much stiffness link i has given up on average and
``sigma_i`` how UNEVENLY it gave it up across the cases. ``KAPPA`` trades the
two. Keep it moderate (0.5 - 1): the sensitivity of ``mu + KAPPA sigma`` to a
case below the mean is ``w_k (1 + KAPPA (U~_k - mu)/sigma)``, which turns
negative -- rewarding a STIFF case for getting softer -- once
``(mu - U~_k)/sigma > 1/KAPPA``.

No stress or deformation-localisation constraint. Heat is now the constraint
and stiffness the objective, so cutting a link -- the failure a heat-minimising
bracket needs a localisation constraint for -- sends ``U~`` to infinity and
the objective forbids it on its own.

Weak (one-way) coupling: the temperature field is solved first and enters the
elastic problem as a thermal strain ``alpha (T - T_REF)``, stress free at
assembly (T_REF = T_HOT). The tank walls (CFRP) are assumed not to contract;
only the link does.

Material: G-10CR glass/epoxy, treated as ISOTROPIC here, constants below.

THE CONDUCTIVITY IS THE MEAN ``k_bar = int k dT / (T_HOT - T_COLD)``. For a
single material between two Dirichlet ends, the Kirchhoff transform
``theta = int k dT`` turns the nonlinear problem into Laplace's equation, so the
heat leak with ``k_bar`` is EXACT for any geometry; only the temperature
distribution in between (used for the thermal strain) is approximate.

SUPERPOSITION. Elasticity is linear, so per link only FOUR elastic solves are
needed, not one per load case:

    u_e_j   unit force e_j on the loaded face, no thermal strain   (j = x,y,z)
    u_th    thermal strain alone, no force
    u_k   = sum_j F[k, j] u_e_j + u_th                              (k = 1..N_CASES)

which is exact.

BATCHED SOLVES (fe.vmap, as in examples/advance/
batched_topology_optimization_shared_pcg_gcmma.py). One evaluation is two
batched solves: the D thermal problems, then the D x 4 elastic ones
(design-major, the traction carried as a surface variable). Each is CG
preconditioned by a cuDSS factorization of the mean operator of every group of
SHARE consecutive cases; with SHARE_TH = 1 and SHARE_ME = 4 every group holds
ONE operator, so the preconditioner is exact. The optimiser is the plain GCMMA
over all D x n_cells variables, NOT BatchedGCMMA: the heat budget couples the
designs, and BatchedGCMMA solves independent per-design subproblems.

FIXED END. ``FIX_MODE = "clamped"`` fixes ux = uy = uz = 0 over the whole face
-- a bonded / bolted fitting. It adds no thermal restraint stress: the face sits
at T_HOT = T_REF, where the thermal strain is zero, and the link's contraction
is free to happen at the loaded (inner-tank) end.

LOADS. Pass ``--loads loads.csv`` with columns ``link,case,Fx,Fy,Fz`` (N, link
local frame, force ACTING ON the inner-tank end, link 0..7, case 0..N_CASES-1). Without
it the link forces come from ``LOAD_FACTOR_TABLE`` (CS-25 load factors) split
statically determinately over the links (``factor_loads``). Cases with a +-
lateral factor are run both ways at half weight, so 7 rows give 10 cases.

Link local frame: z along the link from the outer shell (z = 0) to the inner
tank (z = LZ); x = aircraft longitudinal; y = z cross x.

Usage:
    python lh2_support_link_robust.py
    python lh2_support_link_robust.py --loads my_loads.csv --mass 5000 --qallow 10

Outputs in outputs/lh2_support_link_robust_<FIX_MODE>/: history.csv,
summary.csv (heat, mu, sigma and U~ for each of the 8 links), and per design
density.npy, final.vtu and links.txt in design<d>_<top|bottom|left>/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import time

# spineax keeps its last SPINEAX_FACTOR_CACHE factorizations on the GPU; the
# fe.vmap groups re-factorize one entry in turn, so one is enough. Must be set
# before spineax (feax) is imported.
os.environ.setdefault("SPINEAX_FACTOR_CACHE", "1")

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe
import feax.gene as gene
from feax.experimental.mma import GCMMA
from feax.utils import save_sol


jax.config.update("jax_enable_x64", True)

# ── Command line ───────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--loads", default=None, help="CSV: link,case,Fx,Fy,Fz [N]")
_ap.add_argument("--mass", type=float, default=5000.0, help="inner tank, kg")
_ap.add_argument("--qallow", type=float, default=10.0,
                 help="total heat budget of the 8 links, W")
ARGS, _ = _ap.parse_known_args()

# ── Domain ─────────────────────────────────────────────────────────────────
LX = LY = 50.0                     # mm, link cross-section
LZ = 100.0                         # mm, link length (outer shell -> inner tank)
ELEM_SIZE = 2.5                    # mm
PASSIVE_THICKNESS = 5.0            # mm, solid end fitting at each end

# ── Material: G-10CR, isotropic approximation. mm / N / MPa / K / W ────────
E_SOLID = 28.0e3                   # MPa, warp, room temperature
NU = 0.15
K_MEAN = 0.40e-3                   # W/(mm K), int k dT / dT over 20-293 K
T_HOT, T_COLD = 293.0, 20.0        # K, outer shell and inner tank
T_REF = T_HOT                      # stress free at assembly
CONTRACTION = 0.0024               # integrated dL/L, 293 K -> 20 K (warp)
ALPHA = CONTRACTION / (T_HOT - T_COLD)   # secant CTE, 1/K
E_MIN, K_MIN = 1e-6, 1e-4          # relative floors on the RAMP weights
RAMP_Q_E = 8.0                     # stiffness, q > 0
RAMP_Q_K = -0.8                    # conductivity, q < 0
HEAT_TO_WATT = 2.0 / (T_HOT - T_COLD)    # Q = 2 E / dT, two-terminal problem

LAM = E_SOLID * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_SOLID / (2.0 * (1.0 + NU))

# ── Links and load cases ───────────────────────────────────────────────────
INNER_MASS = ARGS.mass             # kg
G0 = 9.80665
LINKS = ("front_top", "front_bottom", "front_left", "front_right",
         "rear_top", "rear_bottom", "rear_left", "rear_right")
N_LINKS = len(LINKS)
# Load factors n = (n_x, n_y, n_z) on the tank, aircraft frame (X fwd, Y left,
# Z up). n_z = +1 is level flight. "pm" = the lateral factor acts both ways, so
# that case is split into its +n_y and -n_y variants at HALF weight each: every
# row of the table keeps the same total weight, and a left and a right link
# each see the side load in the direction that compresses and stretches them.
LOAD_FACTOR_TABLE = (
    # name                      n_x    n_y    n_z    pm     CS-25
    ("1 manoeuvre +2.5",        0.0,   0.0,   2.5,  False),  # 25.337
    ("2 manoeuvre -1.0",        0.0,   0.0,  -1.0,  False),  # 25.337
    ("3 gust lateral",          0.0,   1.33,  1.0,  True),   # 25.363 (adapted)
    ("4 ground turning",        0.0,   0.5,   1.0,  True),   # 25.495
    ("5 braking",              -0.8,   0.0,   1.2,  False),  # 25.493
    ("6 landing drag",         -0.6,   0.0,   2.5,  False),  # 25.479 (25% of 2.5)
    ("7 crosswind / yaw",       0.0,   1.0,   2.0,  True),   # placeholder, aircraft dependent
)
CASE_NAMES, CASE_N, _w = [], [], []
for _name, _nx, _ny, _nz, _pm in LOAD_FACTOR_TABLE:
    for _s in ((1.0, -1.0) if _pm else (1.0,)):
        CASE_NAMES.append(_name + ("" if not _pm else (" +y" if _s > 0 else " -y")))
        CASE_N.append((_nx, _s * _ny, _nz))
        _w.append(0.5 if _pm else 1.0)
CASE_N = onp.asarray(CASE_N)
N_CASES = len(CASE_N)                              # 10: 7 rows, 3 of them +-
CASE_WEIGHTS = onp.asarray(_w) / onp.sum(_w)       # each table row weighs 1/7

FIX_MODE = "clamped"               # "clamped" (u = 0 on the face) | "slider"

# ── Objective and constraint ───────────────────────────────────────────────
KAPPA = 1.0                        # weight on the spread over load cases
SIGMA_EPS = 1e-8                   # sqrt(var + eps): finite slope at sigma = 0
# Total heat budget of the 8 links, W (--qallow; None -> HEAT_FRACTION of 8
# solid blocks). 10 W is ~46% of 8 solid G-10CR blocks (~2.7 W each).
Q_ALLOW = ARGS.qallow
HEAT_FRACTION = 0.2
REF_FLOOR = 1e-3                   # U_ref_ik >= REF_FLOOR * max_k U_ref_ik
# A case in which a link carries (almost) no force -- a side link under a pure
# vertical load -- leaves only the thermal self-energy in U_ik, and U~_ik then
# measures nothing about stiffness while dominating sigma. Such cases get zero
# weight FOR THAT LINK (LINK_WEIGHTS, renormalised per link).
LOAD_IGNORE = 1e-3                 # |F_ik| < LOAD_IGNORE * max_k |F_ik| -> w_ik = 0

# ── Filter and continuation (as thermax) ───────────────────────────────────
MIN_WIDTH = 6.0                    # mm
FILTER_RADIUS = gene.radius_for_minimum_width(MIN_WIDTH, 0.7)
HEAVISIDE_BETAS = (2, 4, 8, 12, 16)
ITER_PER_STAGE = (6, 6, 10, 20)
assert len(ITER_PER_STAGE) == len(HEAVISIDE_BETAS) - 1
ITER_PER_STAGE_FINAL = 50
FINAL_WINDOW = 10
FINAL_FTOL_REL = 1e-3
FINAL_FEAS_TOL = 1e-3
MMA_MAX_INNER = 5
MMA_MOVE = None
MMA_C_CONSTRAINT = 1e7

# ── Batched solves (fe.vmap) ───────────────────────────────────────────────
# Every batch is CG preconditioned by fe.shared_factor: one cuDSS factorization
# of the MEAN operator of each group of `share` consecutive cases.
#   thermal  8 links, SHARE_TH = 1  -> each link its own factor (exact, CG ~1 it)
#   elastic  8 links x 4 sub-cases, SHARE_ME = 4 -> the 4 sub-cases of a link
#            have the SAME operator, so its mean is exact (CG ~1 it).
# Larger SHARE_ME (8, 16, 32) shares one factor across links: fewer
# factorizations but more CG iterations, since the 8 designs differ.
SHARE_TH = 1
SHARE_ME = 4
CHUNK_ME = 4                       # operators assembled at a time (memory)
CG_TOL = 1e-9
CG_MAXITER = 2000
X_INIT = 0.2
PRINT_EVERY = 1

OUTROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs",
                       f"lh2_support_link_robust_{FIX_MODE}")
os.makedirs(OUTROOT, exist_ok=True)


# ── Loads ──────────────────────────────────────────────────────────────────
def link_frame(name):
    """Rows: local x, y, z axes in aircraft coordinates (X fwd, Y left, Z up).
    Local z points from the outer shell to the inner tank."""
    ez = {"top": (0, 0, -1), "bottom": (0, 0, 1),
          "left": (0, -1, 0), "right": (0, 1, 0)}[name.split("_")[1]]
    ez = onp.asarray(ez, float)
    ex = onp.array([1.0, 0.0, 0.0])
    return onp.stack([ex, onp.cross(ez, ex), ez])


def factor_loads():
    """Link forces from the load-factor table, by a STATICALLY DETERMINATE split.

    Inertial force of the tank on the supports: F = -m g n (n_z = +2.5 pushes
    the tank DOWN with 2.5 m g). The tank sits midway between the two stations,
    so each station takes half, and within a station
        vertical  F_Z  -> top and bottom links, axially, F_Z / 4 each
        lateral   F_Y  -> left and right links, axially, F_Y / 4 each
        fore-aft  F_X  -> all 8 links in shear,  F_X / 8 each
    The clamped links are in fact indeterminate (a side link also takes some
    vertical shear, in proportion to its bending against the top/bottom links'
    axial stiffness); pass --loads for a proper split.
    """
    F = onp.zeros((N_LINKS, N_CASES, 3))
    for li, name in enumerate(LINKS):
        side = name.split("_")[1]
        R = link_frame(name)
        for k, n in enumerate(CASE_N):
            Fg = -INNER_MASS * G0 * n
            share = onp.array([Fg[0] / 8.0,
                               Fg[1] / 4.0 if side in ("left", "right") else 0.0,
                               Fg[2] / 4.0 if side in ("top", "bottom") else 0.0])
            F[li, k] = R @ share
    return F


def read_loads(path):
    F = onp.full((N_LINKS, N_CASES, 3), onp.nan)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            F[int(row["link"]), int(row["case"])] = (
                float(row["Fx"]), float(row["Fy"]), float(row["Fz"]))
    if onp.isnan(F).any():
        missing = onp.argwhere(onp.isnan(F[..., 0])).tolist()
        raise ValueError(f"{path}: missing (link, case) rows {missing}")
    return F


if ARGS.loads:
    LOADS = read_loads(ARGS.loads)
    LOAD_SOURCE = os.path.abspath(ARGS.loads)
else:
    LOADS = factor_loads()
    LOAD_SOURCE = "load-factor table, determinate split"

# ── Per-link case weights and symmetry groups ─────────────────────────────
_Fn = onp.linalg.norm(LOADS, axis=2)                       # (8, N)
LINK_WEIGHTS = CASE_WEIGHTS[None, :] * (_Fn >= LOAD_IGNORE * _Fn.max(axis=1, keepdims=True))
LINK_WEIGHTS /= LINK_WEIGHTS.sum(axis=1, keepdims=True)


def _signature(i):
    """The link's weighted load cases as an unordered set: two links with the
    same set get the same design (front = rear, left = right under +-y)."""
    scale = max(_Fn.max(), 1e-30)
    return tuple(sorted(tuple(onp.round(onp.r_[w, f / scale], 9))
                        for f, w in zip(LOADS[i], LINK_WEIGHTS[i]) if w > 0))


_groups = {}
for _i in range(N_LINKS):
    _groups.setdefault(_signature(_i), []).append(_i)
GROUPS = list(_groups.values())                    # link indices per design
REP = onp.array([g[0] for g in GROUPS])            # representative link
MULT = onp.array([len(g) for g in GROUPS], float)  # how many links it stands for
N_DES = len(GROUPS)
DESIGN_NAMES = ["+".join(LINKS[i] for i in g) for g in GROUPS]
LOADS_D = LOADS[REP]                               # (N_DES, N, 3)
WEIGHTS_D = LINK_WEIGHTS[REP]
DESIGN_OF = onp.empty(N_LINKS, int)
for _d, _g in enumerate(GROUPS):
    DESIGN_OF[_g] = _d


# ── Mesh and masks ─────────────────────────────────────────────────────────
TOL = 1e-6 * max(LX, LY, LZ)
fixed_face = lambda p: jnp.isclose(p[2], 0.0, atol=TOL)
loaded_face = lambda p: jnp.isclose(p[2], LZ, atol=TOL)
face_centre = lambda p: (jnp.isclose(p[0], 0.5 * LX, atol=TOL)
                         & jnp.isclose(p[1], 0.5 * LY, atol=TOL)
                         & jnp.isclose(p[2], 0.0, atol=TOL))
edge_mid = lambda p: (jnp.isclose(p[0], 0.5 * LX, atol=TOL)
                      & jnp.isclose(p[1], LY, atol=TOL)
                      & jnp.isclose(p[2], 0.0, atol=TOL))

mesh = fe.mesh.box_mesh(size=(LX, LY, LZ), mesh_size=ELEM_SIZE,
                        element_type="HEX8")
pts = onp.asarray(mesh.points)
cells = onp.asarray(mesh.cells)
n_nodes, n_cells = pts.shape[0], cells.shape[0]
centroids = pts[cells].mean(axis=1)
NX, NY, NZ = (len(onp.unique(pts[:, d])) - 1 for d in range(3))
if FIX_MODE == "slider" and NX % 2:
    raise ValueError("slider mode needs a node at the face centre: NX must be even")
A_FACE = LX * LY

passive_cell = jnp.asarray((centroids[:, 2] <= PASSIVE_THICKNESS + TOL)
                           | (centroids[:, 2] >= LZ - PASSIVE_THICKNESS - TOL))
design_cell = ~passive_cell
n_design = int(design_cell.sum())
n_total = N_DES * n_cells


# ── Problems ───────────────────────────────────────────────────────────────
def w_E(rho):
    """RAMP, q > 0: stiffness is wanted, so grey must sit below linear."""
    return E_MIN + (1.0 - E_MIN) * rho / (1.0 + RAMP_Q_E * (1.0 - rho))


def w_k(rho):
    """RAMP, -1 < q < 0: conductance is unwanted, so grey must sit ABOVE
    linear -- SIMP p > 1 would make grey the cheapest insulator."""
    return K_MIN + (1.0 - K_MIN) * rho / (1.0 + RAMP_Q_K * (1.0 - rho))


def elastic_strain(u_grad, T):
    return 0.5 * (u_grad + u_grad.T) - ALPHA * (T - T_REF) * jnp.eye(3)


def solid_stress(eps):
    return LAM * jnp.trace(eps) * jnp.eye(3) + 2.0 * MU * eps


class Conduction(fe.Problem):
    """psi = 1/2 k_bar w_k(rho) |grad T|^2; volume var rho (per cell)."""

    def get_energy_density(self):
        def psi(grad_T, rho):
            return 0.5 * K_MEAN * w_k(rho) * jnp.sum(grad_T ** 2)
        return psi


class ThermoElastic(fe.Problem):
    """psi = 1/2 w_E(rho) eps_el : C : eps_el, eps_el = sym grad u - alpha (T -
    T_REF) I; volume vars rho (per cell) and T (per node). The traction on the
    loaded face is a SURFACE variable (tx, ty, tz), so one problem serves every
    load case and the batch can carry it. feax adds the surface map to the
    residual: the map returns MINUS the traction."""

    def get_energy_density(self):
        def psi(u_grad, rho, T):
            eps = elastic_strain(u_grad, T)
            return 0.5 * w_E(rho) * jnp.sum(eps * solid_stress(eps))
        return psi

    def get_surface_maps(self):
        def traction(u, x, tx_, ty_, tz_):
            return jnp.array([tx_, ty_, tz_])
        return [traction]


def true_stress(u_grad, rho, T):
    """Material stress C : eps_el, for output only."""
    return solid_stress(elastic_strain(u_grad, T))

def create_vonmises_fn(problem, stress_at_quad):
    """Quadrature-point von Mises of a solved displacement, for output.

    ``f(u_flat, traced_params) -> (vm, JxW)``, both (num_cells, num_quads).
    Per-cell volume vars are broadcast, nodal ones interpolated."""
    fe0 = problem.fes[0]
    cells = problem.cells_list[0]
    sg, jxw, sval = fe0.shape_grads, fe0.JxW, fe0.shape_vals
    num_cells, num_quads = fe0.num_cells, fe0.num_quads

    def vm(sigma):
        dev = sigma - jnp.eye(3) * (jnp.trace(sigma) / 3.0)
        return jnp.sqrt(jnp.maximum(1.5 * jnp.sum(dev * dev), 1e-30))

    def vonmises(u_flat, traced_params):
        cell_u = jnp.asarray(u_flat).reshape(-1, 3)[cells]
        vars_q = [jnp.tile(v[:, None], (1, num_quads))
                  if v.ndim == 1 and v.shape[0] == num_cells
                  else jnp.einsum("qn,cn->cq", sval, v[cells])
                  for v in traced_params.volume_vars]

        def per_cell(cu, csg, *vq):
            u_grads = jnp.sum(cu[None, :, :, None] * csg[:, :, None, :], axis=1)
            return jax.vmap(lambda ug, *v: vm(stress_at_quad(ug, *v)))(u_grads, *vq)

        return jax.vmap(per_cell)(cell_u, sg, *vars_q), jxw

    return vonmises



_o = jnp.ones((n_cells,))
T_REF_NODES = jnp.full((n_nodes,), T_REF)
cg = fe.KrylovSolverOptions(solver="cg", tol=CG_TOL, atol=1e-14,
                            maxiter=CG_MAXITER)
cudss = fe.solvers.options.has_cudss()
if not cudss:
    print("!! no cuDSS: fe.vmap falls back to Jacobi-preconditioned CG (slow)")

# Thermal: one lhs batch over the N_DES designs.
thermal = Conduction(mesh, vec=1, dim=3, ele_type="HEX8")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(fixed_face, "all", T_HOT),       # outer shell
    fe.DirichletBCSpec(loaded_face, "all", T_COLD),     # inner tank
]).create_bc(thermal)
ts_th = fe.TracedStructure.from_problem(thermal)
# The single-case solvers are only the templates fe.vmap batches; built with
# CG options they hold no factorization of their own.
_solve_T = fe.create_solver(thermal, bc_th, cg, linear=True,
                            traced_structure=ts_th,
                            traced_params=fe.TracedParams(volume_vars=(_o,)))
solve_T_batch = fe.vmap(
    _solve_T, in_axes=fe.TracedParams(volume_vars=(0,)), options=cg,
    preconditioner=fe.shared_factor() if cudss else fe.jacobi(),
    chunk=SHARE_TH, share=SHARE_TH if cudss else None, return_info=True)
guess_T = fe.zero_like_initial_guess(thermal, bc_th)
# E = 1/2 int k |grad T|^2 = 1/2 G dT^2 for this two-terminal problem, so
# Q = G dT = 2 E / dT exactly (HEAT_TO_WATT).
_energy_T = fe.create_energy_fn(thermal)
conduction_energy = lambda T, tp: _energy_T(jnp.asarray(T), tp)

# Elastic: one lhs batch of N_DES designs x 4 sub-cases, design-major
#   [link0: e_x, e_y, e_z, thermal, link1: e_x, ...]
# so the 4 sub-cases of a link are consecutive and share one operator.
mech = ThermoElastic(mesh, vec=3, dim=3, ele_type="HEX8",
                     location_fns=(loaded_face,))
if FIX_MODE == "slider":
    _specs = [fe.DirichletBCSpec(fixed_face, "z", 0.0),
              fe.DirichletBCSpec(face_centre, "x", 0.0),
              fe.DirichletBCSpec(face_centre, "y", 0.0),
              fe.DirichletBCSpec(edge_mid, "x", 0.0)]
elif FIX_MODE == "clamped":
    _specs = [fe.DirichletBCSpec(fixed_face, "all", 0.0)]
else:
    raise ValueError(f"FIX_MODE {FIX_MODE!r}")
bc_me = fe.DirichletBCConfig(_specs).create_bc(mech)
ts_me = fe.TracedStructure.from_problem(mech)
N_SUB = 4                                          # e_x, e_y, e_z, thermal
N_ME = N_DES * N_SUB
_unit_sv = fe.TracedParams.create_uniform_surface_var(mech, 1.0)
# Unit FORCE (1 N resultant) per direction; the map returns minus the traction.
_t_sub = onp.zeros((N_SUB, 3))
_t_sub[:3] = -onp.eye(3) / A_FACE
_t_b = onp.tile(_t_sub, (N_DES, 1))                # (4 N_DES, 3)
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
    options=cg, preconditioner=fe.shared_factor() if cudss else fe.jacobi(),
    chunk=CHUNK_ME, share=SHARE_ME if cudss else None, return_info=True)
guess_u = fe.zero_like_initial_guess(mech, bc_me)
energy_fn = fe.create_energy_fn(mech)
vm_true = create_vonmises_fn(mech, true_stress)
print(f"  batched solves: thermal {solve_T_batch}")
print(f"                  elastic {solve_u_batch}")

# Assembly arrays, initial guesses and filter weights go in as ARGUMENTS,
# never closed over (closed over they are baked into every executable).
STRUCTURES = (ts_th, ts_me, guess_T, guess_u)
filter_rho = gene.create_density_filter(mesh, radius=FILTER_RADIUS,
                                        weight_type="cone", points=centroids)
FILTERS = (filter_rho,)


# ── Design field and states ────────────────────────────────────────────────
def process(xs, beta, filters):
    """Raw designs (8, n_cells) -> physical densities (passive bands solid)."""
    rho = gene.heaviside_projection(filters[0](xs), beta=beta, threshold=0.5)
    return jnp.where(passive_cell[None, :], 1.0, rho)


def solve_states(rho, F, structures):
    """All D = N_DES designs at once: temperature, then the superposed
    displacements of every case.

    rho (D, n_cells), F (D, N_CASES, 3). Returns Q (D,) [W], T (D, n_nodes),
    u (D, N_CASES, ndof), CG iterations (thermal (D,), elastic (4 D,))."""
    ts_th, ts_me, gT, gu = structures
    sol_T, info_T = solve_T_batch(fe.TracedParams(volume_vars=(rho,)), gT, ts_th)
    T = sol_T.dofs                                  # vec = 1: one dof per node
    Q = jax.vmap(lambda Ti, ri: conduction_energy(
        Ti, fe.TracedParams(volume_vars=(ri,))))(T, rho) * HEAT_TO_WATT
    # Sub-cases 0-2 carry no thermal strain (T = T_REF), sub-case 3 carries T.
    T_b = jnp.concatenate([jnp.broadcast_to(T_REF_NODES, (N_DES, 3, n_nodes)),
                           T[:, None, :]], axis=1).reshape(N_ME, n_nodes)
    tp = fe.TracedParams(volume_vars=(jnp.repeat(rho, N_SUB, axis=0), T_b),
                         surface_vars=[SURF_B])
    sol_u, info_u = solve_u_batch(tp, gu, ts_me)
    U = sol_u.dofs.reshape(N_DES, N_SUB, -1)
    u = jnp.einsum("ikj,ijn->ikn", F, U[:, :3]) + U[:, 3:4]
    return Q, T, u, (info_T.iterations, info_u.iterations)


def case_energies(rho, T, u):
    """Stored energy (thermal strain included) of every link and case, (8, N_CASES)."""
    def per_link(r, Ti, ui):
        tp = fe.TracedParams(volume_vars=(r, Ti))
        return jax.vmap(lambda uk: energy_fn(uk, tp))(ui)
    return jax.vmap(per_link)(rho, T, u)


@jax.jit
def reference(F, structures):
    """Solid blocks of all 8 links: heat [W] and raw case energies (8, N_CASES)."""
    rho = jnp.ones((N_DES, n_cells))
    Q, T, u, _its = solve_states(rho, F, structures)
    return Q, case_energies(rho, T, u)


def responses(x, beta, filters, structures, cdata):
    """cdata = (F (D,N,3), U_ref (D,N), w (D,N), mult (D,), heat_ref_W,
    q_allow) over the D = N_DES representative designs, all arrays,
    arrays, passed as arguments rather than closed over."""
    F, U_ref, w, mult, heat_ref_W, q_allow = cdata    # w: rows sum to 1
    rho = process(x.reshape(N_DES, n_cells), beta, filters)
    Q, T, u, its = solve_states(rho, F, structures)
    U = case_energies(rho, T, u) / U_ref                       # (8, N_CASES)
    mu = jnp.sum(w * U, axis=1)
    sig = jnp.sqrt(jnp.sum(w * (U - mu[:, None]) ** 2, axis=1) + SIGMA_EPS)
    # Every design stands for MULT links: weight the mean and count the heat
    # of all 8.
    J = jnp.sum(mult * (mu + KAPPA * sig)) / jnp.sum(mult)
    c = jnp.sum(mult * Q) / q_allow - 1.0
    aux = dict(Q=Q, mu=mu, sigma=sig, U=U,
               volume=(rho * design_cell).sum(axis=1) / n_design,
               heat_ratio=Q / heat_ref_W, cg_T=its[0], cg_u=its[1])
    return J, c, aux


@jax.jit
def value_and_grad(x, beta, filters, structures, cdata):
    """One forward pass, two backward passes (objective and heat budget)."""
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
    """Per link: density, temperature, the displacements of every case, cell von Mises."""
    rho = process(x.reshape(N_DES, n_cells), beta, filters)
    _Q, T, u, _its = solve_states(rho, cdata[0], structures)

    def per_link(r, Ti, ui):
        tp = fe.TracedParams(volume_vars=(r, Ti))
        return jax.vmap(lambda uk: vm_true(uk, tp)[0].mean(axis=1))(ui)
    return rho, T, u, jax.vmap(per_link)(rho, T, u)


# ── References and budget ──────────────────────────────────────────────────
Q_ref, U_raw = (onp.asarray(a) for a in reference(jnp.asarray(LOADS_D), STRUCTURES))
U_REF = onp.maximum(U_raw, REF_FLOOR * U_raw.max(axis=1, keepdims=True))
HEAT_REF_W = float(Q_ref[0])                     # one solid link; same for all
if Q_ALLOW is None:
    Q_ALLOW = HEAT_FRACTION * N_LINKS * HEAT_REF_W
CDATA = (jnp.asarray(LOADS_D), jnp.asarray(U_REF), jnp.asarray(WEIGHTS_D), jnp.asarray(MULT),
         jnp.float64(HEAT_REF_W), jnp.float64(Q_ALLOW))

print(f"LH2 inner-tank support links   {N_LINKS} links -> {N_DES} designs "
      f"by symmetry, {LX:g} x {LY:g} x {LZ:g} mm, {ELEM_SIZE:g} mm HEX8, "
      f"{n_cells:,} cells each, {n_total:,} design variables")
print(f"  G-10CR isotropic E={E_SOLID/1e3:g} GPa nu={NU:g} k_bar={K_MEAN*1e3:g} "
      f"W/mK alpha={ALPHA:.2e}/K;  inner tank {INNER_MASS:g} kg, "
      f"{N_CASES} cases, loads: {LOAD_SOURCE}")
print(f"  fixed end {FIX_MODE};  minimise mean(mu + {KAPPA:g} sigma)  "
      f"s.t. sum over 8 links Q <= {Q_ALLOW:.3f} W "
      f"({Q_ALLOW / (N_LINKS * HEAT_REF_W):.2f} x 8 solid links, "
      f"solid link {HEAT_REF_W:.3f} W)")
print("  cases: " + ", ".join(f"{k}={n} (w {w:.3f})" for k, (n, w)
                            in enumerate(zip(CASE_NAMES, CASE_WEIGHTS))))
print("  axial force F_z [kN] (+ tension) / shear |F_x, F_y| [kN] per case:")
for i, name in enumerate(LINKS):
    print(f"  link {i} {name:13s} "
          + " ".join(f"{f[2]/1e3:+6.1f}/{onp.hypot(f[0], f[1])/1e3:4.1f}"
                     for f in LOADS[i])
          + f"   ({int((LINK_WEIGHTS[i] > 0).sum())} active, "
          f"design {DESIGN_OF[i]})")
for d, name in enumerate(DESIGN_NAMES):
    print(f"  design {d}: {name}  (x{MULT[d]:g})")
print("-" * 78)

# ── GCMMA ──────────────────────────────────────────────────────────────────
opt = GCMMA(value_and_grad, onp.zeros(n_total), onp.ones(n_total),
            n_constraints=1, value_fn=values_only, has_aux=True,
            c=onp.array([MMA_C_CONSTRAINT]),
            max_inner=MMA_MAX_INNER, move=MMA_MOVE)
step = jax.jit(opt.step)

hist = collections.defaultdict(list)
it = 0


def record(state, beta):
    global it
    it += 1
    a = state.aux
    Q = onp.asarray(a["Q"])
    mu, sig = onp.asarray(a["mu"]), onp.asarray(a["sigma"])
    q_tot = float(MULT @ Q)                        # all 8 links
    row = dict(iteration=it, beta=beta, objective=float(state.f0),
               Q_total=q_tot, Q_ratio=q_tot / Q_ALLOW)
    for d in range(N_DES):
        row[f"Q{d}"], row[f"mu{d}"], row[f"sigma{d}"] = Q[d], mu[d], sig[d]
        row[f"V{d}"] = float(a["volume"][d])
    for key, val in row.items():
        hist[key].append(float(val))
    if it % PRINT_EVERY == 0 or it == 1:
        print(f"  it {it:4d} b={beta:4.1f} J={row['objective']:.4f} "
              f"Q8={q_tot:.3f}/{Q_ALLOW:.3f} W  "
              f"Q=[" + " ".join(f"{q:5.3f}" for q in Q) + "] "
              f"mu=[" + " ".join(f"{m:5.2f}" for m in mu) + "] "
              f"sig=[" + " ".join(f"{s:4.2f}" for s in sig) + "] "
              f"in={int(state.inner_iterations)} "
              f"cg[T {int(onp.max(a['cg_T']))} u {int(onp.max(a['cg_u']))}]")
    return row["objective"], row["Q_ratio"] - 1.0


x = jnp.full((n_total,), X_INIT)
t0 = time.perf_counter()
state = None
for stage, beta in enumerate(HEAVISIDE_BETAS):
    beta_t = jnp.float64(beta)
    final = stage == len(HEAVISIDE_BETAS) - 1
    cap = ITER_PER_STAGE_FINAL if final else ITER_PER_STAGE[stage]
    window = collections.deque(maxlen=FINAL_WINDOW)
    print(f"  -- stage {stage+1}/{len(HEAVISIDE_BETAS)}: beta = {beta:g} --")
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
                    print(f"     converged at it {it} (spread {spread:.1e})")
                    break
    x = state.x
dt = time.perf_counter() - t0

# ── Results ────────────────────────────────────────────────────────────────
a = state.aux
Q = onp.asarray(a["Q"])
mu, sig = onp.asarray(a["mu"]), onp.asarray(a["sigma"])
U = onp.asarray(a["U"])
V = onp.asarray(a["volume"])
rho, T, u, vm = (onp.asarray(v) for v in final_fields(
    x, jnp.float64(HEAVISIDE_BETAS[-1]), FILTERS, STRUCTURES, CDATA))
vmax = onp.array([float((vm[d] * rho[d][None, :]).max()) for d in range(N_DES)])
Q8 = float(MULT @ Q)

print("-" * 78)
print(f"  {'design':34s} {'x':>2s} {'Q [W]':>7s} {'share':>6s} {'V':>6s} "
      f"{'mu':>6s} {'sigma':>6s} {'vm_max':>8s}   U~ per case (- = inactive)")
for d, name in enumerate(DESIGN_NAMES):
    print(f"  {d} {name:32s} {MULT[d]:2g} {Q[d]:7.4f} "
          f"{MULT[d]*Q[d]/Q8*100:5.1f}% {V[d]:6.3f} {mu[d]:6.2f} {sig[d]:6.2f} "
          f"{vmax[d]:6.1f}MPa   "
          + " ".join(f"{v:5.2f}" if w > 0 else "  -  "
                     for v, w in zip(U[d], WEIGHTS_D[d])))
print(f"  total heat leak of the 8 links {Q8:.4f} W  (budget {Q_ALLOW:.4f} W, "
      f"8 solid links {N_LINKS * HEAT_REF_W:.4f} W)")
print(f"  objective {float(state.f0):.4f}   evaluations {it}   wall {dt:.0f} s")

with open(os.path.join(OUTROOT, "history.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    keys = list(hist)
    wr.writerow(keys)
    wr.writerows(zip(*(hist[k] for k in keys)))
with open(os.path.join(OUTROOT, "summary.csv"), "w", newline="") as fh:
    # One row per PHYSICAL link (8), each pointing at its design.
    wr = csv.writer(fh)
    wr.writerow(["link", "name", "design", "Q_W", "volume", "mu", "sigma",
                 "vm_max_MPa"]
                + [f"U{k}" for k in range(N_CASES)]
                + [f"F{k}_{c}" for k in range(N_CASES) for c in "xyz"])
    for i, name in enumerate(LINKS):
        d = DESIGN_OF[i]
        wr.writerow([i, name, d, Q[d], V[d], mu[d], sig[d], vmax[d]]
                    + [U[d, k] if WEIGHTS_D[d, k] > 0 else "" for k in range(N_CASES)]
                    + list(LOADS[i].ravel()))
    wr.writerow(["total", "", "", Q8, "", "", "", ""])

_hits = onp.bincount(cells.ravel(), minlength=n_nodes).astype(float)
for d, name in enumerate(DESIGN_NAMES):
    outdir = os.path.join(OUTROOT, f"design{d}_{LINKS[REP[d]].split('_')[1]}")
    os.makedirs(outdir, exist_ok=True)
    onp.save(os.path.join(outdir, "density.npy"), rho[d].astype(onp.float32))
    with open(os.path.join(outdir, "links.txt"), "w") as fh:
        fh.write(name.replace("+", "\n") + "\n")
    try:
        rho_n = onp.bincount(cells.ravel(),
                             weights=onp.repeat(rho[d], cells.shape[1]),
                             minlength=n_nodes) / onp.maximum(_hits, 1.0)
        points = [("density", rho_n), ("temperature", T[d])]
        points += [(f"u_case{k}", u[d, k].reshape(-1, 3) * rho_n[:, None])
                   for k in range(N_CASES)]
        cell_out = [("rho", rho[d])] + [(f"vm_case{k}", vm[d, k] * rho[d])
                                        for k in range(N_CASES)]
        save_sol(mesh, os.path.join(outdir, "final.vtu"),
                 point_infos=points, cell_infos=cell_out)
    except Exception as e:                        # output only
        print(f"  (no vtu for design {d}: {e})")
print(f"wrote history.csv / summary.csv / design*/ to "
      f"{os.path.relpath(OUTROOT, os.getcwd())}/")
