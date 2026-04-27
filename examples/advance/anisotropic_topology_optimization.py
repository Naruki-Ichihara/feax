"""
2D Anisotropic Topology Optimization (MBB beam, half-symmetry).

Simultaneous optimization of density and fiber orientation for 2D orthotropic
materials, ported from libertas (https://github.com/Naruki-Ichihara/libertas)
problem.py to feax.

Design variables (node-based, 4 fields):
    rho        : density (SIMP-penalized material fraction)
    x1, x2, x3 : orientation parameters
        (x1, x2)  -> 8-node isoparametric box-to-triangle map -> (a11, a22)
        x3        -> sign of off-diagonal coupling
                     a12 = sqrt(a11 * a22) * tanh(beta_sgn * x3) / tanh(beta_sgn)
    The orientation tensor is a2 = [[a11, a12], [a12, a22]] (rank-1 closure
    a4 = a2 (x) a2). The orthotropic stiffness tensor is built via the
    Advani-Tucker closure.

Stacking sequence (laminate through-thickness layers) is supported:
    Each layer's stiffness tensor evaluated at theta + phi_k is averaged with
    user-supplied thickness weights w_k. theta itself is encoded in a2 (and
    rotated by phi_k via a 2D rotation).

Optimizer: NLopt LD_MMA on a single packed vector [rho; x1; x2; x3] with
per-variable bounds (rho in [0,1], x_i in [-1,1]).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as np
import numpy as onp
import nlopt

import feax as fe
import feax.gene as gene


# ── Problem configuration ────────────────────────────────────────────────────

# MBB beam (half-domain by vertical-axis symmetry).
# domain [0, LX] x [0, LY] with:
#   - left  edge (x = 0)        : symmetry, u_x = 0
#   - right-bottom corner       : roller,   u_y = 0
#   - top-left load patch       : downward traction
LX, LY = 60.0, 20.0
NX, NY = 180, 60

# CFRP-like orthotropic material constants
E1, E2 = 70.0e3, 5.0e3
G12 = 5.0e3
NU12 = 0.3

# SIMP penalty
SIMP_PENALTY = 3.0
SIMP_EPSILON = 1e-3

# Helmholtz filter radii
FILTER_RHO = 1.5
FILTER_THETA = 1.5

# Smoothed-sign sharpness for off-diagonal coupling
SGN_BETA = 10.0

# Volume fraction target
TARGET_VF = 0.4

# Optimization
MAX_ITER = 200
BETA_INITIAL, BETA_FINAL, BETA_STEP, BETA_UPDATE = 1.0, 16.0, 2.0, 25

# Stacking sequence (offsets in degrees relative to optimized fiber direction).
# None = single layer. Examples:
#   STACKING_DEG = [0.0, 90.0]                # cross-ply
#   STACKING_DEG = [0.0, 45.0, -45.0, 90.0]   # quasi-isotropic
STACKING_DEG: Optional[List[float]] = None
STACKING_WEIGHTS: Optional[List[float]] = None  # None = equal thickness

# When STACKING_DEG implies symmetry about a12 (e.g. [0, 90]), the optimum is
# insensitive to the sign of x3. Set X3_LB to 0.0 to remove the spurious branch.
X3_LB, X3_UB = -1.0, 1.0

# Loading
TRACTION_MAG = 1.0
LOAD_PATCH_FRACTION = 0.05  # length of load patch / LX, applied at top-left

OUTPUT_DIR = Path(__file__).with_name("output_anisotropic_topology_optimization")


# ── Boundary condition predicates ───────────────────────────────────────────

TOL = 1e-5 * max(LX, LY)
left_edge = lambda pt: np.isclose(pt[0], 0.0, atol=TOL)
right_bottom_corner = (
    lambda pt: np.isclose(pt[0], LX, atol=TOL) & np.isclose(pt[1], 0.0, atol=TOL)
)
load_patch = (
    lambda pt: np.isclose(pt[1], LY, atol=TOL) & (pt[0] <= LOAD_PATCH_FRACTION * LX + TOL)
)


# ── Orientation parameterization (libertas isoparametric_2D_box_to_triangle) ─

# Boundary node values (a11, a22) at the 8 serendipity nodes of [-1,1]^2:
#   N1: (-1,-1)  N2: (0,-1)  N3: (1,-1)  N4: (1,0)
#   N5: ( 1, 1)  N6: (0, 1)  N7: (-1,1)  N8: (-1,0)
TRI_TOL = 1e-6
_U = np.array([TRI_TOL, 0.5, 1.0, 0.75, 0.5, 0.25, TRI_TOL, TRI_TOL])
_V = np.array([TRI_TOL, TRI_TOL, TRI_TOL, 0.25, 0.5, 0.75, 1.0, 0.5])


def _serendipity_8(z, e):
    N1 = -(1 - z) * (1 - e) * (1 + z + e) / 4
    N2 = (1 - z**2) * (1 - e) / 2
    N3 = -(1 + z) * (1 - e) * (1 - z + e) / 4
    N4 = (1 + z) * (1 - e**2) / 2
    N5 = -(1 + z) * (1 + e) * (1 - z - e) / 4
    N6 = (1 - z**2) * (1 + e) / 2
    N7 = -(1 - z) * (1 + e) * (1 + z - e) / 4
    N8 = (1 - z) * (1 - e**2) / 2
    return np.stack([N1, N2, N3, N4, N5, N6, N7, N8])


def box_to_triangle(x1, x2):
    """Map (x1, x2) in [-1,1]^2 onto the triangle {a11, a22 >= 0, a11 + a22 <= 1}."""
    N = _serendipity_8(x1, x2)
    a11 = np.sum(_U * N)
    a22 = np.sum(_V * N)
    return a11, a22


def smooth_sgn(x, beta=SGN_BETA):
    return np.tanh(beta * x) / np.tanh(beta)


def orientation_tensor(x1, x2, x3):
    """Build symmetric orientation tensor a2 from 3 design parameters."""
    a11, a22 = box_to_triangle(x1, x2)
    # Clamp to avoid sqrt of (slightly) negative values from tolerance bleed.
    a12 = np.sqrt(np.maximum(a11 * a22, 0.0)) * smooth_sgn(x3)
    a2 = np.array([[a11, a12], [a12, a22]])
    return a2, a11, a22


# ── Orthotropic stiffness (Advani-Tucker closure) ───────────────────────────

def orthotropic_stiffness(a2, a4, E1=E1, E2=E2, G12=G12, nu12=NU12):
    """4th-order in-plane orthotropic stiffness from 2nd/4th orientation tensors."""
    nu21 = nu12 * E2 / E1
    m = 1.0 / (1.0 - nu12 * nu21)
    C1111 = m * E1
    C1122 = m * nu21 * E1
    C2222 = m * E2
    C1212 = G12

    B1 = C1111 + C2222 - 2 * C1122 - 4 * C1212
    B2 = C1122
    B3 = C1212 - C2222 / 2
    B5 = C2222 / 2

    delta = np.eye(2)
    c1 = B1 * a4
    c2 = B2 * (
        np.einsum("ij,kl->ijkl", a2, delta) + np.einsum("kl,ij->ijkl", a2, delta)
    )
    c3 = B3 * (
        np.einsum("ik,jl->ijkl", a2, delta)
        + np.einsum("il,jk->ijkl", a2, delta)
        + np.einsum("jk,il->ijkl", a2, delta)
        + np.einsum("jl,ik->ijkl", a2, delta)
    )
    c5 = B5 * (
        np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    )
    return c1 + c2 + c3 + c5


def rotate_tensor_2(a2, phi):
    """Rotate 2nd-order tensor a2 by angle phi (radians) in-plane."""
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    return R @ a2 @ R.T


# Pre-compute stacking constants
if STACKING_DEG is not None:
    _STACK_PHI = np.array([d * math.pi / 180.0 for d in STACKING_DEG])
    if STACKING_WEIGHTS is not None:
        assert len(STACKING_WEIGHTS) == len(STACKING_DEG)
        _w = np.array(STACKING_WEIGHTS, dtype=np.float64)
    else:
        _w = np.ones(len(STACKING_DEG))
    _STACK_W = _w / np.sum(_w)
else:
    _STACK_PHI = None
    _STACK_W = None


def effective_stiffness(a2, a4):
    """Single-layer or stacking-averaged stiffness tensor."""
    if _STACK_PHI is None:
        return orthotropic_stiffness(a2, a4)

    def _layer(phi, w):
        a2r = rotate_tensor_2(a2, phi)
        a4r = np.einsum("ij,kl->ijkl", a2r, a2r)
        return w * orthotropic_stiffness(a2r, a4r)

    Cs = jax.vmap(_layer)(_STACK_PHI, _STACK_W)
    return np.sum(Cs, axis=0)


# ── Anisotropic elasticity Problem ──────────────────────────────────────────

class AnisotropicElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho, x1, x2, x3):
            a2, a11, a22 = orientation_tensor(x1, x2, x3)
            a4 = np.einsum("ij,kl->ijkl", a2, a2)
            C = effective_stiffness(a2, a4)
            simp = SIMP_EPSILON + (1.0 - SIMP_EPSILON) * rho**SIMP_PENALTY
            mag = np.sqrt(a11**2 + a22**2)
            weight = simp * mag
            strain = 0.5 * (u_grad + u_grad.T)
            return weight * np.einsum("ijkl,kl->ij", C, strain)
        return stress

    def get_surface_maps(self):
        def traction(u, x, *args):
            return np.array([0.0, -TRACTION_MAG])
        return [traction]


# ── Pack / unpack helper ────────────────────────────────────────────────────

def unpack(x_flat: np.ndarray, n_nodes: int):
    """Split flat optimizer vector into (rho, x1, x2, x3) of length n_nodes each."""
    return (
        x_flat[0 * n_nodes:1 * n_nodes],
        x_flat[1 * n_nodes:2 * n_nodes],
        x_flat[2 * n_nodes:3 * n_nodes],
        x_flat[3 * n_nodes:4 * n_nodes],
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Mesh
    mesh = fe.mesh.rectangle_mesh(
        Nx=NX, Ny=NY, domain_x=LX, domain_y=LY, ele_type="QUAD4"
    )
    print(f"Mesh: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

    # Problem
    problem = AnisotropicElasticity(
        mesh=mesh, vec=2, dim=2, ele_type="QUAD4",
        location_fns=(load_patch,),
    )

    # Boundary conditions: symmetry + roller
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left_edge, component=0, value=0.0),
        fe.DirichletBCSpec(location=right_bottom_corner, component=1, value=0.0),
    ]).create_bc(problem)

    # Filters and response functions
    initial_guess = fe.zero_like_initial_guess(problem, bc)
    compliance_fn = gene.create_compliance_fn(problem)
    volume_fn = gene.create_volume_fn(problem)
    filter_rho = gene.create_helmholtz_filter(mesh, radius=FILTER_RHO)
    filter_theta = gene.create_helmholtz_filter(mesh, radius=FILTER_THETA)

    # Solver pre-warming with sample internal vars
    sample_iv = fe.InternalVars(
        volume_vars=(
            fe.InternalVars.create_node_var(problem, TARGET_VF),
            fe.InternalVars.create_node_var(problem, 0.0),
            fe.InternalVars.create_node_var(problem, 0.0),
            fe.InternalVars.create_node_var(problem, 0.0),
        ),
        surface_vars=(),
    )
    solver_options = fe.DirectSolverOptions()
    solver = fe.create_solver(
        problem, bc=bc,
        solver_options=solver_options,
        adjoint_solver_options=solver_options,
        iter_num=1, internal_vars=sample_iv,
    )

    n_nodes = mesh.points.shape[0]

    def _process(x_flat, beta):
        rho, x1, x2, x3 = unpack(x_flat, n_nodes)
        rho_f = filter_rho(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        x1_f = filter_theta(x1)
        x2_f = filter_theta(x2)
        x3_f = filter_theta(x3)
        return rho_p, x1_f, x2_f, x3_f

    def objective_fn(x_flat, beta):
        rho_p, x1_f, x2_f, x3_f = _process(x_flat, beta)
        iv = fe.InternalVars(
            volume_vars=(rho_p, x1_f, x2_f, x3_f),
            surface_vars=(),
        )
        sol = solver(iv, initial_guess)
        return compliance_fn(sol)

    def volume_fn_packed(x_flat, beta):
        rho_p, _, _, _ = _process(x_flat, beta)
        return volume_fn(rho_p)

    obj_and_grad = jax.jit(jax.value_and_grad(objective_fn))
    vol_and_grad = jax.jit(jax.value_and_grad(volume_fn_packed))

    # Bounds: [rho; x1; x2; x3]  (rho in [0,1], x_i in [-1,1])
    n_total = 4 * n_nodes
    lower = onp.empty(n_total)
    upper = onp.empty(n_total)
    lower[0:n_nodes], upper[0:n_nodes] = 0.0, 1.0
    lower[n_nodes:2 * n_nodes], upper[n_nodes:2 * n_nodes] = -1.0, 1.0
    lower[2 * n_nodes:3 * n_nodes], upper[2 * n_nodes:3 * n_nodes] = -1.0, 1.0
    lower[3 * n_nodes:4 * n_nodes], upper[3 * n_nodes:4 * n_nodes] = X3_LB, X3_UB

    # Initial design: rho = target volume fraction; orientation near "no fiber"
    # corner (-1+tol, -1+tol, 0) so the optimizer can grow alignment from scratch.
    ORI_TOL = 1e-2
    x0 = onp.empty(n_total)
    x0[0 * n_nodes:1 * n_nodes] = TARGET_VF
    x0[1 * n_nodes:2 * n_nodes] = -1.0 + ORI_TOL
    x0[2 * n_nodes:3 * n_nodes] = -1.0 + ORI_TOL
    x0[3 * n_nodes:4 * n_nodes] = 0.0

    print(f"Design vars : {n_total} (rho + 3 x orientation, on {n_nodes} nodes)")
    print(f"Stacking    : {STACKING_DEG if STACKING_DEG else 'single layer'}")
    print(f"Max iter    : {MAX_ITER}")
    print(f"beta sched  : {BETA_INITIAL} -> {BETA_FINAL} (+{BETA_STEP} every {BETA_UPDATE} iter)")
    print("-" * 60)

    history = {"iteration": [], "objective": [], "volume": [], "beta": []}
    iter_count = [0]
    beta_state = [BETA_INITIAL]

    a2_vmap = jax.jit(jax.vmap(lambda a, b, c: orientation_tensor(a, b, c)[0]))

    def _save_snapshot(x_flat, tag):
        rho, x1, x2, x3 = unpack(np.array(x_flat), n_nodes)
        rho_f = onp.array(filter_rho(rho))
        x1_f = filter_theta(x1)
        x2_f = filter_theta(x2)
        x3_f = filter_theta(x3)
        a2_nodes = onp.array(a2_vmap(x1_f, x2_f, x3_f))
        fe.utils.save_sol(
            mesh,
            str(OUTPUT_DIR / f"iter_{tag}.vtu"),
            point_infos=[
                ("density", rho_f),
                ("a11", a2_nodes[:, 0, 0]),
                ("a22", a2_nodes[:, 1, 1]),
                ("a12", a2_nodes[:, 0, 1]),
            ],
        )

    def _objective(xx, grad):
        x_jax = np.array(xx)
        beta = np.float64(beta_state[0])
        val, g = obj_and_grad(x_jax, beta)
        if grad.size > 0:
            grad[:] = onp.array(g)
        v = float(volume_fn_packed(x_jax, beta))
        iter_count[0] += 1
        history["iteration"].append(iter_count[0])
        history["objective"].append(float(val))
        history["volume"].append(v)
        history["beta"].append(float(beta_state[0]))
        print(
            f"Iter {iter_count[0]:4d}: obj={float(val):.4e}  "
            f"vol={v:.4f}  beta={beta_state[0]:.2f}"
        )
        if iter_count[0] % 10 == 0:
            _save_snapshot(xx, f"{iter_count[0]:04d}")
        if iter_count[0] % BETA_UPDATE == 0 and beta_state[0] < BETA_FINAL:
            beta_state[0] = min(beta_state[0] + BETA_STEP, BETA_FINAL)
            print(f"  >>> beta = {beta_state[0]:.2f}")
        return float(val)

    def _volume_constraint(xx, grad):
        x_jax = np.array(xx)
        beta = np.float64(beta_state[0])
        v, g = vol_and_grad(x_jax, beta)
        if grad.size > 0:
            grad[:] = onp.array(g)
        return float(v) - TARGET_VF

    opt = nlopt.opt(nlopt.LD_MMA, n_total)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(_objective)
    opt.add_inequality_constraint(_volume_constraint, 1e-6)
    opt.set_maxeval(MAX_ITER)
    opt.set_ftol_rel(1e-5)

    x_opt = opt.optimize(x0)
    print(f"\nOptimization done. final obj = {opt.last_optimum_value():.6e}")

    _save_snapshot(x_opt, "final")

    # History CSV
    import csv
    with open(OUTPUT_DIR / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "objective", "volume", "beta"])
        for i in range(len(history["iteration"])):
            w.writerow([
                history["iteration"][i],
                history["objective"][i],
                history["volume"][i],
                history["beta"][i],
            ])

    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].semilogy(history["iteration"], history["objective"], "b-")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Compliance")
        axes[0].set_title("Compliance history")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["iteration"], history["volume"], "g-")
        axes[1].axhline(y=TARGET_VF, color="r", linestyle="--", label=f"target={TARGET_VF}")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Volume fraction")
        axes[1].set_title("Volume fraction history")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "history.png", dpi=150)
        plt.close()
        print(f"Saved plots and VTU snapshots to {OUTPUT_DIR}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
