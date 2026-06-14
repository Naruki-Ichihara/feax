"""
2D Anisotropic Topology Optimization (MBB beam, half-symmetry).

Simultaneous optimization of density and fiber orientation for 2D orthotropic
materials, ported from libertas (https://github.com/Naruki-Ichihara/libertas)
problem.py to feax.

Design variables (node-based, 4 fields):
    rho        : density (SIMP-penalized material fraction)
    x1, x2, x3 : orientation parameters consumed by
                 :func:`feax.mechanics.orientation.orientation_tensor_2d`,
                 producing the 2-D Advani–Tucker orientation tensor ``a_2``.

The orthotropic in-plane stiffness ``C_ijkl(a_2, a_4)`` is built via the
quadratic Advani–Tucker closure (``a_4 = a_2 ⊗ a_2``) and the helper
:func:`feax.mechanics.orientation.orientation_averaged_stiffness`.

Optimizer: NLopt LD_MMA on a single packed vector [rho; x1; x2; x3] with
per-variable bounds (rho in [0,1], x_i in [-1,1]).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
import nlopt

import feax as fe
import feax.gene as gene
from feax.mechanics.orientation import (
    orientation_tensor_2d,
    principal_direction,
    quadratic_closure,
    orientation_averaged_stiffness,
)


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
MAX_ITER = 100
# Heaviside projection sharpness (constant; no continuation).
HEAVISIDE_BETA = 4.0

# Bounds for x3 (off-diagonal sign control). Use [0, 1] when the problem
# has a12-sign symmetry to remove the spurious branch.
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


# ── Anisotropic elasticity Problem ──────────────────────────────────────────

class AnisotropicElasticity(fe.Problem):
    """2-D plane-stress elasticity with quad-point density and 2-D orientation.

    Material at each quadrature point is built via the libertas-style
    box-bounded design variables ``(rho, x1, x2, x3)``:

    1. ``a_2 = orientation_tensor_2d(x1, x2, x3, sgn_beta=SGN_BETA)``
    2. ``a_4 = quadratic_closure(a_2)``
    3. ``C   = orientation_averaged_stiffness(a_2, a_4, E1, E2, G12, NU12)``
    4. SIMP-penalised weight ``simp(rho) · ||(a11, a22)||``.
    """

    def get_tensor_map(self):
        def stress(u_grad, rho, x1, x2, x3):
            a2, a11, a22 = orientation_tensor_2d(x1, x2, x3, sgn_beta=SGN_BETA)
            a4 = quadratic_closure(a2)
            C = orientation_averaged_stiffness(a2, a4, E1=E1, E2=E2, G12=G12, nu12=NU12)
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

    ts = fe.TracedStructure.from_problem(problem)

    # Filters and response functions
    initial_guess = fe.zero_like_initial_guess(problem, bc)
    compliance_fn = gene.create_compliance_fn(problem)
    volume_fn = gene.create_volume_fn(problem)
    filter_rho = gene.create_helmholtz_filter(mesh, radius=FILTER_RHO)
    filter_theta = gene.create_helmholtz_filter(mesh, radius=FILTER_THETA)

    # Solver pre-warming with sample internal vars
    sample_iv = fe.TracedParams(
        volume_vars=(
            fe.TracedParams.create_node_var(problem, TARGET_VF),
            fe.TracedParams.create_node_var(problem, 0.0),
            fe.TracedParams.create_node_var(problem, 0.0),
            fe.TracedParams.create_node_var(problem, 0.0),
        ),
        surface_vars=(),
    )
    solver_options = fe.DirectSolverOptions()
    solver = fe.create_solver(
        problem, bc=bc,
        solver_options=solver_options,
        adjoint_solver_options=solver_options,
        linear=True, traced_params=sample_iv,
        traced_structure=ts,
    )

    n_nodes = mesh.points.shape[0]

    def _process(x_flat, beta):
        rho, x1, x2, x3 = unpack(x_flat, n_nodes)
        rho_f = filter_rho(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        x1_f = filter_theta(x1)
        x2_f = filter_theta(x2)
        x3_f = filter_theta(x3)
        return rho_f, rho_p, x1_f, x2_f, x3_f

    def objective_fn(x_flat, beta):
        _, rho_p, x1_f, x2_f, x3_f = _process(x_flat, beta)
        tp = fe.TracedParams(
            volume_vars=(rho_p, x1_f, x2_f, x3_f),
            surface_vars=(),
        )
        sol = solver(tp, initial_guess, traced_structure=ts)
        return compliance_fn(sol)

    def volume_fn_packed(x_flat, beta):
        # Use *filtered* rho (no Heaviside projection) for the volume
        # constraint, matching libertas. Mixing the projection into the
        # constraint causes MMA's surrogate to chase a moving target as
        # beta is annealed.
        rho_f, _, _, _, _ = _process(x_flat, beta)
        return volume_fn(rho_f)

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

    # Initial design: rho = target volume fraction; orientation at the
    # "no fiber" corner (-1+tol, -1+tol, 0), matching libertas. The
    # ORI_FLOOR term in the SIMP weight keeps the FE system non-singular
    # while MMA grows the alignment magnitude from scratch.
    ORI_TOL = 1e-2
    x0 = onp.empty(n_total)
    x0[0 * n_nodes:1 * n_nodes] = TARGET_VF
    x0[1 * n_nodes:2 * n_nodes] = -1.0 + ORI_TOL
    x0[2 * n_nodes:3 * n_nodes] = -1.0 + ORI_TOL
    x0[3 * n_nodes:4 * n_nodes] = 0.0

    print(f"Design vars : {n_total} (rho + 3 x orientation, on {n_nodes} nodes)")
    print(f"Max iter    : {MAX_ITER}")
    print(f"Heaviside b : {HEAVISIDE_BETA} (constant)")
    print("-" * 60)

    history = {"iteration": [], "objective": [], "volume": []}
    iter_count = [0]
    beta = np.float64(HEAVISIDE_BETA)

    a2_vmap = jax.jit(jax.vmap(
        lambda a, b, c: orientation_tensor_2d(a, b, c, sgn_beta=SGN_BETA)[0]
    ))

    def _snapshot_fields(x_flat):
        """Return the (name, data) field list for one iteration."""
        rho, x1, x2, x3 = unpack(np.array(x_flat), n_nodes)
        rho_f = onp.array(filter_rho(rho))
        x1_f = filter_theta(x1)
        x2_f = filter_theta(x2)
        x3_f = filter_theta(x3)
        a2_nodes = onp.array(a2_vmap(x1_f, x2_f, x3_f))   # (n, 2, 2)
        # Principal fibre direction at every node, padded with z=0 so
        # ParaView's Glyph filter (which expects 3-D vectors) can render
        # it directly. ``scale_by_alignment=True`` ⇒ length encodes the
        # leading eigenvalue (= alignment magnitude), so isotropic /
        # void regions get short glyphs.
        director_2d = onp.asarray(principal_direction(a2_nodes))
        director_3d = onp.column_stack(
            [director_2d, onp.zeros(director_2d.shape[0])]
        )
        return [
            ("density", rho_f),
            ("a11", a2_nodes[:, 0, 0]),
            ("a22", a2_nodes[:, 1, 1]),
            ("a12", a2_nodes[:, 0, 1]),
            ("director", director_3d),
        ]

    # One XDMF/HDF5 time-series file per optimisation run.  Mesh is
    # stored once, every iteration is appended as a frame so ParaView's
    # Time toolbar can scrub through the design history.
    xdmf_path = OUTPUT_DIR / "history.xdmf"

    with fe.XDMFWriter(mesh, xdmf_path) as xw:

        # Frame 0: initial design (so the time series always has a
        # baseline regardless of how few iterations are run).
        xw.write_iteration(0, point_infos=_snapshot_fields(x0))

        def _objective(xx, grad):
            x_jax = np.array(xx)
            val, g = obj_and_grad(x_jax, beta)
            if grad.size > 0:
                grad[:] = onp.array(g)
            v = float(volume_fn_packed(x_jax, beta))
            iter_count[0] += 1
            history["iteration"].append(iter_count[0])
            history["objective"].append(float(val))
            history["volume"].append(v)
            print(
                f"Iter {iter_count[0]:4d}: obj={float(val):.4e}  vol={v:.4f}"
            )
            xw.write_iteration(
                iter_count[0], point_infos=_snapshot_fields(xx)
            )
            return float(val)

        def _volume_constraint(xx, grad):
            x_jax = np.array(xx)
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

        opt.optimize(x0)
        print(f"\nOptimization done. final obj = {opt.last_optimum_value():.6e}")

    # History CSV
    import csv
    with open(OUTPUT_DIR / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "objective", "volume"])
        for i in range(len(history["iteration"])):
            w.writerow([
                history["iteration"][i],
                history["objective"][i],
                history["volume"][i],
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
