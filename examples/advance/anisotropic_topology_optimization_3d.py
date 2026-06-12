"""
3D Anisotropic Topology Optimization (cantilever beam).

3-D extension of ``anisotropic_topology_optimization.py``: simultaneous
optimisation of density and 3-D fibre orientation tensor on a TET4
cantilever box, using the geometry / BCs / mesh of
``topology_optimization_adaptive.py``.

Design variables (node-based, 7 fields):

    rho                     : density (SIMP-penalised material fraction)
    x1, x2, x3              : diagonal-alignment (libertas tetrahedron map)
    x12, x13, x23           : off-diagonal sign controls (smooth-sgn)

The 3-D Advani-Tucker orientation tensor ``a_2`` is built via
:func:`feax.mechanics.orientation.orientation_tensor_3d` and the
3-D transversely-isotropic stiffness ``C_ijkl(a_2, a_4)`` via
:func:`feax.mechanics.orientation.orientation_averaged_stiffness_3d`
(quadratic closure ``a_4 = a_2 \\otimes a_2``).

Physical setup (= ``topology_optimization_adaptive.py``):

    * Box ``[0, L] x [0, W] x [0, H]`` = ``100 x 5 x 20``
    * Left face ``x = 0`` clamped (all 3 components)
    * Downward traction ``-z`` applied on the lower-quarter of the
      right face (the patch where ``x = L`` and ``z < H/4``)

Optimiser: NLopt LD_MMA on a single packed vector
``[rho; x1; x2; x3; x12; x13; x23]`` with per-variable bounds
(``rho`` in ``[0, 1]``, all ``x_*`` in ``[-1, 1]``).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
import nlopt
import gmsh

import feax as fe
import feax.gene as gene
from feax.mechanics.orientation import (
    orientation_tensor_3d,
    principal_direction,
    quadratic_closure,
    orientation_averaged_stiffness_3d,
)


# ── Geometry & physics (matches topology_optimization_adaptive.py) ─────────

L, W, H = 100.0, 5.0, 20.0
MESH_SIZE = 1.0           # initial element size for gmsh

# Transversely-isotropic CFRP-like lamina (axis 1 = fibre):
E1, E2 = 70.0e3, 5.0e3
G12 = 5.0e3
NU12 = 0.30
NU23 = 0.40              # transverse Poisson ratio (3-D only)

# SIMP penalty
SIMP_PENALTY = 3.0
SIMP_EPSILON = 1e-3

# Helmholtz filter radii (in mesh units)
FILTER_RHO = 3.0
FILTER_THETA = 3.0

# Smoothed-sign sharpness for off-diagonal libertas controls
SGN_BETA = 10.0

# Volume fraction target (constant, matching adaptive example)
TARGET_VF = 0.4

# Heaviside projection sharpness (constant — no continuation here for
# simplicity; the adaptive example anneals β via its Pipeline driver).
HEAVISIDE_BETA = 4.0

# Loading
TRACTION_MAG = 1.0

# Optimisation budget
MAX_ITER = 100

OUTPUT_DIR = Path(__file__).with_name("output_anisotropic_topology_optimization_3d")


# ── Boundary predicates (3-D) ───────────────────────────────────────────────

TOL = 1e-3
left_face = lambda pt: np.isclose(pt[0], 0.0, atol=TOL)
load_patch_3d = (
    lambda pt: np.isclose(pt[0], L, atol=TOL) & (pt[2] < H / 4.0)
)


# ── Anisotropic 3-D elasticity Problem ──────────────────────────────────────

class AnisotropicElasticity3D(fe.Problem):
    """3-D elasticity with quad-point density and 3-D orientation tensor.

    Material at each quadrature point is built via the libertas-style
    box-bounded design variables ``(rho, x1, x2, x3, x12, x13, x23)``:

    1. ``a_2 = orientation_tensor_3d(x1, x2, x3, x12, x13, x23, sgn_beta)``
    2. ``a_4 = quadratic_closure(a_2)``
    3. ``C   = orientation_averaged_stiffness_3d(a_2, a_4, E1, E2, G12, NU12, NU23)``
    4. SIMP-penalised weight ``simp(rho) * sqrt(a11**2 + a22**2 + a33**2)``.
       The alignment-magnitude factor — present in the 2-D libertas
       formulation as well — naturally suppresses spurious "isotropic
       full-density" filler material.
    """

    def get_tensor_map(self):
        def stress(u_grad,
                   rho, x1, x2, x3, x12, x13, x23):
            a2, a11, a22, a33 = orientation_tensor_3d(
                x1, x2, x3, x12, x13, x23, sgn_beta=SGN_BETA,
            )
            a4 = quadratic_closure(a2)
            C = orientation_averaged_stiffness_3d(
                a2, a4,
                E1=E1, E2=E2, G12=G12, nu12=NU12, nu23=NU23,
            )
            simp = SIMP_EPSILON + (1.0 - SIMP_EPSILON) * rho ** SIMP_PENALTY
            mag = np.sqrt(a11 ** 2 + a22 ** 2 + a33 ** 2)
            weight = simp * mag
            strain = 0.5 * (u_grad + u_grad.T)
            return weight * np.einsum("ijkl,kl->ij", C, strain)
        return stress

    def get_surface_maps(self):
        def traction(u, x, *args):
            return np.array([0.0, 0.0, -TRACTION_MAG])
        return [traction]


# ── Gmsh cantilever geometry (matches topology_optimization_adaptive.py) ───

def cantilever_geometry():
    """Build the 100 x 5 x 20 box via Gmsh OCC (called by adaptive_mesh)."""
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()


# ── Pack / unpack helper ────────────────────────────────────────────────────

# Field ordering: rho, x1, x2, x3, x12, x13, x23
N_FIELDS = 7


def unpack(x_flat: np.ndarray, n_nodes: int):
    """Split flat optimiser vector into the 7 nodal fields."""
    return tuple(
        x_flat[k * n_nodes:(k + 1) * n_nodes] for k in range(N_FIELDS)
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Mesh: Gmsh TET4 (no adaptive remeshing for this multi-field run) ──
    print("Generating initial TET4 mesh ...")
    mesh = gene.adaptive.adaptive_mesh(cantilever_geometry, h_max=MESH_SIZE)
    print(f"  {mesh.points.shape[0]} nodes, "
          f"{mesh.cells.shape[0]} {mesh.ele_type} cells")

    # ── Problem ──────────────────────────────────────────────────────────
    problem = AnisotropicElasticity3D(
        mesh=mesh, vec=3, dim=3, ele_type=mesh.ele_type,
        location_fns=(load_patch_3d,),
    )

    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=left_face, component="all", value=0.0),
    ]).create_bc(problem)

    # ── Filters & response functions ─────────────────────────────────────
    initial_guess = fe.zero_like_initial_guess(problem, bc)
    compliance_fn = gene.create_compliance_fn(problem)
    volume_fn = gene.create_volume_fn(problem)
    filter_rho = gene.create_helmholtz_filter(mesh, radius=FILTER_RHO)
    filter_theta = gene.create_helmholtz_filter(mesh, radius=FILTER_THETA)

    # Pre-warm solver with sample InternalVars matching 7 fields.
    sample_iv = fe.InternalVars(
        volume_vars=tuple(
            fe.InternalVars.create_node_var(problem, v) for v in (
                TARGET_VF,                  # rho
                -1.0 + 1e-2,                # x1
                -1.0 + 1e-2,                # x2
                -1.0 + 1e-2,                # x3
                0.0,                        # x12
                0.0,                        # x13
                0.0,                        # x23
            )
        ),
        surface_vars=(),
    )
    solver_opts = fe.DirectSolverOptions()
    solver = fe.create_solver(
        problem, bc=bc,
        solver_options=solver_opts,
        adjoint_solver_options=solver_opts,
        linear=True, internal_vars=sample_iv,
    )

    n_nodes = mesh.points.shape[0]

    def _process(x_flat, beta):
        rho, x1, x2, x3, x12, x13, x23 = unpack(x_flat, n_nodes)
        rho_f = filter_rho(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        return (
            rho_f, rho_p,
            filter_theta(x1), filter_theta(x2), filter_theta(x3),
            filter_theta(x12), filter_theta(x13), filter_theta(x23),
        )

    def objective_fn(x_flat, beta):
        proc = _process(x_flat, beta)
        rho_p = proc[1]
        oris = proc[2:]                # (x1, x2, x3, x12, x13, x23)
        iv = fe.InternalVars(
            volume_vars=(rho_p, *oris),
            surface_vars=(),
        )
        sol = solver(iv, initial_guess)
        return compliance_fn(sol)

    def volume_fn_packed(x_flat, beta):
        # Constraint on filtered (un-projected) ρ — matches libertas;
        # mixing the projection in confuses MMA's surrogate.
        rho_f = _process(x_flat, beta)[0]
        return volume_fn(rho_f)

    obj_and_grad = jax.jit(jax.value_and_grad(objective_fn))
    vol_and_grad = jax.jit(jax.value_and_grad(volume_fn_packed))

    # ── Bounds for the packed vector [rho; x1..x3; x12..x23] ─────────────
    n_total = N_FIELDS * n_nodes
    lower = onp.empty(n_total)
    upper = onp.empty(n_total)
    # rho
    lower[0:n_nodes], upper[0:n_nodes] = 0.0, 1.0
    # x1, x2, x3, x12, x13, x23 — all on [-1, 1]
    for k in range(1, N_FIELDS):
        lower[k * n_nodes:(k + 1) * n_nodes] = -1.0
        upper[k * n_nodes:(k + 1) * n_nodes] = +1.0

    # ── Initial design ───────────────────────────────────────────────────
    # rho = volume target.  Diagonal libertas params start at the
    # "no fibre" tetrahedron vertex (-1+ε, -1+ε, -1+ε); off-diagonal
    # signs at 0 (smooth-sgn(0) = 0 → rank-1 cleanly along x at the
    # vertex limit).  ε = 1e-2 keeps autodiff gradients well-defined.
    ORI_TOL = 1e-2
    init_per_field = (
        TARGET_VF,                # rho
        -1.0 + ORI_TOL,           # x1
        -1.0 + ORI_TOL,           # x2
        -1.0 + ORI_TOL,           # x3
        0.0,                      # x12
        0.0,                      # x13
        0.0,                      # x23
    )
    x0 = onp.empty(n_total)
    for k, v in enumerate(init_per_field):
        x0[k * n_nodes:(k + 1) * n_nodes] = v

    print(f"Design vars : {n_total} ({N_FIELDS} fields x {n_nodes} nodes)")
    print(f"  fields    : rho, x1, x2, x3, x12, x13, x23")
    print(f"Volume tgt  : {TARGET_VF}")
    print(f"Heaviside β : {HEAVISIDE_BETA}")
    print(f"Max iter    : {MAX_ITER}")
    print("-" * 60)

    history = {"iteration": [], "objective": [], "volume": []}
    iter_count = [0]
    beta = np.float64(HEAVISIDE_BETA)

    a2_vmap = jax.jit(jax.vmap(
        lambda *p: orientation_tensor_3d(*p, sgn_beta=SGN_BETA)[0]
    ))

    def _snapshot_fields(x_arr):
        x_jax = np.array(x_arr)
        rho, x1, x2, x3, x12, x13, x23 = unpack(x_jax, n_nodes)
        rho_f = onp.array(filter_rho(rho))
        x1f = filter_theta(x1); x2f = filter_theta(x2); x3f = filter_theta(x3)
        x12f = filter_theta(x12); x13f = filter_theta(x13); x23f = filter_theta(x23)
        a2_nodes = onp.array(a2_vmap(x1f, x2f, x3f, x12f, x13f, x23f))  # (n, 3, 3)
        # Two flavours of director — both as (n, 3) vector fields:
        #   * ``director``       — length = largest eigenvalue λ_max ∈ [0, 1]
        #                          (ParaView Glyph: scale-by-vector-magnitude
        #                          shows alignment strength).
        #   * ``director_unit``  — unit length, pure direction (good for
        #                          orientation streamlines / clean glyphs).
        director       = onp.asarray(
            principal_direction(a2_nodes, scale_by_alignment=True)
        )
        director_unit  = onp.asarray(
            principal_direction(a2_nodes, scale_by_alignment=False)
        )
        # Full a_2 as a 3x3 tensor field (n, 9) for ParaView Tensor Glyph
        # (ellipsoid visualisation of orientation).  Row-major Cartesian
        # ordering: (a11, a12, a13, a21, a22, a23, a31, a32, a33).
        a2_tensor = a2_nodes.reshape(-1, 9)
        return [
            ("density",            rho_f),
            ("director",           director),
            ("director_unit",      director_unit),
            ("orientation_tensor", a2_tensor),
            ("a11",                a2_nodes[:, 0, 0]),
            ("a22",                a2_nodes[:, 1, 1]),
            ("a33",                a2_nodes[:, 2, 2]),
            ("a12",                a2_nodes[:, 0, 1]),
            ("a13",                a2_nodes[:, 0, 2]),
            ("a23",                a2_nodes[:, 1, 2]),
        ]

    xdmf_path = OUTPUT_DIR / "history.xdmf"
    with fe.XDMFWriter(mesh, xdmf_path) as xw:
        # Frame 0: baseline
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
            print(f"Iter {iter_count[0]:4d}: obj={float(val):.4e}  vol={v:.4f}")
            xw.write_iteration(iter_count[0], point_infos=_snapshot_fields(xx))
            return float(val)

        def _volume_constraint(xx, grad):
            x_jax = np.array(xx)
            v, g = vol_and_grad(x_jax, beta)
            if grad.size > 0:
                grad[:] = onp.array(g)
            return float(v) - TARGET_VF

        opt = nlopt.opt(nlopt.LD_MMA, n_total)
        opt.set_lower_bounds(lower); opt.set_upper_bounds(upper)
        opt.set_min_objective(_objective)
        opt.add_inequality_constraint(_volume_constraint, 1e-6)
        opt.set_maxeval(MAX_ITER)
        opt.set_ftol_rel(1e-5)

        opt.optimize(x0)
        print(f"\nOptimization done. final obj = {opt.last_optimum_value():.6e}")

    # ── History CSV + plot ───────────────────────────────────────────────
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

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].semilogy(history["iteration"], history["objective"], "b-")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Compliance")
        axes[0].set_title("Compliance history (3-D anisotropic)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(history["iteration"], history["volume"], "g-")
        axes[1].axhline(y=TARGET_VF, color="r", linestyle="--",
                        label=f"target = {TARGET_VF}")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Volume fraction")
        axes[1].set_title("Volume fraction history")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "history.png", dpi=150)
        plt.close()
        print(f"Saved plot + XDMF time series to {OUTPUT_DIR}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
