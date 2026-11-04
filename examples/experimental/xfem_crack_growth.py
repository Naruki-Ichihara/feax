"""Quasi-static crack growth to rupture with XFEM: SIF-based direction and
load, fixed mesh, no re-JIT — with a load-displacement curve and an animation.

An edge crack in a plate under mixed-mode fixed-grip loading (the top edge is
displaced by ``λ (U_SHEAR, U_TENSION)``, the bottom edge is held; the load
factor ``λ`` is found from the fracture criterion) grows cell by cell until
it breaks the ligament:

.. code-block:: text

    for each step:
        solve for the current polyline crack at the reference grips  (one jitted solve)
        J-vector by autodiff of a virtual crack extension           (two jitted grads)
        K_I, K_II from (J_t, J_n);  kink angle θc by maximum hoop stress
        K_eff = K_θθ(θc);  critical load factor λ = K_Ic / K_eff   (LEFM is linear)
        record (λ·u, λ·F) — one point of the load-displacement curve
        extend the polyline by one cell in direction θc

The mesh, the enriched spaces and the compiled solver never change: the
crack geometry enters only through traced arrays, see
``feax.experimental.xfem_crack``.  The J domain radius shrinks as the tip
approaches a free boundary (the virtual extension must not move boundary
nodes); when the ligament is shorter than two cells the crack is simply
extended to the edge (final rupture) and the force drops to zero.

The first step kinks by about -18 deg (the shear gives K_II > 0); afterwards
the crack follows a locally mode-I path (K_II ~ 0).  Under displacement
control the critical grip displacement first decreases with the crack length
(unstable growth: a snap-back in the load-displacement curve) — this is
what the curve shows.

Output: ``xfem_crack_growth.png`` (final field, crack path, load-displacement
curve, SIF history) and ``xfem_crack_growth.gif`` (one frame per step:
deformed plate at the critical load + the curve so far).  ``--no-gif`` skips
the animation.

ParaView output (``output/xfem_crack_growth/``), written with the standard
feax pipeline (:func:`feax.utils.save_sol`, one VTU per step, indexed by a
``.pvd`` collection because the visualization mesh changes with the crack):

* ``field_XXXX.vtu`` / ``field.pvd`` — the sub-cell visualization mesh of
  :meth:`CrackSpace.visualization` (every sub-triangle has its own vertices,
  so the opening is a real gap in the mesh): point data ``u`` (critical
  load), ``phi``, ``psi``; cell data ``class`` (0 uncut / 1 Heaviside / 2
  tip), ``stress`` (3x3), ``mises``, ``load_factor``, ``K_I``, ``K_II``.
  Apply *Warp By Vector* on ``u`` to see the deformed, opened crack.
* ``crack_XXXX.vtu`` / ``crack.pvd`` — the crack polyline (``LINE2`` cells).
* ``history.csv`` — step, tip, K_I, K_II, kink, load factor, u, F.

Run:  python examples/experimental/xfem_crack_growth.py [--no-gif]
"""
import csv
import os
import sys

import jax
import jax.numpy as np
import numpy as onp

import feax as fe
from feax.experimental.xfem_crack import CrackSpace, crack_problem, plane_strain_energy

jax.config.update("jax_enable_x64", True)

W, H = 1.0, 2.0                 # plate: x in [0, W], y in [-H/2, H/2]
NX, NY = 24, 49                 # odd rows: y = 0 is not a node line
E, NU = 100.0, 0.3
E_PRIME = E / (1 - NU ** 2)     # plane strain
U_SHEAR, U_TENSION = 0.01, 0.005   # reference grip displacement (load factor 1)
K_IC = 0.3                      # fracture toughness
A0 = 0.25                       # initial edge crack length (from x = 0 along y = 0)
MAX_STEPS = 40
MAGNIFY = 8.0                   # displacement magnification in the plots
OUT_DIR = os.path.join("output", "xfem_crack_growth")


def stress_from_eps(eps):
    """Plane-strain stress (n, 3, 3) and von Mises (n,) from strains (n, 2, 2)."""
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    tr = onp.trace(eps, axis1=1, axis2=2)
    sig2 = lam * tr[:, None, None] * onp.eye(2) + 2 * mu * eps
    s_zz = NU * (sig2[:, 0, 0] + sig2[:, 1, 1])
    sig = onp.zeros((len(eps), 3, 3))
    sig[:, :2, :2] = sig2
    sig[:, 2, 2] = s_zz
    mises = onp.sqrt(0.5 * ((sig2[:, 0, 0] - sig2[:, 1, 1]) ** 2 + (sig2[:, 1, 1] - s_zz) ** 2
                            + (s_zz - sig2[:, 0, 0]) ** 2) + 3 * sig2[:, 0, 1] ** 2)
    return sig, mises


def mises_from_eps(eps):
    return stress_from_eps(eps)[1]


def write_step_vtu(space, problem, sol, poly, n, step, lam, KI, KII):
    """One frame of the ParaView series: sub-cell field mesh + crack polyline."""
    vis = space.visualization(problem, sol, poly, n)
    sig, mises = stress_from_eps(vis["eps"])
    n_t = len(vis["cells"])
    field = os.path.join(OUT_DIR, f"field_{step:04d}.vtu")
    fe.utils.save_sol(
        fe.Mesh(vis["points"], vis["cells"], ele_type="TRI3"), field,
        point_infos=[("u", onp.c_[lam * vis["u"], onp.zeros(len(vis["u"]))]),
                     ("phi", vis["phi"]), ("psi", vis["psi"])],
        cell_infos=[("class", vis["class"].astype(onp.float64)), ("stress", lam * sig),
                    ("mises", lam * mises), ("load_factor", onp.full(n_t, lam)),
                    ("K_I", onp.full(n_t, KI)), ("K_II", onp.full(n_t, KII))])
    crack = os.path.join(OUT_DIR, f"crack_{step:04d}.vtu")
    fe.utils.save_sol(space.polyline_mesh(poly, n), crack,
                      cell_infos=[("segment", onp.arange(int(n) - 1, dtype=onp.float64))])
    return field, crack


def main():
    quad = fe.mesh.rectangle_mesh(NX, NY, W, H, origin=(0.0, -H / 2), ele_type="QUAD4")
    c = onp.asarray(quad.cells)
    mesh = fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]),
                   ele_type="TRI3")
    pts = onp.asarray(mesh.points)
    h = W / NX
    space = CrackSpace(mesh, max_points=MAX_STEPS + 4)
    problem = crack_problem(space, plane_strain_energy(E, NU))
    print(f"mesh: {space.num_nodes} nodes, {space.num_cells} TRI3 cells, DOFs {problem.num_total_dofs_all_vars}, "
          f"nq_max {space.nq_max}")

    tol = 1e-8
    bot = lambda p: np.isclose(p[1], -H / 2, tol)
    top = lambda p: np.isclose(p[1], H / 2, tol)
    bc = space.create_bc(problem, fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=bot, component="all", value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=top, component=0, value=U_SHEAR, variable_index=0),
        fe.DirichletBCSpec(location=top, component=1, value=U_TENSION, variable_index=0),
    ]))
    ts = fe.TracedStructure.from_problem(problem)
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    solver = fe.create_solver(problem, bc, solver_options=direct, linear=True, traced_structure=ts)
    J = space.j_vector_fn(problem, solver, bc, ts, radius=4 * h)
    u0 = fe.zero_like_initial_guess(problem, bc)
    top_nodes = onp.flatnonzero(onp.isclose(pts[:, 1], H / 2))

    poly, n = space.polyline([(-0.1, 0.0), (A0, 0.0)])
    history, frames, vtu_frames = [], [], []
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n step   tip (x, y)        K_I      K_II    kink[deg]   lambda    u_y        F_y")
    for step in range(MAX_STEPS):
        tip, prev = onp.asarray(poly[n - 1]), onp.asarray(poly[n - 2])
        dist_edge = min(W - tip[0], H / 2 - abs(tip[1]))
        if dist_edge < 2.0 * h:                      # ligament gone: final rupture
            print(f"  ligament shorter than 2 cells at x = {tip[0]:.3f}: rupture")
            theta = onp.arctan2(tip[1] - prev[1], tip[0] - prev[0])
            poly, n = space.grow(poly, n, theta, dist_edge + 0.5 * h)
            tp = space.traced_params(poly, n)
            sol = solver(tp, u0, traced_structure=ts)
            x_q, u_q, eps_q = (onp.asarray(a) for a in space.quad_fields(problem, sol, tp))
            real = onp.asarray(tp.volume_vars[6]) > 0
            lam = history[-1][4]
            frames.append((onp.asarray(poly[:n]).copy(), x_q[real] + MAGNIFY * lam * u_q[real],
                           lam * mises_from_eps(eps_q[real]), lam, 0.0, 0.0))
            history.append((tip.copy(), onp.nan, onp.nan, 0.0, lam, 0.0, 0.0))
            vtu_frames.append(write_step_vtu(space, problem, sol, poly, n, step, lam, onp.nan, onp.nan))
            break

        tp = space.traced_params(poly, n)
        sol = solver(tp, u0, traced_structure=ts)
        radius = min(4 * h, dist_edge - 0.5 * h)
        theta, KI, KII = space.growth_direction(J, poly, n, E_PRIME, radius=radius)
        t_dir = (tip - prev) / onp.linalg.norm(tip - prev)
        kink = theta - onp.arctan2(t_dir[1], t_dir[0])
        K_eff = float(space.hoop_stress_intensity(KI, KII, kink))
        lam = K_IC / K_eff                            # LEFM: fields scale linearly with the grips
        res = onp.asarray(fe.get_res(problem, problem.unflatten_fn_sol_list(sol), tp, ts)[0])
        Fx, Fy = res[top_nodes].sum(axis=0) * lam
        x_q, u_q, eps_q = (onp.asarray(a) for a in space.quad_fields(problem, sol, tp))
        real = onp.asarray(tp.volume_vars[6]) > 0
        history.append((tip.copy(), KI, KII, kink, lam, Fx, Fy))
        frames.append((onp.asarray(poly[:n]).copy(), x_q[real] + MAGNIFY * lam * u_q[real],
                       lam * mises_from_eps(eps_q[real]), lam, Fx, Fy))
        print(f"  {step:3d}   ({tip[0]:.3f}, {tip[1]: .3f})   {KI:.4f}   {KII: .4f}   {onp.rad2deg(kink): 7.1f}"
              f"   {lam:6.3f}   {lam * U_TENSION:.5f}   {Fy:.4f}")
        vtu_frames.append(write_step_vtu(space, problem, sol, poly, n, step, lam, KI, KII))
        poly, n = space.grow(poly, n, theta, h)

    fe.utils.write_pvd(os.path.join(OUT_DIR, "field.pvd"), [(k, f) for k, (f, _) in enumerate(vtu_frames)])
    fe.utils.write_pvd(os.path.join(OUT_DIR, "crack.pvd"), [(k, c) for k, (_, c) in enumerate(vtu_frames)])
    with open(os.path.join(OUT_DIR, "history.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "tip_x", "tip_y", "K_I", "K_II", "kink_deg", "load_factor", "u_x", "u_y", "F_x", "F_y"])
        for k, (tip_k, KI, KII, kink, lam, Fx, Fy_k) in enumerate(history):
            wr.writerow([k, tip_k[0], tip_k[1], KI, KII, onp.rad2deg(kink), lam,
                         lam * U_SHEAR, lam * U_TENSION, Fx, Fy_k])
    print(f"\nParaView series: {OUT_DIR}/field.pvd, {OUT_DIR}/crack.pvd ({len(vtu_frames)} frames); history.csv")

    lam_hist = onp.array([r[4] for r in history])
    uy = lam_hist * U_TENSION
    Fy = onp.array([r[6] for r in history])
    path = onp.array([r[0] for r in history])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def draw_curve(ax, k=None):
        kk = len(uy) if k is None else k + 1
        ax.plot([0, uy[0]], [0, Fy[0]], "-", color="lightgray", lw=1)         # elastic loading of the initial crack
        ax.plot(uy[:kk], Fy[:kk], "o-", color="tab:red", ms=4, label="crack growth (K_eff = K_Ic)")
        if k is None or kk == len(uy):
            ax.plot(uy, Fy, "o-", color="tab:red", ms=4)
        ax.set_xlim(0, uy.max() * 1.1); ax.set_ylim(0, onp.nanmax(Fy) * 1.15)
        ax.set_xlabel("grip displacement u_y (u_x = 2 u_y)"); ax.set_ylabel("top-edge reaction F_y")
        ax.set_title("load - displacement")

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    pl, xd, mis, *_ = frames[-2]
    vmax = float(onp.percentile(mis, 99))
    sc = ax[0, 0].scatter(xd[:, 0], xd[:, 1], c=mis, s=5, cmap="viridis", vmax=vmax)
    ax[0, 0].plot(pl[:, 0], pl[:, 1], "r-", lw=2, label="crack")
    ax[0, 0].set_aspect("equal"); ax[0, 0].set_title("last step before rupture: von Mises at the critical load")
    ax[0, 0].legend(loc="lower right", fontsize=8); fig.colorbar(sc, ax=ax[0, 0], shrink=0.8)
    ax[0, 1].plot(path[:, 0], path[:, 1], "r.-"); ax[0, 1].set_xlim(0, W); ax[0, 1].set_ylim(-H / 2, H / 2)
    ax[0, 1].set_aspect("equal"); ax[0, 1].set_title("crack path (tip positions)")
    draw_curve(ax[1, 0]); ax[1, 0].legend(fontsize=8)
    steps = onp.arange(len(history))
    ax[1, 1].plot(steps, [r[1] for r in history], "o-", label="K_I (at λ = 1)")
    ax[1, 1].plot(steps, [r[2] for r in history], "s-", label="K_II (at λ = 1)")
    ax[1, 1].plot(steps, lam_hist, "^--", color="gray", label="load factor λ")
    ax[1, 1].set_xlabel("growth step"); ax[1, 1].legend(fontsize=8); ax[1, 1].set_title("SIFs and critical load factor")
    fig.tight_layout()
    fig.savefig("xfem_crack_growth.png", dpi=150)
    print("\nsaved xfem_crack_growth.png")

    if "--no-gif" in sys.argv:
        return
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    vmax = max(float(onp.percentile(m, 99)) for _, _, m, *_ in frames)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [0.9, 1.1]},
                                 constrained_layout=True)
    sc = a1.scatter(frames[0][1][:, 0], frames[0][1][:, 1], c=frames[0][2], s=6, cmap="viridis",
                    norm=Normalize(0, vmax))
    (line,) = a1.plot([], [], "r-", lw=2.5)
    a1.set_xlim(-0.1, W + 0.3); a1.set_ylim(-H / 2 - 0.15, H / 2 + 0.3); a1.set_aspect("equal")
    fig.colorbar(sc, ax=a1, shrink=0.8, pad=0.02, label="von Mises stress (critical load)")
    title = a1.set_title("")
    draw_curve(a2, 0)
    (pt,) = a2.plot([], [], "o", color="k", ms=8)

    def draw(k):
        pl, xd, m, lam, _, fy = frames[k]
        sc.set_offsets(xd); sc.set_array(m)
        line.set_data(pl[:, 0], pl[:, 1])
        a2.lines[1].set_data(uy[:k + 1], Fy[:k + 1])
        pt.set_data([uy[k]], [fy])
        title.set_text(f"step {k}: λ = {lam:.2f}" + ("   (rupture)" if k == len(frames) - 1 else ""))
        return [sc, line, pt, title]

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=500, blit=False)
    anim.save("xfem_crack_growth.gif", writer=PillowWriter(fps=2))
    print(f"saved xfem_crack_growth.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
