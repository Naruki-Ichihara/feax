"""XFEM + cohesive zone: progressive debonding of a bimaterial interface.

Same plate as ``xfem_partial_interface.py`` (unit square, material 1 on top of
material 2, interface on the left half at y = 0.5, tension applied through
the top and bottom edges), but the interface is no longer free: a bilinear
traction-separation law (initial stiffness K, strength sigma_c, fracture
energy Gamma — the NLXP3D / Camanho-type cohesive law) acts on the cut chord
of every enriched cell, so the interface carries traction until it fails.

.. code-block:: text

        t(δ) = K δ            δ < δ0 = σc / K        elastic
        t(δ) = σc (δf-δ)/(δf-δ0)   δ0 ≤ δ < δf        softening
        t(δ) = 0              δ ≥ δf = 2Γ / σc        failed

The load is applied in steps.  Each step is a Newton solve (the interface
term is nonlinear); after convergence the opening history ``delta_max`` is
updated so unloading follows the secant (irreversible damage).  The reaction
force on the top edge, the opening profile and the traction along the
interface are recorded.

Output: ``xfem_cohesive_interface.png`` with
  (1) reaction force vs. applied displacement,
  (2) opening profiles δn(x) at several steps,
  (3) traction profiles t(x) at the same steps,
  (4) von Mises stress at the last step.
With ``--gif`` also ``xfem_cohesive_interface.gif``: one frame per load step
showing the deformed plate (von Mises stress), the interface coloured by its
damage ``delta_max / delta_f`` (blue = intact, red = failed) and the running
force-displacement curve, i.e. the debonding front travelling from the free
edge to the tip.

Run:  python examples/experimental/xfem_cohesive_interface.py [--gif]
"""
import sys

import jax
import jax.numpy as np
import numpy as onp

import feax as fe
from feax.experimental.xfem import CohesiveZone, ShiftedHeaviside, XFEMSpace, bilinear_tsl, xfem_problem

jax.config.update("jax_enable_x64", True)

N_CELLS = 21
ELE_TYPE = "QUAD4"
E_TOP, E_BOT, NU = 100.0, 50.0, 0.3
TIP_X = 0.5
# cohesive law: delta0 = sigma_c/K = 1e-3, delta_f = 2 Gamma / sigma_c = 2e-2
K, SIGMA_C, GAMMA = 1e3, 1.0, 1e-2
D_END, N_STEPS = 0.03, 30        # prescribed |u_y| on the edges, and the number of load steps


def material(u_grad, phi, x):
    E = np.where(phi > 0, E_TOP, E_BOT)
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    eps = 0.5 * (u_grad + u_grad.T)
    return lam * np.trace(eps) * np.eye(2) + 2 * mu * eps


def main():
    mesh = fe.mesh.rectangle_mesh(N_CELLS, N_CELLS, 1.0, 1.0, ele_type=ELE_TYPE)
    pts = onp.asarray(mesh.points)
    cz = CohesiveZone(bilinear_tsl, K=K, sigma_c=SIGMA_C, Gamma=GAMMA, contact_penalty=K)
    space = XFEMSpace(mesh, pts[:, 1] - 0.5, ShiftedHeaviside(), vec=2, ele_type=ELE_TYPE,
                      psi=pts[:, 0] - TIP_X, cohesive=cz)
    problem = xfem_problem(space, material)
    ts = fe.TracedStructure.from_problem(problem)
    print(f"mesh {len(pts)} nodes, interface cells {space.interface_cells.sum()}, "
          f"enriched nodes {space.num_enriched}, DOFs {problem.num_total_dofs_all_vars}")
    print(f"cohesive law: delta0 = {SIGMA_C / K:.1e}, delta_f = {2 * GAMMA / SIGMA_C:.1e}, "
          f"{cz.n_gauss} Gauss points on each of {space.cohesive_cells.sum()} chords")

    tol = 1e-8
    bottom = lambda p: np.isclose(p[1], 0.0, tol)
    top = lambda p: np.isclose(p[1], 1.0, tol)

    def make_bc(d):
        return space.create_bc(problem, fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=bottom, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=bottom, component=1, value=-d, variable_index=0),
            fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=top, component=1, value=d, variable_index=0),
        ]))

    # softening interfaces make the tangent indefinite: use a pivoted LU (cuDSS
    # GENERAL) rather than the default LDLT, which can return a wrong direction
    direct = (fe.DirectSolverOptions(solver="cudss", cudss_options=fe.CUDSSOptions(matrix_type=fe.CUDSSMatrixType.GENERAL))
              if fe.solvers.options.has_cudss() else fe.DirectSolverOptions(solver="umfpack"))
    solver = fe.create_solver(problem, make_bc(0.0), solver_options=direct, linear=False,
                              newton_options=fe.NewtonOptions(tol=1e-9, rel_tol=1e-10, max_iter=100),
                              traced_params=space.traced_params(), traced_structure=ts)

    top_nodes = onp.flatnonzero(onp.isclose(pts[:, 1], 1.0))
    delta_max = space.zero_history()
    u = fe.zero_like_initial_guess(problem, make_bc(0.0))
    steps, forces, profiles, frames = [0.0], [0.0], {}, []
    print("\n step    u_y      reaction   max opening   max traction  failed chords")
    for k in range(1, N_STEPS + 1):
        d = D_END * k / N_STEPS
        bc = make_bc(d)
        u = np.asarray(u).at[bc.bc_rows].set(bc.bc_vals)
        tp = space.traced_params(delta_max=delta_max)
        u = solver(tp, u, bc=bc, traced_structure=ts).dofs
        res = fe.get_res(problem, problem.unflatten_fn_sol_list(u), tp, ts)[0]   # (N, 2), reactions at BC nodes
        force = float(onp.asarray(res)[top_nodes, 1].sum())
        st = space.interface_state(problem, u, delta_max)
        delta_max = space.update_history(delta_max, problem, u)
        failed = int((onp.asarray(delta_max)[space.cohesive_cells] >= 2 * GAMMA / SIGMA_C).all(axis=1).sum())
        steps.append(d); forces.append(force)
        frames.append((d, onp.asarray(u), onp.asarray(delta_max)))
        if k in (N_STEPS // 6, N_STEPS // 3, N_STEPS // 2, 2 * N_STEPS // 3, N_STEPS):
            order = onp.argsort(onp.asarray(st["x"])[:, 0])
            profiles[d] = (onp.asarray(st["x"])[order, 0], onp.asarray(st["delta_n"])[order],
                           onp.asarray(st["traction"])[order])
        print(f"  {k:3d}  {d:.4f}   {force: .4e}   {float(onp.asarray(st['delta_n']).max()):.3e}     "
              f"{float(onp.asarray(st['traction']).max()):.3e}      {failed}/{space.cohesive_cells.sum()}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    ax[0, 0].plot(steps, forces, "o-")
    ax[0, 0].set_xlabel("applied edge displacement u_y"); ax[0, 0].set_ylabel("top-edge reaction F_y")
    ax[0, 0].set_title("force - displacement")
    for d, (x, dn, t) in profiles.items():
        ax[0, 1].plot(x, dn, "o-", ms=3, label=f"u_y = {d:.3f}")
        ax[1, 0].plot(x, t, "o-", ms=3, label=f"u_y = {d:.3f}")
    ax[0, 1].axhline(2 * GAMMA / SIGMA_C, color="gray", ls="--", lw=0.8)
    ax[0, 1].set_xlabel("x along the interface"); ax[0, 1].set_ylabel("opening δn"); ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("opening profile (dashed: δf)")
    ax[1, 0].axhline(SIGMA_C, color="gray", ls="--", lw=0.8)
    ax[1, 0].set_xlabel("x along the interface"); ax[1, 0].set_ylabel("traction t(δ)"); ax[1, 0].legend(fontsize=7)
    ax[1, 0].set_title("cohesive traction (dashed: σc)")
    x_q, u_q, phi_q, grad_u = (onp.asarray(a) for a in space.quad_fields(problem, u, grad=True))
    real = space.qw > 0
    sig = onp.asarray(jax.vmap(material)(np.asarray(grad_u[real]), np.asarray(phi_q[real]), np.asarray(x_q[real])))
    s_zz = NU * (sig[:, 0, 0] + sig[:, 1, 1])
    mises = onp.sqrt(0.5 * ((sig[:, 0, 0] - sig[:, 1, 1]) ** 2 + (sig[:, 1, 1] - s_zz) ** 2
                            + (s_zz - sig[:, 0, 0]) ** 2) + 3 * sig[:, 0, 1] ** 2)
    sc = ax[1, 1].scatter(x_q[real][:, 0], x_q[real][:, 1], c=mises, s=10, cmap="viridis")
    ax[1, 1].plot([0, TIP_X], [0.5, 0.5], "w--", lw=1); ax[1, 1].set_aspect("equal")
    ax[1, 1].set_title(f"von Mises stress at u_y = {D_END:g}"); fig.colorbar(sc, ax=ax[1, 1], shrink=0.8)
    fig.tight_layout()
    fig.savefig("xfem_cohesive_interface.png", dpi=150)
    print("\nsaved xfem_cohesive_interface.png")

    if "--gif" in sys.argv:
        save_gif(space, problem, frames, steps, forces, plt)


def save_gif(space, problem, frames, steps, forces, plt, scale=3.0, fname="xfem_cohesive_interface.gif"):
    """One frame per load step: deformed quadrature points coloured by von Mises
    stress, the interface chords coloured by damage delta_max/delta_f, and the
    force-displacement curve up to the current step."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    real = space.qw > 0
    delta_f = 2 * GAMMA / SIGMA_C
    seg = space.segments[space.cohesive_cells]                 # (nc, 2, 2) undeformed chords
    cells = onp.flatnonzero(space.cohesive_cells)

    # precompute the per-frame fields
    data = []
    for d, u, dm in frames:
        x_q, u_q, phi_q, grad_u = (onp.asarray(a) for a in space.quad_fields(problem, u, grad=True))
        sig = onp.asarray(jax.vmap(material)(np.asarray(grad_u[real]), np.asarray(phi_q[real]),
                                             np.asarray(x_q[real])))
        s_zz = NU * (sig[:, 0, 0] + sig[:, 1, 1])
        mises = onp.sqrt(0.5 * ((sig[:, 0, 0] - sig[:, 1, 1]) ** 2 + (sig[:, 1, 1] - s_zz) ** 2
                                + (s_zz - sig[:, 0, 0]) ** 2) + 3 * sig[:, 0, 1] ** 2)
        damage = onp.clip(dm[cells].mean(axis=1) / delta_f, 0.0, 1.0)
        x_mid, jump = (onp.asarray(a) for a in space.interface_jump(problem, u))
        data.append((d, x_q[real] + scale * u_q[real], mises, damage, jump))
    vmax = max(m.max() for _, _, m, _, _ in data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    sc = ax1.scatter(data[0][1][:, 0], data[0][1][:, 1], c=data[0][2], s=9, cmap="viridis",
                     norm=Normalize(0, vmax))
    fig.colorbar(sc, ax=ax1, shrink=0.85, label="von Mises stress")
    lines = [ax1.plot(seg[i, :, 0], seg[i, :, 1], lw=4, color="tab:blue", solid_capstyle="butt")[0]
             for i in range(len(cells))]
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.1 - scale * D_END, 1.1 + scale * D_END)
    ax1.set_aspect("equal"); title = ax1.set_title("")
    ax2.plot(steps, forces, "-", color="lightgray")
    (curve,) = ax2.plot([], [], "o-", color="tab:red", ms=4)
    ax2.set_xlabel("applied edge displacement u_y"); ax2.set_ylabel("top-edge reaction F_y")
    ax2.set_xlim(0, D_END * 1.05); ax2.set_ylim(0, max(forces) * 1.1)
    ax2.set_title("force - displacement")
    cmap = plt.get_cmap("coolwarm")

    def draw(k):
        d, xd, mises, damage, jump = data[k]
        sc.set_offsets(xd); sc.set_array(mises)
        for i, ln in enumerate(lines):
            # draw each chord at the mean deformed position of its two faces
            # (the opening itself is visible in the scatter)
            ln.set_color(cmap(damage[i]))
            ln.set_linewidth(2 + 4 * damage[i])
        curve.set_data(steps[:k + 2], forces[:k + 2])
        failed = int((damage >= 1.0).sum())
        title.set_text(f"u_y = {d:.4f}   failed chords {failed}/{len(cells)}   (deformation x{scale:g})")
        return [sc, curve, title, *lines]

    anim = FuncAnimation(fig, draw, frames=len(data), interval=150, blit=False)
    anim.save(fname, writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f"saved {fname} ({len(data)} frames)")


if __name__ == "__main__":
    main()
