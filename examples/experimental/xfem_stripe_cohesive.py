"""Stripe pattern -> interface level set -> XFEM with cohesive zones -> loading.

The pipeline a 3D-printed part needs, end to end on one fixed mesh:

1. ``feax.experimental.stripe``: a stripe pattern aligned with a vortex
   (tangential) direction field — concentric print paths — computed as an
   *oriented* field so that the phase ``α`` is globally consistent;
2. the zero set of ``phi = sin α`` is the set of interfaces between
   neighbouring filaments (one closed circle every stripe spacing);
   the innermost circles, under-resolved by the mesh, are left out through
   the tangential level set ``psi = r_min - r``;
3. ``feax.experimental.xfem``: the circles are embedded in the mesh with the
   shifted-sign enrichment (NLXP3D's) and a bilinear cohesive law on every
   cut chord (``CohesiveZone``), the bulk is one isotropic material;
4. tension is applied through the top edge in steps (Newton solve + history
   update of the cohesive opening); the circles debond progressively at
   their poles (where the interface normal is aligned with the load) while
   their flanks carry compression through the contact penalty.

Output: ``xfem_stripe_cohesive.png`` (pattern + interfaces, deformed state
with von Mises stress and interface damage, force-displacement curve), with
``--gif`` an animation, and ``output/xfem_stripe_cohesive/`` with per-step
VTU files (nodal displacement and level set on the mesh, the interface
state as a point cloud, and a crack-aware remesh split along the failed
interfaces — the cracks are real gaps) indexed by ``.pvd`` collections.

Two fields: ``vortex`` (default; concentric circles, the centre left out with
``psi``) and ``bending`` (the smoothly turning field of ``stripe_pattern.py``,
whose pattern needs branch points: there two interfaces merge into one,
i.e. a filament ends — the level set is ``Im ψ`` unnormalised, which passes
smoothly through the branch points).

A third field, ``dislocation``, has horizontal filaments whose spacing
changes from 0.1 (left half) to 0.125 (right half): one filament has to end,
and the branch point (an edge dislocation) is pinned at the centre of the
plate with a local dip of the stripe-energy weight.

Run:  python examples/experimental/xfem_stripe_cohesive.py [vortex|bending|dislocation] [--gif] [--steps=N] [--gamma=G] [--n=N] [--load=x|y] [--dend=D] [--phase-shift=F (fraction of the stripe period)]
"""
import os
import sys

import jax
import jax.numpy as np
import numpy as onp
import pyvista as pv

import feax as fe
from feax.solvers.newton import NewtonLineSearchError
from feax.experimental.stripe import StripePattern, smooth_line_field
from feax.experimental.xfem import CohesiveZone, ShiftedHeaviside, XFEMSpace, bilinear_tsl, exponential_tsl, xfem_problem

jax.config.update("jax_enable_x64", True)

N = 60                          # cells per side (h = 1/60)
STRIPE_SPACING = 0.2            # interfaces (sin alpha = 0) every STRIPE_SPACING / 2
R_MIN = 0.12                    # no interfaces closer than this to the vortex centre
E, NU = 100.0, 0.3
K, SIGMA_C, GAMMA = 2e3, 0.5, 5e-3       # cohesive law: delta0 = 2.5e-4, delta_f = 0.02
D_END, N_STEPS = 0.03, 30
FIELD = next((a for a in sys.argv[1:] if not a.startswith("--")), "vortex")


def _opt(name, default, cast):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


N = _opt("n", 100 if FIELD == "dislocation" else N, int)   # --n=80: cells per side
D_END = _opt("dend", D_END, float)             # --dend=0.12: final loaded-edge displacement
PHASE_SHIFT = _opt("phase-shift", 0.0, float)   # --phase-shift=0.25: interfaces at alpha = 2 pi * 0.25 + k pi
LOAD = _opt("load", "y", str)                  # --load=x: pull the right edge in x (along horizontal stripes)
N_STEPS = _opt("steps", N_STEPS, int)          # --steps=100   (smaller increments for brittle cases)
GAMMA = _opt("gamma", GAMMA, float)            # --gamma=2e-2  (tougher interface)
IMPERFECTION = _opt("imperfection", 0.0, float)   # --imperfection=0.1: +-10 % random strength per chord
TSL = _opt("tsl", "bilinear", str)             # --tsl=exponential (smooth peak, no kink at delta0)
VISCOSITY = _opt("viscosity", 0.0, float)       # --viscosity=30 adds the viscous regularisation
SECANT = _opt("secant", 0, int)                 # --secant=1: explicit damage update (secant potential + outer fixed point)
EXTENT = _opt("extent", None, float)           # bending: --extent=0.35 keeps interfaces to |x - 0.5| < 0.35
SOLVER = _opt("solver", "cudss", str)          # --solver=umfpack (CPU LU)
TAG = ("" if LOAD == "y" else f"_load{LOAD}") + ("" if PHASE_SHIFT == 0 else f"_shift{PHASE_SHIFT:g}")
OUT_DIR = os.path.join("output", f"xfem_stripe_cohesive_{FIELD}{TAG}")


def material(u_grad, phi, x):
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    eps = 0.5 * (u_grad + u_grad.T)
    return lam * np.trace(eps) * np.eye(2) + 2 * mu * eps


def mises(sig):
    s_zz = NU * (sig[:, 0, 0] + sig[:, 1, 1])
    return onp.sqrt(0.5 * ((sig[:, 0, 0] - sig[:, 1, 1]) ** 2 + (sig[:, 1, 1] - s_zz) ** 2
                           + (s_zz - sig[:, 0, 0]) ** 2) + 3 * sig[:, 0, 1] ** 2)


def main():
    quad = fe.mesh.rectangle_mesh(N, N, 1.0, 1.0, ele_type="QUAD4")
    c = onp.asarray(quad.cells)
    mesh = fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]), ele_type="TRI3")
    pts = onp.asarray(mesh.points)
    cx, cy = 0.5 + 1 / (2 * N), 0.5 + 1 / (2 * N)
    r = onp.hypot(pts[:, 0] - cx, pts[:, 1] - cy)

    # --- 1-2. stripes along the field, interfaces on the zero set of Im psi ------
    if FIELD == "vortex":
        angle = onp.arctan2(pts[:, 1] - cy, pts[:, 0] - cx) + onp.pi / 2      # tangential, oriented
        psi = R_MIN - r                                         # interfaces only where psi < 0 (r > R_MIN)
    elif FIELD == "bending":
        angle = smooth_line_field(mesh, 0.9 * pts[:, 0], t=0.002)           # turning field: branch points
        # interfaces spanning the whole width debond all at once (a structural
        # snap under displacement control); --extent keeps them away from the
        # side walls, i.e. the filaments stay bonded there
        psi = None if EXTENT is None else onp.abs(pts[:, 0] - 0.5) - EXTENT
    elif FIELD == "dislocation":
        # horizontal filaments, 10 across the left half and 8 across the right
        # half: the stripe count changes by one period, which forces exactly one
        # branch point (an edge dislocation, a filament that ends); a dip of the
        # stripe-energy weight pins it at the centre of the plate
        angle = onp.zeros(len(pts))
        t = 0.5 * (1.0 + onp.tanh((pts[:, 0] - 0.5) / 0.05))
        frequency = 2 * onp.pi / (0.1 * (1 - t) + 0.125 * t)
        weight = 1.0 - 0.99 * onp.exp(-((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2) / 0.08 ** 2)
        psi = None
    else:
        raise SystemExit(f"unknown field {FIELD!r}: use vortex, bending or dislocation")
    if FIELD == "dislocation":
        stripes = StripePattern.compute(mesh, angle, frequency, oriented=True, weight=weight)
    else:
        stripes = StripePattern.compute(mesh, angle, 2 * onp.pi / STRIPE_SPACING, oriented=True)
    phi = stripes.level_set(raw=True, shift=2 * onp.pi * PHASE_SHIFT)                           # |psi| sin(alpha): 0 on the filament boundaries

    # --- 3. XFEM with cohesive zones on the interfaces -------------------------
    # viscosity: many chords reach their peak together; the viscous term keeps the
    # chord tangent positive so Newton does not cycle between the loading and the
    # unloading branch (eta * delta-increment per step stays << sigma_c here)
    # implicit (default): loading/unloading decided inside Newton (quadratic
    # convergence, 3-4 iterations per step); --secant=1 freezes the damage during
    # each solve and advances it between solves (explicit damage update, more
    # robust on snap-through at the price of an outer fixed point)
    if TSL == "bilinear":
        cz = CohesiveZone(bilinear_tsl, K=K, sigma_c=SIGMA_C, Gamma=GAMMA, contact_penalty=K,
                          viscosity=VISCOSITY, secant=bool(SECANT))
    else:
        cz = CohesiveZone(exponential_tsl, sigma_c=SIGMA_C, Gamma=GAMMA, contact_penalty=K,
                          viscosity=VISCOSITY, secant=bool(SECANT))
    space = XFEMSpace(mesh, phi, ShiftedHeaviside(), vec=2, ele_type="TRI3", psi=psi, cohesive=cz)
    problem = xfem_problem(space, material)
    print(f"mesh {len(pts)} nodes, {space.num_cells} TRI3 cells; stripes: {int((stripes.zero_index != 0).sum())} branch points")
    print(f"interfaces: {space.interface_cells.sum()} interface cells, {space.num_enriched} enriched nodes, "
          f"{space.cohesive_cells.sum()} cohesive chords, DOFs {problem.num_total_dofs_all_vars}")

    tol = 1e-8
    ax_l = 1 if LOAD == "y" else 0                  # load axis
    bottom = lambda p: np.isclose(p[ax_l], 0.0, tol)   # fixed edge (bottom for y, left for x)
    top = lambda p: np.isclose(p[ax_l], 1.0, tol)      # loaded edge (top for y, right for x)

    def make_bc(d):
        return space.create_bc(problem, fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=bottom, component="all", value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=top, component=1 - ax_l, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=top, component=ax_l, value=d, variable_index=0),
        ]))

    ts = fe.TracedStructure.from_problem(problem)
    # softening interfaces make the tangent indefinite: use a pivoted LU (cuDSS
    # GENERAL) rather than the default LDLT, which can return a wrong direction
    direct = (fe.DirectSolverOptions(solver="cudss", cudss_options=fe.CUDSSOptions(matrix_type=fe.CUDSSMatrixType.GENERAL))
              if (SOLVER == "cudss" and fe.solvers.options.has_cudss()) else fe.DirectSolverOptions(solver=SOLVER if SOLVER != "cudss" else "umfpack"))
    solver = fe.create_solver(problem, make_bc(0.0), solver_options=direct, linear=False,
                              newton_options=fe.NewtonOptions(tol=1e-8, rel_tol=1e-10, max_iter=100),
                              traced_params=space.traced_params(), traced_structure=ts)
    top_nodes = onp.flatnonzero(onp.isclose(pts[:, ax_l], 1.0))
    ul, fl = f"u_{LOAD}", f"F_{LOAD}"
    delta_f = 2 * GAMMA / SIGMA_C

    # --- 4. load steps ------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    delta_max = space.zero_history()
    delta_prev = space.zero_history()
    # optional random strength imperfection per chord (--imperfection=0.1); not
    # needed for convergence: a chord-scale checkerboard of strengths only makes
    # the localisation pattern rougher
    rng = onp.random.default_rng(0)
    cz_scale = 1.0 + IMPERFECTION * (2 * rng.random(space.num_cells) - 1)
    u = fe.zero_like_initial_guess(problem, make_bc(0.0))
    steps, forces, frames, vtu_frames, remesh_frames = [0.0], [0.0], [], [], []
    print(f"\n step   {ul}      {fl}        max opening  failed chords  outer its")
    def advance(u_, dm_, d_from, d_to, depth=0):
        """Solve the increment d_from -> d_to; on a Newton failure, bisect the
        increment (up to 4 levels) — the late softening path is close to
        snap-through and occasionally needs a smaller step."""
        bc_ = make_bc(d_to)
        u_try = np.asarray(u_).at[bc_.bc_rows].set(bc_.bc_vals)
        try:
            return space.solve_step(problem, solver, bc_, ts, u_try, dm_, cohesive_scale=cz_scale, relax=0.7)
        except NewtonLineSearchError:
            if depth >= 4:
                raise
            mid = 0.5 * (d_from + d_to)
            print(f"      step cut: {d_from:.5f} -> {mid:.5f} -> {d_to:.5f}")
            u1, dm1, _ = advance(u_, dm_, d_from, mid, depth + 1)
            return advance(u1, dm1, mid, d_to, depth + 1)

    d_prev = 0.0
    for k in range(1, N_STEPS + 1):
        d = D_END * k / N_STEPS
        u, delta_max_new, n_outer = advance(u, delta_max, d_prev, d)
        d_prev = d
        tp = space.traced_params(delta_max=delta_max_new, cohesive_scale=cz_scale, delta_prev=delta_max_new)
        res = onp.asarray(fe.get_res(problem, problem.unflatten_fn_sol_list(u), tp, ts)[0])
        force = float(res[top_nodes, ax_l].sum())
        st = space.interface_state(problem, u, delta_max_new)
        delta_max = delta_max_new
        dm = onp.asarray(delta_max)[space.cohesive_cells]
        failed = int((dm >= delta_f).all(axis=1).sum())
        steps.append(d); forces.append(force)
        x_q, u_q, phi_q, grad_u = (onp.asarray(a) for a in space.quad_fields(problem, u, grad=True))
        real = space.qw > 0
        sig = onp.asarray(jax.vmap(material)(np.asarray(grad_u[real]), np.asarray(phi_q[real]), np.asarray(x_q[real])))
        frames.append((x_q[real] + u_q[real], mises(sig), onp.asarray(st["x"]), dm.mean(axis=1) / delta_f, d, force))
        print(f"  {k:3d}   {d:.4f}   {force: .4e}   {float(onp.asarray(st['delta_n']).max()):.3e}     "
              f"{failed}/{space.cohesive_cells.sum()}   {n_outer}")
        # VTU: nodal fields on the mesh + interface state as a point cloud
        un = onp.asarray(space.nodal_displacement(problem, u))
        f_mesh = os.path.join(OUT_DIR, f"field_{k:04d}.vtu")
        fe.utils.save_sol(mesh, f_mesh, point_infos=[("u", onp.c_[un, onp.zeros(len(un))]), ("phi", phi),
                                                     ("psi", psi if psi is not None else onp.zeros(len(pts))),
                                                     ("cos_alpha", stripes.scalar())],
                          cell_infos=[("interface_cell", space.interface_cells.astype(float))])
        xi = onp.asarray(st["x"])
        f_int = os.path.join(OUT_DIR, f"interface_{k:04d}.vtp")
        cloud = pv.PolyData(onp.c_[xi, onp.zeros(len(xi))])
        cloud.point_data["delta_n"] = onp.asarray(st["delta_n"])
        cloud.point_data["traction"] = onp.asarray(st["traction"])
        cloud.point_data["damage"] = onp.clip(dm.reshape(-1) / delta_f, 0.0, 1.0)
        cloud.save(f_int)
        # crack-aware remesh: the mesh is split along the failed interfaces only,
        # so the cracks are real gaps (Warp By Vector u); intact interfaces stay bonded
        f_rm = os.path.join(OUT_DIR, f"remesh_{k:04d}.vtu")
        space.save_remesh_vtu(f_rm, problem, u, material=material, split="damaged", delta_max=delta_max,
                              crack_filename=os.path.join(OUT_DIR, f"cracks_{k:04d}.vtu"))
        vtu_frames.append((f_mesh, f_int))
        remesh_frames.append(f_rm)
    fe.utils.write_pvd(os.path.join(OUT_DIR, "field.pvd"), [(k, f) for k, (f, _) in enumerate(vtu_frames, 1)])
    fe.utils.write_pvd(os.path.join(OUT_DIR, "interface.pvd"), [(k, f) for k, (_, f) in enumerate(vtu_frames, 1)])
    fe.utils.write_pvd(os.path.join(OUT_DIR, "remesh.pvd"), [(k, f) for k, f in enumerate(remesh_frames, 1)])
    print(f"\nParaView series: {OUT_DIR}/field.pvd (mesh fields), {OUT_DIR}/interface.pvd (interface Gauss points), "
          f"{OUT_DIR}/remesh.pvd (crack-aware remesh, split along failed interfaces)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], stripes.mesh.cells)
    ax[0].tripcolor(tri, stripes.scalar(), shading="gouraud", cmap="Greys_r", vmin=-1, vmax=1)
    seg = space.segments[space.cohesive_cells]
    for s_ in seg:
        ax[0].plot(s_[:, 0], s_[:, 1], "-", color="tab:red", lw=1)
    ax[0].set_aspect("equal"); ax[0].set_title("stripes (cos α), cohesive interfaces (red)")
    xd, m, xi, dmg, d, f = frames[-1]
    sc = ax[1].scatter(xd[:, 0], xd[:, 1], c=m, s=4, cmap="viridis", vmax=onp.percentile(m, 99))
    ax[1].scatter(xi[:, 0], xi[:, 1], c=onp.repeat(dmg, cz.n_gauss), s=6, cmap="coolwarm", vmin=0, vmax=1)
    ax[1].set_aspect("equal"); ax[1].set_title(f"{ul} = {d:g}: von Mises (true deformation)\ninterface damage: blue 0 - red 1")
    fig.colorbar(sc, ax=ax[1], shrink=0.8)
    ax[2].plot(steps, forces, "o-"); ax[2].set_xlabel(f"loaded-edge displacement {ul}"); ax[2].set_ylabel(f"reaction {fl}")
    ax[2].set_title("force - displacement")
    zi = stripes.zero_index != 0
    if zi.any():
        cen = pts[stripes.mesh.cells[zi]].mean(1)
        ax[0].plot(cen[:, 0], cen[:, 1], "o", mfc="none", mec="tab:blue", ms=9, mew=1.5, label="stripe branch point")
        ax[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"xfem_stripe_cohesive_{FIELD}{TAG}.png", dpi=150)
    print(f"saved xfem_stripe_cohesive_{FIELD}{TAG}.png")

    if "--gif" in sys.argv:
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.colors import Normalize
        vmax = max(onp.percentile(m, 99) for _, m, *_ in frames)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        sc = a1.scatter(frames[0][0][:, 0], frames[0][0][:, 1], c=frames[0][1], s=4, cmap="viridis", norm=Normalize(0, vmax))
        si = a1.scatter(frames[0][2][:, 0], frames[0][2][:, 1], c=onp.repeat(frames[0][3], cz.n_gauss), s=6, cmap="coolwarm",
                        norm=Normalize(0, 1))
        a1.set_xlim(-0.05, 1.05); a1.set_ylim(-0.05, 1.15); a1.set_aspect("equal")
        fig.colorbar(sc, ax=a1, shrink=0.8, label="von Mises"); title = a1.set_title("")
        a2.plot(steps, forces, "-", color="lightgray"); (pt,) = a2.plot([], [], "o-", color="tab:red", ms=4)
        a2.set_xlim(0, D_END * 1.05); a2.set_ylim(0, max(forces) * 1.1); a2.set_xlabel(ul); a2.set_ylabel(fl)

        def draw(k):
            xd, m, xi, dmg, d, f = frames[k]
            sc.set_offsets(xd); sc.set_array(m); si.set_offsets(xi); si.set_array(onp.repeat(dmg, cz.n_gauss))
            pt.set_data(steps[:k + 2], forces[:k + 2]); title.set_text(f"{ul} = {d:.4f}")
            return [sc, si, pt, title]

        FuncAnimation(fig, draw, frames=len(frames), interval=200, blit=False).save(
            f"xfem_stripe_cohesive_{FIELD}{TAG}.gif", writer=PillowWriter(fps=5))
        print(f"saved xfem_stripe_cohesive_{FIELD}{TAG}.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()
