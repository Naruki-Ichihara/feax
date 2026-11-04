"""feax.experimental.xfem_crack: traced XFEM crack geometry, tip enrichment and
energy release rates by autodiff.

Reference problem: single-edge-notched strip (W = 1, H = 4, crack from the
left edge along y = 0), fixed grips (u_x = 0, u_y = ±d on the top / bottom
edges), nu = 0.  The energy release rate of the XFEM model (autodiff of the
strain energy with respect to the trial extension) is compared with

* its smoothness across the tip position inside a cell (the pointwise
  derivative of the discrete energy with respect to the tip is not usable:
  it shows a ±50 % sawtooth; the virtual crack extension J does not),
* a conforming FEM model with duplicated nodes along the crack: secant
  ``-(U(a + h) - U(a)) / h`` evaluated at the same mid-length ``a + h/2``
  (the XFEM tip sits there, inside a cell).
"""
import jax
import jax.numpy as np
import numpy as onp
import pytest

import feax as fe
from feax.experimental.xfem_crack import (
    CrackSpace, NQ_MAX, branch_functions, crack_level_sets, crack_problem, plane_strain_energy,
)

jax.config.update("jax_enable_x64", True)

E, NU = 100.0, 0.0
W, H, DELTA = 1.0, 4.0, 0.01
A0 = 0.3
TOL = 1e-8


def _direct():
    return fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")


def _tri_mesh(nx, ny):
    quad = fe.mesh.rectangle_mesh(nx, ny, W, H, origin=(0.0, -H / 2), ele_type="QUAD4")
    c = onp.asarray(quad.cells)
    return fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]),
                   ele_type="TRI3")


def _grip_specs(variable_index=None):
    bot = lambda p: np.isclose(p[1], -H / 2, TOL)
    top = lambda p: np.isclose(p[1], H / 2, TOL)
    kw = {} if variable_index is None else dict(variable_index=variable_index)
    return [fe.DirichletBCSpec(location=bot, component=0, value=0.0, **kw),
            fe.DirichletBCSpec(location=bot, component=1, value=-DELTA, **kw),
            fe.DirichletBCSpec(location=top, component=0, value=0.0, **kw),
            fe.DirichletBCSpec(location=top, component=1, value=DELTA, **kw)]


def _xfem(nx, ny, a):
    mesh = _tri_mesh(nx, ny)
    space = CrackSpace(mesh, max_points=8)
    poly, n = space.polyline([(-0.1, 0.0), (a, 0.0)])
    problem = crack_problem(space, plane_strain_energy(E, NU))
    bc = space.create_bc(problem, fe.DirichletBCConfig(_grip_specs(0)))
    ts = fe.TracedStructure.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=_direct(), linear=True, traced_structure=ts)
    return space, problem, bc, ts, solver, poly, n


def _fem_secant_G(nx, ny, a):
    """Conforming FEM: duplicated nodes on y = 0, x < a; G ≈ -(U(a+h) - U(a))/h."""
    h = W / nx

    def energy(a_len):
        quad = fe.mesh.rectangle_mesh(nx, ny, W, H, origin=(0.0, -H / 2), ele_type="QUAD4")
        pts = onp.asarray(quad.points)
        c = onp.asarray(quad.cells)
        tris = onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]])
        ids = onp.flatnonzero(onp.isclose(pts[:, 1], 0.0) & (pts[:, 0] < a_len - 1e-9))
        mapping = dict(zip(ids, onp.arange(len(pts), len(pts) + len(ids))))
        pts2 = onp.concatenate([pts, pts[ids]])
        tris2 = tris.copy()
        for e in onp.flatnonzero(pts[tris][:, :, 1].mean(1) > 0):
            for k in range(3):
                tris2[e, k] = mapping.get(tris2[e, k], tris2[e, k])
        mesh = fe.Mesh(pts2, tris2, ele_type="TRI3")

        class P(fe.Problem):
            def get_tensor_map(self):
                def stress(g):
                    eps = 0.5 * (g + g.T)
                    return E * eps                                   # nu = 0
                return stress

        problem = P(mesh=mesh, vec=2, dim=2, ele_type="TRI3")
        bc = fe.DirichletBCConfig(_grip_specs()).create_bc(problem)
        ts = fe.TracedStructure.from_problem(problem)
        sol = fe.create_solver(problem, bc, solver_options=_direct(), linear=True, traced_structure=ts)(
            fe.TracedParams(), fe.zero_like_initial_guess(problem, bc), traced_structure=ts)
        res = fe.get_res(problem, problem.unflatten_fn_sol_list(sol), fe.TracedParams(), ts)[0]
        return 0.5 * float((onp.asarray(res) * onp.asarray(sol.dofs).reshape(-1, 2)).sum())

    return -(energy(a + h) - energy(a)) / h


def test_level_sets_of_polyline():
    X = np.array([[0.5, 0.2], [0.5, -0.2], [2.0, 1.0], [2.0, 1.3], [-1.0, 0.1], [1.2, 0.5]])
    space_poly = np.zeros((8, 2)).at[:3].set(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    phi, psi = crack_level_sets(space_poly, np.int32(3), X)
    # first segment: phi = y (left of +x is +y), psi = x - 2 (total length 2)
    assert onp.allclose(onp.asarray(phi[:2]), [0.2, -0.2])
    assert onp.allclose(onp.asarray(psi[:2]), [-1.5, -1.5])
    # beside / ahead of the tip (1, 1) of the upward segment: phi = distance to the
    # extended line x = 1 (positive on the left: x < 1), psi = y - 1
    assert onp.isclose(float(phi[2]), -1.0) and onp.isclose(float(psi[2]), 0.0)
    assert onp.isclose(float(phi[3]), -1.0) and onp.isclose(float(psi[3]), 0.3)
    # behind the mouth: first segment extended, psi < 0
    assert onp.isclose(float(phi[4]), 0.1) and onp.isclose(float(psi[4]), -3.0)
    # nearest to the second segment: phi = -(x - 1) = -0.2, psi = 0.5 - 1 = -0.5
    assert onp.isclose(float(phi[5]), -0.2) and onp.isclose(float(psi[5]), -0.5)


def test_branch_functions_jump_only_across_the_crack():
    above = branch_functions(np.float64(1e-9), np.float64(-0.5))
    below = branch_functions(np.float64(-1e-9), np.float64(-0.5))
    ahead_up = branch_functions(np.float64(1e-9), np.float64(0.5))
    ahead_dn = branch_functions(np.float64(-1e-9), np.float64(0.5))
    jump = onp.asarray(above - below)
    assert onp.isclose(jump[0], 2 * onp.sqrt(0.5), atol=1e-6)          # sqrt(r) sin(theta/2) jumps
    assert onp.allclose(jump[1:], 0.0, atol=1e-6)
    assert onp.allclose(onp.asarray(ahead_up - ahead_dn), 0.0, atol=1e-6)


def test_geometry_classes_and_quadrature():
    # tip at 0.3 + 0.3 h: inside a cell and off the quad diagonal (x = 0.3 + 0.5 h at y = 0)
    space, problem, bc, ts, solver, poly, n = _xfem(20, 81, A0 + 0.015)
    cls = space.classes(poly, n)
    assert (cls == 2).sum() == 1                                   # tip strictly inside one cell
    assert (cls == 1).sum() == 12                                  # 6 columns x 2 triangles
    tp = space.traced_params(poly, n)
    X, phi, psi, mH, mT, qpts, qw = tp.volume_vars
    assert qpts.shape == (space.num_cells, space.nq_max, 2) and qw.shape == (space.num_cells, space.nq_max)
    assert onp.allclose(onp.asarray(qw).sum(1), 0.5)                # reference area of every cell
    assert onp.all(onp.isfinite(onp.asarray(qpts)))
    tip_nodes = space.cells[cls == 2][0]
    assert onp.all(onp.asarray(mT)[cls == 2][0] == 1.0)
    # tip-cell nodes are not Heaviside-enriched, their neighbours on cut cells are
    mH_n = onp.zeros(space.num_nodes); mH_n[space.cells.ravel()] = onp.asarray(mH).ravel()
    assert onp.all(mH_n[tip_nodes] == 0.0)
    assert mH_n.sum() > 0
    assert problem.num_total_dofs_all_vars == 2 * space.num_nodes + (2 + 8) * (space.num_band + 1)


def test_j_vector_is_smooth_in_the_tip_position_and_mode_i():
    """The virtual-crack-extension J must not show the cell-scale sawtooth of
    the pointwise energy derivative: across one cell it varies by a few %."""
    nx, ny = 20, 81
    h = W / nx
    mesh = _tri_mesh(nx, ny)
    space = CrackSpace(mesh, max_points=8, tip_rings=(0.0, 1.0))
    problem = crack_problem(space, plane_strain_energy(E, NU))
    bc = space.create_bc(problem, fe.DirichletBCConfig(_grip_specs(0)))
    ts = fe.TracedStructure.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=_direct(), linear=True, traced_structure=ts)
    J = space.j_vector_fn(problem, solver, bc, ts, radius=4 * h)
    jt, jn = [], []
    for a in A0 + h * onp.array([0.125, 0.375, 0.625, 0.875]):
        poly, n = space.polyline([(-0.1, 0.0), (a, 0.0)])
        a_t, a_n = J(poly, n)
        jt.append(float(a_t)); jn.append(float(a_n))
    jt, jn = onp.array(jt), onp.array(jn)
    assert onp.all(jt > 0)
    # K_I grows with the crack length (~ sqrt(a) F(a)), so J_t increases by a few
    # % per quarter cell; a sawtooth would show as a jump or a sign change
    assert onp.all(onp.diff(jt) > 0)
    assert onp.all(jt[1:] / jt[:-1] < 1.08)
    fit = onp.polyval(onp.polyfit(onp.arange(4), jt, 1), onp.arange(4))
    assert onp.abs(jt - fit).max() < 0.02 * jt.mean()
    assert onp.all(onp.abs(jn) < 0.05 * jt)                        # pure mode I: J_n ~ 0
    KI, KII = space.sif_from_j(np.asarray(jt), np.asarray(jn), E)
    assert onp.all(onp.abs(onp.asarray(KII)) < 0.05 * onp.asarray(KI))
    th = space.hoop_stress_angle(KI, KII)
    assert onp.all(onp.abs(onp.asarray(th)) < onp.deg2rad(3.0))


def test_j_vector_matches_conforming_fem():
    nx, ny = 20, 81
    h = W / nx
    # tip at the secant midpoint a + h/2 (on the quad diagonal: the tip is shared by
    # two triangles, both get the tip treatment)
    space, problem, bc, ts, solver, poly, n = _xfem(nx, ny, A0 + h / 2)
    assert (space.classes(poly, n) == 2).sum() == 2
    J = space.j_vector_fn(problem, solver, bc, ts, radius=4 * h)
    j_t = float(J(poly, n)[0])
    g_fem = _fem_secant_G(nx, ny - 1, A0)                            # even rows: y = 0 is a node line
    assert abs(j_t - g_fem) / g_fem < 0.08


def test_sif_from_j_round_trip():
    E_p = 100.0
    for KI, KII in [(1.0, 0.0), (1.0, 0.3), (0.8, -0.5)]:
        J_t = (KI ** 2 + KII ** 2) / E_p
        J_n = -2 * KI * KII / E_p
        ki, kii = (float(v) for v in CrackSpace.sif_from_j(J_t, J_n, E_p))
        assert onp.isclose(ki, KI) and onp.isclose(kii, KII)
    # hoop stress angle: pure mode II -> ~ -70.5 deg for K_II > 0
    th = float(CrackSpace.hoop_stress_angle(np.float64(0.0), np.float64(1.0)))
    assert onp.isclose(onp.rad2deg(th), -70.53, atol=0.1)
    assert float(CrackSpace.hoop_stress_angle(np.float64(1.0), np.float64(0.0))) == 0.0


def test_growth_updates_polyline_and_keeps_shapes():
    space, problem, bc, ts, solver, poly, n = _xfem(20, 81, A0 + 0.015)
    poly2, n2 = space.grow(poly, n, 0.3, 0.05)
    assert int(n2) == int(n) + 1 and poly2.shape == poly.shape
    tp1, tp2 = space.traced_params(poly, n), space.traced_params(poly2, n2)
    for v1, v2 in zip(tp1.volume_vars, tp2.volume_vars):
        assert v1.shape == v2.shape
    cls = space.classes(poly2, n2)
    assert (cls == 2).sum() == 1


def test_visualization_mesh_and_vtu_series(tmp_path):
    space, problem, bc, ts, solver, poly, n = _xfem(12, 25, 0.3)
    tp = space.traced_params(poly, n)
    sol = solver(tp, fe.zero_like_initial_guess(problem, bc), traced_structure=ts)
    vis = space.visualization(problem, sol, poly, n)
    cls = space.classes(poly, n)
    n_fan = 6 * (1 + 2 * (len(space.tip_rings) - 2))
    assert len(vis["cells"]) == (cls == 0).sum() + 3 * (cls == 1).sum() + n_fan * (cls == 2).sum()
    assert vis["points"].shape == (3 * len(vis["cells"]), 2) and vis["eps"].shape == (len(vis["cells"]), 2, 2)
    P = vis["points"][vis["cells"]]
    area = 0.5 * onp.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                         - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    assert onp.isclose(area.sum(), W * H, rtol=1e-4)
    # the opening is resolved: crack-face vertices from the two sides differ in u_y
    on = (onp.abs(vis["phi"]) < 1e-6) & (vis["psi"] < -1e-6)
    assert on.sum() > 10
    side = onp.sign(vis["phi"][on])
    assert vis["u"][on, 1][side > 0].mean() - vis["u"][on, 1][side < 0].mean() > 1e-3

    out = tmp_path / "series"
    frames = []
    for k in range(2):
        f = str(out / f"field_{k}.vtu")
        fe.utils.save_sol(fe.Mesh(vis["points"], vis["cells"], ele_type="TRI3"), f,
                          point_infos=[("u", onp.c_[vis["u"], onp.zeros(len(vis["u"]))])],
                          cell_infos=[("class", vis["class"].astype(float)), ("eps", vis["eps"])])
        c = str(out / f"crack_{k}.vtu")
        fe.utils.save_sol(space.polyline_mesh(poly, n), c, cell_infos=[("segment", onp.arange(int(n) - 1, dtype=float))])
        frames.append((float(k), f))
    fe.utils.write_pvd(str(out / "field.pvd"), frames)
    import pyvista as pv
    g = pv.read(frames[0][1])
    assert g.n_cells == len(vis["cells"]) and "u" in g.point_data and "eps" in g.cell_data
    g2 = pv.read(str(out / "crack_0.vtu"))
    assert g2.n_cells == int(n) - 1 and set(g2.celltypes.tolist()) == {3}      # VTK_LINE
    pvd = (out / "field.pvd").read_text()
    assert pvd.count("<DataSet") == 2 and 'file="field_1.vtu"' in pvd
