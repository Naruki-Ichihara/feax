"""feax.experimental.xfem: enrichment functions, cut quadrature, XFEMProblem.

Verification problems with closed-form solutions on a 2 x 1 rectangle whose
interface ``x = X_I`` does not line up with any mesh line:

* **uncut** (``phi > 0`` everywhere): the XFEM problem must reproduce the
  standard feax solution (checks the custom shape functions / quadrature).
* **bimaterial bar** (``nu = 0``, ``E`` jumps at the interface): the exact
  solution is piecewise linear with a kink. The weak-discontinuity
  enrichments (:class:`ShiftedAbs`, :class:`ModifiedAbs`) capture it exactly.
* **separated bodies** (free interface): the exact solution is a rigid
  translation of the right body. The strong-discontinuity enrichments
  (:class:`ShiftedHeaviside`, :class:`Heaviside`) capture it exactly.
"""
import jax
import jax.numpy as np
import numpy as onp
import pytest

import feax as fe
from feax.experimental.xfem import (
    Heaviside, ModifiedAbs, ShiftedAbs, ShiftedHeaviside, XFEMSpace, xfem_problem,
    _cut_triangle, _triangle_rule, _sub_triangle_points,
)

jax.config.update("jax_enable_x64", True)

LX, LY = 2.0, 1.0
X_I = 0.73                 # interface, between mesh lines (nodes at multiples of 0.2)
E_L, E_R = 100.0, 25.0
DELTA = 0.1
TOL = 1e-8


def _direct():
    return fe.DirectSolverOptions(
        solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")


def _mesh(ele_type):
    quad = fe.mesh.rectangle_mesh(10, 5, LX, LY, ele_type="QUAD4")
    if ele_type == "QUAD4":
        return quad
    cells = onp.asarray(quad.cells)
    tris = onp.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 2, 3]]])
    return fe.Mesh(onp.asarray(quad.points), tris, ele_type="TRI3")


def _edge_bcs(fix_right_y=False):
    left = lambda p: np.isclose(p[0], 0.0, TOL)
    right = lambda p: np.isclose(p[0], LX, TOL)
    specs = [fe.DirichletBCSpec(location=left, component="all", value=0.0, variable_index=0),
             fe.DirichletBCSpec(location=right, component=0, value=DELTA, variable_index=0)]
    if fix_right_y:
        specs.append(fe.DirichletBCSpec(location=right, component=1, value=0.0, variable_index=0))
    return fe.DirichletBCConfig(specs)


def _solve(space, material, bc_config):
    problem = xfem_problem(space, material)
    bc = space.create_bc(problem, bc_config)
    ts = fe.TracedStructure.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=_direct(), linear=True,
                              traced_structure=ts)
    sol = solver(space.traced_params(), fe.zero_like_initial_guess(problem, bc),
                 traced_structure=ts)
    return problem, sol


def _two_phase(u_grad, phi, x):
    E = np.where(phi < 0, E_L, E_R)
    eps = 0.5 * (u_grad + u_grad.T)
    return E * eps                       # nu = 0: sigma = E eps


# ---------------------------------------------------------------- unit level

def test_cut_triangle_preserves_area_and_sides():
    v = onp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rule = _triangle_rule(3)
    for p in [onp.array([-1.0, 0.5, 2.0]), onp.array([1.0, 1.0, -0.3]), onp.array([0.2, -0.4, 0.9])]:
        subs = _cut_triangle(v, p)
        assert len(subs) == 3
        area = 0.0
        for sub in subs:
            pts, w = _sub_triangle_points(sub, *rule)
            area += w.sum()
            # the linear field has one sign on every point of each sub-triangle
            lam = onp.c_[1 - pts.sum(1), pts]
            assert onp.all(onp.sign(lam @ p) == onp.sign(lam @ p)[0])
        assert onp.isclose(area, 0.5)
    assert len(_cut_triangle(v, onp.array([1.0, 2.0, 3.0]))) == 1


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
def test_space_structure(ele_type):
    mesh = _mesh(ele_type)
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 0] - X_I, ShiftedHeaviside(), vec=2, ele_type=ele_type)

    # cut cells are exactly the column of cells straddling X_I
    xmin = pts[onp.asarray(mesh.cells)][:, :, 0].min(1)
    xmax = pts[onp.asarray(mesh.cells)][:, :, 0].max(1)
    assert onp.array_equal(space.cut_cells, (xmin < X_I) & (xmax > X_I))
    assert space.num_enriched == 2 * 6                  # two node columns x 6 rows
    assert space.emesh.cells.shape == mesh.cells.shape
    assert onp.isnan(space.emesh.points[space.phantom]).all()
    # quadrature covers every cell exactly once in reference measure
    ref_area = 0.5 if ele_type == "TRI3" else 4.0
    assert onp.allclose(space.qw.sum(1), ref_area)

    problem = xfem_problem(space, _two_phase)
    n_nodes = len(pts)
    assert problem.num_total_dofs_all_vars == 2 * n_nodes + 2 * (space.num_enriched + 1)
    bc = space.create_bc(problem, _edge_bcs())
    assert set(space.phantom_rows(problem)).issubset(set(onp.asarray(bc.bc_rows)))


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
def test_uncut_matches_standard_fem(ele_type):
    mesh = _mesh(ele_type)
    pts = onp.asarray(mesh.points)

    class Reference(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad):
                eps = 0.5 * (u_grad + u_grad.T)
                return E_R * eps + 0.3 * E_R * np.trace(eps) * np.eye(2)
            return stress

    ref = Reference(mesh=mesh, vec=2, dim=2, ele_type=ele_type)
    ref_bc = fe.DirichletBCConfig([s for s in _edge_bcs().specs]).create_bc(ref)
    ref_ts = fe.TracedStructure.from_problem(ref)
    u_ref = fe.create_solver(ref, ref_bc, solver_options=_direct(), linear=True, traced_structure=ref_ts)(
        fe.TracedParams(), fe.zero_like_initial_guess(ref, ref_bc), traced_structure=ref_ts)
    u_ref = onp.asarray(u_ref.dofs).reshape(-1, 2)

    def material(u_grad, phi, x):
        eps = 0.5 * (u_grad + u_grad.T)
        return E_R * eps + 0.3 * E_R * np.trace(eps) * np.eye(2)

    space = XFEMSpace(mesh, pts[:, 0] + 10.0, ShiftedHeaviside(), vec=2, ele_type=ele_type)
    assert space.num_enriched == 0 and not space.cut_cells.any()
    problem, sol = _solve(space, material, _edge_bcs())
    u = onp.asarray(space.nodal_displacement(problem, sol))
    assert onp.allclose(u, u_ref, atol=1e-10)


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
@pytest.mark.parametrize("enrichment, atol", [(ModifiedAbs(), 1e-9), (ShiftedAbs(), 2.5e-2 * DELTA)])
def test_bimaterial_weak_discontinuity(ele_type, enrichment, atol):
    """ModifiedAbs reproduces the kinked field exactly. ShiftedAbs does not:
    |phi| - |phi_j| is linear (not zero) on the blending cells, so the
    enriched ansatz there contains spurious quadratic terms (the classic
    blending-element error of the abs enrichment); the error is first order
    in the mesh size (1.9e-3 -> 9.5e-4 -> 4.8e-4 for nx = 10, 20, 40 on TRI3)
    so it is only approximate on this coarse mesh."""
    mesh = _mesh(ele_type)
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 0] - X_I, enrichment, vec=2, ele_type=ele_type)
    problem, sol = _solve(space, _two_phase, _edge_bcs())

    # exact 1D solution: constant stress, piecewise-linear u_x, u_y = 0
    sigma = DELTA / (X_I / E_L + (LX - X_I) / E_R)
    exact = lambda x: onp.where(x < X_I, sigma / E_L * x,
                                sigma / E_L * X_I + sigma / E_R * (x - X_I))

    u = onp.asarray(space.nodal_displacement(problem, sol))
    assert onp.allclose(u[:, 0], exact(pts[:, 0]), atol=atol)
    assert onp.allclose(u[:, 1], 0.0, atol=atol)

    x_q, u_q, phi_q = (onp.asarray(a) for a in space.quad_fields(problem, sol))
    real = space.qw > 0
    assert onp.allclose(u_q[real][:, 0], exact(x_q[real][:, 0]), atol=atol)
    assert onp.allclose(onp.sign(phi_q[real]), onp.sign(x_q[real][:, 0] - X_I))

    # the enrichment is really used (a != 0) on the enriched nodes only
    _, a = space.split(problem, sol)
    assert onp.abs(onp.asarray(a)[:space.num_enriched, 0, 0]).min() > 1e-6
    assert onp.allclose(onp.asarray(a)[space.phantom], 0.0)


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
@pytest.mark.parametrize("enrichment", [ShiftedHeaviside(), Heaviside()])
def test_separated_bodies_strong_discontinuity(ele_type, enrichment):
    mesh = _mesh(ele_type)
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 0] - X_I, enrichment, vec=2, ele_type=ele_type)
    problem, sol = _solve(space, _two_phase, _edge_bcs(fix_right_y=True))

    # exact: left body at rest, right body translated by (DELTA, 0)
    u = onp.asarray(space.nodal_displacement(problem, sol))
    expect = onp.where(pts[:, 0] > X_I, DELTA, 0.0)
    assert onp.allclose(u[:, 0], expect, atol=1e-9)
    assert onp.allclose(u[:, 1], 0.0, atol=1e-9)

    x_q, u_q, phi_q = (onp.asarray(a) for a in space.quad_fields(problem, sol))
    real = space.qw > 0
    assert onp.allclose(u_q[real][:, 0], onp.where(phi_q[real] > 0, DELTA, 0.0), atol=1e-9)
    assert onp.allclose(u_q[real][:, 1], 0.0, atol=1e-9)

    # jump of DELTA carried by the enrichment: |a_x| = DELTA / 2 on every enriched node
    _, a = space.split(problem, sol)
    assert onp.allclose(onp.abs(onp.asarray(a)[:space.num_enriched, 0, 0]), DELTA / 2, atol=1e-9)


def test_tangent_is_symmetric_and_differentiable():
    mesh = _mesh("TRI3")
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 0] - X_I, ShiftedAbs(), vec=2, ele_type="TRI3")
    problem = xfem_problem(space, _two_phase)
    bc = space.create_bc(problem, _edge_bcs())
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    tp = space.traced_params()
    sol0 = fe.zero_like_initial_guess(problem, bc)
    J = fe.get_jacobian(problem, problem.unflatten_fn_sol_list(sol0), tp, ts)
    Jd = onp.asarray(J.todense()) if hasattr(J, "todense") else onp.asarray(J)
    assert onp.allclose(Jd, Jd.T, atol=1e-10)


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
def test_partial_interface_with_psi(ele_type):
    """Edge-crack-like interface on the left half of a unit square (phi = y - 0.5,
    psi = x - 0.5), equal materials, tension on top/bottom (NLXP3D style)."""
    n = 11                                       # odd: y = 0.5 crosses cell centres
    quad = fe.mesh.rectangle_mesh(n, n, 1.0, 1.0, ele_type="QUAD4")
    if ele_type == "QUAD4":
        mesh = quad
    else:
        c = onp.asarray(quad.cells)
        mesh = fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]),
                       ele_type="TRI3")
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 1] - 0.5, ShiftedHeaviside(), vec=2, ele_type=ele_type,
                      psi=pts[:, 0] - 0.5)

    # QUAD4: chord midpoints at x = (k + 0.5)/11 < 0.5 -> columns 0..4 are interface
    # cells (12 enriched nodes, x <= 5/11); column 5 is the tip cell (cut, no new nodes).
    # TRI3 (quads split along the 0-2 diagonal): the (0,2,3) triangle of column 5 has
    # its chord midpoint at x = 5.25/11 < 0.5, so it is an interface cell too and
    # enriches the top-right node of column 5 (x = 6/11); the (0,1,2) triangle of
    # column 5 and the (0,2,3) triangle of column 6 are the tip cells.
    expect = {"QUAD4": (5, 6, 12, 5 / 11), "TRI3": (11, 13, 13, 6 / 11)}[ele_type]
    assert space.interface_cells.sum() == expect[0]
    assert space.cut_cells.sum() == expect[1]
    assert space.num_enriched == expect[2]
    assert onp.isclose(pts[space.enriched_mask, 0].max(), expect[3])
    seg = space.segments[space.cut_cells]
    assert onp.allclose(seg[:, :, 1], 0.5)                     # chord on y = 0.5

    def material(u_grad, phi, x):
        eps = 0.5 * (u_grad + u_grad.T)
        return 0.3 * np.trace(eps) * np.eye(2) + 2 * 0.4 * eps

    problem = xfem_problem(space, material)
    bottom = lambda p: np.isclose(p[1], 0.0, TOL)
    top = lambda p: np.isclose(p[1], 1.0, TOL)
    bc = space.create_bc(problem, fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=bottom, component=0, value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=bottom, component=1, value=-DELTA, variable_index=0),
        fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=top, component=1, value=DELTA, variable_index=0),
    ]))
    ts = fe.TracedStructure.from_problem(problem)
    solver = fe.create_solver(problem, bc, solver_options=_direct(), linear=True, traced_structure=ts)
    sol = solver(space.traced_params(), fe.zero_like_initial_guess(problem, bc), traced_structure=ts)

    # equal materials -> antisymmetric about y = 0.5. Only the QUAD4 mesh is
    # mirror-symmetric (the TRI3 diagonals all run bottom-left to top-right).
    # The interface runs through the QUAD4 cell centres, where the centre
    # level-set value (exactly 0) is nudged to +threshold*scale for the
    # sub-triangulation, so the symmetry holds to O(threshold) ~ 1e-6, not 1e-10.
    u = onp.asarray(space.nodal_displacement(problem, sol))
    assert onp.all(onp.isfinite(u))
    if ele_type == "QUAD4":
        key = onp.round(pts * 1e6).astype(onp.int64)
        index = {(kx, ky): i for i, (kx, ky) in enumerate(map(tuple, key))}
        mirror = onp.array([index[(kx, int(round(1e6)) - ky)] for kx, ky in map(tuple, key)])
        assert onp.allclose(u[:, 0], u[mirror, 0], atol=1e-5)
        assert onp.allclose(u[:, 1], -u[mirror, 1], atol=1e-5)

    # opening: positive, largest at the free edge, decreasing to the tip; no sliding
    x_mid, jump = (onp.asarray(a) for a in space.interface_jump(problem, sol))
    order = onp.argsort(x_mid[:, 0])
    opening = jump[order, 1]
    assert onp.all(opening > 0)
    assert onp.all(onp.diff(opening) < 0)
    assert opening[0] > 0.5 * DELTA
    if ele_type == "QUAD4":
        assert onp.allclose(jump[:, 0], 0.0, atol=1e-5)
    else:
        # the one-directional diagonals bias TRI3: a small sliding (~5% of the
        # opening) appears, and flips sign on the mirrored-diagonal mesh (checked
        # below), so it is mesh bias, not an orientation bug in the XFEM kernel
        assert onp.abs(jump[:, 0]).max() < 0.1 * opening.max()
    _, a = space.split(problem, sol)
    assert onp.allclose(onp.asarray(a)[space.phantom], 0.0)

    if ele_type == "TRI3":
        c = onp.asarray(quad.cells)
        mesh_m = fe.Mesh(pts, onp.concatenate([c[:, [0, 1, 3]], c[:, [1, 2, 3]]]), ele_type="TRI3")
        space_m = XFEMSpace(mesh_m, pts[:, 1] - 0.5, ShiftedHeaviside(), vec=2, ele_type="TRI3",
                            psi=pts[:, 0] - 0.5)
        problem_m = xfem_problem(space_m, material)
        bc_m = space_m.create_bc(problem_m, fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=bottom, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=bottom, component=1, value=-DELTA, variable_index=0),
            fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=top, component=1, value=DELTA, variable_index=0),
        ]))
        ts_m = fe.TracedStructure.from_problem(problem_m)
        sol_m = fe.create_solver(problem_m, bc_m, solver_options=_direct(), linear=True,
                                 traced_structure=ts_m)(
            space_m.traced_params(), fe.zero_like_initial_guess(problem_m, bc_m), traced_structure=ts_m)
        u_m = onp.asarray(space_m.nodal_displacement(problem_m, sol_m))
        key = onp.round(pts * 1e6).astype(onp.int64)
        index = {(kx, ky): i for i, (kx, ky) in enumerate(map(tuple, key))}
        mirror = onp.array([index[(kx, int(round(1e6)) - ky)] for kx, ky in map(tuple, key)])
        # the mirrored-diagonal mesh is the mirror image of the problem
        assert onp.allclose(u[:, 0], u_m[mirror, 0], atol=1e-12)
        assert onp.allclose(u[:, 1], -u_m[mirror, 1], atol=1e-12)
        x_m, jump_m = (onp.asarray(a) for a in space_m.interface_jump(problem_m, sol_m))
        order_m = onp.argsort(x_m[:, 0])
        assert onp.allclose(jump_m[order_m, 1], opening, atol=1e-12)
        assert onp.allclose(jump_m[order_m, 0], -jump[order, 0], atol=1e-12)


# ------------------------------------------------------------------ cohesive zone

from feax.experimental.xfem import CohesiveZone, bilinear_tsl, exponential_tsl


def _czm_setup(ele_type, cz, E=10.0):
    """Full-width interface at y = 0.5 of a unit square, nu = 0 (1D-exact)."""
    n = 11
    quad = fe.mesh.rectangle_mesh(n, n, 1.0, 1.0, ele_type="QUAD4")
    if ele_type == "QUAD4":
        mesh = quad
    else:
        c = onp.asarray(quad.cells)
        mesh = fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]),
                       ele_type="TRI3")
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 1] - 0.5, ShiftedHeaviside(), vec=2, ele_type=ele_type, cohesive=cz)

    def material(u_grad, phi, x):
        return E * 0.5 * (u_grad + u_grad.T)            # nu = 0

    problem = xfem_problem(space, material)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    bottom = lambda p: np.isclose(p[1], 0.0, TOL)
    top = lambda p: np.isclose(p[1], 1.0, TOL)

    def make_bc(d):
        return space.create_bc(problem, fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=bottom, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=bottom, component=1, value=-d, variable_index=0),
            fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=top, component=1, value=d, variable_index=0),
        ]))

    bc0 = make_bc(0.0)
    solver = fe.create_solver(problem, bc0, solver_options=_direct(), linear=False,
                              newton_options=fe.NewtonOptions(tol=1e-11, rel_tol=1e-12, max_iter=60),
                              traced_params=space.traced_params(), traced_structure=ts)

    def solve(d, delta_max, u_prev=None):
        bc = make_bc(d)
        u0 = fe.zero_like_initial_guess(problem, bc) if u_prev is None else \
            np.asarray(u_prev).at[bc.bc_rows].set(bc.bc_vals)
        return solver(space.traced_params(delta_max=delta_max), u0, bc=bc, traced_structure=ts)

    return pts, space, problem, solve


def _exact_opening(cz, E, d, delta_max_scalar):
    """1D series: bar strain (2d - dn) (L = 1) in equilibrium with t(dn)."""
    f = lambda dn: float(cz.traction(np.array([dn]), np.array([delta_max_scalar]))[0]) - E * (2 * d - dn)
    lo, hi = 0.0, 2 * d
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
@pytest.mark.parametrize("cz", [
    CohesiveZone(bilinear_tsl, K=50.0, sigma_c=1.0, Gamma=0.05),
    CohesiveZone(exponential_tsl, sigma_c=1.0, Gamma=0.05),
])
def test_cohesive_interface_1d_exact(ele_type, cz):
    E = 10.0
    pts, space, problem, solve = _czm_setup(ele_type, cz, E)
    assert onp.allclose(space.iqw.sum(), 1.0)                 # chord length = plate width
    dm = space.zero_history()

    # ---- loading step 1: 1D-exact interface opening -----------------------
    d1 = 0.02
    sol1 = solve(d1, dm)
    st = space.interface_state(problem, sol1, dm)
    dn_exact = _exact_opening(cz, E, d1, 0.0)
    assert onp.allclose(onp.asarray(st["delta_n"]), dn_exact, atol=1e-8)
    assert onp.allclose(onp.asarray(st["delta_t"]), 0.0, atol=1e-6)
    assert onp.allclose(onp.asarray(st["traction"]), E * (2 * d1 - dn_exact), atol=1e-7)
    # bar: u_y = sign * dn/2 + eps (y - 0.5), u_x = 0
    u = onp.asarray(space.nodal_displacement(problem, sol1))
    eps = 2 * d1 - dn_exact
    u_exact = onp.sign(pts[:, 1] - 0.5) * dn_exact / 2 + eps * (pts[:, 1] - 0.5)
    assert onp.allclose(u[:, 1], u_exact, atol=1e-8)
    assert onp.allclose(u[:, 0], 0.0, atol=1e-6)      # QUAD4: O(threshold) centre-nudge asymmetry

    # ---- step 2: unload with history -> secant branch -----------------------
    dm = space.update_history(dm, problem, sol1)
    assert onp.allclose(onp.asarray(dm)[space.cohesive_cells], dn_exact, atol=1e-8)
    d2 = 0.5 * d1
    sol2 = solve(d2, dm, sol1)
    st2 = space.interface_state(problem, sol2, dm)
    dn2 = _exact_opening(cz, E, d2, dn_exact)
    assert onp.allclose(onp.asarray(st2["delta_n"]), dn2, atol=1e-8)
    t_max = float(cz.traction(np.array([dn_exact]), np.array([dn_exact]))[0])   # loading traction at delta_max
    t_secant = t_max / dn_exact * dn2                                            # unloading on the secant
    assert onp.allclose(onp.asarray(st2["traction"]), t_secant, atol=1e-7)
    assert dn2 < dn_exact


def test_cohesive_tangent_symmetric():
    cz = CohesiveZone(bilinear_tsl, K=50.0, sigma_c=1.0, Gamma=0.05, contact_penalty=100.0)
    pts, space, problem, solve = _czm_setup("TRI3", cz)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    sol = solve(0.01, space.zero_history())
    J = fe.get_jacobian(problem, problem.unflatten_fn_sol_list(sol), space.traced_params(), ts)
    Jd = onp.asarray(J.todense())
    assert onp.allclose(Jd, Jd.T, atol=1e-9)


def test_cohesive_contact_penalty_resists_interpenetration():
    E = 10.0
    cz_free = CohesiveZone(bilinear_tsl, K=50.0, sigma_c=1.0, Gamma=0.05)
    cz_pen = CohesiveZone(bilinear_tsl, K=50.0, sigma_c=1.0, Gamma=0.05, contact_penalty=1e3)
    for cz, expect_pen in [(cz_free, False), (cz_pen, True)]:
        pts, space, problem, solve = _czm_setup("QUAD4", cz, E)
        sol = solve(-0.01, space.zero_history())                 # compression
        st = space.interface_state(problem, sol)
        dn = float(onp.asarray(st["delta_n"]).mean())
        if expect_pen:
            assert onp.isclose(dn, -2 * 0.01 / (1 + 1e3 / E), atol=1e-8)   # series with kp
        else:
            assert onp.isclose(dn, -2 * 0.01, atol=1e-8)                   # free interpenetration


def test_secant_mode_matches_implicit_solution():
    """Explicit damage update (secant potential + history fixed point) converges
    to the same state as the implicit loading/unloading switch, on the
    1D-exact interface problem, for loading past the peak and for unloading."""
    # softening slope sigma_c/(delta_f - delta_0) = 2.6 < E: no snap-back, so a
    # displacement-controlled equilibrium exists on the softening branch
    E = 10.0
    cz_imp = CohesiveZone(bilinear_tsl, K=50.0, sigma_c=1.0, Gamma=0.2)
    cz_sec = CohesiveZone(bilinear_tsl, K=50.0, sigma_c=1.0, Gamma=0.2, secant=True)
    pts, space_i, problem_i, solve_i = _czm_setup("QUAD4", cz_imp, E)
    pts, space_s, problem_s, solve_s = _czm_setup("QUAD4", cz_sec, E)
    # the secant setup needs the pieces solve_step wants: rebuild bc / solver access
    bottom = lambda p: np.isclose(p[1], 0.0, TOL)
    top = lambda p: np.isclose(p[1], 1.0, TOL)

    def make_bc(d):
        return space_s.create_bc(problem_s, fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=bottom, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=bottom, component=1, value=-d, variable_index=0),
            fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=top, component=1, value=d, variable_index=0),
        ]))

    ts = fe.TracedStructure.from_problem(problem_s)
    solver = fe.create_solver(problem_s, make_bc(0.0), solver_options=_direct(), linear=False,
                              newton_options=fe.NewtonOptions(tol=1e-11, rel_tol=1e-12, max_iter=60),
                              traced_params=space_s.traced_params(), traced_structure=ts)
    dm_i = space_i.zero_history()
    dm_s = space_s.zero_history()
    u_s = fe.zero_like_initial_guess(problem_s, make_bc(0.0))
    # loading in 3 steps up to d = 0.12 (the interface passes the peak delta0 = 0.02 at
    # d = 0.06), then unloading to 0.06
    for d in (0.04, 0.08, 0.12, 0.06):
        sol_i = solve_i(d, dm_i)
        dm_i = space_i.update_history(dm_i, problem_i, sol_i)
        bc = make_bc(d)
        u_s = np.asarray(u_s).at[bc.bc_rows].set(bc.bc_vals)
        # the history fixed point converges linearly on the softening branch
        u_s, dm_s, n_outer = space_s.solve_step(problem_s, solver, bc, ts, u_s, dm_s, tol=1e-8, max_outer=80)
        assert n_outer < 80
        st_i = space_i.interface_state(problem_i, sol_i, dm_i)
        st_s = space_s.interface_state(problem_s, u_s, dm_s)
        assert onp.allclose(onp.asarray(st_s["delta_n"]), onp.asarray(st_i["delta_n"]), atol=1e-6), d
        assert onp.allclose(onp.asarray(dm_s)[space_s.cohesive_cells], onp.asarray(dm_i)[space_i.cohesive_cells], atol=1e-6)
    # unloading: traction on the secant branch, below the loading-branch value
    dn = float(onp.asarray(st_s["delta_n"]).mean())
    assert dn < float(onp.asarray(dm_s)[space_s.cohesive_cells].mean())


@pytest.mark.parametrize("ele_type", ["TRI3", "QUAD4"])
def test_crack_aware_remesh(ele_type, tmp_path):
    """Remesh: conforming, area-preserving; vertices duplicated across the
    interface only where requested; the jump appears as a gap."""
    n = 11
    quad = fe.mesh.rectangle_mesh(n, n, 1.0, 1.0, ele_type="QUAD4")
    if ele_type == "QUAD4":
        mesh = quad
    else:
        c = onp.asarray(quad.cells)
        mesh = fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]), ele_type="TRI3")
    pts = onp.asarray(mesh.points)
    space = XFEMSpace(mesh, pts[:, 1] - 0.5, ShiftedHeaviside(), vec=2, ele_type=ele_type, psi=pts[:, 0] - 0.5)

    def material(u_grad, phi, x):
        eps = 0.5 * (u_grad + u_grad.T)
        return 0.3 * np.trace(eps) * np.eye(2) + 0.8 * eps

    problem = xfem_problem(space, material)
    bottom = lambda p: np.isclose(p[1], 0.0, TOL)
    top = lambda p: np.isclose(p[1], 1.0, TOL)
    bc = space.create_bc(problem, fe.DirichletBCConfig([
        fe.DirichletBCSpec(location=bottom, component="all", value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=top, component=0, value=0.0, variable_index=0),
        fe.DirichletBCSpec(location=top, component=1, value=DELTA, variable_index=0),
    ]))
    ts = fe.TracedStructure.from_problem(problem)
    sol = fe.create_solver(problem, bc, solver_options=_direct(), linear=True, traced_structure=ts)(
        space.traced_params(), fe.zero_like_initial_guess(problem, bc), traced_structure=ts)

    def area(rm):
        P = rm["points"][rm["cells"]]
        return 0.5 * onp.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                             - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])).sum()

    rm_all = space.remesh(problem, sol, split="all")
    rm_none = space.remesh(problem, sol, split="none")
    assert onp.isclose(area(rm_all), 1.0) and onp.isclose(area(rm_none), 1.0)
    assert len(rm_all["cells"]) == len(rm_none["cells"])
    assert len(rm_all["points"]) > len(rm_none["points"])          # duplicated along the interface
    assert len(rm_none["crack_lines"]) == 0 and len(rm_all["crack_lines"]) > 0
    # opening across the free interface: duplicated vertices on the crack faces differ in u_y
    key = onp.round(rm_all["points"] * 1e9).astype(onp.int64)
    _, inv, cnt = onp.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    gaps = [onp.ptp(rm_all["u"][inv == k, 1]) for k in onp.flatnonzero(cnt > 1)]
    assert max(gaps) > 0.5 * DELTA
    # nodal values of the remesh match the XFEM nodal displacement at the original nodes
    un = onp.asarray(space.nodal_displacement(problem, sol))
    for i in range(0, len(pts), 7):
        j = onp.flatnonzero(onp.all(onp.isclose(rm_none["points"], pts[i]), axis=1))
        assert len(j) == 1 and onp.allclose(rm_none["u"][j[0]], un[i], atol=1e-6)
    f = str(tmp_path / "remesh.vtu")
    space.save_remesh_vtu(f, problem, sol, material=material, crack_filename=str(tmp_path / "crack.vtu"))
    import pyvista as pv
    g = pv.read(f)
    assert g.n_cells == len(rm_all["cells"]) and "stress" in g.cell_data and "u" in g.point_data
    assert pv.read(str(tmp_path / "crack.vtu")).n_cells == len(rm_all["crack_lines"])
