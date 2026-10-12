"""Verification of :mod:`feax.gene.length_scale`.

The geometric minimum-length-scale constraints of Zhou, Lazarov, Wang &
Sigmund, CMAME 293 (2015) 266-282.  Checked on four levels:

1. **Sizing formulas** -- the analytic width relations are checked against the
   width the filters actually produce, by bisecting on the filtered field.
2. **Detection** -- the constraints must fire on a member narrower than the
   target and fall silent on a wider one, for both phases.
3. **Projection sharpness** -- a compliant design only reads as compliant once
   ``beta`` is large enough; this pins the ``beta >= 16`` guidance down with
   numbers rather than leaving it as folklore.
4. **Plumbing** -- differentiability, the design mask, parameter validation.
"""

import math

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.optimize import brentq

import feax as fe
import feax.gene as gene
from feax.gene.length_scale import (
    create_geometric_length_scale_fn,
    create_morphological_length_scale_fn,
    epsilon_for_radius,
    helmholtz_equivalent_radius,
    helmholtz_radius_for_minimum_width,
    minimum_beta,
    minimum_width,
    minimum_width_helmholtz,
    radius_for_minimum_width,
)

pytestmark = pytest.mark.cpu

ETA_E, ETA_D = 0.7, 0.3
TARGET = 8.0                     # mm, the minimum width used throughout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    """A 1 mm structured strip, wide enough that a 20 mm bar is interior."""
    nx, ny = 60, 40
    lx, ly = 60.0, 40.0
    mesh = fe.mesh.rectangle_mesh(Nx=nx, Ny=ny, domain_x=lx, domain_y=ly,
                                  ele_type="QUAD4")
    pts = onp.asarray(mesh.points)
    return mesh, pts, lx, ly, 1.0


def _bar(pts, ly, width):
    """A horizontal solid bar of the given width through the middle."""
    return np.asarray((onp.abs(pts[:, 1] - ly / 2) <= width / 2).astype(float))


def _fields(filt, rho, beta):
    rho_tilde = filt(rho)
    rho_bar = gene.heaviside_projection(rho_tilde, beta=beta, threshold=0.5)
    return rho_tilde, rho_bar


# ---------------------------------------------------------------------------
# 1. Sizing formulas
# ---------------------------------------------------------------------------

def test_width_formulas_are_self_inverse():
    for eta in (0.55, 0.7, 0.8):
        r = radius_for_minimum_width(TARGET, eta)
        assert minimum_width(r, eta) == pytest.approx(TARGET, rel=1e-12)
        rh = helmholtz_radius_for_minimum_width(TARGET, eta)
        assert minimum_width_helmholtz(rh, eta) == pytest.approx(TARGET, rel=1e-12)


def test_hat_formula_matches_the_cone_filter(grid):
    """Bisect on the filtered field: the width at which it reaches eta_e is
    what the constraint enforces, and the hat formula must predict it."""
    mesh, pts, _lx, ly, _h = grid
    R = radius_for_minimum_width(TARGET, ETA_E)
    filt = jax.jit(gene.create_density_filter(mesh, radius=R, weight_type="cone"))
    mid = onp.abs(pts[:, 1] - ly / 2) < 1e-9

    def peak(w):
        return float(onp.asarray(filt(_bar(pts, ly, w)))[mid].max())

    w_star = brentq(lambda w: peak(w) - ETA_E, 0.5, 30.0, xtol=1e-3)
    assert w_star == pytest.approx(TARGET, abs=0.5)


def test_helmholtz_formula_matches_the_helmholtz_filter(grid):
    """And the Helmholtz formula predicts the Helmholtz filter -- while the hat
    formula, reached through the equivalent radius, does not."""
    mesh, pts, _lx, ly, _h = grid
    r_h = helmholtz_radius_for_minimum_width(TARGET, ETA_E)
    filt = jax.jit(gene.create_helmholtz_filter(mesh, radius=r_h))
    mid = onp.abs(pts[:, 1] - ly / 2) < 1e-9

    def peak(w):
        return float(onp.asarray(filt(_bar(pts, ly, w)))[mid].max())

    w_star = brentq(lambda w: peak(w) - ETA_E, 0.5, 30.0, xtol=1e-3)
    assert w_star == pytest.approx(TARGET, abs=0.5)

    # The trap this guards: second-moment equivalence does not carry over.
    via_hat = minimum_width(helmholtz_equivalent_radius(r_h), ETA_E)
    assert via_hat > 1.25 * w_star


def test_epsilon_curve_reproduces_the_papers_points():
    """Fig. 8 of the paper: eps = 1e-6, 1e-7, 1e-8 at r = 3, 7, 10.

    The three are not collinear in log eps, so a single log-linear fit misses
    the middle one by 30%; the interpolation has to be piecewise.
    """
    assert epsilon_for_radius(3.0) == pytest.approx(1e-6, rel=1e-9)
    assert epsilon_for_radius(7.0) == pytest.approx(1e-7, rel=1e-9)
    assert epsilon_for_radius(10.0) == pytest.approx(1e-8, rel=1e-9)
    # Monotone: a coarser filter needs more slack.
    rs = [3.0, 5.0, 8.0, 11.0]
    eps = [epsilon_for_radius(r) for r in rs]
    assert all(a > b for a, b in zip(eps, eps[1:]))


# ---------------------------------------------------------------------------
# 2. Detection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def constraint(grid):
    mesh, pts, _lx, ly, h = grid
    R = radius_for_minimum_width(TARGET, ETA_E)
    filt = jax.jit(gene.create_density_filter(mesh, radius=R, weight_type="cone"))
    ls = create_geometric_length_scale_fn(
        mesh, filter_radius=R, ele_type="QUAD4", dim=2,
        eta_e=ETA_E, eta_d=ETA_D, element_size=h)
    return filt, ls, R


def test_thin_solid_member_is_flagged_and_a_wide_one_is_not(grid, constraint):
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, _R = constraint
    g_thin, _ = ls(*_fields(filt, _bar(pts, ly, 4.0), 32.0))
    g_wide, _ = ls(*_fields(filt, _bar(pts, ly, 14.0), 32.0))
    assert float(g_thin) > 1e-7, "a 4 mm member against an 8 mm target must fire"
    assert float(g_wide) < 1e-12, "a 14 mm member must be silent"
    assert float(g_thin) > 1e4 * float(g_wide)


def test_thin_void_gap_is_flagged_and_a_wide_one_is_not(grid, constraint):
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, _R = constraint
    _, g_thin = ls(*_fields(filt, 1.0 - _bar(pts, ly, 4.0), 32.0))
    _, g_wide = ls(*_fields(filt, 1.0 - _bar(pts, ly, 14.0), 32.0))
    assert float(g_thin) > 1e-7
    assert float(g_wide) < 1e-12


def test_solid_and_void_constraints_are_mirror_images(grid, constraint):
    """Complementing the design swaps the two phases, so g_s and g_v swap."""
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, _R = constraint
    rho = _bar(pts, ly, 4.0)
    gs_a, gv_a = ls(*_fields(filt, rho, 32.0))
    gs_b, gv_b = ls(*_fields(filt, 1.0 - rho, 32.0))
    assert float(gs_a) == pytest.approx(float(gv_b), rel=0.05)
    assert float(gv_a) == pytest.approx(float(gs_b), rel=0.05)


def test_violation_grows_as_the_member_thins(grid, constraint):
    """Thinner scores worse -- but only while the member still exists.

    Widths are spaced 2 mm apart because a 1 mm mesh rasterises 7 mm and 6 mm
    to the same seven rows of nodes, which would compare a design with itself.
    """
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, _R = constraint
    vals = [float(ls(*_fields(filt, _bar(pts, ly, w), 32.0))[0])
            for w in (12.0, 10.0, 9.0, 7.0, 5.0)]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals


def test_violation_falls_again_once_the_member_stops_existing(grid, constraint):
    """Below 2R(1 - sqrt(1/2)) the filtered peak drops under the blueprint
    threshold, the member is gone from the physical field, and the indicator
    stops marking it.  g_s must therefore be read as "a feature here is too
    thin", never as a monotone measure of how thin."""
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, R = constraint
    vanish = 2 * R * (1 - math.sqrt(0.5))
    assert 4.0 < vanish < 6.0, vanish
    worst = float(ls(*_fields(filt, _bar(pts, ly, 5.0), 32.0))[0])
    gone = float(ls(*_fields(filt, _bar(pts, ly, 3.0), 32.0))[0])
    assert gone < 0.01 * worst


# ---------------------------------------------------------------------------
# 3. Projection sharpness
# ---------------------------------------------------------------------------

def test_compliant_design_only_reads_compliant_at_high_beta(grid, constraint):
    """The reason :func:`minimum_beta` exists.

    The indicator weights the penalty by the PROJECTED field, so a soft
    projection collects penalty across the whole transition band whether or not
    the design meets the width.  A comfortably wide design must therefore still
    score badly at low beta and cleanly at high beta.
    """
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, R = constraint
    rho = _bar(pts, ly, 16.0)                      # twice the target
    eps = epsilon_for_radius(R / 1.0)
    scores = {b: float(ls(*_fields(filt, rho, float(b)))[0]) / eps
              for b in (2, 4, 8, 16, 32)}

    assert scores[2] > 100, scores          # wide design, still "violating"
    assert scores[32] < 1.0, scores         # and compliant once sharp
    assert all(a > b for a, b in zip(
        [scores[b] for b in (2, 4, 8, 16, 32)][:-1],
        [scores[b] for b in (2, 4, 8, 16, 32)][1:])), scores
    # The documented threshold has to sit where the crossing happens.
    assert scores[int(minimum_beta())] < scores[8]


def test_minimum_beta_is_the_documented_value():
    assert minimum_beta() == 16.0


# ---------------------------------------------------------------------------
# 4. Plumbing
# ---------------------------------------------------------------------------

def test_gradients_are_finite_and_match_finite_differences(grid, constraint):
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, _R = constraint

    def g_of(rho):
        return ls(*_fields(filt, rho, 16.0))[0]

    rho = _bar(pts, ly, 5.0)
    grad = jax.grad(g_of)(rho)
    assert bool(np.all(np.isfinite(grad)))
    assert float(np.max(np.abs(grad))) > 0.0

    # Central differences on the few most sensitive entries.
    idx = onp.argsort(-onp.abs(onp.asarray(grad)))[:5]
    step = 1e-5
    for i in idx:
        plus = rho.at[i].add(step)
        minus = rho.at[i].add(-step)
        fd = (float(g_of(plus)) - float(g_of(minus))) / (2 * step)
        assert fd == pytest.approx(float(grad[i]), rel=2e-3, abs=1e-12)


def test_mask_restricts_the_integral(grid):
    """Masking out the half of the domain holding the thin bar removes it."""
    mesh, pts, _lx, ly, h = grid
    R = radius_for_minimum_width(TARGET, ETA_E)
    filt = jax.jit(gene.create_density_filter(mesh, radius=R, weight_type="cone"))
    rho = _bar(pts, ly, 4.0)

    full = create_geometric_length_scale_fn(
        mesh, filter_radius=R, eta_e=ETA_E, eta_d=ETA_D, element_size=h)
    # Keep only x < lx/4, which the bar also crosses -- so use a y-mask that
    # excludes the bar's neighbourhood entirely.
    away = np.asarray((onp.abs(pts[:, 1] - ly / 2) > 12.0).astype(float))
    masked = create_geometric_length_scale_fn(
        mesh, filter_radius=R, eta_e=ETA_E, eta_d=ETA_D, element_size=h,
        mask=away)

    g_full = float(full(*_fields(filt, rho, 32.0))[0])
    g_mask = float(masked(*_fields(filt, rho, 32.0))[0])
    assert g_full > 1e-7
    assert g_mask < 1e-3 * g_full


def test_reported_metadata(grid, constraint):
    _mesh, _pts, _lx, _ly, h = grid
    _filt, ls, R = constraint
    assert ls.filter_radius == pytest.approx(R)
    assert ls.element_size == pytest.approx(h)
    assert ls.r == pytest.approx(R / h)
    assert ls.c == pytest.approx((R / h) ** 4)
    assert ls.min_solid_width == pytest.approx(TARGET, rel=1e-9)
    assert ls.min_void_width == pytest.approx(minimum_width(R, 1.0 - ETA_D))


def test_is_jittable(grid, constraint):
    _mesh, pts, _lx, ly, _h = grid
    filt, ls, _R = constraint
    rho = _bar(pts, ly, 5.0)
    eager = ls(*_fields(filt, rho, 16.0))
    jitted = jax.jit(lambda z: ls(*_fields(filt, z, 16.0)))(rho)
    for a, b in zip(eager, jitted):
        assert float(a) == pytest.approx(float(b), rel=1e-10)


@pytest.mark.parametrize("eta_d,eta_e", [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.7),
                                         (0.3, 1.0)])
def test_threshold_ordering_is_validated(grid, eta_d, eta_e):
    mesh, _pts, _lx, _ly, h = grid
    with pytest.raises(ValueError):
        create_geometric_length_scale_fn(
            mesh, filter_radius=8.0, eta_e=eta_e, eta_d=eta_d, element_size=h)


@pytest.mark.parametrize("eta", [0.0, 1.0, -0.5, 1.5])
def test_width_formulas_reject_bad_thresholds(eta):
    for fn in (minimum_width, minimum_width_helmholtz,
               radius_for_minimum_width, helmholtz_radius_for_minimum_width):
        with pytest.raises(ValueError):
            fn(8.0, eta)


# ── The morphological measure ──────────────────────────────────────────────
def _sites(n, h, dim):
    """Cell centres of a uniform dim-dimensional grid of n cells per edge."""
    ax = (onp.arange(n) + 0.5) * h
    return onp.stack(onp.meshgrid(*[ax] * dim, indexing="ij"), -1).reshape(-1, dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_morphological_resolution_guard_tracks_radius_over_h(dim):
    """The guard is a statement about radius/h, so it must read the same in 2-D
    and 3-D.  An earlier version dropped the volume of the dim-ball outside 2-D
    and passed 3-D meshes 1.61x coarser than the same criterion allows."""
    h = 1.0
    for min_width, want_ok in ((3.0, False), (5.0, True), (9.0, True)):
        pts = _sites(int(24 / h), h, dim)
        if want_ok:
            fn = create_morphological_length_scale_fn(pts, min_width=min_width)
            assert fn.sites_per_radius == pytest.approx(0.5 * min_width / h,
                                                        rel=0.15)
        else:
            with pytest.raises(ValueError, match="too small for this mesh"):
                create_morphological_length_scale_fn(pts, min_width=min_width)


@pytest.mark.parametrize("rho_value", [0.0, 1.0])
def test_morphological_gradient_is_finite_when_satisfied(rho_value):
    """A uniform field violates nothing, so both measures are exactly zero --
    and ``0 ** (1/p)`` has an infinite derivative.  Without the floor inside the
    p-norm the optimiser is handed nan precisely when the constraint is
    comfortably met."""
    pts = _sites(20, 1.0, 3)
    fn = create_morphological_length_scale_fn(pts, min_width=6.0)
    rho = np.full(pts.shape[0], rho_value)
    g_s, g_v = fn(rho)
    assert float(g_s) == pytest.approx(0.0, abs=1e-12)
    assert float(g_v) == pytest.approx(0.0, abs=1e-12)
    for i in (0, 1):
        grad = jax.grad(lambda r: fn(r)[i])(rho)
        assert bool(np.all(np.isfinite(grad)))


def test_morphological_fires_on_a_thin_member_and_not_on_a_wide_one():
    """The measure it exists for: a slab thinner than the target is flagged, a
    slab comfortably wider is not.

    The wide slab does not read exactly zero -- the softened erosion leaves a
    residue along its two faces -- so the test is on the SEPARATION.  Measured
    at the default sharpness, 0.807 against 0.049, a factor of 17.
    """
    h, n = 1.0, 40
    pts = _sites(n, h, 2)
    fn = create_morphological_length_scale_fn(pts, min_width=7.0)
    out = {}
    for width in (4.0, 12.0):
        centre = 0.5 * n * h
        rho = np.asarray((onp.abs(pts[:, 0] - centre) <= 0.5 * width).astype(float))
        out[width] = float(fn(rho)[0])
    assert out[4.0] > 10.0 * out[12.0]
