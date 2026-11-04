"""feax.experimental.stripe: stripe patterns aligned with a line field (2D port of
Knöppel et al. 2015).

Checks with closed-form patterns on a unit square:

* uniform field at angle phi: the phase is ``λ (−sin φ x + cos φ y) + c`` exactly
  (stripes run ALONG the field, spacing ``2π/λ``), no branch points;
* a line field straddling the angle wrap (sheet crossings on half the edges)
  gives the same stripes (``cos α`` is sheet-invariant);
* a vortex field (tangential directions) gives circular stripes ``α ≈ λ r``,
  with the field singularity detected;
* smoothing removes noise from a field; VTU output round-trips.
"""
import numpy as onp
import pytest

import feax as fe
from feax.experimental.stripe import StripePattern, field_index, line_field_angle, smooth_line_field


def _mesh(n=30):
    quad = fe.mesh.rectangle_mesh(n, n, 1.0, 1.0, ele_type="QUAD4")
    c = onp.asarray(quad.cells)
    return fe.Mesh(onp.asarray(quad.points), onp.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]]), ele_type="TRI3")


def _corner_gradients(spn):
    pts = onp.asarray(spn.mesh.points)
    C = spn.mesh.cells
    P = pts[C]
    cp = spn.corner_phase
    A = onp.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=1)
    rhs = onp.stack([cp[:, 1] - cp[:, 0], cp[:, 2] - cp[:, 0]], axis=1)
    return onp.linalg.solve(A, rhs[..., None])[..., 0]


@pytest.mark.parametrize("phi_deg", [0.0, 30.0, 90.0])
def test_uniform_field_gives_exact_parallel_stripes(phi_deg):
    mesh = _mesh()
    pts = onp.asarray(mesh.points)
    phi = onp.deg2rad(phi_deg)
    lam = 2 * onp.pi / 0.2
    spn = StripePattern.compute(mesh, onp.full(len(pts), phi), lam)
    assert spn.orientable and not spn.zero_index.any() and not spn.field_index.any()
    # the phase is defined up to a global sign (the two roots of the line field)
    # and a constant; the stripes themselves are identical
    expect = lam * (-onp.sin(phi) * pts[:, 0] + onp.cos(phi) * pts[:, 1])
    diffs = [onp.angle(onp.exp(1j * (sgn * spn.nodal_phase() - expect))) for sgn in (1.0, -1.0)]
    assert min(d.std() for d in diffs) < 1e-10
    g = _corner_gradients(spn)
    gexp = lam * onp.array([-onp.sin(phi), onp.cos(phi)])
    assert onp.allclose(g, gexp, atol=1e-9) or onp.allclose(g, -gexp, atol=1e-9)
    # level set sin(alpha) has zeros every half spacing along the normal direction
    s = spn.level_set()
    assert onp.allclose(onp.abs(onp.exp(1j * spn.nodal_phase()) - spn.psi), 0.0)
    assert s.shape == (len(pts),)


def test_sheet_crossings_do_not_change_the_stripes():
    mesh = _mesh()
    pts = onp.asarray(mesh.points)
    lam = 2 * onp.pi / 0.2
    rng = onp.random.default_rng(1)
    # user angle 90 deg +- 0.6 deg: the internal (rotated) angle straddles the wrap
    ang = onp.pi / 2 + 0.01 * (2 * rng.integers(0, 2, len(pts)) - 1)
    spn = StripePattern.compute(mesh, ang, lam)
    assert not spn.orientable and spn.crosses_sheets.sum() > 100
    assert not spn.zero_index.any()
    x = pts[:, 0]
    corr = max(onp.corrcoef(spn.scalar(), onp.cos(lam * x + c0))[0, 1] for c0 in onp.linspace(0, 2 * onp.pi, 73))
    assert corr > 0.999
    # corner phases agree with the nodal field up to the sheet sign
    assert onp.abs(onp.cos(spn.corner_phase) - onp.real(spn.psi[spn.mesh.cells])).max() < 1e-12
    with pytest.raises(ValueError):
        spn.level_set()


def test_vortex_field_gives_circular_stripes():
    mesh = _mesh(40)
    pts = onp.asarray(mesh.points)
    lam = 2 * onp.pi / 0.15
    cx, cy = 0.5 + 1 / 80, 0.5 + 1 / 80                                     # centre inside a cell
    ang = (onp.arctan2(pts[:, 1] - cy, pts[:, 0] - cx) + onp.pi / 2) % onp.pi  # tangential line field
    spn = StripePattern.compute(mesh, ang, lam)
    assert spn.field_index.sum() == 2                                        # the 2-vector index of a vortex
    r = onp.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    far = r > 0.2
    corr = max(onp.corrcoef(spn.scalar()[far], onp.cos(lam * r[far] + c0))[0, 1]
               for c0 in onp.linspace(0, 2 * onp.pi, 73))
    assert corr > 0.999
    g = _corner_gradients(spn)
    cen = pts[spn.mesh.cells].mean(1)
    rc = onp.hypot(cen[:, 0] - cx, cen[:, 1] - cy)
    rad = onp.c_[cen[:, 0] - cx, cen[:, 1] - cy] / rc[:, None]
    ok = (rc > 0.2) & (spn.zero_index == 0)
    assert (onp.abs((g[ok] * rad[ok]).sum(1)) / onp.linalg.norm(g[ok], axis=1)).min() > 0.995
    assert onp.allclose(onp.linalg.norm(g[ok], axis=1), lam, rtol=0.05)   # curved stripes: linear per-triangle phase
    # phase evaluation inside triangles is finite and consistent with the corners
    b = onp.full((len(spn.mesh.cells), 3), 1 / 3)
    ph = spn.phase(onp.arange(len(spn.mesh.cells)), b)
    assert onp.all(onp.isfinite(ph))
    assert onp.allclose(ph[spn.zero_index == 0], spn.corner_phase[spn.zero_index == 0].mean(1))
    m = spn.stripes_at(onp.arange(len(spn.mesh.cells)), b, duty=0.5)
    assert 0.4 < m.mean() < 0.6


def test_oriented_field_gives_single_valued_level_set():
    mesh = _mesh(40)
    pts = onp.asarray(mesh.points)
    lam = 2 * onp.pi / 0.2
    cx, cy = 0.5 + 1 / 80, 0.5 + 1 / 80
    tangent = onp.arctan2(pts[:, 1] - cy, pts[:, 0] - cx) + onp.pi / 2          # oriented (counter-clockwise)
    spn = StripePattern.compute(mesh, tangent, lam, oriented=True)
    assert spn.orientable and not spn.crosses_sheets.any()
    phi = spn.level_set()                                                     # sin(alpha), no sign flips
    r = onp.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    far = r > 0.2
    corr = max(onp.corrcoef(phi[far], onp.sin(lam * r[far] + c0))[0, 1] for c0 in onp.linspace(0, 2 * onp.pi, 73))
    assert corr > 0.999
    # the same field as a line field has sheet crossings and refuses sin(alpha)
    spn_line = StripePattern.compute(mesh, tangent % onp.pi, lam)
    assert spn_line.crosses_sheets.any()
    with pytest.raises(ValueError):
        spn_line.level_set()


def test_line_field_helpers_and_smoothing():
    mesh = _mesh(20)
    pts = onp.asarray(mesh.points)
    v = onp.c_[onp.cos(0.7), onp.sin(0.7)] * onp.ones((len(pts), 1))
    v[::2] *= -1                                                             # sign must not matter
    ang = line_field_angle(v)
    assert onp.allclose(ang, 0.7)
    assert not field_index(pts, mesh.cells, ang).any()
    rng = onp.random.default_rng(0)
    noisy = ang + 0.3 * rng.standard_normal(len(pts))
    sm = smooth_line_field(mesh, noisy, t=0.05)
    err = onp.abs(onp.angle(onp.exp(2j * (sm - ang))) / 2)
    assert err.mean() < 0.05
    assert onp.abs(onp.angle(onp.exp(2j * (noisy - ang))) / 2).mean() > 0.15


def test_save_vtu(tmp_path):
    mesh = _mesh(10)
    pts = onp.asarray(mesh.points)
    spn = StripePattern.compute(mesh, onp.zeros(len(pts)), 2 * onp.pi / 0.3)
    f = str(tmp_path / "stripes.vtu")
    spn.save_vtu(f)
    import pyvista as pv
    g = pv.read(f)
    assert g.n_cells == len(spn.mesh.cells)
    for name in ("cos_alpha", "sin_alpha", "alpha", "angle", "direction"):
        assert name in g.point_data
    assert "zero_index" in g.cell_data and "field_index" in g.cell_data


# ------------------------------------------------------------ differentiable

import jax
import jax.numpy as jnp

from feax.experimental.stripe import DifferentiableStripes

jax.config.update("jax_enable_x64", True)


def _ds_setup(n=16):
    mesh = _mesh(n)
    pts = onp.asarray(mesh.points)
    return mesh, pts, DifferentiableStripes(mesh), 2 * onp.pi / 0.3


@pytest.mark.parametrize("case", ["orientable", "vortex"])
def test_differentiable_stripes_gradients_match_finite_differences(case):
    mesh, pts, ds, lam = _ds_setup()
    if case == "orientable":
        ang = jnp.asarray(0.4 + 0.05 * onp.random.default_rng(1).standard_normal(len(pts)))
    else:
        ang = jnp.asarray((onp.arctan2(pts[:, 1] - 0.53, pts[:, 0] - 0.53) + onp.pi / 2) % onp.pi)
    cw = onp.random.default_rng(0).standard_normal(len(pts))

    def L(a, f):
        z = ds.psi(a, f)
        return jnp.sum(cw * jnp.real(z) ** 2) + 0.3 * jnp.sum(cw[::-1] * jnp.imag(z) * jnp.real(z))

    gA, gF = jax.grad(L, argnums=(0, 1))(ang, lam)
    assert jnp.all(jnp.isfinite(gA)) and jnp.isfinite(gF)
    h = 1e-6
    for k in onp.random.default_rng(2).choice(len(pts), 5, replace=False):
        e = onp.zeros(len(pts)); e[k] = 1.0
        fd = (float(L(ang + h * e, lam)) - float(L(ang - h * e, lam))) / (2 * h)
        assert abs(float(gA[k]) - fd) < 1e-5 * (1 + abs(fd))
    fdF = (float(L(ang, lam + h)) - float(L(ang, lam - h))) / (2 * h)
    assert abs(float(gF) - fdF) < 1e-5 * (1 + abs(fdF))


def test_differentiable_stripes_match_numpy_up_to_gauge_and_jit():
    mesh, pts, ds, lam = _ds_setup()
    ang = (onp.arctan2(pts[:, 1] - 0.53, pts[:, 0] - 0.53) + onp.pi / 2) % onp.pi   # non-orientable: sign gauge only
    z = onp.asarray(ds.psi(jnp.asarray(ang), lam))
    sp_ = StripePattern.compute(mesh, ang, lam)
    assert min(onp.abs(onp.real(z) - s * sp_.scalar()).max() for s in (1, -1)) < 1e-10
    # orientable field: same stripes up to a global phase
    ang_u = onp.full(len(pts), 0.3)
    z = onp.asarray(ds.psi(jnp.asarray(ang_u), lam))
    sp_ = StripePattern.compute(mesh, ang_u, lam)
    rel = z * onp.conj(sp_.psi)
    assert onp.abs(rel - rel[0]).max() < 1e-10
    # the pure_callback primitive works under jit
    f = jax.jit(lambda a: jnp.sum(ds.scalar(a, lam) ** 2))
    assert onp.isfinite(float(f(jnp.asarray(ang))))
    g = jax.jit(jax.grad(lambda a: jnp.sum(ds.scalar(a, lam) ** 2)))(jnp.asarray(ang))
    assert g.shape == (len(pts),) and onp.all(onp.isfinite(onp.asarray(g)))


def test_variable_frequency_and_weight_pin_a_central_dislocation():
    """10 stripes on the left, 8 on the right: exactly one branch point is
    needed; a central dip of the energy weight pins it at the centre."""
    mesh = _mesh(100)
    pts = onp.asarray(mesh.points)
    x, y = pts[:, 0] - 0.5, pts[:, 1] - 0.5
    t = 0.5 * (1 + onp.tanh(x / 0.05))
    lam = 2 * onp.pi / (0.1 * (1 - t) + 0.125 * t)
    w = 1 - 0.99 * onp.exp(-(x ** 2 + y ** 2) / 0.08 ** 2)
    spn = StripePattern.compute(mesh, onp.zeros(len(pts)), lam, oriented=True, weight=w)
    zi = onp.flatnonzero(spn.zero_index != 0)
    assert len(zi) == 1
    centre = pts[spn.mesh.cells[zi[0]]].mean(0)
    assert onp.hypot(*(centre - 0.5)) < 0.02
    # the stripe counts on the two edges differ by the target
    def count(xc):
        col = onp.flatnonzero(onp.isclose(pts[:, 0], xc))
        col = col[onp.argsort(pts[col, 1])]
        a = onp.unwrap(onp.angle(spn.psi[col]))
        return (a.max() - a.min()) / (2 * onp.pi)
    assert abs(count(0.0) - 10) < 0.3 and abs(count(1.0) - 8) < 0.3
    # scalar frequency still accepted; array frequency is stored
    assert onp.ndim(spn.frequency) == 1


def test_level_set_phase_shift_moves_interfaces_to_stripe_edges():
    mesh = _mesh(40)
    pts = onp.asarray(mesh.points)
    spn = StripePattern.compute(mesh, onp.zeros(len(pts)), 2 * onp.pi / 0.2, oriented=True)
    C = spn.mesh.cells

    def mean_abs_cos_on_cut(shift):
        phi = spn.level_set(raw=True, shift=shift)
        s = onp.sign(phi[C])
        cut = onp.any(s != s[:, :1], axis=1)
        return onp.abs(spn.scalar()[C[cut]]).mean()

    assert mean_abs_cos_on_cut(0.0) > 0.85                      # through the stripe centre lines
    assert mean_abs_cos_on_cut(onp.pi / 2) < 0.45               # on the stripe edges
    assert onp.allclose(spn.level_set(raw=True, shift=onp.pi), -spn.level_set(raw=True))   # same zero set
