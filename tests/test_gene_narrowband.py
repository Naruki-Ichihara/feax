"""Tests for feax.gene.narrowband (OC update, moving-band SIMP loop, multires
bootstrap) and the narrow-band sensitivity filter it uses by default.

The generic MMA pipeline driver stays in feax.gene.optimizer; run_oc was
removed (MMA covers the generic case), so the OC machinery here is exercised
only through the narrow-band path it exists for.
"""

import numpy as onp
import pytest

import feax as fe
import feax.gene as gene
from feax.gene.narrowband import _res_ladder, _upsample, oc_update


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_gene_api_surface():
    assert not hasattr(gene, "run_oc")          # removed: use optimizer.run (MMA)
    assert gene.oc_update is gene.narrowband.oc_update
    assert gene.run_narrowband_oc is gene.narrowband.run_narrowband_oc
    assert gene.run_narrowband_multires is gene.narrowband.run_narrowband_multires
    from feax.gene.optimizer import Pipeline, constraint, run  # noqa: F401


# ---------------------------------------------------------------------------
# oc_update
# ---------------------------------------------------------------------------

def test_oc_update_meets_volume_target():
    rng = onp.random.default_rng(0)
    rho = onp.full(200, 0.5)
    dc = -rng.random(200) - 0.1                  # compliance-like: dc < 0
    for volfrac in (0.3, 0.5, 0.7):
        x = oc_update(rho, dc, volfrac, move=0.5)
        assert x.mean() == pytest.approx(volfrac, abs=1e-3)


def test_oc_update_respects_move_and_bounds():
    rho = onp.full(100, 0.5)
    dc = -onp.linspace(0.1, 2.0, 100)
    x = oc_update(rho, dc, 0.5, move=0.1, xmin=1e-3, xmax=1.0)
    assert onp.abs(x - rho).max() <= 0.1 + 1e-12
    assert x.min() >= 1e-3 and x.max() <= 1.0
    # move limit binds when the target forces a big change
    x_lo = oc_update(rho, dc, 0.05, move=0.2, xmin=1e-3)
    assert x_lo.min() >= 0.5 - 0.2 - 1e-12


def test_oc_update_allocates_to_strong_sensitivities():
    # cells with larger |dc| (more compliance benefit) must end up denser
    rho = onp.full(50, 0.5)
    dc = -onp.linspace(0.1, 2.0, 50)
    x = oc_update(rho, dc, 0.5)
    assert (onp.diff(x) >= -1e-12).all()         # monotone in |dc|
    assert x[-1] > x[0]


def test_oc_update_volume_weights():
    # equal sensitivities but 2x cell volumes on the second half: the weighted
    # volume constraint must hold
    rho = onp.full(100, 0.5)
    dc = -onp.ones(100)
    dv = onp.concatenate([onp.ones(50), 2 * onp.ones(50)])
    x = oc_update(rho, dc, 0.4, dv=dv, move=0.5)
    assert (x * dv).sum() / dv.sum() == pytest.approx(0.4, abs=1e-3)


# ---------------------------------------------------------------------------
# multires helpers
# ---------------------------------------------------------------------------

def test_res_ladder():
    assert _res_ladder((32, 16, 16), 3, 2) == [(8, 8, 8), (16, 8, 8), (32, 16, 16)]
    assert _res_ladder((32, 16, 16), 2, 2, floor=4) == [(16, 8, 8), (32, 16, 16)]
    # stops early when a level can get no coarser
    assert _res_ladder((8, 8, 8), 3, 2) == [(8, 8, 8)]


def test_upsample_shape_and_range():
    rng = onp.random.default_rng(1)
    coarse = rng.random((5, 3, 4))
    up = _upsample(coarse, (11, 7, 9))           # odd, non-integer ratios
    assert up.shape == (11, 7, 9)
    assert up.min() >= 0.0 and up.max() <= 1.0
    # constant fields are preserved exactly
    assert onp.allclose(_upsample(onp.full((4, 4, 4), 0.3), (9, 6, 7)), 0.3)


# ---------------------------------------------------------------------------
# narrow-band sensitivity filter
# ---------------------------------------------------------------------------

def test_sensitivity_filter_uniform_invariant():
    grid = fe.StructuredGrid((8, 6, 6))
    filt = gene.create_sensitivity_filter(grid, rmin=1.5)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    rho = onp.full(active.size, 0.5)
    dc = onp.full(active.size, -2.0)
    # uniform density + uniform sensitivity is a fixed point (interior AND
    # boundary cells: Hs counts only in-grid neighbours)
    assert onp.allclose(filt(active, rho, dc), -2.0, atol=1e-12)


def test_sensitivity_filter_smooths_spike():
    grid = fe.StructuredGrid((8, 6, 6))
    filt = gene.create_sensitivity_filter(grid, rmin=1.5)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    rho = onp.full(active.size, 0.5)
    dc = onp.zeros(active.size)
    spike = int(grid.cell_id(4, 3, 3))
    dc[spike] = -10.0
    out = filt(active, rho, dc)
    # the spike is attenuated and spread onto its neighbours
    assert 0 < -out[spike] < 10.0
    neigh = int(grid.cell_id(5, 3, 3))
    assert out[neigh] < 0
    # filtering follows the band: only-band-neighbours are averaged
    band = grid.dilate_cells([spike], margin=1)
    out_band = filt(band, onp.full(band.size, 0.5),
                    onp.where(band == spike, -10.0, 0.0))
    assert 0 < -out_band[onp.searchsorted(band, spike)] < 10.0


# ---------------------------------------------------------------------------
# moving-band OC loop + multires bootstrap (cmg, small grids)
# ---------------------------------------------------------------------------

def _cantilever_build_fn(nx, ny, nz):
    grid = fe.StructuredGrid((nx, ny, nz))
    cmg = fe.NarrowBandCMG(grid, lambda ni, nj, nk, gx, gy, gz: ni == 0,
                           nu=0.3, penal=3.0, cg_tol=1e-7, cg_maxit=300,
                           bucket=64)
    load_node = grid.node_id(nx, ny // 2, nz // 2)
    keep = onp.zeros((nx, ny, nz), bool)
    keep[0, :, :] = True
    keep[nx - 1, max(0, ny // 2 - 1):ny // 2 + 1,
         max(0, nz // 2 - 1):nz // 2 + 1] = True
    load_fn = lambda c, lv: c.load_vector(lv, [load_node], comp=2, value=-1.0)
    return dict(grid=grid, cmg=cmg, keep=keep, load_fn=load_fn)


_KRYLOV = fe.KrylovSolverOptions(solver="cg", tol=1e-9, maxiter=500)


def test_run_narrowband_oc_smoke():
    parts = _cantilever_build_fn(12, 6, 6)
    volfrac = 0.3
    rho, history = gene.run_narrowband_oc(
        parts["grid"], parts["cmg"], parts["keep"], volfrac, parts["load_fn"],
        n_iter=6, grid_update_start=1, solver_options=_KRYLOV, verbose=False)

    NC = parts["grid"].num_cells
    assert rho.shape == (NC,)
    assert rho.min() >= 1e-3 - 1e-12 and rho.max() <= 1.0 + 1e-12
    assert len(history) == 6
    its, comps, actives, vols = zip(*history)
    assert all(onp.isfinite(c) and c > 0 for c in comps)
    # OC holds the volume target from the first update on
    assert vols[-1] == pytest.approx(volfrac, abs=5e-3)
    # material concentrates: compliance improves over the run
    assert min(comps[1:]) < comps[0]
    # NOTE: on a grid this small the band legitimately stays at 100% — the
    # dilation margin covers every void pocket. Band shrinkage is asserted
    # deterministically in test_band_extraction_follows_design.


def test_band_extraction_follows_design():
    # seed a connected beam (solid along the load path, void elsewhere) and
    # start banding immediately: the band must be the dilated beam ∪ keep,
    # not the full grid
    parts = _cantilever_build_fn(16, 6, 6)
    grid = parts["grid"]
    cx, cy, cz = grid.cell_ijk(onp.arange(grid.num_cells))
    beam = (onp.abs(cy - 3) <= 1) & (onp.abs(cz - 3) <= 1)
    x_init = onp.where(beam, 0.6, 1e-3)

    rho, history = gene.run_narrowband_oc(
        parts["grid"], parts["cmg"], parts["keep"], 0.15, parts["load_fn"],
        n_iter=2, grid_update_start=0, margin=1, x_init=x_init,
        solver_options=_KRYLOV, verbose=False)

    its, comps, actives, vols = zip(*history)
    beam_frac = beam.mean()
    # band is a strict sub-grid but covers at least the beam
    assert beam_frac < actives[0] < 1.0
    # the connected load path gives a proper positive compliance
    assert all(onp.isfinite(c) and c > 0 for c in comps)


def test_run_narrowband_multires_smoke():
    volfrac = 0.15
    res = gene.run_narrowband_multires(
        _cantilever_build_fn, (16, 8, 8), volfrac, n_levels=2, coarse_factor=2,
        floor=4, n_iter=3, coarse_iter=6, grid_update_start=1,
        solver_options=_KRYLOV, verbose=False)

    assert res["levels"] == [(8, 4, 4), (16, 8, 8)]
    assert res["x"].shape == (16, 8, 8)
    assert len(res["history"]) == 2
    assert res["history"][0]["dims"] == (8, 4, 4)
    assert len(res["history"][1]["history"]) == 3      # n_iter at the fine level
    # the coarse seed carries over: compliance keeps improving on the fine level
    fine_comps = [c for (_, c, _, _) in res["history"][1]["history"]]
    assert all(onp.isfinite(c) and c > 0 for c in fine_comps)
    assert fine_comps[-1] < fine_comps[0]
    # volume target maintained at the finest level
    assert res["x"].mean() == pytest.approx(volfrac, abs=5e-3)
