"""Tests for feax.narrowband: NarrowBand construction (from_field /
from_structured_grid), solution scatter/gather, the band-solve-equals-full-solve
property with void exterior, and the SupersetBand moving-band lifecycle."""

import numpy as onp
import pytest

import jax.numpy as jnp

import feax as fe


class _Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_from_structured_grid_equals_ctor():
    grid = fe.StructuredGrid((4, 3, 3))
    active = onp.arange(0, grid.num_cells, 3)
    b1 = fe.NarrowBand(grid, active)
    b2 = fe.NarrowBand.from_structured_grid(grid, active)
    assert onp.array_equal(b1.active_cells, b2.active_cells)
    assert onp.array_equal(b1.node_map, b2.node_map)
    assert onp.allclose(onp.asarray(b1.mesh.points), onp.asarray(b2.mesh.points))
    # grid band and mesh band agree
    b3 = fe.NarrowBand(grid.to_mesh(), active)
    assert onp.array_equal(b1.node_map, b3.node_map)
    assert onp.allclose(onp.asarray(b1.mesh.points), onp.asarray(b3.mesh.points))


def test_from_field_cell_and_node():
    grid = fe.StructuredGrid((6, 3, 3))
    mesh = grid.to_mesh()
    cents = onp.asarray(mesh.points)[onp.asarray(mesh.cells)].mean(axis=1)

    # cell field: active where centroid x < 2
    rho = (cents[:, 0] < 2.0).astype(float)
    band = fe.NarrowBand.from_field(mesh, rho, 0.5)
    assert onp.array_equal(onp.sort(band.active_cells), onp.nonzero(rho > 0.5)[0])

    # margin=1 adds the node-adjacent ring (here: the next x-layer of cells)
    band_m = fe.NarrowBand.from_field(mesh, rho, 0.5, margin=1)
    assert set(band.active_cells) < set(band_m.active_cells)
    assert (cents[band_m.active_cells, 0] < 3.0).all()

    # node field seeds every cell touching a hot node
    T = (onp.asarray(mesh.points)[:, 0] < 1.5).astype(float)
    band_n = fe.NarrowBand.from_field(mesh, T, 0.5, cell_field=False)
    assert (cents[band_n.active_cells, 0] < 2.0).all()

    # keep_cells is always retained
    far = int(onp.argmax(cents[:, 0]))
    band_k = fe.NarrowBand.from_field(mesh, rho, 0.5, keep_cells=[far])
    assert far in band_k.active_cells


# ---------------------------------------------------------------------------
# Solution scatter / gather
# ---------------------------------------------------------------------------

def test_gather_scatter_sol_roundtrip():
    grid = fe.StructuredGrid((4, 3, 3))
    mesh = grid.to_mesh()
    band = fe.NarrowBand(mesh, onp.arange(0, onp.asarray(mesh.cells).shape[0], 2))
    vec = 3
    rng = onp.random.default_rng(0)
    full = rng.random(band.num_full_nodes * vec)

    gathered = band.gather_sol(full, vec)
    assert gathered.shape == (band.num_band_nodes * vec,)
    back = band.scatter_sol(gathered, vec)
    # band nodes round-trip; the rest is zero
    mask = onp.zeros(band.num_full_nodes, bool)
    mask[band.node_map] = True
    fullm = full.reshape(-1, vec)
    backm = back.reshape(-1, vec)
    assert onp.allclose(backm[mask], fullm[mask])
    assert onp.allclose(backm[~mask], 0.0)
    # gather(scatter(x)) == x
    assert onp.allclose(band.gather_sol(back, vec), gathered)


def test_gather_scatter_cells():
    grid = fe.StructuredGrid((4, 3, 3))
    band = fe.NarrowBand(grid, onp.arange(0, grid.num_cells, 2))
    full = onp.arange(band.num_full_cells, dtype=float)
    got = band.gather_cells(full)
    assert onp.allclose(got, band.active_cells.astype(float))
    back = band.scatter_cells(got, fill=-1.0)
    assert onp.allclose(back[band.active_cells], got)
    inactive = onp.setdiff1d(onp.arange(band.num_full_cells), band.active_cells)
    assert onp.allclose(back[inactive], -1.0)


# ---------------------------------------------------------------------------
# Band solve reproduces the full solve when the exterior is void
# ---------------------------------------------------------------------------

def test_band_solve_matches_full_solve_with_void_exterior():
    grid = fe.StructuredGrid((6, 3, 3))
    mesh = grid.to_mesh()
    n_cells = onp.asarray(mesh.cells).shape[0]
    cents = onp.asarray(mesh.points)[onp.asarray(mesh.cells)].mean(axis=1)
    active = onp.nonzero(cents[:, 0] < 4.0)[0]          # x-layers 0..3
    assert 0 < active.size < n_cells

    E_solid, E_void = 100.0, 100.0 * 1e-9
    E_full = onp.full(n_cells, E_void)
    E_full[active] = E_solid
    tp_full = fe.TracedParams(volume_vars=(jnp.asarray(E_full),))

    bc_specs = [
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 4.0), "x", 0.05),
    ]
    # the 1e9 solid/void contrast is ill-conditioned for plain CG; use a
    # direct solve (exact, backend-auto) so the comparison isolates the band
    opts = fe.DirectSolverOptions()

    problem_full = _Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    bc_full = fe.DirichletBCConfig(bc_specs).create_bc(problem_full)
    u_full = fe.create_solver(problem_full, bc_full, solver_options=opts, linear=True,
                              traced_params=tp_full)(
        tp_full, fe.zero_like_initial_guess(problem_full, bc_full))

    # band solve: coordinate-based BCs transfer unchanged to the sub-mesh
    band = fe.NarrowBand(mesh, active)
    problem_band = _Elast(band.mesh, vec=3, dim=3, ele_type="HEX8")
    bc_band = fe.DirichletBCConfig(bc_specs).create_bc(problem_band)
    tp_band = band.gather_params(tp_full)
    assert onp.allclose(onp.asarray(tp_band.volume_vars[0]), E_solid)
    u_band = fe.create_solver(problem_band, bc_band, solver_options=opts, linear=True,
                              traced_params=tp_band)(
        tp_band, fe.zero_like_initial_guess(problem_band, bc_band))

    u_scat = band.scatter_sol(onp.asarray(u_band), vec=3).reshape(-1, 3)
    u_ref = onp.asarray(u_full).reshape(-1, 3)
    scale = onp.abs(u_ref).max()
    assert onp.abs(u_scat[band.node_map] - u_ref[band.node_map]).max() < 1e-6 * scale
    # equivalently through gather_sol
    assert onp.allclose(band.gather_sol(u_scat.reshape(-1), 3), onp.asarray(u_band).ravel())


# ---------------------------------------------------------------------------
# SupersetBand lifecycle
# ---------------------------------------------------------------------------

def test_superset_band_lifecycle():
    grid = fe.StructuredGrid((8, 3, 3))
    mesh = grid.to_mesh()
    cents = onp.asarray(mesh.points)[onp.asarray(mesh.cells)].mean(axis=1)

    def layer(k):
        return onp.nonzero((cents[:, 0] > k) & (cents[:, 0] < k + 1))[0]

    mgr = fe.SupersetBand(mesh, margin=2, guard=1)
    assert mgr.needs_reextract(layer(2))            # no superset yet

    band = mgr.reextract(layer(2))
    assert band is mgr.band
    # superset = active dilated by margin=2 -> x-layers 0..4
    xs = cents[band.active_cells, 0]
    assert xs.min() > 0.0 and xs.max() < 5.0
    assert set(layer(2)) <= set(band.active_cells)

    # active well inside the superset: no re-extract needed
    assert not mgr.needs_reextract(layer(2))
    assert not mgr.needs_reextract(layer(3))        # guard ring 2..4 still inside
    # active reaches the superset margin: guard ring 3..5 leaves -> re-extract
    assert mgr.needs_reextract(layer(4))

    # full<->superset maps delegate to the current band
    full_field = onp.arange(cents.shape[0], dtype=float)
    assert onp.allclose(mgr.map_cells(full_field), band.gather_cells(full_field))
    sub = mgr.map_cells(full_field)
    assert onp.allclose(mgr.scatter_cells(sub, fill=-1.0),
                        band.scatter_cells(sub, fill=-1.0))
    sol = onp.random.default_rng(0).random(band.num_band_nodes * 3)
    assert onp.allclose(mgr.scatter_sol(sol, 3), band.scatter_sol(sol, 3))
