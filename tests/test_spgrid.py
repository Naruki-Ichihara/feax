"""Tests for feax.spgrid: StructuredGrid (implicit HEX8 grid), voxelize_mesh,
and SparseDesign (sparse per-cell storage)."""

import numpy as onp
import pytest

import feax as fe


# ---------------------------------------------------------------------------
# StructuredGrid: index arithmetic
# ---------------------------------------------------------------------------

def test_cell_id_ijk_roundtrip():
    grid = fe.StructuredGrid((5, 4, 3))
    ids = onp.arange(grid.num_cells)
    cx, cy, cz = grid.cell_ijk(ids)
    assert onp.array_equal(grid.cell_id(cx, cy, cz), ids)
    assert cx.max() == 4 and cy.max() == 3 and cz.max() == 2


def test_node_id_and_coords():
    grid = fe.StructuredGrid((3, 3, 3), spacing=(0.5, 1.0, 2.0), origin=(1.0, 2.0, 3.0))
    nid = grid.node_id(2, 1, 3)
    assert onp.allclose(grid.node_coords(nid), [1.0 + 2 * 0.5, 2.0 + 1.0, 3.0 + 3 * 2.0])
    # vectorized form
    nids = grid.node_id(onp.array([0, 3]), onp.array([0, 3]), onp.array([0, 3]))
    coords = grid.node_coords(nids)
    assert onp.allclose(coords[0], grid.origin)
    assert onp.allclose(coords[1], grid.origin + onp.array([3, 3, 3]) * grid.spacing)


def test_cell_centroids():
    grid = fe.StructuredGrid((2, 2, 2), spacing=(2.0, 2.0, 2.0))
    c = grid.cell_centroids(onp.array([grid.cell_id(0, 0, 0), grid.cell_id(1, 1, 1)]))
    assert onp.allclose(c, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])


def test_cell_to_nodes_matches_mesh_geometry():
    grid = fe.StructuredGrid((3, 2, 2))
    ids = onp.arange(grid.num_cells)
    cells = grid.cell_to_nodes(ids)
    assert cells.shape == (grid.num_cells, 8)
    # each cell's node coordinates average to its centroid
    cent = grid.node_coords(cells.ravel()).reshape(-1, 8, 3).mean(axis=1)
    assert onp.allclose(cent, grid.cell_centroids(ids))
    # to_mesh materializes the same geometry (renumbered)
    mesh = grid.to_mesh(ids)
    pts, mcells = onp.asarray(mesh.points), onp.asarray(mesh.cells)
    assert onp.allclose(pts[mcells].mean(axis=1), cent)


# ---------------------------------------------------------------------------
# StructuredGrid.fit / point_cells / voxelize
# ---------------------------------------------------------------------------

def test_fit_with_h_contains_points():
    rng = onp.random.default_rng(0)
    pts = rng.random((50, 3)) * onp.array([3.0, 2.0, 1.0]) + onp.array([1.0, -1.0, 0.5])
    grid = fe.StructuredGrid.fit(pts, h=0.4, pad_cells=1)
    assert onp.allclose(grid.spacing, 0.4)
    # every point falls in a valid cell
    assert (grid.point_cells(pts) >= 0).all()
    # the pad leaves ~one empty cell of margin around the bounding box
    lo, hi = pts.min(0), pts.max(0)
    extent = grid.origin + onp.array([grid.nx, grid.ny, grid.nz]) * grid.spacing
    assert (grid.origin <= lo - 0.5 * 0.4).all()
    assert (extent >= hi + 0.5 * 0.4).all()


def test_fit_with_dims_spans_bbox():
    pts = onp.array([[0.0, 0.0, 0.0], [4.0, 2.0, 1.0]])
    grid = fe.StructuredGrid.fit(pts, dims=(8, 4, 2))
    assert (grid.nx, grid.ny, grid.nz) == (8, 4, 2)
    assert onp.allclose(grid.origin, [0.0, 0.0, 0.0])
    assert onp.allclose(grid.spacing, [0.5, 0.5, 0.5])


def test_fit_align_rounds_dims_up():
    pts = onp.random.default_rng(1).random((20, 3)) * 5.0
    grid = fe.StructuredGrid.fit(pts, h=1.0, align=8)
    assert grid.nx % 8 == 0 and grid.ny % 8 == 0 and grid.nz % 8 == 0
    with pytest.raises(ValueError):
        fe.StructuredGrid.fit(pts)  # neither h nor dims


def test_point_cells_outside_is_minus_one():
    grid = fe.StructuredGrid((2, 2, 2))
    cid = grid.point_cells([[0.5, 0.5, 0.5], [-0.1, 0.5, 0.5], [0.5, 2.5, 0.5]])
    assert cid[0] == grid.cell_id(0, 0, 0)
    assert cid[1] == -1 and cid[2] == -1


def test_voxelize_marks_containing_cells():
    grid = fe.StructuredGrid((4, 4, 4))
    pts = [[0.5, 0.5, 0.5], [0.6, 0.4, 0.5], [3.5, 3.5, 3.5], [-1.0, 0.0, 0.0]]
    active = grid.voxelize(pts)
    expect = onp.unique([grid.cell_id(0, 0, 0), grid.cell_id(3, 3, 3)])
    assert onp.array_equal(active, expect)


# ---------------------------------------------------------------------------
# StructuredGrid.dilate_cells
# ---------------------------------------------------------------------------

def test_dilate_cells_interior_and_clipped():
    grid = fe.StructuredGrid((5, 5, 5))
    center = grid.cell_id(2, 2, 2)
    assert grid.dilate_cells([center], margin=1).size == 27
    corner = grid.cell_id(0, 0, 0)
    assert grid.dilate_cells([corner], margin=1).size == 8  # clipped to bounds
    # margin=0 is just unique()
    assert onp.array_equal(grid.dilate_cells([center, center], margin=0), [center])


# ---------------------------------------------------------------------------
# voxelize_mesh
# ---------------------------------------------------------------------------

def test_voxelize_mesh_covers_embedded_mesh():
    mesh = fe.mesh.box_mesh((2, 2, 2), mesh_size=1.0)
    grid = fe.StructuredGrid.fit(onp.asarray(mesh.points), h=0.5, pad_cells=1)
    active = fe.voxelize_mesh(grid, mesh)
    # every source node's containing cell is active (boundary nodes may sit on
    # cell interfaces; the centroid samples guarantee interior coverage)
    cents = onp.asarray(mesh.points)[onp.asarray(mesh.cells)].mean(axis=1)
    assert onp.isin(grid.point_cells(cents), active).all()
    # nothing outside the mesh bounding box is active
    lo, hi = onp.asarray(mesh.points).min(0), onp.asarray(mesh.points).max(0)
    c = grid.cell_centroids(active)
    assert ((c > lo - grid.spacing) & (c < hi + grid.spacing)).all()


# ---------------------------------------------------------------------------
# SparseDesign
# ---------------------------------------------------------------------------

def test_sparse_design_gather_default_and_sorting():
    sd = fe.SparseDesign([5, 1, 3], [0.5, 0.1, 0.3])   # unsorted input
    assert onp.array_equal(sd.ids, [1, 3, 5])          # stored sorted
    got = sd.gather([1, 2, 3, 5, 9], default=-1.0)
    assert onp.allclose(got, [0.1, -1.0, 0.3, 0.5, -1.0])


def test_sparse_design_update_merges_and_overwrites():
    sd = fe.SparseDesign.uniform([0, 1, 2], 0.5)
    sd2 = sd.update([2, 7], [0.9, 0.7])
    assert onp.allclose(sd2.gather([0, 1, 2, 7]), [0.5, 0.5, 0.9, 0.7])
    # functional: original unchanged
    assert onp.allclose(sd.gather([2, 7]), [0.5, 0.0])


def test_sparse_design_active_ids_and_band_cells():
    grid = fe.StructuredGrid((5, 5, 5))
    center = grid.cell_id(2, 2, 2)
    edge = grid.cell_id(0, 0, 0)
    sd = fe.SparseDesign([center, edge], [0.8, 0.01])
    assert onp.array_equal(sd.active_ids(0.1), [center])
    band = sd.band_cells(grid, 0.1, margin=1)
    assert onp.array_equal(band, grid.dilate_cells([center], margin=1))
    # keep_ids always included
    band_k = sd.band_cells(grid, 0.1, margin=1, keep_ids=[edge])
    assert edge in band_k and band_k.size == band.size + 1


def test_sparse_design_nbytes():
    sd = fe.SparseDesign.uniform(onp.arange(10), 1.0)
    assert sd.nbytes() == sd.ids.nbytes + sd.vals.nbytes
    assert sd.nbytes() == 10 * (4 + 8)  # int32 id + float64 value
