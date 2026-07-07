"""Structured-grid (SPGrid-style) domain representation.

feax has two ways to store a computational domain:

* :class:`feax.Mesh` — explicit unstructured mesh (points + cells arrays).
* :class:`StructuredGrid` (this module) — an *implicit* uniform HEX8 grid:
  connectivity and coordinates are index arithmetic, so the full grid is never
  materialized and a giga-cell background domain costs O(1) memory. Cell data
  is stored sparsely with :class:`SparseDesign` (only the cells that carry it).

Any mesh-defined space embeds into an enclosing grid: :meth:`StructuredGrid.fit`
places a grid over the bounding box of arbitrary points, and
:func:`voxelize_mesh` marks the grid cells an unstructured mesh occupies.
The resulting active-cell set feeds :class:`feax.NarrowBand` (explicit O(band)
sub-mesh for any feax solver) or :class:`feax.solvers.cmg.NarrowBandCMG`
(matrix-free geometric multigrid directly on the grid).
"""

import numpy as onp

# HEX8 local node ordering used by feax (matches box_mesh / Gmsh), as integer
# (dx,dy,dz) corner offsets from a cell's base (min) corner.
_HEX8_OFFSETS = onp.array([(0, 1, 1), (0, 0, 1), (0, 0, 0), (0, 1, 0),
                           (1, 1, 1), (1, 0, 1), (1, 0, 0), (1, 1, 0)],
                          dtype=onp.int64)


class StructuredGrid:
    """Implicit uniform HEX8 grid — connectivity/coordinates computed on the fly.

    Conventions (nx,ny,nz = cells per axis):
        cell  e(cx,cy,cz) = (cx*ny + cy)*nz + cz
        node  n(i,j,k)    = (i*(ny+1) + j)*(nz+1) + k,  i in [0,nx] ...

    Node ordering of :meth:`cell_to_nodes` matches feax's box_mesh HEX8
    convention, so an extracted band is a valid feax ``Mesh``.
    """

    ele_type = 'HEX8'

    def __init__(self, dims, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
        self.nx, self.ny, self.nz = (int(d) for d in dims)
        self.spacing = onp.asarray(spacing, float)
        self.origin = onp.asarray(origin, float)

    @classmethod
    def fit(cls, points, *, h=None, dims=None, pad_cells=1, origin=None, align=None):
        """Place a grid over the bounding box of ``points`` (e.g. an arbitrary
        unstructured mesh's nodes), so the point set lies inside the grid.

        Give exactly one of:

        ``h``     target cell size; dims are derived, and ``pad_cells`` empty
                  cells of margin are added on every side.
        ``dims``  cell counts per axis; the spacing is derived so the grid
                  exactly spans the bounding box (``pad_cells`` has no effect).

        ``align`` rounds each dim UP to a multiple of ``align`` (e.g. 8) so
        geometric multigrid can coarsen; the extra cells are empty.
        """
        pts = onp.asarray(points, float).reshape(-1, 3)
        lo, hi = pts.min(0), pts.max(0)
        if h is not None:
            h = onp.broadcast_to(onp.asarray(h, float), (3,)).copy()
            org = (lo - pad_cells * h) if origin is None else onp.asarray(origin, float)
            dims = onp.maximum(1, onp.ceil((hi + pad_cells * h - org) / h)).astype(int)
            spacing = h
        elif dims is not None:
            dims = onp.asarray(dims, int)
            org = lo if origin is None else onp.asarray(origin, float)
            spacing = (hi - lo) / onp.maximum(dims, 1)
        else:
            raise ValueError("fit requires either h (cell size) or dims")
        if align:
            dims = ((dims + align - 1) // align) * align
        return cls(tuple(int(d) for d in dims), spacing=tuple(spacing), origin=tuple(org))

    def point_cells(self, points):
        """Cell id containing each point; -1 for points outside the grid."""
        p = onp.asarray(points, float).reshape(-1, 3)
        ijk = onp.floor((p - self.origin) / self.spacing).astype(onp.int64)
        inb = ((ijk[:, 0] >= 0) & (ijk[:, 0] < self.nx) & (ijk[:, 1] >= 0)
               & (ijk[:, 1] < self.ny) & (ijk[:, 2] >= 0) & (ijk[:, 2] < self.nz))
        cid = (ijk[:, 0] * self.ny + ijk[:, 1]) * self.nz + ijk[:, 2]
        return onp.where(inb, cid, -1)

    def voxelize(self, points):
        """Active cells = those containing any of ``points`` (occupancy). O(points);
        sample the source densely enough relative to the cell size."""
        cid = self.point_cells(points)
        return onp.unique(cid[cid >= 0])

    def cells_where(self, predicate):
        """Active cells whose centroid satisfies ``predicate(centroids)->bool``
        (e.g. an SDF ``lambda c: sdf(c) < 0``). Enumerates all cells — O(num_cells);
        use :meth:`voxelize` for large grids where the active set is small."""
        allc = onp.arange(self.num_cells, dtype=onp.int64)
        mask = onp.asarray(predicate(self.cell_centroids(allc)))
        return allc[mask]

    @property
    def num_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def num_nodes(self):
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def cell_ijk(self, cell_ids):
        e = onp.asarray(cell_ids, onp.int64)
        nynz = self.ny * self.nz
        cx, r = e // nynz, e % nynz
        return cx, r // self.nz, r % self.nz

    def cell_to_nodes(self, cell_ids):
        """(n,8) global node ids of each cell, in feax HEX8 local order."""
        cx, cy, cz = self.cell_ijk(cell_ids)
        ny1, nz1 = self.ny + 1, self.nz + 1
        ox, oy, oz = _HEX8_OFFSETS[:, 0], _HEX8_OFFSETS[:, 1], _HEX8_OFFSETS[:, 2]
        I = cx[:, None] + ox[None, :]
        J = cy[:, None] + oy[None, :]
        K = cz[:, None] + oz[None, :]
        return (I * ny1 + J) * nz1 + K

    def node_coords(self, node_ids):
        n = onp.asarray(node_ids, onp.int64)
        ny1, nz1 = self.ny + 1, self.nz + 1
        i, r = n // (ny1 * nz1), n % (ny1 * nz1)
        j, k = r // nz1, r % nz1
        return self.origin + onp.stack([i, j, k], axis=-1) * self.spacing

    def cell_centroids(self, cell_ids):
        cx, cy, cz = self.cell_ijk(cell_ids)
        return self.origin + (onp.stack([cx, cy, cz], axis=-1) + 0.5) * self.spacing

    def node_id(self, i, j, k):
        """Global node id(s) at grid index (i,j,k)."""
        i, j, k = (onp.asarray(v, onp.int64) for v in (i, j, k))
        return (i * (self.ny + 1) + j) * (self.nz + 1) + k

    def cell_id(self, ex, ey, ez):
        """Global cell id(s) at grid index (ex,ey,ez)."""
        ex, ey, ez = (onp.asarray(v, onp.int64) for v in (ex, ey, ez))
        return (ex * self.ny + ey) * self.nz + ez

    def nodes_where(self, predicate):
        """Global node ids whose grid index (i,j,k) satisfies ``predicate(i,j,k)``
        — for placing loads/BCs by grid position. O(nodes)."""
        i = onp.arange(self.nx + 1)
        j = onp.arange(self.ny + 1)
        k = onp.arange(self.nz + 1)
        I, J, K = onp.meshgrid(i, j, k, indexing='ij')
        m = onp.asarray(predicate(I, J, K))
        return self.node_id(I[m], J[m], K[m])

    def to_mesh(self, cell_ids=None):
        """Materialize an explicit feax :class:`~feax.mesh.Mesh` of ``cell_ids``
        (default: the whole grid — costs O(num_cells), defeating the implicit
        representation; prefer passing an active subset for large grids).

        Returns only the mesh; use ``feax.NarrowBand(grid, cell_ids)`` instead
        when the band<->full maps (``node_map``, scatter/gather) are needed.
        """
        from .mesh import Mesh
        ids = (onp.arange(self.num_cells, dtype=onp.int64) if cell_ids is None
               else onp.asarray(cell_ids, onp.int64).ravel())
        cells = self.cell_to_nodes(ids)
        used = onp.unique(cells)
        return Mesh(self.node_coords(used), onp.searchsorted(used, cells),
                    ele_type=self.ele_type)

    def dilate_cells(self, cell_ids, margin=1):
        """Cell ids dilated by ``margin`` in the grid (box neighbourhood), clipped
        to bounds. Pure arithmetic on the implicit grid — O(n·(2m+1)^3)."""
        if margin <= 0:
            return onp.unique(onp.asarray(cell_ids, onp.int64))
        cx, cy, cz = self.cell_ijk(cell_ids)
        pieces = []
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                for dz in range(-margin, margin + 1):
                    jx, jy, jz = cx + dx, cy + dy, cz + dz
                    m = ((jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
                         & (jz >= 0) & (jz < self.nz))
                    pieces.append((jx[m] * self.ny + jy[m]) * self.nz + jz[m])
        return onp.unique(onp.concatenate(pieces))


def voxelize_mesh(grid, mesh, subsamples=2):
    """Active grid cells occupied by an unstructured ``mesh`` (any element type).

    Samples each source element (nodes + centroid + ``subsamples`` interior
    points between centroid and nodes) and marks the containing grid cells.
    Increase ``subsamples`` when the grid is finer than the source mesh.
    Returns active cell ids — feed to :class:`feax.NarrowBand` or a cmg build.
    """
    pts = onp.asarray(mesh.points, float)
    cells = onp.asarray(mesh.cells)
    cent = pts[cells].mean(axis=1)                       # (ne, 3)
    samples = [pts, cent]
    for f in onp.linspace(0.25, 0.75, max(0, subsamples)):
        samples.append((f * pts[cells] + (1.0 - f) * cent[:, None, :]).reshape(-1, 3))
    return grid.voxelize(onp.concatenate(samples, axis=0))


class SparseDesign:
    """Per-cell values stored only on the cells that carry them, keyed by global
    cell id — the companion to :class:`StructuredGrid` for extreme resolution.

    Costs O(stored) memory (12 B/cell: int32 id + float64 value) instead of a
    dense O(num_cells) array. Ids are kept sorted for searchsorted lookup.
    """

    def __init__(self, ids, vals):
        ids = onp.asarray(ids, onp.int64).ravel()
        assert ids.size == 0 or ids.max() < 2 ** 31, "grid too large for int32 cell ids"
        o = onp.argsort(ids)
        self.ids = ids[o].astype(onp.int32)
        self.vals = onp.asarray(vals, float).ravel()[o]

    @classmethod
    def uniform(cls, cell_ids, value):
        cell_ids = onp.asarray(cell_ids).ravel()
        return cls(cell_ids, onp.full(cell_ids.size, float(value)))

    def gather(self, query_ids, default=0.0):
        """Values for ``query_ids``; ids not in the store get ``default``."""
        q = onp.asarray(query_ids, onp.int64).ravel().astype(onp.int32)
        if self.ids.size == 0:
            return onp.full(q.shape, default, float)
        pos = onp.searchsorted(self.ids, q)
        posc = onp.clip(pos, 0, self.ids.size - 1)
        hit = (pos < self.ids.size) & (self.ids[posc] == q)
        return onp.where(hit, self.vals[posc], default)

    def update(self, ids, vals):
        """Return a new store with ``(ids, vals)`` merged in (new overwrites old)."""
        ids = onp.asarray(ids, onp.int64).ravel()
        vals = onp.asarray(vals, float).ravel()
        if self.ids.size:
            keep = ~onp.isin(self.ids.astype(onp.int64), ids)
            ids = onp.concatenate([self.ids.astype(onp.int64)[keep], ids])
            vals = onp.concatenate([self.vals[keep], vals])
        return SparseDesign(ids, vals)

    def active_ids(self, threshold):
        """Cell ids whose stored value exceeds ``threshold``."""
        return self.ids[self.vals > threshold].astype(onp.int64)

    def band_cells(self, grid, threshold, *, margin=1, keep_ids=None):
        """Active band = dilate({value>threshold}, margin) ∪ keep, on ``grid``."""
        active = grid.dilate_cells(self.active_ids(threshold), margin)
        if keep_ids is not None:
            active = onp.union1d(active, onp.asarray(keep_ids, onp.int64).ravel())
        return active

    # --- TracedParams bridge -------------------------------------------------
    def traced_params(self, active_cells, *, default=0.0, extra_vars=()):
        """Gather the stored design onto ``active_cells`` as a
        :class:`feax.TracedParams` — the container every feax solver (including
        :class:`~feax.solvers.cmg.NarrowBandCMG`'s) accepts:
        ``volume_vars = (design_on_band, *extra_vars)``."""
        import jax.numpy as jnp
        from .traced_params import TracedParams
        rho = jnp.asarray(self.gather(active_cells, default))
        return TracedParams(volume_vars=(rho,) + tuple(extra_vars))

    def updated(self, active_cells, values):
        """Write band values back into the store (returns a new SparseDesign).

        ``values``: a bare ``(n_active,)`` array or a ``TracedParams`` (its
        ``volume_vars[0]``) — the inverse of :meth:`traced_params`.
        """
        from .traced_params import TracedParams
        if isinstance(values, TracedParams):
            values = values.volume_vars[0]
        return self.update(onp.asarray(active_cells), onp.asarray(values))

    def nbytes(self):
        return int(self.ids.nbytes + self.vals.nbytes)
