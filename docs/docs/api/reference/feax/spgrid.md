---
sidebar_label: spgrid
title: feax.spgrid
---

Structured-grid (SPGrid-style) domain representation.

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

## StructuredGrid Objects

```python
class StructuredGrid()
```

Implicit uniform HEX8 grid — connectivity/coordinates computed on the fly.

Conventions (nx,ny,nz = cells per axis):
    cell  e(cx,cy,cz) = (cx*ny + cy)*nz + cz
    node  n(i,j,k)    = (i*(ny+1) + j)*(nz+1) + k,  i in [0,nx] ...

Node ordering of :meth:`cell_to_nodes` matches feax&#x27;s box_mesh HEX8
convention, so an extracted band is a valid feax ``Mesh``.

#### fit

```python
@classmethod
def fit(cls,
        points,
        *,
        h=None,
        dims=None,
        pad_cells=1,
        origin=None,
        align=None)
```

Place a grid over the bounding box of ``points`` (e.g. an arbitrary
unstructured mesh&#x27;s nodes), so the point set lies inside the grid.

Give exactly one of:

``h``     target cell size; dims are derived, and ``pad_cells`` empty
          cells of margin are added on every side.
``dims``  cell counts per axis; the spacing is derived so the grid
          exactly spans the bounding box (``pad_cells`` has no effect).

``align`` rounds each dim UP to a multiple of ``align`` (e.g. 8) so
geometric multigrid can coarsen; the extra cells are empty.

#### point\_cells

```python
def point_cells(points)
```

Cell id containing each point; -1 for points outside the grid.

#### voxelize

```python
def voxelize(points)
```

Active cells = those containing any of ``points`` (occupancy). O(points);
sample the source densely enough relative to the cell size.

#### cells\_where

```python
def cells_where(predicate)
```

Active cells whose centroid satisfies ``predicate(centroids)-&gt;bool``
(e.g. an SDF ``lambda c: sdf(c) &lt; 0``). Enumerates all cells — O(num_cells);
use :meth:`voxelize` for large grids where the active set is small.

#### cell\_to\_nodes

```python
def cell_to_nodes(cell_ids)
```

(n,8) global node ids of each cell, in feax HEX8 local order.

#### node\_id

```python
def node_id(i, j, k)
```

Global node id(s) at grid index (i,j,k).

#### cell\_id

```python
def cell_id(ex, ey, ez)
```

Global cell id(s) at grid index (ex,ey,ez).

#### nodes\_where

```python
def nodes_where(predicate)
```

Global node ids whose grid index (i,j,k) satisfies ``predicate(i,j,k)``
— for placing loads/BCs by grid position. O(nodes).

#### to\_mesh

```python
def to_mesh(cell_ids=None)
```

Materialize an explicit feax :class:`~feax.mesh.Mesh` of ``cell_ids``
(default: the whole grid — costs O(num_cells), defeating the implicit
representation; prefer passing an active subset for large grids).

Returns only the mesh; use ``feax.NarrowBand(grid, cell_ids)`` instead
when the band&lt;-&gt;full maps (``node_map``, scatter/gather) are needed.

#### dilate\_cells

```python
def dilate_cells(cell_ids, margin=1)
```

Cell ids dilated by ``margin`` in the grid (box neighbourhood), clipped
to bounds. Pure arithmetic on the implicit grid — O(n·(2m+1)^3).

#### voxelize\_mesh

```python
def voxelize_mesh(grid, mesh, subsamples=2)
```

Active grid cells occupied by an unstructured ``mesh`` (any element type).

Samples each source element (nodes + centroid + ``subsamples`` interior
points between centroid and nodes) and marks the containing grid cells.
Increase ``subsamples`` when the grid is finer than the source mesh.
Returns active cell ids — feed to :class:`feax.NarrowBand` or a cmg build.

## SparseDesign Objects

```python
class SparseDesign()
```

Per-cell values stored only on the cells that carry them, keyed by global
cell id — the companion to :class:`StructuredGrid` for extreme resolution.

Costs O(stored) memory (12 B/cell: int32 id + float64 value) instead of a
dense O(num_cells) array. Ids are kept sorted for searchsorted lookup.

#### gather

```python
def gather(query_ids, default=0.0)
```

Values for ``query_ids``; ids not in the store get ``default``.

#### update

```python
def update(ids, vals)
```

Return a new store with ``(ids, vals)`` merged in (new overwrites old).

#### active\_ids

```python
def active_ids(threshold)
```

Cell ids whose stored value exceeds ``threshold``.

#### band\_cells

```python
def band_cells(grid, threshold, *, margin=1, keep_ids=None)
```

Active band = dilate({`value&gt;threshold`}, margin) ∪ keep, on ``grid``.

#### traced\_params

```python
def traced_params(active_cells, *, default=0.0, extra_vars=())
```

Gather the stored design onto ``active_cells`` as a
:class:`feax.TracedParams` — the container every feax solver (including
:class:`~feax.solvers.cmg.NarrowBandCMG`&#x27;s) accepts:
``volume_vars = (design_on_band, *extra_vars)``.

#### updated

```python
def updated(active_cells, values)
```

Write band values back into the store (returns a new SparseDesign).

``values``: a bare ``(n_active,)`` array or a ``TracedParams`` (its
``volume_vars[0]``) — the inverse of :meth:`traced_params`.

