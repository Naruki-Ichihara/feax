---
sidebar_label: narrowband
title: feax.narrowband
---

Narrow-band solves on an active subset of a domain.

A *narrow band* is a user-chosen subset of the domain&#x27;s cells. feax solves only
on the sub-mesh those cells span — the reduced system has DOFs only for the
nodes the active cells touch. The band works on both domain representations:

* an unstructured :class:`feax.Mesh` (any element type), or
* an implicit :class:`feax.StructuredGrid` (the full grid is never built —
  only the O(band) sub-mesh is materialized).

The band is defined by the user from any field or policy (a density, a
level-set, an error indicator, ...): the mechanism only consumes
``active_cells`` (integer cell indices). If the excluded cells are void
(stiffness → 0), the band solve reproduces the full solve to O(void
stiffness); in general it is the solve on the sub-domain the active cells form,
with a free surface on the band boundary.

Everything downstream is stock feax: build a ``Problem`` on ``band.mesh``
(coordinate-based ``location_fns`` transfer unchanged), assemble, and solve
with the usual differentiable ``create_solver``; ``jax.grad`` flows through
via feax&#x27;s existing implicit-diff adjoint.

Usage
-----
&gt;&gt;&gt; active = np.nonzero(rho &gt; 1e-2)[0]                 # any user policy
&gt;&gt;&gt; band = fe.NarrowBand(mesh, active)                 # Mesh or StructuredGrid
&gt;&gt;&gt; sub_problem = MyProblem(band.mesh, vec=3, dim=3, location_fns=[...])
&gt;&gt;&gt; ...                                                # bc, tp, ts, solver as usual
&gt;&gt;&gt; sol_band = solver(tp_band, initial_band, traced_structure=ts_band)
&gt;&gt;&gt; u_full = band.scatter_sol(sol_band, vec=3)         # back to the full domain

## NarrowBand Objects

```python
@dataclass(frozen=True)
class NarrowBand()
```

An active-cell subset of a domain and the maps to/from the full domain.

``domain`` is either an unstructured :class:`feax.Mesh` or an implicit
:class:`feax.StructuredGrid`; both yield the same kind of band.

Attributes
----------
- **mesh** (*Mesh*): The sub-mesh spanned by the active cells (renumbered nodes). Build a ``Problem`` on this directly.
- **active_cells** (*ndarray, (n_active,)*): Global indices of the active cells (band cell -&gt; global cell).
- **node_map** (*ndarray, (n_band_nodes,)*): Band node index -&gt; global node index.


#### from\_structured\_grid

```python
@classmethod
def from_structured_grid(cls, grid, active_cells)
```

Alias for ``NarrowBand(grid, active_cells)`` (kept for clarity).

#### from\_field

```python
@classmethod
def from_field(cls,
               mesh,
               field,
               threshold,
               *,
               margin=0,
               keep_cells=None,
               cell_field=True)
```

Build a band from a scalar field on an unstructured mesh via
``field &gt; threshold`` (+ dilation). For a :class:`StructuredGrid` use
:meth:`StructuredGrid.cells_where` / :meth:`SparseDesign.band_cells`.

``field``      : per-cell (``cell_field=True``) or per-node scalar array.
``margin``     : node-adjacency dilation rings added around the seed
                 (``margin=1`` is the usual narrow-band safety layer).
``keep_cells`` : extra cell indices always kept active (e.g. near loads /
                 supports so the band never disconnects the load path).

#### gather\_cells

```python
def gather_cells(full_cell_field)
```

Full per-cell array -&gt; band per-cell array (active cells only).

#### scatter\_cells

```python
def scatter_cells(band_cell_field, fill=0.0)
```

Band per-cell array -&gt; full per-cell array (inactive filled).

#### scatter\_sol

```python
def scatter_sol(sol_band, vec)
```

Band solution (flat, band_node*vec+comp) -&gt; full flat solution vector.

Inactive DOFs are left zero. Accepts a numpy or JAX array.

#### gather\_sol

```python
def gather_sol(sol_full, vec)
```

Full flat solution vector -&gt; band flat solution (active nodes only).

#### gather\_params

```python
def gather_params(tp_full)
```

Full-domain :class:`feax.TracedParams` -&gt; band TracedParams.

Each volume var is gathered by its leading axis: ``(num_full_cells, ...)``
by ``active_cells``, ``(num_full_nodes, ...)`` by ``node_map`` (cells take
precedence if the counts coincide). ``surface_vars`` must be empty — the
band&#x27;s boundary faces differ from the full mesh&#x27;s; rebuild surface vars
on the band ``Problem`` instead. Differentiable (pure gathers).

#### scatter\_params

```python
def scatter_params(tp_band, fill=0.0)
```

Band TracedParams -&gt; full-domain TracedParams (inactive entries
``fill``). Inverse of :meth:`gather_params`; same shape dispatch.

## SupersetBand Objects

```python
class SupersetBand()
```

Fixed-superset band manager for moving bands (e.g. topology optimization).

A *superset* is a NarrowBand that contains the current active band plus
``margin`` rings of dilation cells. While the superset is held fixed, the
active band may move freely inside it: the design lives in ``traced_params``
on the superset (inactive-but-in-superset cells are just low-stiffness), so
every solve has the same shapes and reuses one compiled feax solver with any
``solver_options``. Only when the active band migrates within ``guard``
rings of the superset boundary is the superset re-extracted (one recompile,
amortized).

This class owns only the superset lifecycle + full&lt;-&gt;superset maps; the user
builds the feax ``Problem``/solver on ``band.mesh`` when told to re-extract::

    mgr = SupersetBand(mesh, margin=2)
    for it in range(n_iter):
        active = onp.nonzero(rho &gt; threshold)[0]          # any policy
        if mgr.needs_reextract(active):
            band = mgr.reextract(active)                  # recompile point
            solver, run = build_solver(band.mesh)         # user physics/solver
        sol = run(mgr.map_cells(rho))                     # warm solve
        ...                                               # update rho

#### reextract

```python
def reextract(active_cells)
```

(Re)build the superset = dilate(active_cells, margin). Returns the band.

Call when :meth:`needs_reextract` is True — this is the (rare) recompile
point since the sub-mesh shapes change.

#### needs\_reextract

```python
def needs_reextract(active_cells)
```

True if no superset yet, or the active band (dilated by ``guard``) is
not fully contained in the current superset (it reached the margin).

#### map\_cells

```python
def map_cells(full_cell_field)
```

Full per-cell field -&gt; superset per-cell field (for traced_params).

