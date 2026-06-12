---
sidebar_label: copv
title: feax.mechanics.copv
---

Mesh and model generation for double-wall composite pressure vessels (COPV).

A *double-wall* (sandwich) COPV is an axisymmetric tank whose wall is a radial
stack

.. code-block:: text

    [ inner CFRP shell ] [ solid fill ] [ outer CFRP shell ]

This module builds the full-ring structured ``HEX8`` mesh of the **axial half
model** (equator at ``z = 0``, one polar dome at the top) and the companion
data needed to drive a :mod:`feax.mechanics.layered_solid_element` analysis:

* :class:`DoubleWallCopvGeometry` — geometry + discretization parameters.
* :func:`create_double_wall_copv_mesh` — the structured mesh, radial *zone*
  labels (inner shell / fill / outer shell) and node-id bookkeeping, returned
  as a :class:`DoubleWallCopvMesh`.
* :func:`create_winding_ply_stiffness` — per-cell, per-ply stiffness with the
  Clairaut (geodesic) winding orientation on the shells and an isotropic fill.
* surface helpers (:meth:`DoubleWallCopvMesh.outer_surface_location_fn`,
  :meth:``0) and DoF helpers for the reference
  static solve / buckling boundary conditions.

The wall is described by a reference meridian on the **inner** surface (a
cylinder of radius ``a`` and length ``Lc`` capped by a spherical dome down to a
polar opening of radius ``r_p``); every radial layer is offset outward along
the local meridian normal, so the dome thickness follows the mould line.

## DoubleWallCopvGeometry Objects

```python
@dataclass(frozen=True)
class DoubleWallCopvGeometry()
```

Geometry and discretization of a double-wall (sandwich) COPV half model.

Lengths are in consistent units (mm in the bundled examples). The reference
surface is the **inner** mould line: a cylinder of radius ``a`` and length
``cyl_len`` above the equator, closed by a spherical dome down to a polar
opening of radius ``polar_radius``.

Parameters
----------
- **a** (*float*): Inner-tank inner radius (the reference surface radius).
- **shell_t** (*float*): Thickness of each CFRP shell (inner and outer are equal).
- **gap** (*float*): Radial thickness of the solid fill between the two shells.
- **cyl_len** (*float*): Cylinder length above the equator (the dome sits on top of it).
- **polar_radius** (*float*): Polar-opening radius at the reference (inner) surface; sets the maximum winding latitude via Clairaut&#x27;s relation.
- **n_circum** (*int*): Circumferential divisions of the full ring (max wavenumber ~``n_circum/4``).
- **n_fill** (*int*): Radial elements through the solid fill (each shell is a single radial element, so the wall has ``n_fill + 2`` radial elements total).
- **axis** (*tuple of float*): Tank axis (unit vector); default global ``+z``.


#### center\_z

```python
@property
def center_z() -> float
```

Axial coordinate of the dome&#x27;s sphere center (top of the cylinder).

#### d\_layers

```python
@property
def d_layers() -> onp.ndarray
```

Radial offsets (from the inner surface) of every layer interface.

``[0, t]`` (inner shell) + ``n_fill`` fill stations + ``t+gap+t`` (outer
shell); length ``n_fill + 3`` → ``n_fill + 2`` radial elements.

## DoubleWallCopvMesh Objects

```python
@dataclass
class DoubleWallCopvMesh()
```

Result of :func:`create_double_wall_copv_mesh`.

Bundles the :class:`feax.Mesh`, per-cell radial ``zone`` labels and the
structured node-id bookkeeping used to assemble materials, surfaces and
boundary conditions.

Attributes
----------
- **mesh** (*feax.Mesh*): Full-ring ``HEX8`` mesh of the half model.
- **cell_zone** (*(n_cells,) int ndarray*): Radial zone of each cell: :data:`ZONE_INNER_SHELL`, :data:`ZONE_FILL` or :data:`ZONE_OUTER_SHELL`.
- **geom** (*DoubleWallCopvGeometry*): The generating geometry.
- **n_merid** (*int*): Number of meridional element rows (``= n_merid_cyl + n_merid_dome``).
- **meridian** (*tuple of ndarray*): ``(r, z, n_r, n_z)`` reference-meridian stations and normals.


#### node\_id

```python
def node_id(i: int, j: int, kk: int) -> int
```

Node id at meridian row ``i``, circumferential ``j`` (wraps), layer ``kk``.

#### cell\_nodes

```python
@property
def cell_nodes() -> jnp.ndarray
```

``(n_cells, 8, 3)`` physical node coordinates per cell.

#### equator\_node\_ids

```python
def equator_node_ids() -> onp.ndarray
```

All nodes on the equator ring (``i = 0``), every column and layer.

#### rim\_inner\_node\_ids

```python
def rim_inner_node_ids() -> onp.ndarray
```

Inner two layers of the polar-rim ring (top meridian row ``i = n_merid``).

#### outer\_surface\_node\_ids

```python
def outer_surface_node_ids() -> onp.ndarray
```

All nodes on the outermost mould line (last layer).

#### inner\_surface\_node\_ids

```python
def inner_surface_node_ids() -> onp.ndarray
```

All nodes on the innermost mould line (first layer, the tank cavity).

#### rim\_inner\_radius

```python
def rim_inner_radius() -> float
```

Minimum cylindrical radius of the inner rim nodes (polar-boss radius).

#### dome\_cell\_mask

```python
def dome_cell_mask() -> onp.ndarray
```

Boolean ``(n_cells,)`` mask: ``True`` for cells on the polar dome.

Cells are emitted meridian-row-major (row ``i``, column ``j``, radial
``ke``), so a cell&#x27;s meridian row is ``index // (n_circum · n_rad_el)``;
rows ``&gt;= n_merid_cyl`` are the dome.

#### equator\_axial\_dofs

```python
def equator_axial_dofs() -> onp.ndarray
```

``u_z`` DoFs of the equator ring (symmetry-plane constraint).

#### rigid\_inplane\_pin\_dofs

```python
def rigid_inplane_pin_dofs() -> onp.ndarray
```

Three DoFs that remove the in-plane rigid modes (x,y at A; y at B).

``A`` and ``B`` are two diametrically opposite equator inner-surface
nodes; pinning ``A_x, A_y, B_y`` removes the two in-plane translations
and the rotation about the tank axis.

#### buckling\_free\_dofs

```python
def buckling_free_dofs(num_total_dofs: int) -> onp.ndarray
```

Free DoFs for the buckling eigenproblem.

Equator ``u_z = 0`` plus the three in-plane pins, removed from the full
``num_total_dofs`` set. The result is the ``free_dofs`` argument of
:func:`feax.create_linear_buckling_solver`.

#### outer\_surface\_location\_fn

```python
def outer_surface_location_fn(tol: float = 1e-4) -> Callable
```

``location_fn(point) -&gt; bool`` selecting the outer mould-line surface.

#### inner\_surface\_location\_fn

```python
def inner_surface_location_fn(tol: float = 1e-4) -> Callable
```

``location_fn(point) -&gt; bool`` selecting the inner (cavity) mould-line surface.

#### rim\_location\_fn

```python
def rim_location_fn(tol: float = 1e-4) -> Callable
```

``location_fn(point) -&gt; bool`` selecting the inner polar-rim surface.

#### build\_meridian

```python
def build_meridian(
    geom: DoubleWallCopvGeometry
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]
```

Reference (inner-surface) meridian: cylinder + spherical dome.

Returns ``(r, z, n_r, n_z)`` arrays of length ``n_merid + 1`` giving the
radius, axial coordinate and the (outward) meridian-normal components at
each meridional station. Radial layers are offset along ``(n_r, n_z)``.

#### create\_double\_wall\_copv\_mesh

```python
def create_double_wall_copv_mesh(
        geom: DoubleWallCopvGeometry = DoubleWallCopvGeometry(),
        check_jacobian: bool = True) -> DoubleWallCopvMesh
```

Build the full-ring ``HEX8`` mesh of a double-wall COPV half model.

Parameters
----------
- **geom** (*DoubleWallCopvGeometry*): Geometry and discretization (defaults to the bundled example tank).
- **check_jacobian** (*bool*): Assert all cell Jacobian determinants are positive (no inverted HEX8).


Returns
-------
DoubleWallCopvMesh
    Mesh, per-cell radial zone labels and structured node-id helpers.

#### create\_winding\_ply\_stiffness

```python
def create_winding_ply_stiffness(copv_mesh: DoubleWallCopvMesh,
                                 ply_C: jnp.ndarray,
                                 layup: Sequence[Tuple[str, float]],
                                 fill_C: jnp.ndarray = None,
                                 drop_dome_hoop: bool = True) -> jnp.ndarray
```

Per-cell, per-ply stiffness with Clairaut winding on the shells.

For every shell cell each ply is rotated into the wall triad: a ``&quot;hoop&quot;``
ply aligns with the circumferential direction, a ``&quot;helical&quot;`` ply is wound
at the geodesic angle ``α = arcsin(r_p / r)`` (Clairaut&#x27;s relation) about the
meridian, with ``sign`` giving the ``±`` winding. Fill cells (if ``fill_C``
is given) take the isotropic ``fill_C`` for every ply.

Hoop windings physically exist only on the cylinder (real filament winding
terminates the hoop band at the cylinder–dome tangent line). With
``drop_dome_hoop=True`` (default), ``&quot;hoop&quot;`` plies on the **dome** are
therefore re-oriented as helical continuations at the local geodesic angle
instead of circumferentially — the ply count and wall thickness are fixed by
the mesh, so the hoop reinforcement is removed without changing the geometry.
Successive hoop plies alternate ``±α`` to keep the dome laminate balanced.

Parameters
----------
- **copv_mesh** (*DoubleWallCopvMesh*): The generated mesh (provides cell geometry and zone labels).
- **ply_C** (*(3,3,3,3) array*): Single-ply (unidirectional) stiffness in ply axes.
- **layup** (*sequence of ``(kind, sign)``*): Per-ply stacking, ``kind`` in ``{`&quot;helical&quot;, &quot;hoop&quot;`}`` and ``sign = ±1`` the winding orientation (ignored for hoop).
- **fill_C** (*(3,3,3,3) array, optional*): Isotropic fill stiffness. If ``None`` every cell is treated as a shell.
- **drop_dome_hoop** (*bool*): Re-orient hoop plies on the dome as helical continuations (default ``True``). Set ``False`` to keep hoop windings circumferential everywhere.


Returns
-------
- **C_cell_ply** (*(n_cells, n_ply, 3, 3, 3, 3) array*): Rotated stiffness per cell and ply, ready for :func:``4.


#### create\_winding\_cte

```python
def create_winding_cte(copv_mesh: DoubleWallCopvMesh,
                       ply_alpha: jnp.ndarray,
                       layup: Sequence[Tuple[str, float]],
                       fill_alpha: jnp.ndarray = None,
                       drop_dome_hoop: bool = True) -> jnp.ndarray
```

Per-cell, per-ply thermal-expansion (CTE) tensor with Clairaut winding.

Thermal analogue of :func:`create_winding_ply_stiffness`: each shell ply&#x27;s CTE
is rotated into the **same** wall triad / winding frame used for the stiffness,
so the anisotropic contraction (small ``α₁`` along the fibre, large ``α₂``
transverse) follows the winding direction. Fill cells (if ``fill_alpha`` is
given) take the isotropic ``fill_alpha`` for every ply.

The result feeds the layered-solid thermal eigenstrain: expand it to quadrature
points and multiply by the temperature change ``ΔT`` to get ``ε_th = α·ΔT``
(see :func:``3).

Parameters
----------
- **copv_mesh** (*DoubleWallCopvMesh*): The generated mesh (provides cell geometry and zone labels).
- **ply_alpha** (*(3, 3) array*): Single-ply CTE tensor in ply (material) axes, fibre = 1-axis, e.g. ``transverse_isotropic_cte_3d(alpha_1, alpha_2)``.
- **layup** (*sequence of ``(kind, sign)``*): Per-ply stacking, identical to :func:`create_winding_ply_stiffness` so the CTE and stiffness frames stay in lock-step.
- **fill_alpha** (*(3, 3) array, optional*): Isotropic fill CTE. If ``None`` every cell is treated as a shell.
- **drop_dome_hoop** (*bool*): Re-orient hoop plies on the dome as helical continuations (default ``True``); must match the value used for the stiffness.


Returns
-------
- **cte_cell_ply** (*(n_cells, n_ply, 3, 3) array*): Rotated CTE per cell and ply, in **global** axes.


#### cell\_fiber\_directions

```python
def cell_fiber_directions(copv_mesh: DoubleWallCopvMesh,
                          layup: Sequence[Tuple[str, float]],
                          drop_dome_hoop: bool = True,
                          shell_only: bool = True) -> onp.ndarray
```

Per-cell, per-ply primary fiber direction in **global** coordinates.

Uses the same orientation as :func:`create_winding_ply_stiffness` (so the
vectors visualise the laminate actually assembled). Fill cells carry no
fibers and are zeroed when ``shell_only`` (the default).

Returns
-------
- **fiber** (*(n_cells, n_ply, 3) ndarray*): Unit fiber direction ``f1`` of each ply, per cell (zero on fill cells).


#### cell\_winding\_angle\_deg

```python
def cell_winding_angle_deg(copv_mesh: DoubleWallCopvMesh) -> onp.ndarray
```

Per-cell helical winding angle ``α = arcsin(r_p / r)`` in degrees (Clairaut).

#### cell\_triads

```python
def cell_triads(
    copv_mesh: DoubleWallCopvMesh
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]
```

Per-cell wall triad ``(t_meridian, t_hoop, e_normal)``, each ``(n_cells, 3)``.

The same geometric directions used for the winding orientation — handy for
projecting per-cell stresses (e.g. the hoop prestress ``t_hoop·σ·t_hoop``).

#### save\_orientation\_vtk

```python
def save_orientation_vtk(copv_mesh: DoubleWallCopvMesh,
                         sol_file: str,
                         layup: Sequence[Tuple[str, float]],
                         drop_dome_hoop: bool = True) -> None
```

Write a VTU visualising the fiber orientation of the COPV laminate.

The output carries, as **cell** data (view as glyphs in ParaView):

* ``fiber_ply{`k`}`` — primary fiber direction of ply ``k`` (global coords);
* ``dir_meridian`` / ``dir_hoop`` / ``dir_normal`` — the wall triad;
* ``winding_angle_deg`` — the helical Clairaut angle;
* ``zone`` (0/1/2 inner-shell/fill/outer-shell) and ``is_dome`` (0/1).

In ParaView: open the ``.vtu``, ``Glyph`` filter, orient by e.g.
``fiber_ply0`` to see the winding (steepening over the dome, hoop on the
cylinder), and colour by ``winding_angle_deg`` or ``zone``.

Parameters
----------
- **copv_mesh** (*DoubleWallCopvMesh*)
- **sol_file** (*str*): Output path (``.vtu``).

