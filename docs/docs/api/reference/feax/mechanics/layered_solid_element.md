---
sidebar_label: layered_solid_element
title: feax.mechanics.layered_solid_element
---

Layered solid (3D) element with exact per-ply integration.

This module provides a small-strain, linear, fully-anisotropic 3D continuum
element for laminated composites.  An arbitrary number of plies can live
inside a *single* element: each ply is integrated with its own Gauss rule so
the through-thickness integral

    K = Σ_k ∫_{`z_k`}^{`z_{k+1`}} Bᵀ C_k B dz

is evaluated **exactly** regardless of ply count.  The implementation lives
in ``get_universal_kernel`` — feax handles global assembly and autodiff
linearisation.

Scope
-----
* small-strain linear elasticity, ``σ_ij = C_ijkl ε_kl``
* full 3D anisotropic stiffness, arbitrary number of plies per element
* ply stacking along an arbitrary (default global-z) normal, each ply
  rotated about that normal by its own angle

Not included (see :mod:`feax.mechanics.shell` for the FSDT plate):

* no ANS / EAS — thin laminates with HEX8 still exhibit shear / thickness
  locking in the in-plane directions.
* no geometric nonlinearity.

Kinematics &amp; constitutive law
------------------------------

.. code-block:: text

    ε   = sym(∇u)              (3×3 small strain)
    σ_k = C_k : (ε − ε_th,k)   (per-ply stress; ε_th = α·ΔT, optional thermal
                                eigenstrain — zero unless thermal coupling is on)

#### voigt\_to\_tensor

```python
def voigt_to_tensor(C6: np.ndarray) -> np.ndarray
```

Expand a ``(6, 6)`` Voigt stiffness to the ``(3,3,3,3)`` tensor.

Uses the convention ``σ_ij = C_ijkl ε_kl`` (tensor strains) together
with engineering-shear Voigt order ``(11, 22, 33, 23, 13, 12)``. The
expansion ``C_ijkl = C6[v(i,j), v(k,l)]`` reproduces both the
engineering shear factors and the minor symmetries automatically.

#### tensor\_to\_voigt

```python
def tensor_to_voigt(C: np.ndarray) -> np.ndarray
```

Contract a ``(3,3,3,3)`` stiffness tensor to its ``(6, 6)`` Voigt form.

#### isotropic\_stiffness\_3d

```python
def isotropic_stiffness_3d(E: float, nu: float) -> np.ndarray
```

Isotropic 3D stiffness tensor ``C_ijkl`` of shape ``(3,3,3,3)``.

.. code-block:: text

    σ_ij = λ δ_ij tr(ε) + 2 μ ε_ij
    λ = E ν / ((1+ν)(1-2ν)),   μ = E / (2(1+ν))

#### orthotropic\_stiffness\_3d

```python
def orthotropic_stiffness_3d(E1: float, E2: float, E3: float, G12: float,
                             G13: float, G23: float, nu12: float, nu13: float,
                             nu23: float) -> np.ndarray
```

Orthotropic 3D stiffness tensor in *material* axes (fibre = 1-axis).

Built by inverting the engineering compliance matrix

.. code-block:: text

    S = [[ 1/E1,    -ν12/E1, -ν13/E1, 0,     0,     0    ],
         [-ν12/E1,   1/E2,   -ν23/E2, 0,     0,     0    ],
         [-ν13/E1,  -ν23/E2,  1/E3,   0,     0,     0    ],
         [ 0,        0,        0,      1/G23, 0,     0    ],
         [ 0,        0,        0,      0,     1/G13, 0    ],
         [ 0,        0,        0,      0,     0,     1/G12]]

(symmetric, using ``ν21/E2 = ν12/E1`` etc.) with Voigt shear order
``(23, 13, 12)``, then expanded to the 4th-order tensor.

#### transverse\_isotropic\_stiffness\_3d

```python
def transverse_isotropic_stiffness_3d(E1: float, E2: float, G12: float,
                                      nu12: float, nu23: float) -> np.ndarray
```

Transversely-isotropic 3D stiffness for a unidirectional lamina.

The fibre direction is the 1-axis and the 2-3 plane is isotropic, so

.. code-block:: text

    E3  = E2,   G13 = G12,   ν13 = ν12,
    G23 = E2 / (2 (1 + ν23)).

A convenience wrapper around :func:`orthotropic_stiffness_3d` for the
common single-ply UD-composite case.

#### isotropic\_cte\_3d

```python
def isotropic_cte_3d(alpha: float) -> np.ndarray
```

Isotropic thermal-expansion tensor ``α_ij = α δ_ij`` of shape ``(3, 3)``.

#### transverse\_isotropic\_cte\_3d

```python
def transverse_isotropic_cte_3d(alpha_1: float, alpha_2: float) -> np.ndarray
```

CTE tensor of a UD lamina in material axes (fibre = 1-axis).

Transverse isotropy in the 2-3 plane gives ``α = diag(α₁, α₂, α₂)``. For CFRP
the fibre-direction ``α₁`` is small (often slightly negative) while the
transverse ``α₂`` is large — the source of large cool-down eigenstrains and
inter-ply stresses in cryogenic laminates.

#### rotation\_matrix\_axis

```python
def rotation_matrix_axis(
    theta: float, axis: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray
```

Right-handed rotation matrix of angle ``theta`` (rad) about ``axis``.

Uses Rodrigues&#x27; formula ``R = I + sinθ K + (1−cosθ) K²`` with ``K`` the
skew-symmetric matrix of the unit ``axis``.

#### rotate\_stiffness\_3d

```python
def rotate_stiffness_3d(C: np.ndarray, R: np.ndarray) -> np.ndarray
```

Rotate a ``(3,3,3,3)`` stiffness tensor by rotation matrix ``R``.

.. math:: \bar C_{`ijkl`} = R_{`ip`} R_{`jq`} R_{`kr`} R_{`ls`} C_{`pqrs`}.

#### rotate\_cte\_3d

```python
def rotate_cte_3d(alpha: np.ndarray, R: np.ndarray) -> np.ndarray
```

Rotate a ``(3, 3)`` symmetric CTE tensor by rotation matrix ``R``.

.. math:: \bar\alpha_{`ij`} = R_{`ip`} R_{`jq`} \alpha_{`pq`}.

Use the **same** ``R`` as :func:`rotate_stiffness_3d` so the ply&#x27;s stiffness
and thermal expansion are expressed in a consistent (global) frame.

#### layered\_reference\_quadrature

```python
def layered_reference_quadrature(
        ply_thicknesses: Sequence[float],
        n_inplane: int = 2,
        n_thick_per_ply: int = 2
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]
```

Per-ply through-thickness quadrature on the reference hex ``[0,1]³``.

The in-plane directions (ξ, η) use an ``n_inplane`` Gauss rule; the
thickness direction (ζ) is split at the ply interfaces and each ply gets
its own ``n_thick_per_ply`` Gauss rule with weights scaled by the ply&#x27;s
ζ-extent. The total weight sums to 1 (the reference-cell volume), matching
basix&#x27;s convention.

Returns
-------
- **points** (*(nq, 3) array       Reference quadrature points in ``[0,1]³``.*)
- **weights** (*(nq,) array        Quadrature weights (Σ = 1).*)


#### interpolate\_nodal\_to\_layered\_quad

```python
def interpolate_nodal_to_layered_quad(nodal_cells: np.ndarray,
                                      ply_thicknesses: Sequence[float],
                                      *,
                                      ele_type: str = "HEX8",
                                      n_inplane: int = 2,
                                      n_thick_per_ply: int = 2) -> np.ndarray
```

Interpolate a per-cell nodal field to the layered-solid quadrature points.

Maps a scalar nodal field gathered per cell — e.g. a solved temperature
``T[cells]`` of shape ``(num_cells, n_nodes)`` — onto the **same** per-ply
through-thickness quadrature as :func:`create_oriented_layered_solid`, giving
``(num_cells, nq)``. Combine with :func:`expand_cte_to_quad` to form the
thermal eigenstrain at each quad point::

    T_quad   = interpolate_nodal_to_layered_quad(T[cells], ply_thicknesses)
    cte_quad = expand_cte_to_quad(cte_cell_ply, ply_thicknesses)
    eps_th   = cte_quad * (T_quad - T_ref)[..., None, None]

## LayeredSolid Objects

```python
class LayeredSolid(fe.Problem)
```

3D composite solid with exact per-ply (layer-wise) through-thickness
integration — many plies inside a single element.

Displacement-only continuum element (``vec=3``, ``dim=3``) implemented as
a custom element via :meth:`get_universal_kernel`. Build it with
:func:`create_layered_solid`, which precomputes the per-ply quadrature and
rotated stiffness and supplies per-cell node coordinates as the single
volume internal variable.

``additional_info = (dNdxi_ref, weights, C_quad)`` where

* ``dNdxi_ref`` : ``(nq, num_nodes, 3)`` reference shape gradients,
* ``weights``   : ``(nq,)`` reference quadrature weights,
* ``C_quad``    : ``(nq, 3,3,3,3)`` rotated stiffness at each quad point.

#### create\_layered\_solid

```python
def create_layered_solid(
    mesh,
    ply_C: np.ndarray,
    ply_angles: Sequence[float],
    ply_thicknesses: Sequence[float],
    *,
    ele_type: str = "HEX8",
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
    location_fns: Optional[Iterable[Callable]] = None,
    normal: Sequence[float] = (0.0, 0.0, 1.0)
) -> Tuple[LayeredSolid, InternalVars]
```

Build a :class:`LayeredSolid` with exact per-ply through-thickness
integration (many plies in one element).

One element is assumed to span the whole laminate thickness, with the
laminate stacked along the element&#x27;s local ζ-axis (the last spatial axis
for a structured :func:`feax.mesh.box_mesh`). Each ply is integrated with
its own ``n_thick_per_ply`` Gauss rule, so an *n*-ply laminate is captured
exactly by a single layer of elements regardless of ply count.

Parameters
----------
- **mesh** (*feax.Mesh*): 3D mesh, one element through the laminate thickness.
- **ply_C** (*array*): Lamina stiffness in material axes, ``(3,3,3,3)`` (broadcast) or ``(n_ply, 3,3,3,3)``.
- **ply_angles** (*sequence of float*): Per-ply rotation angles (radians) about ``normal``, bottom → top.
- **ply_thicknesses** (*sequence of float*): Per-ply thicknesses, bottom → top.
- **ele_type** (*str*): Element type (only ``&quot;HEX8&quot;`` is supported).
- **n_inplane** (*int*): In-plane Gauss points per direction (default 2).
- **n_thick_per_ply** (*int*): Through-thickness Gauss points *per ply* (default 2).
- **location_fns** (*iterable of callables, optional*): Boundary-location predicates for surface integrals.
- **normal** (*sequence of float*): Physical stacking direction used to rotate each ply (default ``z``).


Returns
-------
- **problem** (*LayeredSolid*)
- **internal_vars** (*feax.InternalVars*): Holds per-cell node coordinates ``(num_cells, num_nodes, 3)``.


## OrientedLayeredSolid Objects

```python
class OrientedLayeredSolid(fe.Problem)
```

Layered composite solid whose per-ply stiffness varies **from element to
element** — for curved laminates where the stacking normal and the fibre
orientation follow the geometry (e.g. a filament-wound pressure vessel).

Identical in spirit to :class:`LayeredSolid` (exact per-ply through-thickness
integration, many plies in one element), but the rotated stiffness at each
quadrature point is supplied **per cell** as a volume internal variable rather
than shared across the mesh. Build it with :func:`create_oriented_layered_solid`.

Volume internal variables (in order): ``(cell_nodes, C_quad)`` — or, when the
element is built with ``with_thermal=True``, ``(cell_nodes, C_quad, eps_th)`` —
where

* ``cell_nodes`` : ``(num_cells, num_nodes, 3)`` physical node coordinates,
* ``C_quad``     : ``(num_cells, nq, 3,3,3,3)`` rotated stiffness per quad point,
* ``eps_th``     : ``(num_cells, nq, 3, 3)`` thermal eigenstrain per quad point
  (``= α·ΔT`` in global axes). Only present when ``with_thermal=True``; the
  stress is then ``σ = C : (ε − ε_th)``, which adds a constant cool-down
  thermal load to the residual.

``additional_info = (dNdxi_ref, weights, with_thermal)`` carries the reference
shape gradients ``(nq, num_nodes, 3)``, quadrature weights ``(nq,)`` and the
thermal-coupling flag.

#### expand\_cte\_to\_quad

```python
def expand_cte_to_quad(cte_cell_ply: np.ndarray,
                       ply_thicknesses: Sequence[float],
                       *,
                       n_inplane: int = 2,
                       n_thick_per_ply: int = 2) -> np.ndarray
```

Per-cell, per-ply CTE (global axes) -&gt; per-quad CTE ``(num_cells, nq, 3, 3)``.

Uses the **same** per-ply through-thickness quadrature as
:func:`create_oriented_layered_solid`, so the result lines up with ``C_quad``.
Multiply by the temperature change to get the thermal eigenstrain to feed the
element / :func:`layered_quad_stress`::

    cte_quad = expand_cte_to_quad(cte_cell_ply, ply_thicknesses)
    eps_th   = cte_quad * dT                       # (num_cells, nq, 3, 3)

``dT`` may be a scalar, a per-cell array (broadcast as ``dT[:, None, None, None]``)
or any field — keep it separate so the temperature can be a load/design input.

#### create\_oriented\_layered\_solid

```python
def create_oriented_layered_solid(
    mesh,
    C_cell_ply: np.ndarray,
    ply_thicknesses: Sequence[float],
    *,
    ele_type: str = "HEX8",
    n_inplane: int = 2,
    n_thick_per_ply: int = 2,
    location_fns: Optional[Iterable[Callable]] = None,
    problem_class: type = OrientedLayeredSolid,
    with_thermal: bool = False,
    cte_cell_ply: Optional[np.ndarray] = None
) -> Tuple[OrientedLayeredSolid, InternalVars]
```

Build an :class:`OrientedLayeredSolid` from *already-rotated*, per-cell,
per-ply stiffness tensors (global axes).

The orientation logic (per-element local triad, winding angle, ply rotation) is
left to the caller — this keeps the element general. The caller supplies, for
every cell and ply, the lamina stiffness expressed in **global** axes; this
function only handles the exact per-ply through-thickness quadrature and packs
the per-quad stiffness as an internal variable.

Parameters
----------
- **mesh** (*feax.Mesh*): 3D mesh, one element through the laminate thickness, laminate stacked along the element&#x27;s local ζ-axis (last reference axis, i.e. nodes 0-3 = inner thickness face, nodes 4-7 = outer face for a HEX8).
- **C_cell_ply** (*array*): ``(num_cells, n_ply, 3,3,3,3)`` rotated lamina stiffness in global axes, bottom → top through the thickness.
- **ply_thicknesses** (*sequence of float*): Per-ply thicknesses, bottom → top (length ``n_ply``).
- **with_thermal** (*bool*): Enable thermal coupling. When ``True`` the element uses ``σ = C : (ε − ε_th)`` and expects a **third** volume internal variable, the per-quad thermal eigenstrain ``eps_th`` ``(num_cells, nq, 3, 3)``. The returned ``internal_vars`` carry a default ``eps_th`` (zero, or ``cte_quad · 1`` from ``cte_cell_ply``); recompute it per solve as ``cte_quad * dT`` and rebuild :class:``4 with it.
- **cte_cell_ply** (*array, optional*): ``(num_cells, n_ply, 3, 3)`` per-ply CTE in **global** axes (rotate with :func:``7 using the same frame as the stiffness). Only used when ``with_thermal=True`` to seed the default ``eps_th`` at ``ΔT = 1``; if omitted the default ``eps_th`` is zero.


Returns
-------
- **problem** (*OrientedLayeredSolid*)
- **internal_vars** (*feax.InternalVars*): ``volume_vars = (cell_nodes, C_quad)``, or ``(cell_nodes, C_quad, eps_th)`` when ``with_thermal=True``.


## GeometricStiffnessSolid Objects

```python
class GeometricStiffnessSolid(fe.Problem)
```

Initial-stress (geometric) stiffness element for linear buckling.

Given a prestress field ``σ₀`` (from a reference static solution), this element&#x27;s
residual is the gradient of the geometric energy

.. code-block:: text

    E_g(u) = ∫ σ₀_ij · ½ · (∂u_k/∂x_i)(∂u_k/∂x_j) dV,

so its (constant) Jacobian is the geometric stiffness matrix ``K_g``. The linear
buckling problem is then the generalized eigenproblem ``(K + λ K_g) φ = 0`` where
``K`` is the material tangent stiffness; the lowest ``λ`` scales the reference load
to the critical load.

Volume internal variables: ``(cell_nodes, sigma0)`` with
``sigma0`` of shape ``(num_cells, nq, 3, 3)``. ``additional_info = (dNdxi_ref,
weights)``. Assemble ``K_g`` by calling :func:``0 at ``u = 0``
(the residual is linear in ``u``, so the Jacobian is exact and constant).

#### create\_layered\_solid\_geometric\_stiffness

```python
def create_layered_solid_geometric_stiffness(
        mesh,
        cell_sigma0: np.ndarray,
        ply_thicknesses: Sequence[float],
        *,
        ele_type: str = "HEX8",
        n_inplane: int = 2,
        n_thick_per_ply: int = 2
) -> Tuple[GeometricStiffnessSolid, InternalVars]
```

Build a :class:`GeometricStiffnessSolid` from a per-quad prestress field.

Uses the **same** per-ply through-thickness quadrature as
:func:`create_oriented_layered_solid`, so ``cell_sigma0`` must be evaluated at those
quadrature points (e.g. via :func:`layered_quad_stress` on the reference solution).

Parameters
----------
- **mesh** (*feax.Mesh*)
- **cell_sigma0** (*array*): ``(num_cells, nq, 3, 3)`` Cauchy prestress at each quadrature point.
- **ply_thicknesses** (*sequence of float*): Per-ply thicknesses (defines the through-thickness quadrature split).


Returns
-------
- **problem** (*GeometricStiffnessSolid*)
- **internal_vars** (*feax.InternalVars*): ``volume_vars = (cell_nodes, sigma0)``.


#### layered\_quad\_stress

```python
def layered_quad_stress(
        dNdxi_ref: np.ndarray,
        cell_nodes: np.ndarray,
        u_cells: np.ndarray,
        C_quad: np.ndarray,
        eps_th_quad: Optional[np.ndarray] = None) -> np.ndarray
```

Per-cell, per-quad Cauchy stress for a layered solid.

With no thermal field this is the purely-mechanical ``σ = C : ε(u)``; passing
``eps_th_quad`` gives ``σ = C : (ε(u) − ε_th)`` so the prestress fed to the
geometric-stiffness / buckling solve includes the cool-down thermal stress
(the same constitutive law as the ``with_thermal=True`` element).

Parameters
----------
- **dNdxi_ref** (*array*): ``(nq, n_nodes, 3)`` reference shape gradients (``problem._dNdxi_ref``).
- **cell_nodes** (*array*): ``(num_cells, n_nodes, 3)`` physical node coordinates.
- **u_cells** (*array*): ``(num_cells, n_nodes, 3)`` nodal displacements gathered per cell.
- **C_quad** (*array*): ``(num_cells, nq, 3,3,3,3)`` per-quad stiffness (``internal_vars`` from :func:``0).
- **eps_th_quad** (*array, optional*): ``(num_cells, nq, 3, 3)`` thermal eigenstrain ``α·ΔT`` in global axes (e.g. ``expand_cte_to_quad(...) * dT``). ``None`` → no thermal term.


Returns
-------
- **sigma** (*array*): ``(num_cells, nq, 3, 3)`` stress at each quadrature point.

