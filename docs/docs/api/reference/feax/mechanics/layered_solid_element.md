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

    ε   = sym(∇u)          (3×3 small strain)
    σ_k = C_k : ε          (per-ply stress)

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

