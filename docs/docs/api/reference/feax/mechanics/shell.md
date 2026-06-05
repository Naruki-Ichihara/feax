---
sidebar_label: shell
title: feax.mechanics.shell
---

Mindlin/Reissner (FSDT) flat plate elements.

This module provides building blocks for first-order shear-deformation
plate theory (Mindlin/Reissner) on a flat 2D mesh embedded in the
``z = 0`` plane. Curved-shell (curvilinear basis, geometric stiffness,
drilling DOF) extensions are not implemented yet.

Variable layout
---------------
The plate is treated as a *two-variable* problem so that the transverse
shear coupling ``γ_α = ∂w/∂x_α + θ_α`` can be expressed as a mixed
&quot;gradient + value&quot; term in feax&#x27;s :meth:`~feax.Problem.get_weak_form`:

* variable 0 — translations  ``(u, v, w)``         (vec = 3)
* variable 1 — section rotations ``(θ_x, θ_y)``    (vec = 2)

Both variables share the same 2D mesh.

Kinematics (FSDT, flat plate)
-----------------------------

.. code-block:: text

    ε_αβ  = sym(∇u_in)         (membrane strain,  2×2)
    κ_αβ  = sym(∇θ)            (curvature,        2×2)
    γ_α   = ∂w/∂x_α + θ_α      (transverse shear, 2)

Constitutive law (homogeneous symmetric plate, no membrane-bending coupling):

.. code-block:: text

    N = A : ε        (membrane stress resultant,  2×2)
    M = D : κ        (bending moment,             2×2)
    Q = G_s · γ      (transverse shear force,     2)

with thickness-integrated stiffness

.. code-block:: text

    A   = h        · C_in                  (2,2,2,2)
    D   = h^3 / 12 · C_in                  (2,2,2,2)
    G_s = κ_s h    · G_transverse          (2,2)

where ``κ_s = 5/6`` is the standard shear-correction factor.

Notes
-----
Low-order interpolations (QUAD4 with full integration) suffer from
transverse-shear locking on thin plates. A reduced/MITC integration
scheme is not yet implemented; for the time being prefer moderately
thick plates (``h / L ≳ 0.05``) or use higher-order elements.

#### isotropic\_in\_plane\_stiffness

```python
def isotropic_in_plane_stiffness(E: float, nu: float) -> np.ndarray
```

Plane-stress isotropic stiffness tensor ``C_ijkl`` of shape ``(2,2,2,2)``.

.. code-block:: text

    σ_ij = λ* δ_ij tr(ε) + 2 μ ε_ij     (plane stress)
    λ*   = E ν / (1 - ν^2)
    μ    = E / (2 (1 + ν))

#### orthotropic\_in\_plane\_stiffness

```python
def orthotropic_in_plane_stiffness(E1: float, E2: float, G12: float,
                                   nu12: float) -> np.ndarray
```

Plane-stress orthotropic stiffness tensor in *material* axes.

The fibre direction is the local 1-axis. The Voigt matrix is

.. code-block:: text

    Q = 1 / (1 - ν12 ν21) · [[ E1,        ν21 E1,  0       ],
                              [ ν12 E2,   E2,      0       ],
                              [ 0,        0,       G12·(1-ν12 ν21) ]]

(with ``ν21 = ν12 E2 / E1``) and is converted to the 4th-order tensor
``C_ijkl`` with the Voigt convention σ_voigt = (σ11, σ22, σ12).

#### plate\_stiffness

```python
def plate_stiffness(
        C_in: np.ndarray,
        h: float,
        G_transverse: Union[float, np.ndarray],
        kappa_s: float = 5.0 / 6.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Build ``(A, D, G_s)`` for a homogeneous symmetric flat plate.

Parameters
----------
- **C_in** (*(2,2,2,2) array*): In-plane plane-stress stiffness tensor.
- **h** (*float*): Plate thickness.
- **G_transverse** (*float or (2,2) array*): Transverse shear modulus. A scalar is interpreted as isotropic (``G·I``); a ``(2,2)`` array is used directly (e.g. ``diag(G13, G23)`` for an orthotropic lamina).
- **kappa_s** (*float*): Shear correction factor (default ``5/6``).


Returns
-------
- **A** (*(2,2,2,2) array        Membrane stiffness.*)
- **D** (*(2,2,2,2) array        Bending stiffness.*)


#### z\_layer\_coordinates

```python
def z_layer_coordinates(thicknesses: np.ndarray) -> np.ndarray
```

Interface z-coordinates of an n-layer laminate, midplane at ``z = 0``.

Layers are ordered from bottom (``k = 0``) to top (``k = n − 1``).
The returned array has length ``n + 1`` with ``z[0] = -h/2`` and
``z[n] = +h/2`` (where ``h = sum(thicknesses)``).

#### rotate\_in\_plane\_stiffness

```python
def rotate_in_plane_stiffness(C_in: np.ndarray, theta: float) -> np.ndarray
```

Rotate a 4th-order in-plane stiffness tensor by ``theta`` (radians).

.. math::

    \bar C_{`ijkl`} = R_{`ip`}\, R_{`jq`}\, R_{`kr`}\, R_{`ls`}\, C_{`pqrs`},
    \qquad R(\theta) = \begin{`pmatrix`} c &amp; -s \\ s &amp; c \end{`pmatrix`}.

This is the tensor-form equivalent of the Voigt 3×3 transformation
used in Reddy 1997, §2.3.7.

#### rotate\_shear\_stiffness

```python
def rotate_shear_stiffness(G_shear: np.ndarray, theta: float) -> np.ndarray
```

Rotate a (2,2) transverse-shear stiffness tensor by ``theta`` (radians).

#### laminate\_stiffness

```python
def laminate_stiffness(
    C_in: np.ndarray,
    G_shear: np.ndarray,
    thetas: np.ndarray,
    thicknesses: np.ndarray,
    *,
    kappa_s: float = 5.0 / 6.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Compute ``(A, B, D, G_s)`` for an n-layer laminate (CLT integrals).

For each layer ``k`` with material stiffness rotated by ``thetas[k]``
(in addition to the lamina&#x27;s own material orientation):

.. code-block:: text

    A   = Σ_k Q̄_k         (z_{`k+1`} − z_k)
    B   = (1/2) Σ_k Q̄_k   (z_{`k+1`}² − z_k²)
    D   = (1/3) Σ_k Q̄_k   (z_{`k+1`}³ − z_k³)
    G_s = κ_s · Σ_k Q̄_s,k (z_{`k+1`} − z_k)

where ``z_k`` are the interface coordinates from
:func:`z_layer_coordinates` (midplane at z = 0). This matches Reddy
1997 §1.3.71 (in-plane) and §3.4.18 (transverse shear).

Parameters
----------
- **C_in** (*array*): In-plane stiffness in lamina material axes. Either a single ``(2, 2, 2, 2)`` tensor (broadcast to every layer) or a ``(n_layers, 2, 2, 2, 2)`` array of per-layer materials.
- **G_shear** (*array*): Transverse-shear stiffness, ``(2, 2)`` for a single material or ``(n_layers, 2, 2)`` per layer. Use ``G * np.eye(2)`` for an isotropic lamina or ``np.diag([G13, G23])`` for an orthotropic one.
- **thetas** (*array, shape ``(n_layers,)``*): Layer rotation angles in radians (additional to the lamina&#x27;s own material axes).
- **thicknesses** (*array, shape ``(n_layers,)``*): Layer thicknesses, ordered bottom → top.
- **kappa_s** (*float*): Shear correction factor (default ``5/6``).


Returns
-------
- **A** (*(2, 2, 2, 2)   Membrane stiffness.*)
- **B** (*(2, 2, 2, 2)   Membrane–bending coupling. Vanishes for*): midplane-symmetric laminates such as                  ``[θ / -θ]_s`` or ``[0 / 90]_s``.
- **D** (*(2, 2, 2, 2)   Bending stiffness.*)


#### thermal\_expansion\_from\_orientation

```python
def thermal_expansion_from_orientation(a2: np.ndarray, alpha_fibre: float,
                                       alpha_transverse: float) -> np.ndarray
```

Build a 2D thermal-expansion tensor from an orientation tensor.

Rank-2 closure analogue of the libertas Advani–Tucker stiffness:

.. math::

    \alpha(a_2) = \alpha_t \, I + (\alpha_f - \alpha_t)\, a_2.

Limits:

* ``a_2 = e_1 ⊗ e_1`` (full alignment) → ``α = diag(α_f, α_t)``
* ``a_2 = (1/2)·I`` (uniform random) → ``α = ((α_f+α_t)/2)·I``

The expansion tensor field rotates and decays naturally as ``a_2``
varies in space, making it suitable for libertas-style continuous
fibre-orientation optimisation with thermal loading.

#### rotate\_thermal\_expansion

```python
def rotate_thermal_expansion(alpha: np.ndarray, theta: float) -> np.ndarray
```

Rotate a ``(2, 2)`` thermal-expansion tensor by ``theta`` (radians).

.. math:: \bar\alpha = R\,\alpha\,R^T.

#### laminate\_thermal\_loads

```python
def laminate_thermal_loads(
        C_in: np.ndarray,
        alpha: np.ndarray,
        thetas: np.ndarray,
        thicknesses: np.ndarray,
        *,
        dT_avg: float = 1.0,
        dT_grad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]
```

Thermal force/moment resultants ``(N_T, M_T)`` for an n-layer laminate.

For a through-thickness temperature distribution
``ΔT(z) = dT_avg + z · dT_grad``,

.. code-block:: text

    N_T = Σ_k (C̄_k : α̅_k) · ∫_{`z_k`}^{`z_{k+1`}} ΔT(z) dz
    M_T = Σ_k (C̄_k : α̅_k) · ∫_{`z_k`}^{`z_{k+1`}} z · ΔT(z) dz

where ``C̄_k`` and ``α̅_k`` are the layer stiffness and CTE rotated
by ``thetas[k]`` to laminate axes (Reddy 1997, §1.4).

For a midplane-symmetric laminate at uniform ``ΔT``, ``M_T = 0``.

Parameters
----------
- **C_in** (*array*): In-plane stiffness, ``(2, 2, 2, 2)`` (broadcast to all layers) or ``(n_layers, 2, 2, 2, 2)``.
- **alpha** (*array*): Lamina CTE in material axes, ``(2, 2)`` (broadcast) or ``(n_layers, 2, 2)``. Build with :func:``2 or ``np.diag([alpha_1, alpha_2])``.
- **thetas** (*``(n_layers,)``*): Layer rotation angles (radians).
- **thicknesses** (*``(n_layers,)``*): Layer thicknesses, ordered bottom → top.


Returns
-------
- **N_T** (*``(2, 2)`` array — membrane thermal force resultant.*)


#### mindlin\_strains

```python
def mindlin_strains(
        grad_uvw: np.ndarray,
        grad_theta: np.ndarray,
        theta: np.ndarray,
        *,
        nonlinear: str = "linear"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Compute membrane strain, curvature, and transverse shear strain.

Parameters
----------
- **grad_uvw** (*(3, 2) array*): Spatial gradient of ``(u, v, w)``.
- **grad_theta** (*(2, 2) array*): Spatial gradient of ``(θ_x, θ_y)``.
- **theta** (*(2,) array*): Section rotations ``(θ_x, θ_y)``.
- **nonlinear** (*{`&quot;linear&quot;, &quot;von_karman&quot;`}*): ``&quot;linear&quot;`` (default) uses ``ε = sym(∇u_in)``. ``&quot;von_karman&quot;`` adds the moderate-rotation membrane term ``½ ∇w ⊗ ∇w``, capturing membrane stretching due to bending for rotations up to ~10–15 degrees. The curvature ``κ`` and transverse shear ``γ`` remain linear in their arguments.


Returns
-------
- **eps** (*(2, 2) — membrane strain*)
- **kappa** (*(2, 2) — sym(∇θ)           (curvature)*)


#### mindlin\_resultants

```python
def mindlin_resultants(
    eps: np.ndarray,
    kappa: np.ndarray,
    gamma: np.ndarray,
    A: np.ndarray,
    D: np.ndarray,
    G_s: np.ndarray,
    *,
    B: Optional[np.ndarray] = None,
    N_T: Optional[np.ndarray] = None,
    M_T: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Linear constitutive law for an FSDT plate / laminate.

Without optional terms:

.. code-block:: text

    N = A : ε,    M = D : κ,    Q = G_s · γ.

With ``B`` (membrane–bending coupling, e.g. asymmetric laminate):

.. code-block:: text

    N = A : ε + B : κ,    M = B : ε + D : κ.

With ``N_T`` and/or ``M_T`` (thermal pre-force / pre-moment from
:func:`laminate_thermal_loads`, or computed locally from a quad-point
eigenstrain ``α(x)·ΔT``), they are subtracted from the elastic
resultants:

.. code-block:: text

    N ← N − N_T,    M ← M − M_T.

#### mindlin\_weak\_form

```python
def mindlin_weak_form(A: np.ndarray,
                      D: np.ndarray,
                      G_s: np.ndarray,
                      *,
                      B: Optional[np.ndarray] = None,
                      N_T: Optional[np.ndarray] = None,
                      M_T: Optional[np.ndarray] = None,
                      nonlinear: str = "linear") -> Callable
```

Return a quad-point weak form for the FSDT (Mindlin) plate.

The callable is meant to be returned by
:meth:`feax.Problem.get_weak_form` for a two-variable problem with
layout ``var0 = (u, v, w)``, ``var1 = (θ_x, θ_y)``.

Per quadrature point (with test variations ``δ(u,v,w), δθ``):

.. code-block:: text

    δ(u,v) :  ∫ N_αβ ∂_β δu_α dΩ          → grad_term[0][0:2, :] = N
    δw    :  ∫ q_β   ∂_β δw   dΩ          → grad_term[0][2,   :] = q
    δθ    :  ∫ M_αβ ∂_β δθ_α + Q_α δθ_α dΩ → grad_term[1] = M, mass_term[1] = Q

where ``q = Q`` for ``nonlinear=&quot;linear&quot;`` and
``q = Q + N · ∇w`` for ``nonlinear=&quot;von_karman&quot;`` — the latter is
the membrane–bending coupling term arising from the variation of
``½ ∇w ⊗ ∇w``.

Parameters
----------
- **G_s** (*(2,2) array           Transverse shear stiffness.*)
- **nonlinear** (*{`&quot;linear&quot;, &quot;von_karman&quot;`}*): Strain measure (see :func:``7). ``&quot;von_karman&quot;`` makes the residual cubic in ``w`` and requires Newton iteration (``iter_num != 1`` when calling :func:``4).


Returns
-------
- **weak_form** (*callable*): ``(vals, grads, x, *internal_vars) -&gt; (mass_terms, grad_terms)``.


## MindlinPlate Objects

```python
class MindlinPlate(fe.Problem)
```

Two-variable FSDT plate problem (translations + rotations).

Use :func:`make_mindlin_plate` to construct one cleanly from a single
mesh; this class itself is a vanilla :class:`feax.Problem` subclass
that expects ``additional_info=(A, D, G_s, nonlinear)``.

#### make\_mindlin\_plate

```python
def make_mindlin_plate(mesh,
                       A: np.ndarray,
                       D: np.ndarray,
                       G_s: np.ndarray,
                       ele_type: str = "QUAD4",
                       location_fns: Optional[Iterable[Callable]] = None,
                       gauss_order: Optional[int] = None,
                       nonlinear: str = "linear",
                       B: Optional[np.ndarray] = None,
                       N_T: Optional[np.ndarray] = None,
                       M_T: Optional[np.ndarray] = None) -> MindlinPlate
```

Construct a :class:`MindlinPlate` from a single 2D mesh.

Variable layout is fixed: ``var0 = (u, v, w)`` (``vec=3``) and
``var1 = (θ_x, θ_y)`` (``vec=2``), both on the same mesh.

Parameters
----------
- **mesh** (*feax.Mesh*): Single 2D mesh in the ``z = 0`` plane (only x, y coordinates used).
- **ele_type** (*str*): Element type for both variables (default ``&quot;QUAD4&quot;``).
- **location_fns** (*iterable of callables, optional*): Boundary-location predicates for surface integrals.
- **gauss_order** (*int, optional*): Gaussian quadrature order; ``None`` lets feax choose per element.
- **nonlinear** (*{`&quot;linear&quot;, &quot;von_karman&quot;`}*): Strain measure. ``&quot;von_karman&quot;`` requires a Newton solve (``iter_num != 1`` in :func:``0).
- **B** (*(2,2,2,2) array, optional*): Membrane–bending coupling stiffness for an asymmetric laminate. Defaults to ``None`` (uncoupled, symmetric plate).

