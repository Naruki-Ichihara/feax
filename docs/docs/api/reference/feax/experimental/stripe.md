---
sidebar_label: stripe
title: feax.experimental.stripe
---

Stripe patterns aligned with a direction field on a planar triangle mesh —
experimental.

A port of the core of Knöppel, Crane, Pinkall &amp; Schröder, *Stripe Patterns on
Surfaces* (ACM TOG 2015; code: Keenan Crane&#x27;s ``StripePatterns``) to flat 2D
``TRI3`` meshes.  Given a nodal *line field* (a direction defined up to sign,
e.g. the filament / fibre orientation of a 3D print) and a frequency, it
computes a phase ``α`` whose iso-lines ``α = 2πk`` are evenly spaced stripes
running **along** the field.  The stripes are the print paths; the zeros of
``sin α`` are the boundaries between neighbouring filaments.

On a flat domain the tangent spaces share one global frame, so the parallel
transport of the original algorithm is the identity and the method reduces to:

1. per edge ``ij``: the phase the stripes should advance between the two
   nodes, ``ω_ij = λ (l_ij / 2) [cos(φ_i − β_ij) + cos(φ_j − β_ij)]`` (the
   trapezoid rule of ``λ X · e_ij``; ``φ`` field angle, ``β`` edge angle,
   ``λ`` frequency).  If the two roots of the line field point in opposite
   directions the edge *crosses sheets* and ``φ_j`` is flipped by ``π``;
2. the stripe energy ``Σ_ij w_ij |ψ_i − e^{`iω_ij`} ψ_j|²`` (cotan weights,
   complex conjugation across sheet crossings) as a real ``2n × 2n`` matrix;
   its smallest generalized eigenvector w.r.t. the lumped mass matrix is the
   unit complex field ``ψ_i = e^{`iα_i`}``.  Relaxing ``|ψ| = 1`` is what lets
   the pattern place its own branch points (zeros of ``ψ``) where the field
   has singularities or the spacing cannot be kept;
3. per triangle, the three corner phases are unwrapped consistently with
   ``ω`` and the net turning number ``n`` of the phase around the triangle is
   the stripe *zero index* (a stripe starts or ends there).

Usage
-----

.. code-block:: python

    from feax.experimental.stripe import StripePattern

    sp = StripePattern.compute(mesh, angle, frequency=2 * np.pi / spacing)
    s = sp.scalar()                 # nodal cos α   (1 on stripe centre lines)
    phi = sp.level_set()            # nodal sin α   (0 on the filament boundaries) — orientable fields
    a = sp.phase(face_ids, bary)    # exact phase anywhere, branch points included
    sp.save_vtu(&quot;stripes.vtu&quot;)

``smooth_line_field`` regularises a noisy field (e.g. principal stress
directions of a topology-optimised part) before computing the pattern.

:class:``2 is the same algorithm as a JAX function
``psi(angle, frequency)`` with gradients: the matrix assembly is traced, the
eigenvector is computed by scipy inside a ``pure_callback`` and its VJP comes
from the implicit function theorem (one sparse solve of the bordered
eigen-system per backward pass).  It turns the chain *field angle → stripes →
interface level set → XFEM solve* into one differentiable pipeline, i.e. the
print path becomes an optimisation variable.

Limitations: planar ``TRI3`` meshes; the eigenvector is only differentiable
where the smallest eigenvalue is simple (after fixing the global phase, which
is a free gauge for orientable fields); the sheet-crossing and singular-face
decisions are piecewise constant.

#### line\_field\_angle

```python
def line_field_angle(vectors) -> onp.ndarray
```

Angle in ``[0, π)`` of a nodal vector field (sign ignored: a line field).

#### field\_index

```python
def field_index(points, cells, angle) -> onp.ndarray
```

Index of the line field in every triangle (``±1/2`` singularities give
``±1`` for the doubled angle, i.e. the value returned is the index of the
2-vector field ``e^{`2iφ`}``; nonsingular faces give 0).

#### smooth\_line\_field

```python
def smooth_line_field(mesh: Mesh, angle, t: float) -> onp.ndarray
```

Smooth a nodal line field by one implicit step of the 2-vector
Dirichlet flow: ``(L + M/t) z = M z0`` with ``z = e^{`2iφ`}`` (cotan
Laplacian ``L``, lumped mass ``M``).  Larger ``t`` smooths more; the
result is re-normalised and returned as an angle in ``[0, π)``.

## StripePattern Objects

```python
@dataclass
class StripePattern()
```

Result of :meth:`StripePattern.compute`.

Attributes
----------
- **mesh** (*Mesh*): The (CCW-oriented copy of the) input mesh.
- **angle** (*(N,)*): The nodal line-field angle used.
- **frequency** (*float*): ``λ``: stripe phase advance per unit length (spacing ``2π/λ``).
- **psi** (*(N,) complex*): Unit complex field ``e^{`iα`}`` at the nodes.
- **psi_raw** (*(N,) complex*): The eigenvector before the per-node normalisation: smooth through the branch points (where it vanishes), hence the better level set.
- **corner_phase** (*(F, 3)*): Unwrapped phase at the three corners of each triangle (consistent within the triangle; triangles are independent &quot;islands&quot;).
- **zero_index** (*(F,)*): Turning number of the phase around each triangle: a stripe starts or ends in triangles with ``zero_index != 0``.
- **field_index** (*(F,)*): Index of the input line field in each triangle (singularities).
- **omega** (*(E,) and edges : (E, 2)*): Target phase advance per edge and the edge list.
- **crosses_sheets** (*(E,) bool*): Edges where the two roots of the line field point opposite ways.


#### compute

```python
@classmethod
def compute(cls,
            mesh: Mesh,
            angle,
            frequency: float,
            shift: float = 1e-4,
            oriented: bool = False) -> "StripePattern"
```

Compute the stripe pattern.

Parameters
----------
- **mesh** (*Mesh*): Planar ``TRI3`` mesh.
- **angle** (*(N,) array*): Nodal field angle in radians.  By default a *line* field (only ``angle mod π`` matters; use :func:`line_field_angle` for a vector field).  With ``oriented=True`` the angle is a *direction* (mod 2π): no sheet crossings are introduced, the phase ``α`` is globally consistent and ``sin α`` is single-valued — use this when the field has a consistent orientation (e.g. a print direction) and the interface level set is needed.
- **frequency** (*float*): ``λ`` (radians per unit length): the stripe spacing is ``2π / λ``.
- **shift** (*float*): Positive diagonal shift that keeps the stripe energy definite (does not change the eigenvector).
- **oriented** (*bool*): Treat ``angle`` as an oriented direction field (see above).


#### orientable

```python
@property
def orientable() -> bool
```

True when no edge crosses sheets (the line field has a global sign).

#### nodal\_phase

```python
def nodal_phase() -> onp.ndarray
```

Wrapped nodal phase ``α_i = arg ψ_i`` in (−π, π].

#### scalar

```python
def scalar() -> onp.ndarray
```

Nodal ``cos α``: 1 on the stripe centre lines, −1 half-way between.
Well defined for any line field (invariant under ``α → −α``).

#### level\_set

```python
def level_set(raw: bool = False) -> onp.ndarray
```

Nodal ``sin α``: zero on the filament boundaries ``α = kπ`` with
alternating sign (the XFEM / NLXP3D-style interface level set).
Requires an orientable field: across a sheet crossing the sign of
``α`` flips and ``sin α`` is not single-valued.

``raw=True`` returns ``Im ψ_raw`` instead (``|ψ| sin α``): the same
zero set, but smooth through the stripe branch points, where the
normalised phase is undefined — use it when the pattern has
``zero_index != 0`` triangles and the level set feeds an interface
model.

#### phase

```python
def phase(face_ids, bary) -> onp.ndarray
```

Exact phase at points given by triangle ids (m,) and barycentric
coordinates (m, 3) w.r.t. the CCW corners of ``self.mesh.cells``,
including the branch-point correction of the original renderer.

#### stripes\_at

```python
def stripes_at(face_ids, bary, duty: float = 0.5) -> onp.ndarray
```

Binary filament mask at points: 1 where ``cos α ≥ cos(π duty)``
(a stripe of relative width ``duty`` around each centre line).

#### save\_vtu

```python
def save_vtu(
    filename: str, extra_point_fields=(), extra_cell_fields=()) -> None
```

Write the pattern with :func:`feax.utils.save_sol`: nodal ``cos_alpha``,
``sin_alpha`` (if orientable), ``alpha``, ``angle``, the field as a vector,
and per-cell ``zero_index`` / ``field_index``.

## DifferentiableStripes Objects

```python
class DifferentiableStripes()
```

Stripe pattern as a differentiable JAX function of the field and frequency.

``psi(angle, frequency)`` returns the unit complex nodal field ``e^{`iα`}``
(same algorithm and conventions as :meth:`StripePattern.compute`), and
``jax.grad`` works through it:

* the edge phase advances ``ω``, the matrix entries ``a = w cos ω``,
  ``b = w sin ω`` and the final per-node normalisation are ordinary JAX;
* the smallest generalized eigenvector ``g`` of ``(A(a, b), M)`` is a
  ``custom_vjp`` primitive: forward with ``scipy.sparse.linalg.eigsh`` in a
  ``pure_callback``; backward by the adjoint of the eigen-equation
  ``(A − μM) g = 0, gᵀMg = 1``: solve the bordered system
  ``[[A − μM, Mg], [−(Mg)ᵀ, 0]] [λ; ν] = [ḡ; 0]`` and ``∂L/∂A_kl = −λ_k g_l``.
  For an *orientable* field the eigenvalue is double (the global phase of
  ``ψ`` is free), so the phase is gauged by making ``ψ`` real and positive
  at the node of largest amplitude and the system gets a second border
  row for that gauge.

Decisions that are piecewise constant in the parameters (sheet crossings,
singular faces, the gauge node) are made with zero-gradient operations.

#### psi

```python
def psi(angle, frequency)
```

Unit complex nodal field ``e^{`iα`}`` (N,), differentiable in both arguments.

#### scalar

```python
def scalar(angle, frequency)
```

Nodal ``cos α`` (differentiable).

#### level\_set

```python
def level_set(angle, frequency)
```

Nodal ``sin α`` (differentiable; meaningful for orientable fields).

