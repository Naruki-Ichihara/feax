---
sidebar_label: length_scale
title: feax.gene.length_scale
---

Minimum length scale by geometric constraints.

Implements the two differentiable geometric constraints of

    M. Zhou, B. S. Lazarov, F. Wang &amp; O. Sigmund,
    &quot;Minimum length scale in topology optimization by geometric constraints&quot;,
    Comput. Methods Appl. Mech. Engrg. 293 (2015) 266-282.
    https://doi.org/10.1016/j.cma.2015.05.003

The idea, in one paragraph.  In a filter-and-project scheme the *filtered*
field ``rho_tilde`` is smooth, so the centreline of a thin solid member is a
local maximum of ``rho_tilde`` and the centreline of a thin void gap is a local
minimum -- in both cases ``grad rho_tilde = 0``.  Those stationary points are
the &quot;inflection regions&quot; of the paper.  A member is at least as wide as the
length scale implied by the threshold ``eta_e`` exactly when ``rho_tilde`` at
its centreline has already reached ``eta_e``; a gap is wide enough when
``rho_tilde`` at its centreline has dropped below ``eta_d``.  So it is enough
to (a) mark the stationary points and (b) penalise the ones that fail the
threshold test.

Marking is done with the structural indicators, Eqs. (8-9)::

    I_s = rho_bar       * exp(-c |grad rho_tilde|^2)
    I_v = (1 - rho_bar) * exp(-c |grad rho_tilde|^2)

and the penalties are the aggregated constraints, Eqs. (10-12)::

    g_s = &lt;I_s * min(rho_tilde - eta_e, 0)^2&gt;  &lt;=  eps
    g_v = &lt;I_v * min(eta_d - rho_tilde, 0)^2&gt;  &lt;=  eps

where ``&lt;.&gt;`` is the volume average over the design domain.  ``g_s`` controls
the minimum width of the SOLID phase and ``g_v`` the minimum width of the VOID
phase (equivalently, the minimum gap and the minimum radius of curvature).

``g_s`` is **not** monotone in member width, and should not be read as &quot;how
badly the length scale is missed&quot;.  The penalty grows as a member thins, but
the indicator ``I_s`` is weighted by ``rho_bar``, which falls away once the
member stops surviving the projection at all -- below ``2R(1 - sqrt(1/2))``,
where the filtered peak drops under the blueprint threshold 0.5.  Measured with
``R = 8.84`` mm (an 8 mm target), ``g_s`` rises from 2.6e-15 at 9 mm to 2.9e-4
at 5 mm and then falls back to 2.1e-8 at 3 mm.  That is correct: a 3 mm member
is not a thin member, it is no member.  The constraint polices features that
exist; it does not reward making them disappear, because removing a member
that the objective wants costs the objective.

Both are evaluated at the quadrature points of the mesh rather than at element
centres, so the aggregation is a proper volume integral and the result does not
change when the mesh is graded.  On a uniform mesh this reduces to the paper&#x27;s
element sum.

.. warning::
   The paper is explicit that these constraints must be switched on only
   *after* an initial topology has formed: &quot;starting with a grey initial guess
   and meanwhile imposing the geometric constraints from the beginning can
   result in a local minima very quickly.  In such a case, it is difficult to
   generate a meaningful topology at the end.&quot; (Section 3.)

.. warning::
   ``g_s`` and ``g_v`` are ``O(1e-7)``.  Feeding them to an optimiser as
   ``g - eps`` gives a constraint whose multiplier is ``1e7`` times the
   objective&#x27;s scale, which breaks MMA&#x27;s elastic penalty.  Pass
   ``g / eps - 1`` instead; :func:``6 returns
   the raw ``g`` values and leaves that scaling to the caller.

#### helmholtz\_equivalent\_radius

```python
def helmholtz_equivalent_radius(helmholtz_radius: float) -> float
```

Hat-filter radius ``R`` equivalent to a Helmholtz filter parameter.

:func:`feax.gene.create_helmholtz_filter` solves ``rt - r^2 grad^2 rt = rho``
where ``r`` is the *Helmholtz length parameter*, not a support radius.
Matching the second moment of the two kernels (Lazarov &amp; Sigmund 2011)
gives ``r = R / (2 sqrt(3))``, i.e. ``R = 2 sqrt(3) r``.

Every length-scale formula here is written for ``R``, so convert first.

#### minimum\_width

```python
def minimum_width(radius: float, eta: float) -> float
```

Minimum member width for a **linear hat** filter of radius ``R``.

For a 1-D linear hat filter of radius ``R``, a solid bar of half-width
``a`` filters to ``rho_tilde(centre) = 1 - (1 - a/R)^2``.  Requiring that
to reach ``eta`` gives ``a = R (1 - sqrt(1 - eta))``, so

    w_min = 2 R (1 - sqrt(1 - eta)).

Applied with ``eta = eta_e`` this is the minimum solid width enforced by
``g_s``; applied with ``eta = 1 - eta_d`` it is the minimum void width
enforced by ``g_v``.

This is the paper&#x27;s filter.  For a Helmholtz filter use
:func:``0 -- the two differ by 30-60%, see its
docstring.

#### radius\_for\_minimum\_width

```python
def radius_for_minimum_width(width: float, eta: float) -> float
```

Inverse of :func:`minimum_width`: the hat radius ``R`` giving ``width``.

#### minimum\_width\_helmholtz

```python
def minimum_width_helmholtz(helmholtz_radius: float, eta: float) -> float
```

Minimum member width for a **Helmholtz** filter of parameter ``r``.

The 1-D Green&#x27;s function of ``rt - r^2 grad^2 rt = rho`` is the exponential
kernel ``exp(-|d|/r) / (2r)``, so a solid bar of half-width ``a`` filters to
``rho_tilde(centre) = 1 - exp(-a/r)``.  Requiring that to reach ``eta``:

    w_min = 2 r ln(1 / (1 - eta)).

Do **not** reach this number by converting to an equivalent hat radius and
calling :func:``2.  The equivalence
:func:``3 matches the two kernels&#x27; *second
moment*, which governs smooth fields, not the peak value a narrow bar
retains.  At the origin the exponential kernel is ``sqrt(3)/R`` against the
hat&#x27;s ``1/R``, i.e. 1.73 times as peaked, so a bar keeps more density and
crosses ``eta`` at a smaller width.  Measured on a 1 mm mesh with
``eta = 0.7``, this formula tracks the enforced width to within 5% while
the hat formula overestimates it by 30-60%.

#### helmholtz\_radius\_for\_minimum\_width

```python
def helmholtz_radius_for_minimum_width(width: float, eta: float) -> float
```

Inverse of :func:`minimum_width_helmholtz`.

#### epsilon\_for\_radius

```python
def epsilon_for_radius(r: float) -> float
```

Relaxation ``eps`` of Eq. (12) for a filter radius of ``r`` elements.

``g_s`` and ``g_v`` never reach zero: with a finite ``c`` the indicators
keep small values outside the stationary regions, so the equality
constraints of Eqs. (10-11) are relaxed to ``g &lt;= eps``.  How much slack is
needed depends on how well the discretisation resolves the filter, i.e. on
``r = R / h`` -- a coarse filter needs more.

This interpolates the paper&#x27;s Fig. 8, which records the working ``eps``
against ``r`` (with ``c = r**4``) for its mechanism example::

    r  =  3     7      10
    eps = 1e-6  1e-7   1e-8

The three points are not collinear in ``log eps`` -- the slope steepens
from -0.25 to -0.33 decades per element -- so this is a piecewise linear
interpolation in ``(r, log10 eps)``, clamped to the end slopes outside
``[3, 10]``.  Hard-coding a single number instead is a trap: the same model
on a mesh twice as coarse halves ``r`` and needs roughly twenty times the
slack.

The paper warns that over-relaxing fails to impose the length scale, so
treat this as a starting point -- but it IS only a starting point.  The fit
comes from one mechanism example with element-wise design variables, and
the tail the indicators leave outside the stationary regions depends on the
problem, the boundary conditions and whether the design field is nodal.
Measured on the 2-D thermal bracket in ``thermax``, a design whose narrowest
solid member is 25 mm and whose narrowest gap is 12 mm -- against an 8 mm
target -- still reports ``g_s = 6.4 eps``.  That is a false positive from a
too-tight ``eps``, not a violated length scale.

So do not read ``g &gt; eps`` as &quot;the length scale failed&quot;.  Verify the
achieved width directly: threshold the blueprint, take a Euclidean distance
transform of the solid and of the void, and twice the per-component maximum
is that feature&#x27;s width.  Then set ``eps`` just above what a design you
accept actually scores.

#### minimum\_beta

```python
def minimum_beta(eta_e: float = 0.7) -> float
```

Projection sharpness below which the constraints cannot be satisfied.

The indicator ``I_s = rho_bar * exp(-c |grad rho_tilde|^2)`` weights the
penalty by the *projected* field, so a soft projection leaves ``rho_bar``
around 0.5 across the whole transition band and the penalty is collected
there whether or not the design actually meets the length scale.  Measured
on a design of clean 12 mm members with an 8 mm target, ``g_s`` falls seven
orders of magnitude between ``beta = 2`` and ``beta = 32``, and only from
``beta &gt;= 16`` does a compliant design register as compliant:

    beta =  2     4      8      16     32
    g_s  = 5e-3  1e-3   5e-5   1e-7   8e-10

The paper uses ``beta = 32`` and ``beta = 64`` throughout.  Continuation
that stops at ``beta = 8`` cannot satisfy these constraints at any width.

#### create\_geometric\_length\_scale\_fn

```python
def create_geometric_length_scale_fn(
        mesh,
        *,
        filter_radius: float,
        ele_type: str = "QUAD4",
        dim: int = 2,
        eta_e: float = 0.7,
        eta_d: float = 0.3,
        c: Optional[float] = None,
        element_size: Optional[float] = None,
        mask: Optional[Array] = None,
        field: str = "node") -> Callable[[Array, Array], Tuple[Array, Array]]
```

Build the geometric length-scale constraints of Zhou et al. (2015).

Parameters
----------
- **mesh** (*Mesh*): The design mesh.  The density fields are nodal on this mesh.
- **filter_radius** (*float*): The **hat-filter-equivalent** radius ``R`` in physical units.  For a Helmholtz filter pass ``helmholtz_equivalent_radius(r)``, not ``r``.
- **c** (*float, optional*): Decay rate of the indicators.  Default ``r**4`` with ``r = R / h``, the paper&#x27;s recommendation (Section 3), validated there for ``r`` in ``[3, 10]``.  Workable range ``[0.4, 1.4] * r**4``.  Too large and the indicators vanish at the very regions they should mark; too small and they bleed into non-stationary regions.
- **element_size** (*float, optional*): ``h``, used only to form ``r = R / h`` and to make the gradient dimensionless.  Defaults to the mean element size implied by the mesh volume, which is exact for a uniform mesh.
- **field** (*{`&quot;node&quot;, &quot;cell&quot;`}*): Whether the density fields are given per node or per element.  Element fields are piecewise constant, so their gradient vanishes INSIDE every element and the indicators would mark the whole domain.  For ``&quot;cell&quot;`` the field is therefore first recovered to the nodes by a volume-weighted average and the gradient taken from that, which is the continuum analogue of the finite differences the paper takes between neighbouring elements.  ``rho_tilde`` and ``rho_bar`` themselves are still used at their own resolution.
- **mask** (*array of shape (num_nodes,) or (num_cells,), optional*): Weight in ``[0, 1]`` restricting the integral to the design region. Pass the design mask when the model has passive solid or void regions: their filtered field is not under the optimiser&#x27;s control, so including them only injects a constant violation.


Returns
-------
callable
    ``fn(rho_tilde, rho_bar) -&gt; (g_s, g_v)``, both scalars, where
    ``rho_tilde`` is the filtered nodal field and ``rho_bar`` the projected
    (physical) nodal field.  Differentiable in both.

Notes
-----
The gradient is made dimensionless as ``h * grad rho_tilde`` so that ``c``
keeps the paper&#x27;s meaning (their finite difference is per element).

#### create\_morphological\_length\_scale\_fn

```python
def create_morphological_length_scale_fn(
        points,
        *,
        min_width: float,
        sharpness: float = 40.0,
        p: float = 12.0,
        mask: Optional[Array] = None,
        volumes: Optional[Array] = None
) -> Callable[[Array], Tuple[Array, Array]]
```

Minimum width by morphological opening and closing of the blueprint.

A complement to :func:`create_geometric_length_scale_fn`, which it exists
to cover rather than replace.  The geometric constraints of Zhou et al.
police only the STATIONARY points of the filtered field -- their Eq. (1)
defines the region as ``{`x : rho_bar = 1 and grad rho_tilde = 0`}`` -- which
is exactly what their argument needs, because a topology change under a
shifting threshold has to happen at a stationary point.  It is not the same
as a minimum width.  A member that tapers along its length has no
stationary point anywhere on it, so its indicator is killed by
``exp(-c |grad rho_tilde|^2)`` and it may thin without limit.  Measured on
a bracket with a strut tapering from 8 mm to 2 mm, the indicator&#x27;s decay
factor fell to 1e-17 exactly where the member was thinnest, and ``g_s``
stayed inside its bound.

This measure asks the question directly.  A feature is at least ``w`` wide
iff it survives an erosion by a disc of radius ``w/2`` followed by a
dilation by the same disc -- the morphological OPENING.  Material present
in the design but removed by the opening sits in something thinner than
``w``::

    v_solid = max(rho - opening(rho), 0)
    g_solid = ( &lt; v_solid^p &gt; )^(1/p)

and the mirror image for the void.  Both vanish exactly when every solid
feature and every void gap is at least ``min_width`` across, with no
reference to gradients or stationary points.

The aggregate is a p-norm and NOT a volume average, which matters more than
it looks.  Averaged, a violation confined to a few elements is divided by
the whole domain: an optimiser handed that constraint finds a string of
beads joined by one-element necks, because the necks are too small a
fraction to register.  Measured here, exactly that happened -- the design
came out in eight pieces with a 1 mm neck against an 8 mm target while the
averaged constraint sat at 0.9998 of its bound, tuned to four digits. The
p-norm is insensitive to how much domain surrounds the violation.

Erosion and dilation are made differentiable by the usual log-sum-exp
softening over the disc neighbourhood, which is one sparse mat-vec each::

    erode(rho)  = -log( W exp(-k rho) ) / k
    dilate(rho) =  log( W exp( k rho) ) / k

with ``W`` the row-normalised indicator of &quot;within w/2&quot;.  As ``k`` grows
these approach the true min and max; ``k = 24`` recovers a step edge to
about 3% while keeping the gradient usable.

Parameters
----------
- **points** (*array of shape (N, dim)*): Coordinates of the design sites -- element centroids for an element field, nodes for a nodal one.  The measure is defined on whatever the blueprint lives on.
- **min_width** (*float*): The erosion diameter, in the same units as ``points``.  Set it about one element BELOW the width you want to keep: at ``min_width`` exactly equal to the target, a member of precisely that width erodes to zero thickness and the softened min removes it, so the target itself reads as a violation.  Measured on a 1 mm mesh against an 8 mm target, ``min_width = 7`` gives 1.1e-4 for every width from 8 mm up and 4.6e-2 from 6 mm down -- a factor of 410 -- while ``min_width = 8`` flags the 8 mm member itself at 5.8e-2.
- **p** (*float*): Exponent of the aggregation.  Large ``p`` approaches the worst element and refuses to dilute; 12 is the same value the stress p-norm uses here and already tracks the max closely.
- **sharpness** (*float*): ``k`` above.  Too small blurs the min/max and lets thin features pass; too large and ``exp(k)`` loses precision.  The separation above is flat in ``k``; what changes is the floor a compliant design sits at -- 2.8e-4 at ``k = 24``, 1.0e-4 at 40, 4.5e-5 at 60 -- so 40 is a reasonable place to stop. The floor is the softening, not a violation: put the constraint&#x27;s ``eps`` above it.
- **mask** (*array of shape (N,), optional*): Weight in ``[0, 1]`` restricting the aggregate to the design region.
- **volumes** (*array of shape (N,), optional*): Per-site volume for the average.  Defaults to uniform, which is exact on a uniform mesh.


Returns
-------
callable
    ``fn(rho) -&gt; (g_solid, g_void)``, both scalars, differentiable in
    ``rho``.  Scale them by an ``eps`` before handing them to an optimiser,
    as with the geometric pair.

