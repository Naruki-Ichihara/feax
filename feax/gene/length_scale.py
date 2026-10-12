"""Minimum length scale by geometric constraints.

Implements the two differentiable geometric constraints of

    M. Zhou, B. S. Lazarov, F. Wang & O. Sigmund,
    "Minimum length scale in topology optimization by geometric constraints",
    Comput. Methods Appl. Mech. Engrg. 293 (2015) 266-282.
    https://doi.org/10.1016/j.cma.2015.05.003

The idea, in one paragraph.  In a filter-and-project scheme the *filtered*
field ``rho_tilde`` is smooth, so the centreline of a thin solid member is a
local maximum of ``rho_tilde`` and the centreline of a thin void gap is a local
minimum -- in both cases ``grad rho_tilde = 0``.  Those stationary points are
the "inflection regions" of the paper.  A member is at least as wide as the
length scale implied by the threshold ``eta_e`` exactly when ``rho_tilde`` at
its centreline has already reached ``eta_e``; a gap is wide enough when
``rho_tilde`` at its centreline has dropped below ``eta_d``.  So it is enough
to (a) mark the stationary points and (b) penalise the ones that fail the
threshold test.

Marking is done with the structural indicators, Eqs. (8-9)::

    I_s = rho_bar       * exp(-c |grad rho_tilde|^2)
    I_v = (1 - rho_bar) * exp(-c |grad rho_tilde|^2)

and the penalties are the aggregated constraints, Eqs. (10-12)::

    g_s = <I_s * min(rho_tilde - eta_e, 0)^2>  <=  eps
    g_v = <I_v * min(eta_d - rho_tilde, 0)^2>  <=  eps

where ``<.>`` is the volume average over the design domain.  ``g_s`` controls
the minimum width of the SOLID phase and ``g_v`` the minimum width of the VOID
phase (equivalently, the minimum gap and the minimum radius of curvature).

``g_s`` is **not** monotone in member width, and should not be read as "how
badly the length scale is missed".  The penalty grows as a member thins, but
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
change when the mesh is graded.  On a uniform mesh this reduces to the paper's
element sum.

.. warning::
   The paper is explicit that these constraints must be switched on only
   *after* an initial topology has formed: "starting with a grey initial guess
   and meanwhile imposing the geometric constraints from the beginning can
   result in a local minima very quickly.  In such a case, it is difficult to
   generate a meaningful topology at the end." (Section 3.)

.. warning::
   ``g_s`` and ``g_v`` are ``O(1e-7)``.  Feeding them to an optimiser as
   ``g - eps`` gives a constraint whose multiplier is ``1e7`` times the
   objective's scale, which breaks MMA's elastic penalty.  Pass
   ``g / eps - 1`` instead; :func:`create_geometric_length_scale_fn` returns
   the raw ``g`` values and leaves that scaling to the caller.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import jax.numpy as np
import numpy as onp

from ..fe import FiniteElement

__all__ = [
    "create_geometric_length_scale_fn",
    "helmholtz_equivalent_radius",
    "minimum_width",
    "minimum_width_helmholtz",
    "radius_for_minimum_width",
    "helmholtz_radius_for_minimum_width",
    "epsilon_for_radius",
    "minimum_beta",
    "create_morphological_length_scale_fn",
]

Array = np.ndarray


# ---------------------------------------------------------------------------
# Length-scale bookkeeping
# ---------------------------------------------------------------------------

def helmholtz_equivalent_radius(helmholtz_radius: float) -> float:
    """Hat-filter radius ``R`` equivalent to a Helmholtz filter parameter.

    :func:`feax.gene.create_helmholtz_filter` solves ``rt - r^2 grad^2 rt = rho``
    where ``r`` is the *Helmholtz length parameter*, not a support radius.
    Matching the second moment of the two kernels (Lazarov & Sigmund 2011)
    gives ``r = R / (2 sqrt(3))``, i.e. ``R = 2 sqrt(3) r``.

    Every length-scale formula here is written for ``R``, so convert first.
    """
    return 2.0 * math.sqrt(3.0) * float(helmholtz_radius)


def minimum_width(radius: float, eta: float) -> float:
    """Minimum member width for a **linear hat** filter of radius ``R``.

    For a 1-D linear hat filter of radius ``R``, a solid bar of half-width
    ``a`` filters to ``rho_tilde(centre) = 1 - (1 - a/R)^2``.  Requiring that
    to reach ``eta`` gives ``a = R (1 - sqrt(1 - eta))``, so

        w_min = 2 R (1 - sqrt(1 - eta)).

    Applied with ``eta = eta_e`` this is the minimum solid width enforced by
    ``g_s``; applied with ``eta = 1 - eta_d`` it is the minimum void width
    enforced by ``g_v``.

    This is the paper's filter.  For a Helmholtz filter use
    :func:`minimum_width_helmholtz` -- the two differ by 30-60%, see its
    docstring.
    """
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    return 2.0 * float(radius) * (1.0 - math.sqrt(1.0 - eta))


def radius_for_minimum_width(width: float, eta: float) -> float:
    """Inverse of :func:`minimum_width`: the hat radius ``R`` giving ``width``."""
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    return 0.5 * float(width) / (1.0 - math.sqrt(1.0 - eta))


def minimum_width_helmholtz(helmholtz_radius: float, eta: float) -> float:
    """Minimum member width for a **Helmholtz** filter of parameter ``r``.

    The 1-D Green's function of ``rt - r^2 grad^2 rt = rho`` is the exponential
    kernel ``exp(-|d|/r) / (2r)``, so a solid bar of half-width ``a`` filters to
    ``rho_tilde(centre) = 1 - exp(-a/r)``.  Requiring that to reach ``eta``:

        w_min = 2 r ln(1 / (1 - eta)).

    Do **not** reach this number by converting to an equivalent hat radius and
    calling :func:`minimum_width`.  The equivalence
    :func:`helmholtz_equivalent_radius` matches the two kernels' *second
    moment*, which governs smooth fields, not the peak value a narrow bar
    retains.  At the origin the exponential kernel is ``sqrt(3)/R`` against the
    hat's ``1/R``, i.e. 1.73 times as peaked, so a bar keeps more density and
    crosses ``eta`` at a smaller width.  Measured on a 1 mm mesh with
    ``eta = 0.7``, this formula tracks the enforced width to within 5% while
    the hat formula overestimates it by 30-60%.
    """
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    return 2.0 * float(helmholtz_radius) * math.log(1.0 / (1.0 - eta))


def helmholtz_radius_for_minimum_width(width: float, eta: float) -> float:
    """Inverse of :func:`minimum_width_helmholtz`."""
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    return 0.5 * float(width) / math.log(1.0 / (1.0 - eta))


def epsilon_for_radius(r: float) -> float:
    """Relaxation ``eps`` of Eq. (12) for a filter radius of ``r`` elements.

    ``g_s`` and ``g_v`` never reach zero: with a finite ``c`` the indicators
    keep small values outside the stationary regions, so the equality
    constraints of Eqs. (10-11) are relaxed to ``g <= eps``.  How much slack is
    needed depends on how well the discretisation resolves the filter, i.e. on
    ``r = R / h`` -- a coarse filter needs more.

    This interpolates the paper's Fig. 8, which records the working ``eps``
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

    So do not read ``g > eps`` as "the length scale failed".  Verify the
    achieved width directly: threshold the blueprint, take a Euclidean distance
    transform of the solid and of the void, and twice the per-component maximum
    is that feature's width.  Then set ``eps`` just above what a design you
    accept actually scores.
    """
    knots_r = (3.0, 7.0, 10.0)
    knots_e = (-6.0, -7.0, -8.0)
    r = float(r)
    if r <= knots_r[0]:
        slope = (knots_e[1] - knots_e[0]) / (knots_r[1] - knots_r[0])
        exponent = knots_e[0] + slope * (r - knots_r[0])
    elif r >= knots_r[-1]:
        slope = (knots_e[2] - knots_e[1]) / (knots_r[2] - knots_r[1])
        exponent = knots_e[2] + slope * (r - knots_r[2])
    else:
        i = 0 if r < knots_r[1] else 1
        slope = ((knots_e[i + 1] - knots_e[i])
                 / (knots_r[i + 1] - knots_r[i]))
        exponent = knots_e[i] + slope * (r - knots_r[i])
    return 10.0 ** exponent


def minimum_beta(eta_e: float = 0.7) -> float:
    """Projection sharpness below which the constraints cannot be satisfied.

    The indicator ``I_s = rho_bar * exp(-c |grad rho_tilde|^2)`` weights the
    penalty by the *projected* field, so a soft projection leaves ``rho_bar``
    around 0.5 across the whole transition band and the penalty is collected
    there whether or not the design actually meets the length scale.  Measured
    on a design of clean 12 mm members with an 8 mm target, ``g_s`` falls seven
    orders of magnitude between ``beta = 2`` and ``beta = 32``, and only from
    ``beta >= 16`` does a compliant design register as compliant:

        beta =  2     4      8      16     32
        g_s  = 5e-3  1e-3   5e-5   1e-7   8e-10

    The paper uses ``beta = 32`` and ``beta = 64`` throughout.  Continuation
    that stops at ``beta = 8`` cannot satisfy these constraints at any width.
    """
    return 16.0


# ---------------------------------------------------------------------------
# The constraints
# ---------------------------------------------------------------------------

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
    field: str = "node",
) -> Callable[[Array, Array], Tuple[Array, Array]]:
    """Build the geometric length-scale constraints of Zhou et al. (2015).

    Parameters
    ----------
    mesh : Mesh
        The design mesh.  The density fields are nodal on this mesh.
    filter_radius : float
        The **hat-filter-equivalent** radius ``R`` in physical units.  For a
        Helmholtz filter pass ``helmholtz_equivalent_radius(r)``, not ``r``.
    ele_type, dim :
        Element type and spatial dimension of ``mesh``.
    eta_e, eta_d : float
        Eroded and dilated thresholds, ``eta_d < eta_i < eta_e``.  The paper
        uses ``(0.3, 0.7)`` and ``(0.25, 0.75)``; a wider range means a larger
        enforced length scale.  See :func:`minimum_width`.
    c : float, optional
        Decay rate of the indicators.  Default ``r**4`` with ``r = R / h``,
        the paper's recommendation (Section 3), validated there for
        ``r`` in ``[3, 10]``.  Workable range ``[0.4, 1.4] * r**4``.  Too large
        and the indicators vanish at the very regions they should mark; too
        small and they bleed into non-stationary regions.
    element_size : float, optional
        ``h``, used only to form ``r = R / h`` and to make the gradient
        dimensionless.  Defaults to the mean element size implied by the mesh
        volume, which is exact for a uniform mesh.
    field : {"node", "cell"}
        Whether the density fields are given per node or per element.  Element
        fields are piecewise constant, so their gradient vanishes INSIDE every
        element and the indicators would mark the whole domain.  For
        ``"cell"`` the field is therefore first recovered to the nodes by a
        volume-weighted average and the gradient taken from that, which is the
        continuum analogue of the finite differences the paper takes between
        neighbouring elements.  ``rho_tilde`` and ``rho_bar`` themselves are
        still used at their own resolution.
    mask : array of shape (num_nodes,) or (num_cells,), optional
        Weight in ``[0, 1]`` restricting the integral to the design region.
        Pass the design mask when the model has passive solid or void regions:
        their filtered field is not under the optimiser's control, so including
        them only injects a constant violation.

    Returns
    -------
    callable
        ``fn(rho_tilde, rho_bar) -> (g_s, g_v)``, both scalars, where
        ``rho_tilde`` is the filtered nodal field and ``rho_bar`` the projected
        (physical) nodal field.  Differentiable in both.

    Notes
    -----
    The gradient is made dimensionless as ``h * grad rho_tilde`` so that ``c``
    keeps the paper's meaning (their finite difference is per element).
    """
    if not 0.0 < eta_d < eta_e < 1.0:
        raise ValueError(
            f"need 0 < eta_d < eta_e < 1, got eta_d={eta_d}, eta_e={eta_e}")

    if field not in ("node", "cell"):
        raise ValueError(f"field must be 'node' or 'cell', got {field!r}")

    fes = FiniteElement(mesh=mesh, vec=1, dim=dim, ele_type=ele_type)
    cells = np.asarray(fes.cells)                  # (nc, nn)
    shape_vals = np.asarray(fes.shape_vals)        # (nq, nn)
    shape_grads = np.asarray(fes.shape_grads)      # (nc, nq, nn, dim)
    JxW = np.asarray(fes.JxW)                      # (nc, nq)

    volume = float(onp.sum(onp.asarray(fes.JxW)))
    num_cells = int(fes.num_cells)
    if element_size is None:
        element_size = (volume / num_cells) ** (1.0 / dim)
    h = float(element_size)
    r = float(filter_radius) / h
    if c is None:
        c = r ** 4

    # Cell -> node recovery: each node averages the cells it belongs to,
    # weighted by their volume. Built once as scatter indices.
    cell_volume = np.sum(JxW, axis=1)                           # (nc,)
    num_nodes = int(fes.num_total_nodes)
    node_weight = np.zeros(num_nodes).at[cells.reshape(-1)].add(
        np.repeat(cell_volume, cells.shape[1]))

    def to_nodes(cell_field: Array) -> Array:
        acc = np.zeros(num_nodes).at[cells.reshape(-1)].add(
            np.repeat(cell_field * cell_volume, cells.shape[1]))
        return acc / np.maximum(node_weight, 1e-30)

    if mask is None:
        weight = JxW
    else:
        mask = np.asarray(mask)
        if field == "cell":
            mask_q = np.broadcast_to(mask[:, None], JxW.shape)
        else:
            mask_q = np.einsum("qn,cn->cq", shape_vals, mask[cells])
        weight = JxW * np.clip(mask_q, 0.0, 1.0)
    total = np.sum(weight)

    def length_scale_constraints(rho_tilde: Array, rho_bar: Array):
        if field == "cell":
            rt_q = np.broadcast_to(rho_tilde[:, None], JxW.shape)   # (nc, nq)
            rb_q = np.broadcast_to(rho_bar[:, None], JxW.shape)
            rt_nodal = to_nodes(rho_tilde)
        else:
            rt_nodal = rho_tilde
            rt_q = np.einsum("qn,cn->cq", shape_vals, rho_tilde[cells])
            rb_q = np.einsum("qn,cn->cq", shape_vals, rho_bar[cells])
        rt = rt_nodal[cells]                                        # (nc, nn)
        grad_q = np.einsum("cqnd,cn->cqd", shape_grads, rt)         # (nc, nq, dim)
        grad2 = (h * h) * np.sum(grad_q * grad_q, axis=-1)      # dimensionless
        decay = np.exp(-c * grad2)                              # Eqs. (8-9)

        pen_s = np.minimum(rt_q - eta_e, 0.0) ** 2              # Eq. (10)
        pen_v = np.minimum(eta_d - rt_q, 0.0) ** 2              # Eq. (11)
        g_s = np.sum(rb_q * decay * pen_s * weight) / total
        g_v = np.sum((1.0 - rb_q) * decay * pen_v * weight) / total
        return g_s, g_v

    length_scale_constraints.r = r
    length_scale_constraints.c = c
    length_scale_constraints.element_size = h
    length_scale_constraints.filter_radius = float(filter_radius)
    # Reported for the paper's hat filter; with a Helmholtz filter use
    # minimum_width_helmholtz on its own radius parameter instead.
    length_scale_constraints.min_solid_width = minimum_width(filter_radius, eta_e)
    length_scale_constraints.min_void_width = minimum_width(filter_radius, 1.0 - eta_d)
    return length_scale_constraints


# ---------------------------------------------------------------------------
# Minimum length scale by morphological opening / closing
# ---------------------------------------------------------------------------

def create_morphological_length_scale_fn(
    points,
    *,
    min_width: float,
    sharpness: float = 40.0,
    p: float = 12.0,
    mask: Optional[Array] = None,
    volumes: Optional[Array] = None,
) -> Callable[[Array], Tuple[Array, Array]]:
    """Minimum width by morphological opening and closing of the blueprint.

    A complement to :func:`create_geometric_length_scale_fn`, which it exists
    to cover rather than replace.  The geometric constraints of Zhou et al.
    police only the STATIONARY points of the filtered field -- their Eq. (1)
    defines the region as ``{x : rho_bar = 1 and grad rho_tilde = 0}`` -- which
    is exactly what their argument needs, because a topology change under a
    shifting threshold has to happen at a stationary point.  It is not the same
    as a minimum width.  A member that tapers along its length has no
    stationary point anywhere on it, so its indicator is killed by
    ``exp(-c |grad rho_tilde|^2)`` and it may thin without limit.  Measured on
    a bracket with a strut tapering from 8 mm to 2 mm, the indicator's decay
    factor fell to 1e-17 exactly where the member was thinnest, and ``g_s``
    stayed inside its bound.

    This measure asks the question directly.  A feature is at least ``w`` wide
    iff it survives an erosion by a disc of radius ``w/2`` followed by a
    dilation by the same disc -- the morphological OPENING.  Material present
    in the design but removed by the opening sits in something thinner than
    ``w``::

        v_solid = max(rho - opening(rho), 0)
        g_solid = ( < v_solid^p > )^(1/p)

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

    with ``W`` the row-normalised indicator of "within w/2".  As ``k`` grows
    these approach the true min and max; ``k = 24`` recovers a step edge to
    about 3% while keeping the gradient usable.

    Parameters
    ----------
    points : array of shape (N, dim)
        Coordinates of the design sites -- element centroids for an element
        field, nodes for a nodal one.  The measure is defined on whatever the
        blueprint lives on.
    min_width : float
        The erosion diameter, in the same units as ``points``.  Set it about
        one element BELOW the width you want to keep: at ``min_width`` exactly
        equal to the target, a member of precisely that width erodes to zero
        thickness and the softened min removes it, so the target itself reads
        as a violation.  Measured on a 1 mm mesh against an 8 mm target,
        ``min_width = 7`` gives 1.1e-4 for every width from 8 mm up and 4.6e-2
        from 6 mm down -- a factor of 410 -- while ``min_width = 8`` flags the
        8 mm member itself at 5.8e-2.
    p : float
        Exponent of the aggregation.  Large ``p`` approaches the worst element
        and refuses to dilute; 12 is the same value the stress p-norm uses here
        and already tracks the max closely.
    sharpness : float
        ``k`` above.  Too small blurs the min/max and lets thin features pass;
        too large and ``exp(k)`` loses precision.  The separation above is
        flat in ``k``; what changes is the floor a compliant design sits at --
        2.8e-4 at ``k = 24``, 1.0e-4 at 40, 4.5e-5 at 60 -- so 40 is a
        reasonable place to stop. The floor is the softening, not a violation:
        put the constraint's ``eps`` above it.
    mask : array of shape (N,), optional
        Weight in ``[0, 1]`` restricting the aggregate to the design region.
    volumes : array of shape (N,), optional
        Per-site volume for the average.  Defaults to uniform, which is exact
        on a uniform mesh.

    Returns
    -------
    callable
        ``fn(rho) -> (g_solid, g_void)``, both scalars, differentiable in
        ``rho``.  Scale them by an ``eps`` before handing them to an optimiser,
        as with the geometric pair.
    """
    from .filters import _compute_density_filter_weights_sparse

    if min_width <= 0.0:
        raise ValueError(f"min_width must be positive, got {min_width}")
    if sharpness <= 0.0:
        raise ValueError(f"sharpness must be positive, got {sharpness}")

    pts = onp.asarray(points)
    radius = 0.5 * float(min_width)
    W, row_sums = _compute_density_filter_weights_sparse(
        pts, radius, weight_type="constant")

    # The erosion disc has to be resolved. Measured against an 8 mm target on
    # a square grid, sweeping the element size: the separation between a 6 mm
    # member (a violation) and an 8 mm one (compliant) is three orders at
    # radius/h = 3.5, still two at 2.33, and GONE at 1.75 -- there the 6 mm
    # member scores exactly what a 16 mm one does, because the disc spans too
    # few sites to erode anything. Below about 1.2 it over-corrects instead and
    # flags the target width itself.
    # ``row_sums`` with constant weights is the number of sites inside the
    # disc, so the spacing follows from inverting the volume of a dim-ball:
    # n = V_dim (r/h)**dim.  Dropping V_dim, as an earlier version did outside
    # 2-D, understates h by 1.61 in 3-D and lets a mesh through that the same
    # criterion rejects in 2-D.
    _ball = {1: 2.0, 2: onp.pi, 3: 4.0 * onp.pi / 3.0}.get(pts.shape[1], 1.0)
    spacing = (_ball ** (1.0 / pts.shape[1]) * radius
               * float(onp.mean(row_sums)) ** (-1.0 / pts.shape[1]))
    sites_per_radius = radius / max(spacing, 1e-30)
    if sites_per_radius < 2.0:
        raise ValueError(
            f"min_width={min_width} is too small for this mesh: the erosion "
            f"disc spans only {sites_per_radius:.2f} sites per radius and the "
            f"measure stops discriminating below about 2.3. Refine the mesh or "
            f"raise min_width.")
    inv = 1.0 / np.maximum(np.asarray(row_sums), 1.0)
    k = float(sharpness)

    def _erode(f):
        return -np.log(np.maximum((W @ np.exp(-k * f)) * inv, 1e-300)) / k

    def _dilate(f):
        return np.log(np.maximum((W @ np.exp(k * f)) * inv, 1e-300)) / k

    pp = float(p)
    n = pts.shape[0]
    w = np.ones(n) if volumes is None else np.asarray(volumes)
    if mask is not None:
        w = w * np.clip(np.asarray(mask), 0.0, 1.0)
    total = np.sum(w)

    def _pnorm(v):
        """Volume-weighted p-norm, normalised by its own max so that ``v**p``
        cannot underflow before the sum is taken.

        The inner sum is floored before the root.  A design that satisfies the
        constraint everywhere makes ``v`` identically zero, and ``0**(1/p)`` has
        an INFINITE derivative: without the floor the gradient handed to the
        optimiser is nan exactly when the constraint is comfortably met, and one
        nan poisons the whole MMA step.  Since ``scaled`` is normalised to a max
        of 1, the sum is at least ``min(w)/total`` whenever ``v`` is not all
        zero, so the floor can only ever bite in that one case.
        """
        vmax = np.max(v)
        scaled = v / np.maximum(vmax, 1e-30)
        mean_p = np.maximum(np.sum(w * scaled ** pp) / total, 1e-300)
        return vmax * mean_p ** (1.0 / pp)

    def morphological_constraints(rho: Array):
        opened = _dilate(_erode(rho))          # removes solid thinner than w
        closed = _erode(_dilate(rho))          # fills void thinner than w
        return (_pnorm(np.maximum(rho - opened, 0.0)),
                _pnorm(np.maximum(closed - rho, 0.0)))

    morphological_constraints.min_width = float(min_width)
    morphological_constraints.radius = radius
    morphological_constraints.sharpness = k
    morphological_constraints.sites_per_radius = sites_per_radius
    morphological_constraints.p = pp
    return morphological_constraints
