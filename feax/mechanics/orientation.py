"""Fibre-orientation tensor API (2-D and 3-D).

Reusable building blocks for continuous fibre-orientation modelling and
optimisation, based on Advani–Tucker orientation tensors.

A fibre-orientation distribution is summarised by its second-order
orientation tensor

.. math::

    a_{ij} = \\langle p_i p_j \\rangle,

where ``p`` is the unit fibre direction and ``⟨·⟩`` averages over the
local fibre population.  ``a_2`` is symmetric and (for normalised
distributions) ``tr(a_2) = 1``.  Two limiting cases:

* ``a_2 = e_1 ⊗ e_1`` — perfect alignment along the 1-axis;
* ``a_2 = (1/d) I`` (with ``d = 2`` or ``3``) — fully isotropic.

For 4th-order constitutive laws (orthotropic stiffness, anisotropic
thermal expansion, etc.) we also need ``a_4 = ⟨p p p p⟩``.  ``a_4`` is
not determined by ``a_2`` alone and must be approximated through a
*closure* — see :func:`quadratic_closure`, :func:`linear_closure`,
:func:`hybrid_closure` (all dimension-aware).

Module contents
---------------

1. **Design parameterisations**

   * :func:`box_to_triangle` — libertas 2-D 8-node serendipity map
     ``[-1, 1]² → {a11, a22 ≥ 0, a11+a22 ≤ 1}``;
   * :func:`box_to_tetrahedron` — Nomura *et al.* 2019 3-D 8-node
     trilinear-hex map ``[-1, 1]³ → {a_ii ≥ 0, ∑a_ii ≤ 1}``;
   * :func:`smooth_sgn` — smoothed sign for off-diagonal control;
   * :func:`orientation_tensor_2d`, :func:`orientation_tensor_3d` —
     convenience builders that combine the above into a full ``a_2``.

2. **Closures** — :func:`quadratic_closure`, :func:`linear_closure`,
   :func:`hybrid_closure` for ``a_4`` from ``a_2``.

3. **Constitutive builders** — :func:`orthotropic_stiffness` (2-D
   plane-stress, 4 moduli) and :func:`orthotropic_stiffness_3d`
   (full 3-D transverse-isotropic, 5 moduli).

References
----------
* Advani, S. G. & Tucker, C. L. (1987). The use of tensors to describe
  and predict fiber orientation in short fiber composites.
  *J. Rheol.* 31(8):751–784.
* Advani, S. G. & Tucker, C. L. (1990). Closure approximations for
  three-dimensional structure tensors.  *J. Rheol.* 34(3):367–386.
* Nomura, T., Kawamoto, A., Kondoh, T., Dede, E. M., Lee, J., Song,
  Y., Kikuchi, N. (2019).  Inverse design of structure and fiber
  orientation by means of topology optimization with tensor field
  variables.  *Composites Part B* 176, 107187.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as np


# ---------------------------------------------------------------------------
# 2-D libertas isoparametric box-to-triangle map (8-node serendipity)
# ---------------------------------------------------------------------------

# Boundary node values for (a11, a22) at the 8 serendipity nodes of [-1, 1]²:
#   N1 (-1,-1)  N2 (0,-1)  N3 (1,-1)  N4 (1, 0)
#   N5 ( 1, 1)  N6 (0, 1)  N7 (-1,1)  N8 (-1, 0)
# The image is the triangle ``{a11, a22 >= 0, a11 + a22 <= 1}``.
_TRI_TOL = 1e-6
_U = np.array([_TRI_TOL, 0.5, 1.0, 0.75, 0.5, 0.25, _TRI_TOL, _TRI_TOL])
_V = np.array([_TRI_TOL, _TRI_TOL, _TRI_TOL, 0.25, 0.5, 0.75, 1.0, 0.5])


def _serendipity_8(z: float, e: float) -> np.ndarray:
    """Eight 2-D serendipity shape functions on ``[-1, 1]²``."""
    N1 = -(1 - z) * (1 - e) * (1 + z + e) / 4
    N2 = (1 - z**2) * (1 - e) / 2
    N3 = -(1 + z) * (1 - e) * (1 - z + e) / 4
    N4 = (1 + z) * (1 - e**2) / 2
    N5 = -(1 + z) * (1 + e) * (1 - z - e) / 4
    N6 = (1 - z**2) * (1 + e) / 2
    N7 = -(1 - z) * (1 + e) * (1 + z - e) / 4
    N8 = (1 - z) * (1 - e**2) / 2
    return np.stack([N1, N2, N3, N4, N5, N6, N7, N8])


def box_to_triangle(x1: float, x2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Map the 2-D design square ``[-1, 1]²`` onto the orientation triangle.

    Smooth (C¹), nearly-bijective map onto
    ``{a11, a22 ≥ 0, a11 + a22 ≤ 1}`` via 8-node quadratic serendipity
    interpolation.  The standard libertas parameterisation for box-
    bounded design variables that respect the positivity and trace
    constraints on the diagonal of ``a_2``.

    Parameters
    ----------
    x1, x2 : scalar or array, in ``[-1, 1]``
        Box-bounded design variables.

    Returns
    -------
    a11, a22 : same shape as inputs
    """
    N = _serendipity_8(x1, x2)
    return np.sum(_U * N), np.sum(_V * N)


# ---------------------------------------------------------------------------
# 3-D Nomura 2019 box-to-tetrahedron map (8-node trilinear hex)
# ---------------------------------------------------------------------------

# Image vertices for the 8 corners of ``[-1, 1]^3`` (Nomura et al. 2019,
# Eq. 32 — notation v_i):
#
#   ξ η ζ                     image (a11, a22, a33)
#   --------                  ------------------------
#   N1: -,-,-     v1          (0,    0,    0)         tet vertex
#   N2: +,-,-     v2          (1,    0,    0)         tet vertex
#   N3: +,+,-     v3          (1/2,  1/2,  0)         edge midpoint
#   N4: -,+,-     v4          (0,    1,    0)         tet vertex
#   N5: -,-,+     v5          (0,    0,    1)         tet vertex
#   N6: +,-,+     v6          (1/2,  0,    1/2)       edge midpoint
#   N7: +,+,+     v7          (1/3,  1/3,  1/3)       face centroid
#   N8: -,+,+     v8          (0,    1/2,  1/2)       edge midpoint
#
# (the ξ η ζ ordering matches the trilinear shape functions in
# :func:`_trilinear_hex_8`).
_TET_TOL = 1e-6
_TET_VERTICES = np.array([
    [_TET_TOL, _TET_TOL, _TET_TOL],   # v1
    [1.0,      _TET_TOL, _TET_TOL],   # v2
    [0.5,      0.5,      _TET_TOL],   # v3
    [_TET_TOL, 1.0,      _TET_TOL],   # v4
    [_TET_TOL, _TET_TOL, 1.0],        # v5
    [0.5,      _TET_TOL, 0.5],        # v6
    [1.0/3.0,  1.0/3.0,  1.0/3.0],    # v7
    [_TET_TOL, 0.5,      0.5],        # v8
])


def _trilinear_hex_8(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Eight trilinear hex shape functions on ``[-1, 1]³``.

    Order: ``(ξ, η, ζ) ∈ {(-,-,-), (+,-,-), (+,+,-), (-,+,-),
    (-,-,+), (+,-,+), (+,+,+), (-,+,+)}`` (Nomura et al. 2019, Eq. 30).
    """
    N1 = (1 - xi) * (1 - eta) * (1 - zeta) / 8.0
    N2 = (1 + xi) * (1 - eta) * (1 - zeta) / 8.0
    N3 = (1 + xi) * (1 + eta) * (1 - zeta) / 8.0
    N4 = (1 - xi) * (1 + eta) * (1 - zeta) / 8.0
    N5 = (1 - xi) * (1 - eta) * (1 + zeta) / 8.0
    N6 = (1 + xi) * (1 - eta) * (1 + zeta) / 8.0
    N7 = (1 + xi) * (1 + eta) * (1 + zeta) / 8.0
    N8 = (1 - xi) * (1 + eta) * (1 + zeta) / 8.0
    return np.stack([N1, N2, N3, N4, N5, N6, N7, N8])


def box_to_tetrahedron(
    x1: float, x2: float, x3: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the 3-D design cube ``[-1, 1]³`` onto the orientation tetrahedron.

    Smooth, surjective map onto the closed tetrahedron
    ``{a11, a22, a33 ≥ 0, a11 + a22 + a33 ≤ 1}`` via the Nomura *et al.*
    2019 trilinear-hex isoparametric construction.  Four of the eight
    box corners go to the four tetrahedron vertices; the remaining four
    go to edge midpoints / a face centroid (see source for the exact
    table of image vertices).  This is the 3-D analogue of
    :func:`box_to_triangle` and serves the same role: it lets all six
    diagonal/off-diagonal design variables for ``a_2`` live on simple
    box bounds while still respecting the physical positivity and
    trace constraints.

    Parameters
    ----------
    x1, x2, x3 : scalar or array, in ``[-1, 1]``
        Box-bounded design variables.

    Returns
    -------
    a11, a22, a33 : same shape as inputs
    """
    N = _trilinear_hex_8(x1, x2, x3)
    a11 = np.sum(_TET_VERTICES[:, 0] * N)
    a22 = np.sum(_TET_VERTICES[:, 1] * N)
    a33 = np.sum(_TET_VERTICES[:, 2] * N)
    return a11, a22, a33


# ---------------------------------------------------------------------------
# Smoothed sign for off-diagonal control
# ---------------------------------------------------------------------------

def smooth_sgn(x: np.ndarray, beta: float = 10.0) -> np.ndarray:
    r"""Smooth sign function ``\\tanh(\\beta x) / \\tanh(\\beta)``.

    Maps ``[-1, 1] → [-1, 1]`` smoothly and monotonically, with
    sharpness controlled by ``beta``.  Used to drive off-diagonal
    signs in :func:`orientation_tensor_2d` and
    :func:`orientation_tensor_3d` while keeping the design variable
    on a box.

    ``beta = 10`` is a good default: ``smooth_sgn(±0.5) ≈ ±0.987``.
    Larger ``beta`` approaches a hard sign at the cost of stiffness in
    the optimisation gradients.
    """
    return np.tanh(beta * x) / np.tanh(beta)


# ---------------------------------------------------------------------------
# Orientation-tensor builders
# ---------------------------------------------------------------------------

def orientation_tensor_2d(
    x1: float, x2: float, x3: float, sgn_beta: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 2-D orientation tensor ``a_2`` from libertas design parameters.

    The construction:

    .. code-block:: text

        (a11, a22) = box_to_triangle(x1, x2)
        a12        = sqrt(max(a11 * a22, 0)) * smooth_sgn(x3)
        a_2        = [[a11, a12], [a12, a22]]

    The ``sqrt(a11 * a22)`` factor makes ``a_2`` rank-deficient at the
    triangle vertices (perfect alignment) and gives ``|a12|`` its
    largest value at ``a11 = a22`` (45°).  ``smooth_sgn`` selects the
    ``+45°`` / ``-45°`` branch.

    Parameters
    ----------
    x1, x2 : in ``[-1, 1]``
        Diagonal-alignment design variables (mapped to the triangle).
    x3 : in ``[-1, 1]``
        Off-diagonal sign control.
    sgn_beta : float
        Sharpness for :func:`smooth_sgn`.

    Returns
    -------
    a2 : (2, 2) array
    a11, a22 : scalars (also returned for SIMP-style "alignment
        magnitude" weights ``sqrt(a11**2 + a22**2)``).
    """
    a11, a22 = box_to_triangle(x1, x2)
    a12 = np.sqrt(np.maximum(a11 * a22, 0.0)) * smooth_sgn(x3, sgn_beta)
    a2 = np.array([[a11, a12], [a12, a22]])
    return a2, a11, a22


def orientation_tensor_3d(
    x1: float, x2: float, x3: float,
    x12: float, x13: float, x23: float,
    sgn_beta: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the 3-D orientation tensor ``a_2`` from 6 libertas design parameters.

    The construction (Nomura *et al.* 2019, §2.2.2 + Eq. 34):

    .. code-block:: text

        (a11, a22, a33) = box_to_tetrahedron(x1, x2, x3)
        a_ij            = sqrt(max(a_ii * a_jj, 0)) * smooth_sgn(x_ij)   (i ≠ j)
        a_2             = symmetric(a_11, a_22, a_33, a_12, a_13, a_23)

    The ``sqrt(a_ii * a_jj)`` factor enforces ``a_ij² = a_ii a_jj`` at
    the limit ``smooth_sgn → ±1`` — Nomura's second-tensor-invariant
    constraint, which together with the trace constraint embedded in
    :func:`box_to_tetrahedron` projects six box-bounded design
    variables onto the manifold of physically admissible
    rank-1-or-2 orientation tensors.

    Parameters
    ----------
    x1, x2, x3 : in ``[-1, 1]``
        Diagonal-alignment design variables (mapped to the tetrahedron).
    x12, x13, x23 : in ``[-1, 1]``
        Off-diagonal sign controls.
    sgn_beta : float
        Sharpness for :func:`smooth_sgn`.

    Returns
    -------
    a2 : (3, 3) array
    a11, a22, a33 : scalars (returned for SIMP-style "alignment
        magnitude" weights ``sqrt(a11**2 + a22**2 + a33**2)``).
    """
    a11, a22, a33 = box_to_tetrahedron(x1, x2, x3)
    a12 = np.sqrt(np.maximum(a11 * a22, 0.0)) * smooth_sgn(x12, sgn_beta)
    a13 = np.sqrt(np.maximum(a11 * a33, 0.0)) * smooth_sgn(x13, sgn_beta)
    a23 = np.sqrt(np.maximum(a22 * a33, 0.0)) * smooth_sgn(x23, sgn_beta)
    a2 = np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33],
    ])
    return a2, a11, a22, a33


# ---------------------------------------------------------------------------
# Principal direction extraction (visualization helper)
# ---------------------------------------------------------------------------

def principal_direction(
    a2: np.ndarray,
    scale_by_alignment: bool = True,
) -> np.ndarray:
    """Extract the principal fibre direction from ``a_2``.

    Returns the eigenvector of ``a_2`` associated with its **largest**
    eigenvalue.  This is the direction along which the orientation
    distribution is most concentrated:

    * Unidirectional state (``a_2 = p p^T``): the unit vector ``p``.
    * Isotropic state (``a_2 = (1/d) I``): all eigenvalues equal — the
      "principal" direction is degenerate; the routine returns whichever
      eigenvector ``eigh`` happens to pick.

    Supports both the unbatched ``(d, d)`` and batched ``(N, d, d)`` (or
    higher-rank) layouts; the eigendecomposition runs over the last two
    axes so for a ``(N, d, d)`` field the result has shape ``(N, d)``.

    Parameters
    ----------
    a2 : (..., d, d) array
        Symmetric orientation tensor(s).  ``d`` is 2 or 3 in practice.
    scale_by_alignment : bool, default True
        Multiply the unit eigenvector by its eigenvalue ``λ_max ∈ [0, 1]``.
        Fully isotropic → length ``1/d``; unidirectional → unit length.
        This is what most glyph visualisations (ParaView "Orient by
        Vector") want as input.

    Returns
    -------
    director : (..., d) array

    Notes
    -----
    Eigenvectors are defined only up to a sign — fibres are
    bidirectional, so this is physically meaningful.  ``eigh`` does
    not enforce any sign convention; adjacent nodes whose tensors
    differ slightly may yield directors that point in opposite
    directions.  Glyph visualisations are usually robust to this
    ambiguity, but if continuity matters (e.g. for streamline
    integration) you should run a sign-disambiguation pass after this.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> a2 = jnp.array([[1.0, 0.0], [0.0, 0.0]])     # aligned along axis 1
    >>> principal_direction(a2)
    Array([1., 0.], dtype=float32)
    >>> # Pad to 3-D for VTU vector output:
    >>> import numpy as onp
    >>> d2 = onp.asarray(principal_direction(a2_nodes))   # (N, 2)
    >>> d3 = onp.column_stack([d2, onp.zeros(len(d2))])   # (N, 3)
    """
    w, v = np.linalg.eigh(a2)
    # ``eigh`` returns eigenvalues sorted ascending and eigenvectors as
    # columns of ``v``: the leading (largest-eigenvalue) eigenvector is
    # the last column ``v[..., :, -1]``.
    director = v[..., -1]
    if scale_by_alignment:
        director = director * w[..., -1:]
    return director


# ---------------------------------------------------------------------------
# Closures: 4th-order orientation tensor a_4 from a_2
# ---------------------------------------------------------------------------

def quadratic_closure(a2: np.ndarray) -> np.ndarray:
    """Quadratic (Advani–Tucker) closure ``a_4 = a_2 ⊗ a_2``.

    Exact for *unidirectional* orientation states (when ``a_2`` is
    rank-1, ``a_4 = p p p p``) and a reasonable approximation for
    near-aligned distributions.  Substantially overestimates anisotropy
    for nearly-isotropic distributions — for which :func:`linear_closure`
    or :func:`hybrid_closure` is preferable.

    Dimension-agnostic: works for any ``(d, d)`` ``a_2``.
    """
    return np.einsum("ij,kl->ijkl", a2, a2)


# Linear-closure coefficients ``(α_1, α_2)`` such that
#   a_4 = α_1 (δδ + δδ + δδ) + α_2 (aδ × 6 sym permutations)
# is exact for the d-dimensional isotropic state ``a_2 = (1/d) I`` and
# satisfies the contraction identity ``a_4_ijkk = a_2_ij``.  Derivations:
# d=2 from Advani & Tucker 1987 §3.3; d=3 from the same paper Eq. 30.
_LINEAR_COEFFS = {
    2: (-1.0 / 24.0, 1.0 / 6.0),
    3: (-1.0 / 35.0, 1.0 / 7.0),
}

# Hybrid-closure mixing factor scale ``c_d`` such that
#   f = 1 - c_d · det(a_2) ∈ [0, 1]
# (=0 fully isotropic, =1 unidirectional).  ``c_d = d^d`` reaches 1
# when ``a_2 = (1/d) I``.
_HYBRID_DET_SCALE = {2: 4.0, 3: 27.0}


def linear_closure(a2: np.ndarray) -> np.ndarray:
    """Linear (Advani–Tucker) closure for 2-D or 3-D ``a_2``.

    Exact for the *isotropic* (random) orientation state ``(1/d) I``.
    Strongly inaccurate for unidirectional states — pair with
    :func:`hybrid_closure` for general use.

    .. code-block:: text

        a_4 = α_1 (δ_ij δ_kl + δ_ik δ_jl + δ_il δ_jk)
              + α_2 (a_ij δ_kl + a_kl δ_ij
                    + a_ik δ_jl + a_il δ_jk
                    + a_jk δ_il + a_jl δ_ik)

    with ``(α_1, α_2) = (-1/24, 1/6)`` in 2-D and ``(-1/35, 1/7)`` in 3-D.

    See Advani & Tucker (1987), J. Rheol. 31(8):751–784.
    """
    d = a2.shape[0]
    if d not in _LINEAR_COEFFS:
        raise ValueError(
            f"linear_closure: only dim 2 or 3 supported, got a_2 shape {a2.shape}"
        )
    c1, c2 = _LINEAR_COEFFS[d]
    delta = np.eye(d)
    iso = (
        np.einsum("ij,kl->ijkl", delta, delta)
        + np.einsum("ik,jl->ijkl", delta, delta)
        + np.einsum("il,jk->ijkl", delta, delta)
    )
    aniso = (
        np.einsum("ij,kl->ijkl", a2, delta)
        + np.einsum("kl,ij->ijkl", a2, delta)
        + np.einsum("ik,jl->ijkl", a2, delta)
        + np.einsum("il,jk->ijkl", a2, delta)
        + np.einsum("jk,il->ijkl", a2, delta)
        + np.einsum("jl,ik->ijkl", a2, delta)
    )
    return c1 * iso + c2 * aniso


def hybrid_closure(a2: np.ndarray) -> np.ndarray:
    """Hybrid (Advani–Tucker) closure: linear + quadratic blend.

    Combines the two limiting closures with a scalar measure ``f`` of
    distance from isotropy:

    .. code-block:: text

        f   = 1 - d^d · det(a_2)               (=0 isotropic, =1 aligned)
        a_4 = (1 - f) · a_4_linear + f · a_4_quadratic.

    More accurate than either closure on its own across the full range
    of orientation states.  ``d^d`` evaluates to ``4`` in 2-D and
    ``27`` in 3-D.

    Reference: Advani & Tucker (1990), J. Rheol. 34(3):367–386.
    """
    d = a2.shape[0]
    if d not in _HYBRID_DET_SCALE:
        raise ValueError(
            f"hybrid_closure: only dim 2 or 3 supported, got a_2 shape {a2.shape}"
        )
    f = 1.0 - _HYBRID_DET_SCALE[d] * np.linalg.det(a2)
    return (1.0 - f) * linear_closure(a2) + f * quadratic_closure(a2)


# ---------------------------------------------------------------------------
# Orthotropic stiffness builders from (a_2, a_4)
# ---------------------------------------------------------------------------

def orthotropic_stiffness(
    a2: np.ndarray,
    a4: np.ndarray,
    E1: float,
    E2: float,
    G12: float,
    nu12: float,
) -> np.ndarray:
    r"""Build the 2-D plane-stress in-plane stiffness ``C_ijkl(a_2, a_4)``.

    Advani–Tucker decomposition for a 2-D orthotropic lamina (4
    independent moduli):

    .. math::

        C_{ijkl} = B_1 a_{4,ijkl}
                 + B_2 (a_{ij}\\delta_{kl} + a_{kl}\\delta_{ij})
                 + B_3 (a_{ik}\\delta_{jl} + a_{il}\\delta_{jk}
                      + a_{jk}\\delta_{il} + a_{jl}\\delta_{ik})
                 + B_5 (\\delta_{ik}\\delta_{jl} + \\delta_{il}\\delta_{jk}),

    with ``B_n`` from the principal-axis plane-stress moduli:

    .. code-block:: text

        Q11 = m E1, Q12 = m ν21 E1, Q22 = m E2, Q66 = G12
        m   = 1 / (1 - ν12 ν21),     ν21 = ν12 E2 / E1
        B1  = Q11 + Q22 - 2 Q12 - 4 Q66
        B2  = Q12
        B3  = Q66 - Q22 / 2
        B5  = Q22 / 2

    (No ``B_4`` term: the 2-D plane-stress reduction folds the
    bulk-like ``δδ`` contribution into ``B_5`` via the missing
    out-of-plane direction.)

    Parameters
    ----------
    a2, a4 : (2,2) and (2,2,2,2) arrays
        Orientation tensors.
    E1, E2 : float
        Lamina principal Young's moduli (1 = fibre, 2 = transverse).
    G12 : float
        In-plane shear modulus.
    nu12 : float
        Major Poisson's ratio.

    Returns
    -------
    C : (2, 2, 2, 2) array
    """
    nu21 = nu12 * E2 / E1
    m = 1.0 / (1.0 - nu12 * nu21)
    Q11 = m * E1
    Q22 = m * E2
    Q12 = m * nu21 * E1
    Q66 = G12

    B1 = Q11 + Q22 - 2.0 * Q12 - 4.0 * Q66
    B2 = Q12
    B3 = Q66 - Q22 / 2.0
    B5 = Q22 / 2.0

    delta = np.eye(2)
    return (
        B1 * a4
        + B2 * (
            np.einsum("ij,kl->ijkl", a2, delta)
            + np.einsum("kl,ij->ijkl", a2, delta)
        )
        + B3 * (
            np.einsum("ik,jl->ijkl", a2, delta)
            + np.einsum("il,jk->ijkl", a2, delta)
            + np.einsum("jk,il->ijkl", a2, delta)
            + np.einsum("jl,ik->ijkl", a2, delta)
        )
        + B5 * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
    )


def orthotropic_stiffness_3d(
    a2: np.ndarray,
    a4: np.ndarray,
    E1: float,
    E2: float,
    G12: float,
    nu12: float,
    nu23: float,
) -> np.ndarray:
    r"""Build the 3-D in-plane stiffness ``C_ijkl(a_2, a_4)`` from (5 moduli).

    Full Advani–Tucker decomposition (Nomura *et al.* 2019, Eq. 14)
    onto basis tensors built from the orientation tensors:

    .. math::

        C_{ijkl} = B_1 a_{4,ijkl}
                 + B_2 (a_{ij}\\delta_{kl} + a_{kl}\\delta_{ij})
                 + B_3 (a_{ik}\\delta_{jl} + a_{il}\\delta_{jk}
                      + a_{jk}\\delta_{il} + a_{jl}\\delta_{ik})
                 + B_4 \\delta_{ij}\\delta_{kl}
                 + B_5 (\\delta_{ik}\\delta_{jl} + \\delta_{il}\\delta_{jk}).

    with the ``B_n`` invariants computed from the principal-axis
    transversely-isotropic moduli (axis 1 = fibre direction):

    .. code-block:: text

        ν21 = ν12 E2 / E1
        m   = 1 / [(1 + ν23)(1 - ν23 - 2 ν12 ν21)]
        C1111 = E1 (1 - ν23²) m
        C1122 = E1 ν21 (1 + ν23) m
        C2222 = E2 (1 - ν12 ν21) m
        C2233 = E2 (ν23 + ν12 ν21) m
        C1212 = G12

        B1 = C1111 + C2222 - 2 C1122 - 4 C1212
        B2 = C1122 - C2233
        B3 = C1212 + (C2233 - C2222) / 2
        B4 = C2233
        B5 = (C2222 - C2233) / 2

    The transverse shear modulus ``G_23 = (C2222 - C2233)/2 = E2 / (2(1+ν23))``
    is fixed by the transverse-isotropy assumption, so only 5 of the
    engineering constants are independent.

    Limits:

    * Perfect alignment (``a_2 = e_1 ⊗ e_1``, ``a_4 = e_1⁴``) recovers
      the lamina stiffness in its principal axes.
    * 3-D isotropic (``a_2 = (1/3) I``) gives the 3-D isotropic average.

    Parameters
    ----------
    a2, a4 : (3,3) and (3,3,3,3) arrays
        Orientation tensors.
    E1, E2 : float
        Axial (fibre) and transverse Young's moduli (E2 = E3 by
        transverse isotropy).
    G12 : float
        Axial shear modulus (G12 = G13).
    nu12 : float
        Major (axial) Poisson's ratio (ν12 = ν13).
    nu23 : float
        Transverse Poisson's ratio.

    Returns
    -------
    C : (3, 3, 3, 3) array
    """
    nu21 = nu12 * E2 / E1
    m = 1.0 / ((1.0 + nu23) * (1.0 - nu23 - 2.0 * nu12 * nu21))

    C1111 = E1 * (1.0 - nu23**2) * m
    C1122 = E1 * nu21 * (1.0 + nu23) * m
    C2222 = E2 * (1.0 - nu12 * nu21) * m
    C2233 = E2 * (nu23 + nu12 * nu21) * m
    C1212 = G12

    B1 = C1111 + C2222 - 2.0 * C1122 - 4.0 * C1212
    B2 = C1122 - C2233
    B3 = C1212 + (C2233 - C2222) / 2.0
    B4 = C2233
    B5 = (C2222 - C2233) / 2.0

    delta = np.eye(3)
    return (
        B1 * a4
        + B2 * (
            np.einsum("ij,kl->ijkl", a2, delta)
            + np.einsum("kl,ij->ijkl", a2, delta)
        )
        + B3 * (
            np.einsum("ik,jl->ijkl", a2, delta)
            + np.einsum("il,jk->ijkl", a2, delta)
            + np.einsum("jk,il->ijkl", a2, delta)
            + np.einsum("jl,ik->ijkl", a2, delta)
        )
        + B4 * np.einsum("ij,kl->ijkl", delta, delta)
        + B5 * (
            np.einsum("ik,jl->ijkl", delta, delta)
            + np.einsum("il,jk->ijkl", delta, delta)
        )
    )


__all__ = [
    "box_to_triangle",
    "box_to_tetrahedron",
    "smooth_sgn",
    "orientation_tensor_2d",
    "orientation_tensor_3d",
    "principal_direction",
    "quadratic_closure",
    "linear_closure",
    "hybrid_closure",
    "orthotropic_stiffness",
    "orthotropic_stiffness_3d",
]
