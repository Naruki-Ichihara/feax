---
sidebar_label: orientation
title: feax.mechanics.orientation
---

Fibre-orientation tensor API (2-D and 3-D).

Reusable building blocks for continuous fibre-orientation modelling and
optimisation, based on Advani–Tucker orientation tensors.

A fibre-orientation distribution is summarised by its second-order
orientation tensor

.. math::

    a_{`ij`} = \langle p_i p_j \rangle,

where ``p`` is the unit fibre direction and ``⟨·⟩`` averages over the
local fibre population.  ``a_2`` is symmetric and (for normalised
distributions) ``tr(a_2) = 1``.  Two limiting cases:

* ``a_2 = e_1 ⊗ e_1`` — perfect alignment along the 1-axis;
* ``a_2 = (1/d) I`` (with ``d = 2`` or ``3``) — fully isotropic.

For 4th-order constitutive laws (orthotropic stiffness, anisotropic
thermal expansion, etc.) we also need ``a_4 = ⟨p p p p⟩``.  ``a_4`` is
not determined by ``a_2`` alone and must be approximated through a
*closure* — see :func:``2, :func:``3,
:func:``4 (all dimension-aware).

Module contents
---------------

1. **Design parameterisations**

   * :func:``5 — libertas 2-D 8-node serendipity map
     ``[-1, 1]² → {`a11, a22 ≥ 0, a11+a22 ≤ 1`}``;
   * :func:``8 — Nomura *et al.* 2019 3-D 8-node
     trilinear-hex map ``[-1, 1]³ → {`a_ii ≥ 0, ∑a_ii ≤ 1`}``;
   * :func:``1 — smoothed sign for off-diagonal control;
   * :func:``2, :func:``3 —
     convenience builders that combine the above into a full ``a_2``.

2. **Closures** — :func:``2, :func:``3,
   :func:``4 for ``a_4`` from ``a_2``.

3. **Constitutive builders** — :func:``3
   (2-D plane-stress, 4 moduli) and
   :func:``4 (full 3-D
   transverse-isotropic, 5 moduli).  These produce the *ODF-averaged*
   stiffness over an orthotropic base material — the resulting
   macroscopic stiffness is **not** in general orthotropic (it is
   orthotropic only when the ODF is symmetric about the principal
   direction of ``a_2``; closure-free :func:``7 with
   ``y_2 != 0`` deliberately admits non-orthotropic macro states).

References
----------
* Advani, S. G. &amp; Tucker, C. L. (1987). The use of tensors to describe
  and predict fiber orientation in short fiber composites.
  *J. Rheol.* 31(8):751–784.
* Advani, S. G. &amp; Tucker, C. L. (1990). Closure approximations for
  three-dimensional structure tensors.  *J. Rheol.* 34(3):367–386.
* Nomura, T., Kawamoto, A., Kondoh, T., Dede, E. M., Lee, J., Song,
  Y., Kikuchi, N. (2019).  Inverse design of structure and fiber
  orientation by means of topology optimization with tensor field
  variables.  *Composites Part B* 176, 107187.

#### box\_to\_triangle

```python
def box_to_triangle(x1: float, x2: float) -> Tuple[np.ndarray, np.ndarray]
```

Map the 2-D design square ``[-1, 1]²`` onto the orientation triangle.

Smooth (C¹), nearly-bijective map onto
``{`a11, a22 ≥ 0, a11 + a22 ≤ 1`}`` via 8-node quadratic serendipity
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

#### box\_to\_tetrahedron

```python
def box_to_tetrahedron(x1: float, x2: float,
                       x3: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Map the 3-D design cube ``[-1, 1]³`` onto the orientation tetrahedron.

Smooth, surjective map onto the closed tetrahedron
``{`a11, a22, a33 ≥ 0, a11 + a22 + a33 ≤ 1`}`` via the Nomura *et al.*
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

#### smooth\_sgn

```python
def smooth_sgn(x: np.ndarray, beta: float = 10.0) -> np.ndarray
```

Smooth sign function ``\\tanh(\\beta x) / \\tanh(\\beta)``.

Maps ``[-1, 1] → [-1, 1]`` smoothly and monotonically, with
sharpness controlled by ``beta``.  Used to drive off-diagonal
signs in :func:`orientation_tensor_2d` and
:func:`orientation_tensor_3d` while keeping the design variable
on a box.

``beta = 10`` is a good default: ``smooth_sgn(±0.5) ≈ ±0.987``.
Larger ``beta`` approaches a hard sign at the cost of stiffness in
the optimisation gradients.

#### orientation\_tensor\_2d

```python
def orientation_tensor_2d(
        x1: float,
        x2: float,
        x3: float,
        sgn_beta: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Build the 2-D orientation tensor ``a_2`` from libertas design parameters.

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
- **x3** (*in ``[-1, 1]``*): Off-diagonal sign control.
- **sgn_beta** (*float*): Sharpness for :func:``0.


Returns
-------
- **a2** (*(2, 2) array*)


#### orientation\_tensor\_3d

```python
def orientation_tensor_3d(
    x1: float,
    x2: float,
    x3: float,
    x12: float,
    x13: float,
    x23: float,
    sgn_beta: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Build the 3-D orientation tensor ``a_2`` from 6 libertas design parameters.

The construction (Nomura *et al.* 2019, §2.2.2 + Eq. 34):

.. code-block:: text

    (a11, a22, a33) = box_to_tetrahedron(x1, x2, x3)
    a_ij            = sqrt(max(a_ii * a_jj, 0)) * smooth_sgn(x_ij)   (i ≠ j)
    a_2             = symmetric(a_11, a_22, a_33, a_12, a_13, a_23)

The ``sqrt(a_ii * a_jj)`` factor enforces ``a_ij² = a_ii a_jj`` at
the limit ``smooth_sgn → ±1`` — Nomura&#x27;s second-tensor-invariant
constraint, which together with the trace constraint embedded in
:func:`box_to_tetrahedron` projects six box-bounded design
variables onto the manifold of physically admissible
rank-1-or-2 orientation tensors.

Parameters
----------
- **sgn_beta** (*float*): Sharpness for :func:``3.


Returns
-------
- **a2** (*(3, 3) array*)


#### principal\_direction

```python
def principal_direction(a2: np.ndarray,
                        scale_by_alignment: bool = True) -> np.ndarray
```

Extract the principal fibre direction from ``a_2``.

Returns the eigenvector of ``a_2`` associated with its **largest**
eigenvalue.  This is the direction along which the orientation
distribution is most concentrated:

* Unidirectional state (``a_2 = p p^T``): the unit vector ``p``.
* Isotropic state (``a_2 = (1/d) I``): all eigenvalues equal — the
  &quot;principal&quot; direction is degenerate; the routine returns whichever
  eigenvector ``eigh`` happens to pick.

Supports both the unbatched ``(d, d)`` and batched ``(N, d, d)`` (or
higher-rank) layouts; the eigendecomposition runs over the last two
axes so for a ``(N, d, d)`` field the result has shape ``(N, d)``.

Parameters
----------
- **a2** (*(..., d, d) array*): Symmetric orientation tensor(s).  ``d`` is 2 or 3 in practice.
- **scale_by_alignment** (*bool, default True*): Multiply the unit eigenvector by its eigenvalue ``λ_max ∈ [0, 1]``. Fully isotropic → length ``1/d``; unidirectional → unit length. This is what most glyph visualisations (ParaView &quot;Orient by Vector&quot;) want as input.


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
```python
>>> import jax.numpy as jnp
>>> a2 = jnp.array([[1.0, 0.0], [0.0, 0.0]])     # aligned along axis 1
>>> principal_direction(a2)
```
Array([1., 0.], dtype=float32)
```python
>>> # Pad to 3-D for VTU vector output:
>>> import numpy as onp
>>> d2 = onp.asarray(principal_direction(a2_nodes))   # (N, 2)
>>> d3 = onp.column_stack([d2, onp.zeros(len(d2))])   # (N, 3)
```

#### quadratic\_closure

```python
def quadratic_closure(a2: np.ndarray) -> np.ndarray
```

Quadratic (Advani–Tucker) closure ``a_4 = a_2 ⊗ a_2``.

Exact for *unidirectional* orientation states (when ``a_2`` is
rank-1, ``a_4 = p p p p``) and a reasonable approximation for
near-aligned distributions.  Substantially overestimates anisotropy
for nearly-isotropic distributions — for which :func:`linear_closure`
or :func:`hybrid_closure` is preferable.

Dimension-agnostic: works for any ``(d, d)`` ``a_2``.

#### linear\_closure

```python
def linear_closure(a2: np.ndarray) -> np.ndarray
```

Linear (Advani–Tucker) closure for 2-D or 3-D ``a_2``.

Exact for the *isotropic* (random) orientation state ``(1/d) I``.
Strongly inaccurate for unidirectional states — pair with
:func:`hybrid_closure` for general use.

.. code-block:: text

    a_4 = α_1 (δ_ij δ_kl + δ_ik δ_jl + δ_il δ_jk)
          + α_2 (a_ij δ_kl + a_kl δ_ij
                + a_ik δ_jl + a_il δ_jk
                + a_jk δ_il + a_jl δ_ik)

with ``(α_1, α_2) = (-1/24, 1/6)`` in 2-D and ``(-1/35, 1/7)`` in 3-D.

See Advani &amp; Tucker (1987), J. Rheol. 31(8):751–784.

#### build\_a4\_2d

```python
def build_a4_2d(a2: np.ndarray,
                y1: float,
                y2: float,
                eps: float = 1e-12) -> np.ndarray
```

Closure-free 2-D ``a_4`` parameterised by two extra DOFs.

Given a (possibly sub-normalised) 2-D second-order orientation
tensor ``a_2`` with trace :math:`\tau = a_{`11`} + a_{`22`} \in [0, 1]`
and two scalar design parameters :math:`(y_1, y_2)` with
:math:`y_1^2 + y_2^2 \le 1`, build the fully-symmetric fourth-order
tensor ``a_4`` that

* is *admissible*, i.e. representable as :math:`\tau \cdot
  \langle p_i p_j p_k p_l \rangle` for some probability
  distribution on :math:``0;
* satisfies the contraction identity :math:``1
  exactly (which neither :func:``2 nor
  :func:``3 do for :math:``4);
* has full index symmetry :math:``5.

Construction (Schur reflection coefficient).  Let :math:``6 be the normalised orientation tensor (trace
1) and define its trigonometric moments

.. math::

    \nu_1 \;=\; \hat a_{`11`} - \hat a_{`22`} + 2 i \hat a_{`12`},
    \qquad
    \rho_2 \;=\; y_1 + i y_2.

The 4th-order trigonometric moment of the normalised distribution
is then taken to be

.. math::

    \nu_2 \;=\; \nu_1^2 + \rho_2 \,(1 - |\nu_1|^2),

which sweeps the *entire* admissible region for the 4th moment as
:math:``7 ranges over the closed unit disk (Toeplitz-PSD
parameterisation of the trigonometric moment problem).  The five
independent components of :math:``8 are then

.. math::

    \hat a_{`1111`}, \hat a_{`2222`} &amp;= (3 \pm 4 \hat c_2 + \hat c_4)/8,\\
    \hat a_{`1122`} &amp;= (1 - \hat c_4)/8,\\
    \hat a_{`1112`}, \hat a_{`1222`} &amp;= (2 \hat s_2 \pm \hat s_4)/8,

with :math:``9 and likewise for :math:``0.  Finally
:math:``1 so that ``a_4 → 0``
as ``τ → 0`` and the contraction :math:``1 is
preserved at any :math:``7.

Limiting cases:

* :math:``8 — the *maximum-determinant* (Bingham-like)
  baseline: ``a_4`` is the contraction-consistent 4th moment of the
  &quot;most spread-out&quot; admissible distribution for the given ``a_2``.
  This is the natural drop-in replacement for
  :func:``2 when one wants the contraction
  identity satisfied at sub-trace-1 ``a_2``.
* :math:``6 — singular limit: the inferred distribution
  collapses to two Dirac peaks on :math:``0 (a
  cross-fibre lattice).  Approach this edge with care: the
  stiffness remains finite but the inferred distribution is
  degenerate.
* :math:``8 (rank 1) —
  :math:``9, so :math:``7 drops out and ``a_4 = τ ·
  n^{`\otimes 4`}`` regardless of ``(y_1, y_2)``.

Use this in place of ``a_4 = quadratic_closure(a_2)`` when the
downstream model can express more than purely aligned states
(e.g. cross-ply or near-isotropic lattices).

Parameters
----------
- **a2** (*(2, 2) array*): Symmetric 2-D second-order orientation tensor with :math:`\tau = a_{`11`} + a_{`22`} \in [0, 1]`7, :math:`\tau = a_{`11`} + a_{`22`} \in [0, 1]`8, :math:`\tau = a_{`11`} + a_{`22`} \in [0, 1]`9.
- **eps** (*float*): Numerical floor on the trace, used as ``τ + eps`` in the normalisation to keep the function well-defined at fully empty elements (``τ → 0``).  At ``τ = 0`` the output is ``a_4 = 0``.


Returns
-------
- **a4** (*(2, 2, 2, 2) array*): Fully-symmetric, contraction-consistent fourth-order orientation tensor.


#### hybrid\_closure

```python
def hybrid_closure(a2: np.ndarray) -> np.ndarray
```

Hybrid (Advani–Tucker) closure: linear + quadratic blend.

Combines the two limiting closures with a scalar measure ``f`` of
distance from isotropy:

.. code-block:: text

    f   = 1 - d^d · det(a_2)               (=0 isotropic, =1 aligned)
    a_4 = (1 - f) · a_4_linear + f · a_4_quadratic.

More accurate than either closure on its own across the full range
of orientation states.  ``d^d`` evaluates to ``4`` in 2-D and
``27`` in 3-D.

Reference: Advani &amp; Tucker (1990), J. Rheol. 34(3):367–386.

#### orientation\_averaged\_stiffness

```python
def orientation_averaged_stiffness(a2: np.ndarray, a4: np.ndarray, E1: float,
                                   E2: float, G12: float,
                                   nu12: float) -> np.ndarray
```

Build the 2-D plane-stress in-plane stiffness ``C_ijkl(a_2, a_4)``.

Advani–Tucker orientation-averaged decomposition: ``C`` is the
expectation of the orthotropic lamina stiffness under an ODF on
:math:`\mathbb{`R`}P^1`, expressed in closed form via the second-
and fourth-order orientation tensors ``(a_2, a_4)``.  The **base
material** is orthotropic (4 in-plane moduli below); the resulting
**macro stiffness is not in general orthotropic** — it is
orthotropic only when the ODF is symmetric about the principal
direction of ``a_2``.  With the closure-free :func:`build_a4_2d`
parameterisation, ``y_2 != 0`` (in the eigenframe of ``a_2``)
produces ``a_{`4,1112`}, a_{`4,1222`} != 0``, which propagates into
Voigt ``D_{`13`}, D_{`23`} != 0`` and yields a fully (monoclinic /
general) anisotropic plane-stress stiffness with no orthotropic
principal frame.

Form (4 independent moduli):

.. math::

    C_{`ijkl`} = B_1 a_{`4,ijkl`}
             + B_2 (a_{`ij`}\\delta_{`kl`} + a_{`kl`}\\delta_{`ij`})
             + B_3 (a_{`ik`}\\delta_{`jl`} + a_{`il`}\\delta_{`jk`}
                  + a_{`jk`}\\delta_{`il`} + a_{`jl`}\\delta_{`ik`})
             + B_5 (\\delta_{`ik`}\\delta_{`jl`} + \\delta_{`il`}\\delta_{`jk`}),

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
- **G12** (*float*): In-plane shear modulus.
- **nu12** (*float*): Major Poisson&#x27;s ratio.


Returns
-------
C : (2, 2, 2, 2) array

#### orientation\_averaged\_stiffness\_3d

```python
def orientation_averaged_stiffness_3d(a2: np.ndarray, a4: np.ndarray,
                                      E1: float, E2: float, G12: float,
                                      nu12: float, nu23: float) -> np.ndarray
```

Build the 3-D stiffness ``C_ijkl(a_2, a_4)`` from a transversely-
isotropic base material averaged over an ODF on the unit sphere.

Full Advani–Tucker orientation-averaged decomposition (Nomura
*et al.* 2019, Eq. 14) onto basis tensors built from the
orientation tensors.  As with the 2-D variant, the **base
material** is transversely isotropic, but the **macro stiffness**
is in general anisotropic and matches a transversely-isotropic
symmetry class only when the ODF respects that symmetry.

.. math::

    C_{`ijkl`} = B_1 a_{`4,ijkl`}
             + B_2 (a_{`ij`}\\delta_{`kl`} + a_{`kl`}\\delta_{`ij`})
             + B_3 (a_{`ik`}\\delta_{`jl`} + a_{`il`}\\delta_{`jk`}
                  + a_{`jk`}\\delta_{`il`} + a_{`jl`}\\delta_{`ik`})
             + B_4 \\delta_{`ij`}\\delta_{`kl`}
             + B_5 (\\delta_{`ik`}\\delta_{`jl`} + \\delta_{`il`}\\delta_{`jk`}).

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
- **G12** (*float*): Axial shear modulus (G12 = G13).
- **nu12** (*float*): Major (axial) Poisson&#x27;s ratio (ν12 = ν13).
- **nu23** (*float*): Transverse Poisson&#x27;s ratio.


Returns
-------
C : (3, 3, 3, 3) array

