---
sidebar_label: xfem_crack
title: feax.experimental.xfem_crack
---

LEFM crack growth with XFEM on a fixed mesh: traced geometry, tip enrichment
and energy release rates by autodiff — experimental.

The crack is a polyline.  Everything downstream of it is a fixed-shape JAX
function, so one compilation serves every crack configuration:

.. code-block:: text

    polyline (M_max, 2), n_pts ──► phi, psi (nodal level sets)
        ──► per-cell class (uncut / Heaviside-cut / tip), nodal enrichment
            masks, interface-conforming sub-cell quadrature  (fixed nq_max)
        ──► XFEM solve  ──► strain energy  Π
    J-vector = −∂Π/∂a of a *virtual crack extension*: the nodes around the
    tip move with it (smooth bump q), so the derivative is the domain-form
    J integral of the discrete solution                        (jax.grad)

Enrichment (Moës–Dolbow–Belytschko node sets, shifted basis):

* **Heaviside** ``sign(phi(x)) − sign(phi_j)`` on the nodes of the cells the
  crack crosses completely (variable 1, 2 DOFs per node),
* **branch functions** ``F_k(r, θ) − F_k(r_j, θ_j)``,
  ``F = √r {`sin θ/2, cos θ/2, sin θ/2 sin θ, cos θ/2 sin θ`}``, on the nodes
  of the cell containing the tip (variable 2, 8 DOFs per node);
  ``r, θ`` come from the two level sets (``θ = atan2(phi, psi)``), their
  gradients from ``jax.jacfwd``.

Both enriched spaces live on a static *band* of candidate nodes (plus one
phantom node each, ``with_jxw=False``).  Which band nodes are actually
enriched is a traced 0/1 mask: inactive enriched DOFs are held by a small
quadratic penalty inside the element energy, so no Dirichlet row ever
changes.  The sub-cell quadrature is padded to a fixed ``nq_max`` (the tip
fan: 6 triangles × 6 points) and the three candidate rules (uncut / cut /
tip fan) are computed for every cell and selected by its class, so the
geometry is one ``vmap`` over cells.

Scope: 2D ``TRI3`` meshes, linear elasticity given as an energy density
``W(eps, x)``, displacement-controlled loading (``Π`` = strain energy, so
``J = −∂Π/∂a`` at fixed grips).  ``K_I, K_II`` follow from the J-vector
(``J_t = (K_I²+K_II²)/E&#x27;``, ``J_n = −2K_I K_II/E&#x27;``) and the growth direction
from the maximum circumferential stress criterion.

Why not differentiate with respect to the tip position directly?  The
discrete energy is not smooth at the cell scale (the enriched node sets and
the sub-cell quadrature change from cell to cell), so ``∂Π/∂x_tip`` shows a
sawtooth of ±50 % within a cell even though the one-cell secant of ``Π`` is
accurate to a few percent.  Moving the whole neighbourhood with the tip
removes that pollution (it is the discrete form of the J domain integral).

Usage
-----

.. code-block:: python

    from feax.experimental.xfem_crack import CrackSpace, crack_problem, plane_strain_energy

    space = CrackSpace(mesh, max_points=32)              # TRI3 mesh
    problem = crack_problem(space, plane_strain_energy(E, nu))
    bc = space.create_bc(problem, fe.DirichletBCConfig([...]))
    ts = fe.TracedStructure.from_problem(problem)
    solver = fe.create_solver(problem, bc, linear=True, traced_structure=ts)
    crack = space.polyline([(-0.05, 0.0), (0.3, 0.0)])  # (poly, n_pts)

    J = space.j_vector_fn(problem, solver, bc, ts, radius=4 * h)   # jitted
    J_t, J_n = J(poly, n_pts)                                     # energy release rate + J2
    theta, K_I, K_II = space.growth_direction(J, poly, n_pts, E_prime)
    poly, n_pts = space.grow(poly, n_pts, theta, da)

#### DEFAULT\_TIP\_RINGS

radial grading of the tip fan toward the tip

#### nq\_max\_for

```python
def nq_max_for(rings) -> int
```

Quadrature slots per cell: the tip fan (6 fan triangles, each split into
``1 + 2 (m-1)`` sub-triangles by ``m`` radial rings, 6-point rule) is the
largest of the three candidate rules.

#### crack\_level\_sets

```python
def crack_level_sets(poly, n_pts, X)
```

Nodal level sets of a polyline crack.

``poly`` (M, 2) holds ``n_pts`` valid points (the rest is padding); ``X``
(N, 2) are the node coordinates.  Returns ``phi`` (signed distance to the
polyline, positive on the left of the crack direction; the first and last
segments are extended to infinity so ``phi = 0`` continues straight ahead
of the tip and behind the mouth) and ``psi`` (arc length of the nearest
point minus the crack length: negative on the crack, positive ahead of the
tip, where it is the distance along the last segment&#x27;s direction).

#### branch\_functions

```python
def branch_functions(phi, psi)
```

``F_k(r, θ)`` of a crack tip, ``r = sqrt(phi² + psi²)``, ``θ = atan2(phi, psi)``.

#### plane\_strain\_energy

```python
def plane_strain_energy(E: float, nu: float) -> Callable
```

Isotropic linear elastic energy density ``W(eps, x)`` (plane strain).

## CrackSpace Objects

```python
class CrackSpace()
```

XFEM crack space on a TRI3 mesh with traced (fixed-shape) geometry.

Parameters
----------
- **mesh** (*Mesh*): ``TRI3`` background mesh.
- **band** (*bool array (num_nodes,), optional*): Candidate nodes for enrichment (default: all nodes).  Enriched DOFs exist on these nodes only (``2 + 8`` per band node, plus 2 phantom nodes); which of them are active is a traced mask.
- **max_points** (*int*): Capacity of the crack polyline (fixed shape).
- **threshold** (*float*): Relative clamp for near-zero nodal ``phi`` (sliver sub-cells).
- **penalty** (*float*): Stiffness of the quadratic penalty holding inactive enriched DOFs.
- **tip_rings** (*tuple of float*): Radial grading ``0 = r_0 &lt; r_1 &lt; ... &lt; r_m = 1`` of the tip-cell fan quadrature (sub-triangles concentrate toward the tip, where the energy density behaves like 1/r).  ``nq_max`` follows from it.


#### polyline

```python
def polyline(points: Sequence[Tuple[float, float]])
```

``(poly, n_pts)`` padded to ``max_points``.

#### grow

```python
@staticmethod
def grow(poly, n_pts, theta, da)
```

Append a segment of length ``da`` at angle ``theta`` from the tip.

#### geometry

```python
def geometry(poly, n_pts, X=None)
```

Traced per-cell arrays for ``TracedParams`` (fixed shapes).  ``X``
(num_nodes, 2) overrides the node coordinates (used by the virtual
crack extension, which moves the nodes around the tip with it).

#### classes

```python
def classes(poly, n_pts)
```

Per-cell class (0 uncut, 1 Heaviside, 2 tip) as a numpy array.

#### visualization

```python
def visualization(problem: Problem,
                  sol,
                  poly,
                  n_pts,
                  nudge: float = 1e-6) -> dict
```

Sub-cell visualization mesh with the discontinuity resolved.

Every sub-triangle of the interface-conforming sub-triangulation gets
its own three vertices, so the displacement jump across the crack and
the kink at the tip appear in the output instead of being smeared by
a nodal average.  Fields are evaluated at the vertices nudged by
``nudge`` toward the sub-triangle centroid, which puts a vertex lying
on the crack on the correct side.

Returns a dict (host numpy) with ``points`` (P, 2), ``cells`` (T, 3),
``cell_of`` (T,) parent cell index, ``class`` (T,), ``u`` (P, 2),
``phi`` (P,), ``psi`` (P,) and ``eps`` (T, 2, 2) (strain at the
sub-triangle centroid).  Padding sub-triangles are dropped.  Use it
with :func:``8 on ``Mesh(points, cells, &quot;TRI3&quot;)``.

#### polyline\_mesh

```python
@staticmethod
def polyline_mesh(poly, n_pts) -> Mesh
```

The crack polyline as a ``LINE2`` mesh for :func:`feax.utils.save_sol`.

#### nodal\_displacement

```python
def nodal_displacement(problem: Problem, sol)
```

Total nodal displacement (shifted enrichments vanish at the nodes).

#### strain\_energy

```python
def strain_energy(problem: Problem, sol, tp: TracedParams)
```

``Π`` = Σ_cells element energy (strain energy + inactive-DOF penalty).

#### quad\_fields

```python
def quad_fields(problem: Problem, sol, tp: TracedParams)
```

``(x_q, u_q, eps_q)`` at the sub-cell quadrature points (num_cells, nq_max, ...).

#### j\_vector\_fn

```python
def j_vector_fn(problem: Problem,
                solver: Callable,
                bc: DirichletBC,
                ts,
                radius: float,
                inner: float = 0.5) -> Callable
```

Jitted ``J(poly, n_pts, radius=radius) -&gt; (J_t, J_n)``: the J-vector
(configurational force on the tip) by **virtual crack extension with
autodiff**.  ``radius`` may be overridden per call (traced), e.g. to
shrink the domain when the tip approaches a boundary.

The nodes within ``radius`` of the tip are moved with the tip by the
smooth bump ``q`` (``q = 1`` for ``r &lt; inner * radius``, smoothstep to 0
at ``radius``) and the strain energy is differentiated with respect to
the extension length at zero:  ``J_e = −∂Π/∂a``.  Because the mesh and
the sub-cell geometry around the tip translate rigidly, this is the
domain-integral J of the discrete solution and is free of the cell-
scale sawtooth that the pointwise derivative of the discrete energy
with respect to the tip position shows.  ``J_t`` is along the crack
direction at the tip (the energy release rate for straight growth),
``J_n`` along the left normal.  ``radius`` must keep ``q = 0`` on the
outer boundary.  Displacement-controlled loading is assumed.

#### sif\_from\_j

```python
@staticmethod
def sif_from_j(J_t, J_n, E_prime)
```

``(K_I, K_II)`` from the J-vector of a straight crack tip:
``J_t = (K_I² + K_II²)/E&#x27;``, ``J_n = −2 K_I K_II/E&#x27;`` (K_I ≥ 0 branch).

#### hoop\_stress\_angle

```python
@staticmethod
def hoop_stress_angle(KI, KII)
```

Maximum circumferential stress kink angle (relative to the crack
direction, positive toward the left normal).

#### hoop\_stress\_intensity

```python
@staticmethod
def hoop_stress_intensity(KI, KII, theta)
```

``K_θθ(θ) = cos(θ/2) [K_I cos²(θ/2) − 1.5 K_II sin θ]``: the mode-I
intensity of the kinked crack; growth when it reaches ``K_Ic``.

#### growth\_direction

```python
def growth_direction(J, poly, n_pts, E_prime, **j_kwargs)
```

``(theta_abs, K_I, K_II)``: absolute growth angle from the SIFs of
the J-vector and the maximum hoop stress criterion.

## CrackProblem Objects

```python
@jax.tree_util.register_pytree_node_class

@dataclass
class CrackProblem(Problem)
```

Three-variable XFEM crack problem; build with :func:`crack_problem`.
``additional_info = (space, W)``, ``W(eps, x)`` the energy density.

