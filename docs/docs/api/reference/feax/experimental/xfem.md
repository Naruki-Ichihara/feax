---
sidebar_label: xfem
title: feax.experimental.xfem
---

XFEM (eXtended FEM) on a fixed mesh — experimental.

An interface described by a nodal level set ``phi`` (zero iso-line = the
interface) is embedded in a mesh that does not conform to it.  The cells the
interface crosses (&quot;cut&quot; cells) get *enriched*: the nodes of those cells carry
extra DOFs ``a_j`` and the displacement is

.. code-block:: text

    u(x) = Σ_j N_j(x) u_j  +  Σ_{`j enriched`} N_j(x) ψ_j(x) a_j

with an enrichment function ``ψ_j`` chosen for the physics of the interface
(:class:`ShiftedHeaviside` for a displacement jump, :class:`ShiftedAbs` /
:class:`ModifiedAbs` for a strain jump, ...).  Cut cells are integrated with a
sub-triangulation that follows the interface, so no quadrature point straddles
the discontinuity.

Building blocks
---------------

* :class:`Enrichment` and the defaults :class:`ShiftedHeaviside`,
  :class:``1, :class:`ShiftedAbs`, :class:`ModifiedAbs` — pluggable
  ``ψ_j(x)``, ``∇ψ_j(x)``.
* :class:``8 — host-side preprocessing: clamps ``phi``, finds the cut
  cells and the enriched nodes, builds the *enrichment mesh* (enriched nodes +
  one phantom node, so the enriched DOFs cost exactly ``vec × n_enriched``
  DOFs, not ``vec × n_nodes``) and the padded sub-cell quadrature.
* :class:``5 — a two-variable :class:``6 (variable 0:
  standard displacement; variable 1: enriched DOFs on a DOF-only space,
  ``with_jxw=False``) whose universal kernel integrates a user constitutive
  law ``sigma = material(u_grad, phi, x, *extra)`` on the cut quadrature.

The standard feax machinery (Dirichlet BCs, direct / Krylov solvers,
``TracedStructure``, autodiff tangents and adjoints) applies unchanged.

Usage
-----

.. code-block:: python

    from feax.experimental.xfem import XFEMSpace, ShiftedAbs, xfem_problem

    phi = mesh.points[:, 0] - 0.73                    # interface at x = 0.73
    space = XFEMSpace(mesh, phi, ShiftedAbs(), vec=2, ele_type=&quot;QUAD4&quot;)

    def material(u_grad, phi, x):                     # two-phase, nu = 0
        E = np.where(phi &lt; 0, E_left, E_right)
        eps = 0.5 * (u_grad + u_grad.T)
        return E * eps

    problem = xfem_problem(space, material)
    bc = space.create_bc(problem, fe.DirichletBCConfig([...]))   # adds the phantom rows
    solver = fe.create_solver(problem, bc, linear=True)
    sol = solver(space.traced_params(), fe.zero_like_initial_guess(problem, bc))
    u_nodes = space.nodal_displacement(problem, sol)
    x_q, u_q, phi_q = space.quad_fields(problem, sol)  # both sides of the interface

Scope of this first version: 2D (``TRI3``, ``QUAD4``), one level set, one
enrichment per node (``Enrichment.n_funcs == 1``; the DOF layout already
allows more, e.g. crack-tip branch functions), bulk terms only (an interface
cohesive law is the natural next addition: the cut segments are available from
the same preprocessing).

## Enrichment Objects

```python
class Enrichment()
```

Enrichment function ``ψ_j(x)`` of the partition-of-unity ansatz.

Subclasses implement :meth:`__call__` returning, for the ``nq`` quadrature
points of one cell and its ``nn`` nodes, the value ``psi[q, a, f]`` and the
physical gradient ``gpsi[q, a, f, :]`` of the ``f``-th enrichment function
attached to node ``a`` (``f &lt; n_funcs``).  :meth:``7 gives the value
``ψ_j(x_j)`` at the node itself (zero for *shifted* enrichments), used to
recover nodal displacements.

Parameters passed to ``__call__``:

* ``phi_q``  (nq,)        level set at the quadrature points
* ``gphi_q`` (nq, dim)    its physical gradient
* ``phi_n``  (nn,)        nodal level-set values of the cell
* ``N``      (nq, nn)     shape functions
* ``dNdx``   (nq, nn, dim) their physical gradients

#### at\_nodes

```python
def at_nodes(phi_n)
```

``ψ_j(x_j)`` for an array of nodal level-set values (zero if shifted).

## ShiftedHeaviside Objects

```python
class ShiftedHeaviside(Enrichment)
```

Strong discontinuity (displacement jump), shifted so ``ψ_j(x_j) = 0``:

``ψ_j(x) = sign(phi(x)) − sign(phi_j)``.  The jump across the interface is
``2 a_j``; nodal displacements stay ``u_j``; blending cells are enrichment-
free automatically.  This is the enrichment of NLXP3D&#x27;s CUT elements.

## Heaviside Objects

```python
class Heaviside(Enrichment)
```

Strong discontinuity, classical unshifted form ``ψ_j(x) = sign(phi(x))``.

Nodal displacements are ``u_j + sign(phi_j) a_j`` (see :meth:`at_nodes`).

## ShiftedAbs Objects

```python
class ShiftedAbs(Enrichment)
```

Weak discontinuity (strain jump, continuous displacement), shifted:

``ψ_j(x) = |phi(x)| − |phi_j|``,  ``∇ψ_j = sign(phi) ∇phi``.

Note the shift only zeroes ``ψ_j`` *at* node j: on a blending cell (one
that holds enriched nodes but is not cut) ``|phi|`` is linear, so ``ψ_j``
is linear there and the enriched ansatz carries spurious quadratic terms.
This is the classic blending-element error of the abs enrichment: a
kinked piecewise-linear field is reproduced only approximately (the
error vanishes with refinement).  Prefer :class:``0, which is
blending-free and exact for such fields.

## ModifiedAbs Objects

```python
class ModifiedAbs(Enrichment)
```

Weak discontinuity, the blending-free ridge function of Moës et al. (2003):

``ψ(x) = Σ_i N_i(x) |phi_i| − |Σ_i N_i(x) phi_i|`` (the same function for
every node of the cell).  It vanishes identically on cells the interface
does not cross, so blending cells need no special treatment.

#### bilinear\_tsl

```python
def bilinear_tsl(delta, delta_max, *, K, sigma_c, Gamma, smooth=0.0)
```

Bilinear traction-separation potential with irreversibility (NLXP3D /
Camanho style), as a function of the effective opening ``delta``.

.. code-block:: text

    t(δ) = K δ                               δ &lt; δ0 = σc / K      (elastic)
    t(δ) = σc (δf − δ) / (δf − δ0)           δ0 ≤ δ &lt; δf          (softening)
    t(δ) = 0                                 δ ≥ δf = 2Γ / σc     (failed)

``φ = ∫ t dδ`` is returned; unloading from ``delta_max`` is a secant to
the origin (``φ = φ(δmax)/δmax² · δ²``).  ``K`` is the initial (penalty)
stiffness, ``sigma_c`` the strength, ``Gamma`` the fracture energy;
``δf &gt; δ0`` is required (``2ΓK &gt; σc²``).  ``smooth &gt; 0`` blends the kinks
of the law (at ``δ0``, ``δf`` and the loading/unloading switch) over the
absolute band ``± smooth δ0`` (continuous tangent); the default is the
exact piecewise law.  For robust load stepping prefer the secant mode of
:class:``6 (``secant=True``) over smoothing.

#### exponential\_tsl

```python
def exponential_tsl(delta, delta_max, *, sigma_c, Gamma, smooth=0.0)
```

Xu–Needleman exponential potential ``φ = Γ[1 − (1 + δ/δc) e^{`−δ/δc`}]``,
``δc = Γ/(e σc)``, with secant unloading from ``delta_max`` (the loading
law of :func:`feax.mechanics.cohesive.exponential_potential`; the
loading/unloading switch is blended over ``± smooth δc``, see
:func:`bilinear_tsl`).

## CohesiveZone Objects

```python
class CohesiveZone()
```

Cohesive zone model integrated on the cut chord of the enriched cells.

The displacement jump ``[[u]] = u⁺ − u⁻`` across the interface (``u±`` the
limits from ``phi ≷ 0``) is split into the normal opening ``δn = [[u]]·n``
(``n = ∇phi/|∇phi|``, pointing to ``phi &gt; 0``) and the sliding ``δt``, and
the effective opening ``δ = sqrt(⟨δn⟩₊² + β² δt²)`` drives the potential
``tsl(δ, δmax, **params)`` (see :func:``8,
:func:``9, or any ``feax.mechanics.cohesive`` potential).
``contact_penalty`` adds ``½ kp ⟨−δn⟩₊²`` against interpenetration.

The interface energy is added to the element kernel and its tangent comes
from autodiff, so the tangent stays symmetric.  Irreversibility needs the
history ``delta_max`` (max effective opening so far, per interface Gauss
point): pass it through :meth:``8 and update it
after each converged load step with :meth:``9.

Parameters
----------
- **tsl** (*callable  ``tsl(delta, delta_max, **params) -&gt; energy per unit length``*)
- **beta** (*float    mode-mixity weight of the sliding (default 1.0)*)
- **contact_penalty** (*float   penalty stiffness for ``δn &lt; 0`` (default 0: none)*)
- **n_gauss** (*int   Gauss points per chord (default 2)*)
- **on** (*str        ``&quot;interface&quot;`` (default, the ``psi &lt; 0`` cells) or ``&quot;cut&quot;``*): (every sub-divided cell, i.e. including the tip cell)
- **viscosity** (*float   viscous regularisation (see :meth:``0)*)
- **secant** (*bool   use the secant potential ``k_sec(δmax) δ²`` with the damage*): frozen during a solve (explicit damage update); the             history is advanced between solves and the solve repeated             until it is consistent (:meth:``3).             Robust where the implicit loading/unloading switch makes             Newton stall (many chords at ``δ = δmax`` at a step start).


#### secant\_potential

```python
def secant_potential(delta, delta_max)
```

``φ_L(δmax) + ½ k_sec (δ² − δmax²)`` with ``k_sec = t_L(δmax)/δmax``
(``K`` for a virgin chord): the cohesive potential with the damage
frozen at the history ``delta_max`` (same traction as the loading law
at ``δmax``).  Quadratic in the opening, so a Newton step on it is
essentially linear; the history is advanced *between* solves
(explicit damage update, see :meth:``0).

#### opening

```python
def opening(jump, normal)
```

``(delta_eff, delta_n, delta_t)`` from jumps (g, 2) and normals (g, 2).

#### energy\_density

```python
def energy_density(jump, normal, delta_max, scale=1.0, delta_prev=None)
```

Cohesive energy per unit length at the Gauss points.

``scale`` multiplies the traction-separation potential (strength,
stiffness and toughness together, so ``δ0`` and ``δf`` are
unchanged) — a per-chord imperfection that lets damage localise
instead of bifurcating when many identical interfaces reach their
peak simultaneously.  ``delta_prev`` (effective opening at the previous
converged step) activates the viscous regularisation
``½ η (δ − δ_prev)²`` per step (``viscosity`` = η, a traction per unit
opening increment): it adds a positive stiffness ``η`` to every chord,
which keeps Newton from cycling between the loading and the secant
(unloading) branch when many chords sit at their peak, and
regularises snap-backs; keep ``η Δδ`` small against ``σc``.

#### traction

```python
def traction(delta, delta_max)
```

Effective traction ``t = dφ/dδ`` (vectorised over Gauss points).

## XFEMSpace Objects

```python
class XFEMSpace()
```

Host-side XFEM preprocessing for one level set on a 2D mesh.

Parameters
----------
- **mesh** (*Mesh*): The (non-conforming) background mesh; variable 0 lives on it.
- **phi** (*array (num_nodes,)*): Nodal level set; the interface is ``phi = 0``.  Values closer to zero than ``threshold`` times the local scale are pushed away from zero (sign preserved, ``phi = 0`` goes positive) to avoid sliver sub-cells — the same role as NLXP3D&#x27;s ``threshold``.  The clamped copy is used consistently for both the cut quadrature and the kernel.
- **psi** (*array (num_nodes,), optional*): Tangential level set limiting the *extent* of the interface, as in NLXP3D: the interface is the part of ``phi = 0`` where ``psi &lt; 0``. A cell crossed by ``phi = 0`` is an *interface cell* (its nodes get enriched) when ``psi &lt; 0`` at the midpoint of the cut chord.  With a sign-type enrichment and no tip functions the jump does not stop at the chord of the last interface cell: it tapers linearly to zero across the next crossed cell (that cell is sub-divided for quadrature too, see ``cut_cells``), so the effective tip is up to one cell beyond ``psi = 0``.  ``None`` (default): the whole ``phi = 0`` line.
- **enrichment** (*Enrichment*): ``ψ_j`` (see the defaults in this module).
- **vec** (*int*): Components of the displacement (2 for 2D elasticity).
- **ele_type** (*str*): ``&quot;TRI3&quot;`` or ``&quot;QUAD4&quot;``.
- **cohesive** (*CohesiveZone, optional*): Cohesive zone model on the interface chord of the enriched cells (``iqpts`` / ``iqw``: its Gauss points in reference coords and physical length weights, zero for cells without an interface).  Needs a sign-type enrichment (a jump); with a continuous enrichment the interface never opens and the term vanishes.
- **tri_rule** (*int*): Points of the triangle rule used on every sub-triangle (1, 3 or 6). Uncut cells are triangulated the same way (QUAD4: 4 triangles about the centre), so the rule is uniform across the mesh.
- **threshold** (*float*): Relative clamp for near-zero nodal level-set values.


Attributes
----------
- **interface_cells** (*bool (num_cells,)*): Cells crossed by the interface (within its ``psi`` extent); their nodes are enriched.
- **cut_cells** (*bool (num_cells,)*): Cells integrated with the interface-conforming sub-triangulation: every cell crossed by ``phi = 0`` that holds at least one enriched node (``interface_cells`` plus the tip cell(s)).
- **segments** (*float (num_cells, 2, 2)*): Physical end points of the cut chord of each ``cut_cells`` cell (``nan`` elsewhere); ``segments_ref`` holds them in reference coords.
- **enriched_mask** (*bool (num_nodes,)*)
- **local** (*int (num_nodes,)*): Node of the enrichment mesh for each background node (``phantom`` where not enriched).
- **emesh** (*Mesh*): Enrichment mesh: enriched nodes (real coordinates) + the phantom node (coordinates ``nan`` so no location-based BC ever selects it).


#### zero\_history

```python
def zero_history()
```

Initial ``delta_max`` (num_cells, n_gauss) for the cohesive zone.

#### problem\_kwargs

```python
def problem_kwargs() -> dict
```

Constructor arguments for the two-variable :class:`~feax.Problem`.

#### traced\_params

```python
def traced_params(*extra,
                  delta_max=None,
                  cohesive_scale=None,
                  delta_prev=None) -> TracedParams
```

``TracedParams`` carrying the geometry, level set and quadrature.

``extra`` arrays are forwarded to the material as cell-level arguments:
each must have leading dimension ``num_cells`` (or be ≥3-D, which feax
passes through as-is) and is given to ``material`` per cell.
With a cohesive zone, ``delta_max`` (num_cells, n_gauss) is the opening
history (default: :meth:``0) and ``cohesive_scale``
(num_cells,) an optional per-chord scaling of the cohesive potential
(default 1; see :meth:``3); ``delta_prev``
(num_cells, n_gauss) the opening at the previous converged step for
the viscous regularisation (:meth:``6).

#### interface\_state

```python
def interface_state(problem: Problem, sol, delta_max=None)
```

Cohesive-zone state at the interface Gauss points of the cohesive
cells: dict with ``x`` (n, 2), ``delta`` (effective opening), ``delta_n``,
``delta_t``, ``traction`` (effective, from ``delta_max``; default
zero history), ``cell`` (cell index), all (n_cohesive_cells * n_gauss,).

#### current\_opening

```python
def current_opening(problem: Problem, sol)
```

Effective opening at the interface Gauss points as a (num_cells,
n_gauss) array (zero outside the cohesive cells) — the ``delta_prev``
of the next step for the viscous regularisation.

#### update\_history

```python
def update_history(delta_max, problem: Problem, sol)
```

``max(delta_max, current effective opening)`` (num_cells, n_gauss).

#### solve\_step

```python
def solve_step(problem: Problem,
               solver: Callable,
               bc,
               ts,
               u0,
               delta_max,
               *extra,
               cohesive_scale=None,
               max_outer: int = 30,
               tol: float = 1e-6,
               relax: float = 1.0)
```

One load step with an explicit damage update (secant cohesive zone).

Solves with the history ``delta_max`` frozen, advances the history
with the new openings (``max``), and repeats until the history no
longer changes (relative ``tol`` on the opening scale).  ``relax &lt; 1``
damps the update when it oscillates (snap-through).  Returns
``(sol, delta_max, n_outer)``.  With ``secant=False`` cohesive zones
the single implicit solve is returned unchanged.

#### phantom\_rows

```python
def phantom_rows(problem: Problem) -> onp.ndarray
```

Global DOF rows of the phantom node (always Dirichlet-fixed to 0).

#### create\_bc

```python
def create_bc(problem: Problem,
              config: Optional[DirichletBCConfig] = None) -> DirichletBC
```

Dirichlet BC from ``config`` (on variable 0 and/or 1) plus the phantom rows.

#### split

```python
def split(problem: Problem, sol)
```

``(u, a)``: standard nodal values (num_nodes, vec) and enriched values
(num_enriched + 1, n_funcs, vec) (last row = phantom).

#### nodal\_displacement

```python
def nodal_displacement(problem: Problem, sol)
```

Total displacement at the nodes, ``u_j + ψ_j(x_j) a_j`` (num_nodes, vec).

#### cell\_solution

```python
def cell_solution(problem: Problem, sol)
```

Per-cell flat solution in the kernel&#x27;s layout (num_cells, nn*(vec+vec_enriched)).

#### interface\_jump

```python
def interface_jump(problem: Problem, sol, eps: float = 1e-12)
```

Displacement jump ``u⁺ − u⁻`` across the interface at the midpoint
of the cut chord of every ``cut_cells`` cell.

Returns ``(x_mid, jump)``: (num_cut, 2) physical points and
(num_cut, vec) jumps (zero for continuous enrichments).  ``u⁺`` /
``u⁻`` are the limits from ``phi &gt; 0`` / ``phi &lt; 0``, evaluated by
calling the enrichment with ``phi = ±eps``.

#### remesh

```python
def remesh(problem: Problem,
           sol,
           *extra,
           split: str = "all",
           delta_max=None,
           damage_threshold: float = 1.0,
           nudge: float = 1e-6) -> dict
```

Crack-aware conforming remesh of the XFEM solution.

Every cell crossed by the interface is split along it (the same
sub-triangulation as the quadrature); vertices are shared between
neighbouring triangles *except* across the interface, where they are
duplicated — one copy per side — so the displacement jump becomes a
real gap in the mesh.  ``split`` selects where the mesh is opened:

* ``&quot;all&quot;``: along every interface (default);
* ``&quot;damaged&quot;``: only along chords whose cohesive damage
  (``max δmax / δf`` over the chord&#x27;s Gauss points, needs a cohesive
  zone and ``delta_max``) reaches ``damage_threshold`` — i.e. the
  cracks; intact interfaces stay connected (their vertices take the
  average of the two sides);
* ``&quot;none&quot;``: a plain conforming refinement.

Fields are evaluated at the vertices nudged by ``nudge`` toward the
sub-triangle centroid (picks the correct side on the interface).

Returns host numpy arrays: ``points`` (P, 2), ``cells`` (T, 3),
``u`` (P, vec), ``phi`` (P,), ``parent`` (T,), ``side`` (T,),
``grad_u`` (T, 2, 2) at the centroids, ``damage`` (T,) of the parent
chord (0 where none), ``opened`` (T,) bool, and ``crack_lines``
(L, 2) vertex pairs along the opened interface (both faces).
Write it with :func:``6 on ``Mesh(points, cells, &quot;TRI3&quot;)``.

#### save\_remesh\_vtu

```python
def save_remesh_vtu(filename: str,
                    problem: Problem,
                    sol,
                    material: Optional[Callable] = None,
                    crack_filename: Optional[str] = None,
                    **kwargs) -> dict
```

Write :meth:`remesh` with :func:`feax.utils.save_sol`: point data ``u``
(3 components, for Warp By Vector) and ``phi``; cell data ``parent``,
``side``, ``damage``, ``opened`` and, if ``material(u_grad, phi, x)`` is
given, ``stress`` (3x3, plane) at the sub-triangle centroids.  With
``crack_filename`` the opened interface faces are written as ``LINE2``
cells.  ``kwargs`` go to :meth:`remesh`.  Returns the remesh dict.

#### quad\_fields

```python
def quad_fields(problem: Problem, sol, grad: bool = False)
```

``(x_q, u_q, phi_q)`` at the sub-cell quadrature points, each
(num_cells, nq_max, ...), plus the displacement gradient ``grad_u``
(num_cells, nq_max, dim, dim) when ``grad=True``.  Padded points carry
``qw == 0``; use ``space.qw &gt; 0`` to select the real ones.  Handy for
looking at the fields (and strains / stresses) on both sides of the
interface.

## XFEMProblem Objects

```python
@jax.tree_util.register_pytree_node_class

@dataclass
class XFEMProblem(Problem)
```

Two-variable XFEM problem; build it with :func:`xfem_problem`.

``additional_info = (space, material)`` where
``material(u_grad, phi, x, *extra) -&gt; sigma`` (``u_grad``: (dim, dim)
displacement gradient at a quadrature point, ``phi``: level-set value there,
so a two-phase law is ``np.where(phi &lt; 0, ...)``, ``x``: its coordinates,
``extra``: the cell-level arrays given to :meth:``5).

#### xfem\_problem

```python
def xfem_problem(space: XFEMSpace, material: Callable,
                 **kwargs) -> XFEMProblem
```

Create the :class:`XFEMProblem` for ``space`` with constitutive law ``material``.

``kwargs`` (e.g. ``matrix_view``, ``location_fns``) go to :class:``1.

