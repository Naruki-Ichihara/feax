---
sidebar_label: cmg
title: feax.experimental.cmg
---

Compact geometric multigrid (cmg) on a structured narrow band — experimental.

A matrix-free, O(band) geometric-multigrid preconditioned CG (MGPCG) for
LINEAR, SYMMETRIC POSITIVE DEFINITE problems on a uniform HEX8
:class:`feax.StructuredGrid` — the memory-scalable solver for the giga-voxel
narrow-band regime, where an assembled factorization or an algebraic-MG
hierarchy would exhaust memory.

What is fixed and what is not
-----------------------------
Fixed (the reasons cmg is O(band)):

* a uniform HEX8 structured grid — connectivity and coordinates are index
  arithmetic, never materialised;
* a linear operator ``K(params) u`` that is symmetric positive definite once the
  Dirichlet DOFs are removed — CG, the Gauss-Seidel smoother and the
  &quot;adjoint = one more forward solve&quot; of the design gradient all rely on it.
  One-way coupled physics (e.g. temperature -&gt; thermal strain -&gt; displacement)
  is NOT one SPD system; solve it as two cmg instances sharing the same band
  geometry (see :func:`build_geometry`).

Free (chosen through a :class:`Material`):

* the number of DOFs per node ``vec`` (3 = elasticity, 1 = diffusion, ...);
* the constitutive law. :class:`ScalarMaterial` is the fast path — one
  reference element matrix ``KE`` times a per-cell scalar (SIMP, RAMP, ...),
  one float of state per cell. :class:``0 takes any *linear* feax
  ``tensor_map`` (``u_grad (vec,3) -&gt; flux (vec,3)``) and evaluates it through
  the per-cell constitutive matrix ``D = d flux / d u_grad`` — anisotropic,
  orientation-dependent, multi-material, ... at ``(3*vec)**2`` floats per cell
  and roughly 5x the matvec cost. :meth:``9 pulls the
  ``tensor_map`` (or the gradient of the energy density) out of an ordinary
  :class:``2, so the cmg operator is generated from the same kernel
  feax assembles from.

Conventions
-----------
Global node/DOF numbering matches ``StructuredGrid`` (node
``n = i*(ny+1)*(nz+1) + j*(nz+1) + k``, DOF ``vec*n + c``), so the returned
field is on the same band-node layout as ``NarrowBand(grid, active)``
(``node_map``). The element-local corner ordering here is internal (the 8
corners ``(dx,dy,dz)`` with ``dz`` fastest) and differs from feax&#x27;s HEX8 order;
the assembled operator and the solution are identical.

Smoother: 8-colour block (``vec x vec`` node) Gauss-Seidel. Transfers:
trilinear prolongation / full-weighting restriction on the compact active set.
Coarsest level: cuDSS (via spineax, default) or a matrix-free block-Jacobi
Krylov solve — selected by the feax ``solver_options`` passed to
:meth:`Material`1.

Usage::

    from feax.experimental import NarrowBandCMG, ScalarMaterial, KernelMaterial, simp

    cmg    = NarrowBandCMG(grid, fixed_pred)                # isotropic SIMP elasticity
    cmg    = NarrowBandCMG(grid, fixed_pred, material=ScalarMaterial.isotropic_conduction(
                 grid.spacing, scale_fn=simp(1e-3, 1.0, 3.0)))          # vec=1 diffusion
    cmg    = NarrowBandCMG(grid, fixed_pred, material=KernelMaterial.from_problem(
                 MyProblem, grid, vec=3, additional_info=(...)))       # any linear Problem

    levels = cmg.build(active_cells)
    b      = cmg.load_vector(levels, node_ids, comp=2, value=-1.0)
    b      = cmg.face_load_vector(levels, axis=1, side=0, traction=(0, -1.0, 0))
    solver = cmg.create_solver(levels, b)          # None -&gt; cuDSS coarsest
    u      = solver(params)                        # differentiable (implicit vjp)
    u      = solver(params, b=other_rhs)           # ... in the design AND the rhs
    # non-zero Dirichlet values: lift them (u_p holds the values on fixed DOFs)
    u      = u_p + solver(params, b=cmg.dirichlet_rhs(levels, params, u_p))

A grid built by :meth:`Material`2 from a ``box_mesh`` shares
its node and cell numbering, so a full-grid cmg field and a feax Problem on the
box mesh use one layout (no permutation).

#### hex\_shape\_grads

```python
def hex_shape_grads(spacing=1.0)
```

Shape-function gradients of the trilinear hex on a box cell of size
``spacing`` (scalar or (3,)), at its 8 Gauss points.

Returns ``(dNdx, w)``: ``dNdx`` of shape ``(8 gp, 8 nodes, 3)`` (nodes in
cmg corner order) and the quadrature weights ``w`` of shape ``(8,)`` (sum =
cell volume).

#### gradient\_operator

```python
def gradient_operator(spacing=1.0, vec=3)
```

``B`` of shape ``(8 gp, 3*vec, 8*vec)`` mapping the element DOF vector
(index ``vec*a + c``: node-major, component-minor) to the flattened
gradient ``u_grad.reshape(-1)`` (row-major ``(vec, 3)``: index ``3*c + i``),
plus the weights ``w`` ``(8,)``.

#### constitutive\_matrix

```python
def constitutive_matrix(tensor_map, vec, *params)
```

``D = d flux / d u_grad`` at zero gradient, as ``(3*vec, 3*vec)`` on the
row-major flattened gradient (JAX; differentiable in ``params``).

#### element\_matrix

```python
def element_matrix(B, w, D)
```

``K_e = sum_g w_g B_g^T D B_g`` for one constitutive matrix.

#### isotropic\_stress

```python
def isotropic_stress(nu)
```

feax-style ``tensor_map`` of unit-modulus isotropic linear elasticity.

#### make\_KE\_3d

```python
def make_KE_3d(nu=0.3, E=1.0, spacing=1.0)
```

``24x24`` stiffness of an isotropic linear-elastic box HEX8 cell (unit
cube by default), in cmg corner order.

#### make\_KE\_conduction

```python
def make_KE_conduction(spacing=1.0, k=1.0)
```

``8x8`` conductivity matrix of a box HEX8 cell (``div(k grad T)``).

#### unit\_cell\_mesh

```python
def unit_cell_mesh(grid)
```

One-cell :class:`feax.Mesh` (feax HEX8 order) with the grid&#x27;s spacing.

#### simp

```python
def simp(Emin=1e-9, E0=1.0, penal=3.0)
```

SIMP scale ``rho -&gt; Emin + rho**penal * (E0 - Emin)``.

#### ramp

```python
def ramp(Emin=1e-9, E0=1.0, q=8.0)
```

RAMP scale ``rho -&gt; Emin + (E0 - Emin) * rho / (1 + q (1 - rho))``.

## Material Objects

```python
class Material()
```

Constitutive interface of cmg. Everything is JAX over a leading cell axis.

``state0(params)``            per-cell material state on level 0 from the
                              design parameters (an array or a tuple of
                              arrays over the band cells);
``matvec(state, ue)``         ``(n, 8*vec) -&gt; (n, 8*vec)`` element matvec;
``node_blocks(state)``        ``(n, 8, vec, vec)`` diagonal corner blocks;
``element_matrices(state)``   ``(n, 8*vec, 8*vec)`` (coarsest level only).

``coarsen`` (average children into the parent) and ``pad`` are generic.

## ScalarMaterial Objects

```python
class ScalarMaterial(Material)
```

``K_e = scale_fn(*params_e) * KE`` — one float of state per cell.

``KE``: the reference element matrix ``(8*vec, 8*vec)`` in cmg corner
order; ``scale_fn``: JAX ``(*params_cell) -&gt; scalar`` (default SIMP).

#### isotropic\_elastic

```python
@classmethod
def isotropic_elastic(cls, spacing=1.0, nu=0.3, scale_fn=None)
```

``vec=3`` isotropic linear elasticity (unit modulus times ``scale_fn``).

#### isotropic\_conduction

```python
@classmethod
def isotropic_conduction(cls, spacing=1.0, scale_fn=None)
```

``vec=1`` steady diffusion ``div(k grad T)`` (unit conductivity times
``scale_fn``).

## KernelMaterial Objects

```python
class KernelMaterial(Material)
```

Any linear feax ``tensor_map`` — state is the per-cell constitutive
matrix ``D_e = d flux / d u_grad`` of shape ``(3*vec, 3*vec)``.

``tensor_map(u_grad, *params_cell) -&gt; flux`` follows feax&#x27;s
``Problem.get_tensor_map`` convention (``u_grad``/``flux`` of shape
``(vec, 3)``); the design ``params`` given to the solver is a tuple of
per-cell arrays (or one array) in the same order. Cell-constant parameters
only (no nodal internal variables).

Memory: ``(3*vec)**2`` floats per cell (81 for elasticity) on every level.
Cost: about 5x the :class:``0 matvec.

``sample_params``: per-cell parameter values used to verify at construction
that the map is linear in ``u_grad`` and that ``D`` is symmetric (SPD is
what cmg relies on). Pass it whenever the law is not obviously linear.

#### from\_problem

```python
@classmethod
def from_problem(cls,
                 problem,
                 grid,
                 vec=None,
                 *,
                 additional_info=(),
                 sample_params=None,
                 **problem_kw)
```

The material of a :class:`feax.Problem`.

``problem`` is either an INSTANCE (its maps are used as they are; the
mesh it lives on is irrelevant, ``vec`` is read from it) or a SUBCLASS,
which is instantiated on a one-cell mesh with the grid&#x27;s spacing
(``problem(mesh, vec=vec, dim=3, ele_type=&#x27;HEX8&#x27;,
additional_info=additional_info, **problem_kw)``).

The constitutive map is ``get_tensor_map()`` when the Problem defines
it, else the gradient of ``get_energy_density()`` with respect to the
field gradient (``flux = d psi / d u_grad``) — so energy-based Problems
work unchanged. Either way it must be linear in the field
(``sample_params`` checks that).

#### check\_linear

```python
def check_linear(*params_cell, rtol=1e-6)
```

Raise ``ValueError`` unless ``tensor_map`` is linear in ``u_grad``
with a symmetric ``D`` at ``params_cell``.

#### auto\_levels

```python
def auto_levels(dims, floor=16)
```

Geometric MG depth: ceil-halve each dim (works for arbitrary, incl. odd,
grid sizes) while the smallest dim of the next level would still be
``&gt;= floor``, so every level has at least ``floor`` cells per axis.

The default ``floor=16`` stops coarsening while the coarsest level is still
a few thousand cells per short dimension, and leaves the rest to the
coarsest-level solver (cuDSS factors ~1e5 DOF in milliseconds). Deeper
hierarchies cost more sequential V-cycle work per iteration AND converge
worse for anisotropic / high-contrast materials, whose coarse operators
(averaged material state) degrade with every level: on a 33^3 anisotropic
RAMP elasticity problem, 5 levels (coarsest 3^3) took 71 PCG iterations and
2 levels (coarsest 17^3) took 16, ten times faster overall.

#### build\_geometry

```python
def build_geometry(grid, cells0, n_levels=None)
```

Per-level band geometry (cells, nodes, corner-&gt;node map, transfers) for
an active-cell set — everything cmg needs that does not depend on the
physics. Pass the result to :meth:`NarrowBandCMG.build` of several cmg
instances (e.g. a ``vec=1`` and a ``vec=3`` one) to share it.

## NarrowBandCMG Objects

```python
class NarrowBandCMG()
```

Matrix-free O(band) geometric-MG (MGPCG) solver for a structured band.

``grid``       : a :class:`feax.StructuredGrid`.
``fixed_pred`` : ``(ni,nj,nk,nx,ny,nz) -&gt; bool`` over band nodes (by grid
                 index): ``(n,)`` fixes every component of a node,
                 ``(n, vec)`` fixes components individually.
``material``   : a :class:``3 (default: isotropic SIMP elasticity
                 built from ``nu``/``Emin``/``E0``/``penal`` and the grid
                 spacing). Sets ``vec``.
``n_levels``   : MG levels (default from :func:`feax.StructuredGrid`6: coarsen while
                 the smallest dim stays &gt;= 16; shallow is faster here).
``bucket``     : per-level padded capacities are rounded up to a multiple of
                 this, so a moving band reuses the compiled MGPCG while its
                 sizes stay in the same bucket.
``precision``  : ``&quot;double&quot;`` runs everything in float64. ``&quot;mixed&quot;`` runs
                 the multigrid PRECONDITIONER (smoother, transfers, coarse
                 Krylov) in float32 and keeps the outer CG — residual,
                 search direction, the level-0 matvec and the cuDSS coarse
                 solve — in float64, so the converged solution is a float64
                 solution to ``cg_tol``. On GPUs with slow float64 (consumer
                 / GB10 class) this makes a V-cycle several times cheaper.

Usage (feax create_solver convention — one differentiable solver)::

    cmg    = NarrowBandCMG(grid, fixed_pred, nu=0.3, penal=3.0)
    levels = cmg.build(active_cells)             # or cmg.build(build_geometry(...))
    b      = cmg.load_vector(levels, node_ids, comp=2, value=-1.0)
    solver = cmg.create_solver(levels, b)        # None -&gt; cuDSS coarsest
    u      = solver(params)                      # array / tuple / TracedParams
    dc     = jax.grad(lambda r: jnp.dot(b, solver(r)))(params)

#### build

```python
def build(cells_or_geometry)
```

Per-level band structure for an active-cell set (global cell ids) or
for a :func:`build_geometry` result (shared between instances).

#### compact\_nodes

```python
def compact_nodes(levels, node_ids)
```

Global grid node ids -&gt; compact band node indices (level 0).

#### load\_vector

```python
def load_vector(levels, node_ids, comp=None, value=-1.0)
```

Nodal-load RHS on the compact band DOFs: set component ``comp`` (in
``range(vec)``; default: the last one — ``z`` for elasticity, the only
one for a scalar field) to ``value`` at every global node in
``node_ids`` (must be in the band).

#### face\_load\_vector

```python
def face_load_vector(levels, axis, side, traction)
```

Consistent nodal loads of a uniform traction on one boundary face of
the grid (``axis`` 0/1/2, ``side`` 0 = the ``index == 0`` face, 1 = the
``index == n`` face), restricted to the band&#x27;s cells on that face.
``traction`` is the force per unit area, ``(vec,)``, applied in the
POSITIVE coordinate directions. feax&#x27;s ``surface_map`` convention adds
the map to the residual, so a feax traction ``t`` corresponds to
``traction = -t`` here.

For bilinear faces under uniform traction the consistent load is one
quarter of ``traction * face_area`` at each of the face&#x27;s 4 nodes.

#### dirichlet\_rhs

```python
def dirichlet_rhs(levels, params, prescribed)
```

Right-hand side of the lifting for NON-ZERO Dirichlet values.

``prescribed`` is a band DOF vector holding the prescribed values on the
fixed DOFs (anything on free DOFs is ignored). With ``u_0 = solver(params,
b=dirichlet_rhs(...))`` the field is ``u = prescribed_on_fixed + u_0``:
``K_ff u_0 = -K_fc u_c``. Differentiable in ``params`` (the RHS depends
on the design), so ``jax.grad`` through ``solver(params, b=...)`` is
complete.

#### compliance\_and\_dc

```python
def compliance_and_dc(levels, params, u)
```

Analytic self-adjoint compliance and sensitivity on the band cells:
``c = u^T K(params) u``, ``dc = -u^T (dK/dparams) u`` (no adjoint solve).
``params``: array, tuple of arrays or TracedParams; ``dc`` has the same
structure (numpy).

#### create\_solver

```python
def create_solver(levels,
                  b=None,
                  solver_options=None,
                  jit=True,
                  return_solution=True)
```

Build the cmg solver for this band, matching feax&#x27;s
``create_solver`` convention: returns a single callable
``solver(params, b=None) -&gt; u`` that is differentiable (implicit-diff
``custom_vjp``) in the design AND in the right-hand side. ``b`` (compact
band DOF layout) given here is the default load; a ``b`` passed to the
call overrides it (e.g. a design-dependent :meth:``0).
Compliance sensitivity for the analytic/OC path is
``compliance_and_dc(levels, params, solver(params))``.

``solver_options`` selects the coarsest-level solver: ``None`` /
``DirectSolverOptions`` → cuDSS; ``KrylovSolverOptions`` → matrix-free
block-Jacobi Krylov (cg/bicgstab/gmres), no cuDSS dependency. The
geometric-MG smoother/transfers are intrinsic to cmg.

``jit=True`` (default) wraps the MGPCG in ``jax.jit``; the compiled
executable is reused across bands whose padded (bucketed) shapes match.
``jit=False`` runs eagerly — no compilation, so a band whose shape
changes every iteration avoids the per-iteration recompile (pair with
``bucket=1`` for an exact-size band). Result and gradient are identical.

The result is a :class:``9 (one field,
``(num_band_nodes, vec)``) by default; ``return_solution=False`` gives
the raw flat band vector.

