---
sidebar_label: cmg
title: feax.solvers.cmg
---

Compact geometric multigrid (cmg) solver for structured narrow bands.

A matrix-free, O(band) geometric multigrid preconditioned CG (MGPCG) for
LINEAR ELASTICITY on a uniform HEX8 :class:`feax.StructuredGrid` — the
memory-scalable solver for the giga-voxel structured narrow-band regime, where
an assembled direct factorization (3D fill) or an algebraic-MG hierarchy would
exhaust memory.

Scope: uniform structured grid, linear elasticity with per-cell SIMP scaling
E = Emin + rho^p (E0-Emin). The global node/DOF numbering matches
``StructuredGrid`` (node n = i*(ny+1)*(nz+1)+j*(nz+1)+k, DOF 3n+c), so the
returned displacement is on the same band-node layout as
``NarrowBand(grid, active)`` (node_map). The element-local corner ordering here
is internal (paired with make_KE_3d) and independent of feax&#x27;s HEX8 order — the
assembled operator and the solution are identical.

Smoother: 8-colour block (3x3 node) Gauss-Seidel. Transfers: trilinear
prolongation / full-weighting restriction on the compact active set. Coarsest
level: cuDSS (via spineax, default) or a matrix-free block-Jacobi Krylov solve
— selected by the feax ``solver_options`` passed to
:meth:`NarrowBandCMG.create_solver`.

#### make\_KE\_3d

```python
def make_KE_3d(nu=0.3, E=1.0)
```

24x24 stiffness of a unit-cube trilinear hex; local node order = the 8
corners (dx,dy,dz), dz fastest — matches the compact edof in this module.

#### auto\_levels

```python
def auto_levels(dims, floor=4)
```

Geometric MG depth: ceil-halve each dim (works for arbitrary, incl. odd,
grid sizes) until the smallest dim reaches ``floor``.

## NarrowBandCMG Objects

```python
class NarrowBandCMG()
```

Matrix-free O(band) geometric-MG (MGPCG) solver for a structured band.

``grid``       : a :class:`feax.StructuredGrid`.
``fixed_pred`` : ``(ni,nj,nk,nx,ny,nz) -&gt; bool`` marking fixed (Dirichlet)
                 nodes by grid index (e.g. ``lambda ni,nj,nk,nx,ny,nz: nk==0``).
``n_levels``   : MG levels (default from :func:``1).
``bucket``     : per-level padded capacities are rounded up to a multiple of
                 this, so a moving band reuses the compiled MGPCG while its
                 sizes stay in the same bucket (coarser bucket = fewer
                 recompiles, more padding per solve).

Usage (feax create_solver convention — one differentiable solver)::

    cmg    = fe.NarrowBandCMG(grid, fixed_pred, nu=0.3, penal=3.0)
    levels = cmg.build(active_cells)
    b      = cmg.load_vector(levels, node_ids, comp=2, value=-1.0)
    solver = cmg.create_solver(levels, b)      # None -&gt; cuDSS coarsest
    u      = solver(rho_cells)                 # bare array OR TracedParams
    u      = solver(fe.TracedParams(volume_vars=(rho_cells,)))   # same
    dc     = jax.grad(lambda r: jnp.dot(b, solver(r)))(rho_cells)

#### compact\_nodes

```python
def compact_nodes(levels, node_ids)
```

Global grid node ids -&gt; compact band node indices (level 0).

#### load\_vector

```python
def load_vector(levels, node_ids, comp=2, value=-1.0)
```

Nodal-load RHS on the compact band DOFs: set component ``comp`` (0/1/2)
to ``value`` at every global node in ``node_ids`` (must be in the band).

#### compliance\_and\_dc

```python
def compliance_and_dc(levels, rho_cells, u)
```

Analytic self-adjoint compliance and SIMP sensitivity on the band cells:
c = Σ E_e uₑᵀKₑuₑ ,  dc = -p ρ^(p-1)(E0-Emin) uₑᵀKₑuₑ.
``rho_cells``: bare array or TracedParams (volume_vars[0]).

#### create\_solver

```python
def create_solver(levels,
                  b,
                  solver_options=None,
                  jit=True,
                  return_solution=True)
```

Build the cmg solver for this band + load ``b`` (compact band DOF
layout), matching feax&#x27;s ``create_solver`` convention: returns a single
callable ``solver(rho_cells) -&gt; u`` that is differentiable (implicit-diff
``custom_vjp``). Compliance sensitivity for the analytic/OC path is
``compliance_and_dc(levels, rho, solver(rho))``.

``solver_options`` selects the coarsest-level solver: ``None`` /
``DirectSolverOptions`` → cuDSS; ``KrylovSolverOptions`` → matrix-free
block-Jacobi Krylov (cg/bicgstab/gmres), no cuDSS dependency. The
geometric-MG smoother/transfers are intrinsic to cmg.

``jit=True`` (default) wraps the MGPCG in ``jax.jit``; the compiled
executable is reused across bands whose padded (bucketed) shapes match.
``jit=False`` runs eagerly — no compilation, so a band whose shape
changes every iteration avoids the per-iteration recompile (pair with
``bucket=1`` for an exact-size band). Result and gradient are identical.

The result is a :class:``6 (one field,
``(num_band_nodes, 3)``) by default; ``return_solution=False`` gives
the raw flat band vector.

