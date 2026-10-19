---
sidebar_label: cadjoint_backend
title: feax.experimental.cadjoint_backend
---

feax as a cadjoint FEM solver backend — experimental.

Registers ``&quot;feax&quot;`` in :mod:`cadjoint.fem.backends` so a cadjoint study, the
playground viewer included, solves with feax instead of jax-fem. The mesh and
the boundary conditions arrive as plain arrays over cadjoint&#x27;s ABI, so nothing
here knows about SDFs or the viewer.

cadjoint resolves ``&quot;feax&quot;`` to this class through a lazy factory and defaults
to it, so the playground and every ``Study.solve`` use it with no configuration.
:func:`register` is only needed to register under a different name.

FORWARD SOLVES ONLY. cadjoint&#x27;s ABI asks for a solve differentiable in
``points``, because its shape optimization moves the mesh. feax is built the
other way round: :class:``0 bakes ``JxW`` and ``shape_grads`` at
construction and traces material fields instead, which is what makes its
structured-grid solvers index arithmetic. Differentiating one of these calls
w.r.t. ``points`` raises rather than returning a wrong gradient.

For shape sensitivity with feax, keep the grid fixed and let the SDF move
through the material weight instead -- see
``examples/experimental/sdf_shape_optimization.py``, which is the same geometry
source with the derivative taken somewhere feax can honestly provide it.

The two ABI conventions this relies on, both shared with feax:

* a location function may take ``(point, index)``, so a node-index set becomes
  a predicate directly;
* a boundary face belongs to a patch when ALL of its vertices do.

## FeaxBackend Objects

```python
class FeaxBackend()
```

Forward-only cadjoint solver backend built on feax.

#### thermal

```python
def thermal(points, cells, bcs, *, conductivity, source, base_points=None)
```

Solve -div(k grad T) = q; returns per-node temperature ``(N,)``.

#### elastic

```python
def elastic(points,
            cells,
            bcs,
            *,
            youngs,
            poisson,
            base_points=None,
            body_force=None)
```

Solve small-strain linear elasticity; returns displacement ``(N, 3)``.

#### register

```python
def register(name: str = "feax", **kwargs) -> None
```

Register this backend with cadjoint under ``name``.

