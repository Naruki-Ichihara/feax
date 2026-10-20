---
sidebar_label: update
title: feax.gene.update
---

Design updates and material interpolation for topology optimization.

Pure JAX, and shape-agnostic in the leading axis: every function takes one
design field ``(n,)`` or a batch of them ``(B, n)`` (the designs of a
``feax.vmap`` batch), with per-case targets where that makes sense.

#### simp

```python
def simp(rho, E0, penal: float = 3.0, eps: float = 1e-3)
```

SIMP material interpolation ``E = E0 (eps + (1 - eps) rho^penal)``.

Elementwise, so it applies to one field or a batch alike; ``E0`` may be a
scalar or an array broadcastable to ``rho``.

#### oc\_update

```python
def oc_update(rho,
              dc,
              volfrac,
              *,
              dv=1.0,
              move=0.2,
              xmin=0.0,
              xmax=1.0,
              damping=0.5,
              n_bisect: int = 60,
              l_min: float = 1e-20,
              l_max: float = 1e20)
```

Optimality-criteria update with the volume constraint met per design.

The Bendsøe multiplicative update for compliance-type objectives (``dc`` is
the objective sensitivity, negative for compliance): density moves toward
the optimality condition within a move limit, and the Lagrange multiplier
is found by a bisection (fixed length, on a log scale, so it traces and
jits) such that the weighted volume meets the target::

    rho_new = clip(rho * max(0, -dc / (dv * lam))**damping,
                   [rho - move, rho + move] ∩ [xmin, xmax])
    sum(rho_new * dv) = volfrac * sum(dv)

Parameters
----------
- **volfrac** (*float or array (B,)*): Target volume fraction; one per case for a batch.
- **dv** (*scalar or array (n,)*): Volume weight of each design variable (equal cells = 1.0; the nodal volume weights of ``create_volume_fn`` for a nodal field).
- **n_bisect** (*int*): Bisection steps for the multiplier (60 covers 40 decades to ~1e-12).


Returns
-------
rho_new : array, same shape as ``rho``

