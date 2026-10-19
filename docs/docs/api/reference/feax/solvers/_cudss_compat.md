---
sidebar_label: _cudss_compat
title: feax.solvers._cudss_compat
---

Compatibility shim mapping feax&#x27;s cuDSS calls onto upstream spineax.

feax historically targeted a fork of spineax that exposed
``spineax.cudss.factor_solve`` (``factorize`` / ``solve_with``) and
``spineax.cudss.solver.CuDSSSolver``. Upstream spineax (johnviljoen / the
official PyPI release) provides the same capability through
``spineax.cudss.solver.analyze`` / ``solve`` instead. This module adapts the
signatures so the rest of feax is unchanged and the spineax API surface feax
depends on lives in exactly one place.

Notes on the mapping
--------------------
* ``factorize(values, offsets, columns, device_id=, mtype_id=, mview_id=)`` -&gt;
  ``analyze(values, offsets, columns, mtype_id=, mview_id=, device_id=)``.
  The keyword names already match; ``analyze`` accepts the raw cuDSS enum ints
  feax passes (mtype: general=0/symmetric=1/spd=3; mview: full=0/upper=1/lower=2).
  Upstream splits analysis (``analyze``) from re-factorization with new values
  (``factorize``/``refactorize`` on an existing token); mapping feax&#x27;s one-shot
  ``factorize`` onto ``analyze`` preserves the current analyze-every-call
  behaviour exactly (a future optimisation could reuse the analysed token).
* ``solve_with(token, b, device_id=)`` -&gt; ``solve(token, b)``. The device is
  baked into the token at ``analyze`` time, so ``device_id`` is accepted for
  signature compatibility and ignored here.
* ``CuDSSSolver`` is reproduced as an ``equinox`` module with the same static
  fields and a ``__call__(b, csr_values) -&gt; (x, aux)`` returning a 2-tuple, as
  the fork did.

#### factorize

```python
def factorize(csr_values,
              csr_offsets,
              csr_columns,
              *,
              device_id: int = 0,
              mtype_id=1,
              mview_id=1)
```

Analyze + numerically factorize a CSR matrix, returning a solve-ready token.

#### solve\_with

```python
def solve_with(token, b_values, device_id: int = 0)
```

Solve ``A x = b`` reusing the factorization referenced by ``token``.

``device_id`` is accepted for backward-compatible call sites but ignored:
upstream binds the device to the token at ``analyze`` time.

## CuDSSSolver Objects

```python
class CuDSSSolver(eqx.Module)
```

cuDSS direct solver wrapper (sparsity pattern static for vmap).

Backed by upstream ``spineax.cudss.solver.analyze`` / ``solve``. ``__call__``
returns ``(x, aux)`` to match the fork&#x27;s tuple-returning contract; ``aux`` is
the factorization token (``None`` was previously discarded at the call site).

