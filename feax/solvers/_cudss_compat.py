"""Compatibility shim mapping feax's cuDSS calls onto upstream spineax.

feax historically targeted a fork of spineax that exposed
``spineax.cudss.factor_solve`` (``factorize`` / ``solve_with``) and
``spineax.cudss.solver.CuDSSSolver``. Upstream spineax (johnviljoen / the
official PyPI release) provides the same capability through
``spineax.cudss.solver.analyze`` / ``solve`` instead. This module adapts the
signatures so the rest of feax is unchanged and the spineax API surface feax
depends on lives in exactly one place.

Notes on the mapping
--------------------
* ``factorize(values, offsets, columns, device_id=, mtype_id=, mview_id=)`` ->
  ``analyze(values, offsets, columns, mtype_id=, mview_id=, device_id=)``.
  The keyword names already match; ``analyze`` accepts the raw cuDSS enum ints
  feax passes (mtype: general=0/symmetric=1/spd=3; mview: full=0/upper=1/lower=2).
  Upstream splits analysis (``analyze``) from re-factorization with new values
  (``factorize``/``refactorize`` on an existing token); mapping feax's one-shot
  ``factorize`` onto ``analyze`` preserves the current analyze-every-call
  behaviour exactly (a future optimisation could reuse the analysed token).
* ``solve_with(token, b, device_id=)`` -> ``solve(token, b)``. The device is
  baked into the token at ``analyze`` time, so ``device_id`` is accepted for
  signature compatibility and ignored here.
* ``CuDSSSolver`` is reproduced as an ``equinox`` module with the same static
  fields and a ``__call__(b, csr_values) -> (x, aux)`` returning a 2-tuple, as
  the fork did.
"""

import equinox as eqx
from jax import Array

# Imported eagerly so that importing this shim fails with ImportError when
# spineax is absent — exactly as the old ``from spineax.cudss.factor_solve
# import ...`` did, keeping every caller's try/except (and cmg's ``is None``
# sentinel) working unchanged.
#
# Upstream splits the fork's one-shot ``factorize`` into two phases: ``analyze``
# does the symbolic analysis + pivot order only (an un-factorized token), and
# ``factorize(token, values)`` performs the numeric factorization. The fork's
# ``factor_solve.factorize`` did both, so the shim chains them.
from spineax.cudss.solver import (
    analyze as _analyze,
    factorize as _numeric_factorize,
    refactorize as _numeric_refactorize,
    solve as _solve,
)


def factorize(csr_values, csr_offsets, csr_columns, *,
              device_id: int = 0, mtype_id=1, mview_id=1):
    """Analyze + numerically factorize a CSR matrix, returning a solve-ready token."""
    token = _analyze(csr_values, csr_offsets, csr_columns,
                     mtype_id=mtype_id, mview_id=mview_id, device_id=device_id)
    return _numeric_factorize(token, csr_values)


def refactorize(token, csr_values):
    """Numerically re-factorize ``token``'s matrix with new values on the same
    pattern, keeping its symbolic analysis (reordering) and its memory.

    The returned token replaces ``token`` (spineax renames the entry; using
    the old token afterwards re-factorizes it from scratch).
    """
    return _numeric_refactorize(token, csr_values)


def solve_with(token, b_values, device_id: int = 0):
    """Solve ``A x = b`` reusing the factorization referenced by ``token``.

    ``device_id`` is accepted for backward-compatible call sites but ignored:
    upstream binds the device to the token at ``analyze`` time.
    """
    return _solve(token, b_values)


class CuDSSSolver(eqx.Module):
    """cuDSS direct solver wrapper (sparsity pattern static for vmap).

    Backed by upstream ``spineax.cudss.solver.analyze`` / ``solve``. ``__call__``
    returns ``(x, aux)`` to match the fork's tuple-returning contract; ``aux`` is
    the factorization token (``None`` was previously discarded at the call site).
    """

    csr_offsets: Array = eqx.field(static=True)
    csr_columns: Array = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)

    def __call__(self, b, csr_values):
        token = _analyze(csr_values, self.csr_offsets, self.csr_columns,
                         mtype_id=self.mtype_id, mview_id=self.mview_id,
                         device_id=self.device_id)
        token = _numeric_factorize(token, csr_values)
        return _solve(token, b), token
