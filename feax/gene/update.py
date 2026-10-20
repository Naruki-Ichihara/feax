"""Design updates and material interpolation for topology optimization.

Pure JAX, and shape-agnostic in the leading axis: every function takes one
design field ``(n,)`` or a batch of them ``(B, n)`` (the designs of a
``feax.vmap`` batch), with per-case targets where that makes sense.
"""

import jax
import jax.numpy as np


def simp(rho, E0, penal: float = 3.0, eps: float = 1e-3):
    """SIMP material interpolation ``E = E0 (eps + (1 - eps) rho^penal)``.

    Elementwise, so it applies to one field or a batch alike; ``E0`` may be a
    scalar or an array broadcastable to ``rho``.
    """
    return E0 * (eps + (1.0 - eps) * rho ** penal)


def oc_update(rho, dc, volfrac, *, dv=1.0, move=0.2, xmin=0.0, xmax=1.0,
              damping=0.5, n_bisect: int = 60, l_min: float = 1e-20, l_max: float = 1e20):
    """Optimality-criteria update with the volume constraint met per design.

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
    rho, dc : array (n,) or (B, n)
        Design field(s) and the objective sensitivity w.r.t. them.
    volfrac : float or array (B,)
        Target volume fraction; one per case for a batch.
    dv : scalar or array (n,)
        Volume weight of each design variable (equal cells = 1.0; the nodal
        volume weights of ``create_volume_fn`` for a nodal field).
    move, xmin, xmax, damping :
        Move limit, bounds and damping exponent of the OC update.
    n_bisect : int
        Bisection steps for the multiplier (60 covers 40 decades to ~1e-12).

    Returns
    -------
    rho_new : array, same shape as ``rho``
    """
    rho = np.asarray(rho)
    dc = np.asarray(dc)
    if rho.ndim == 2:
        volfrac = np.broadcast_to(np.asarray(volfrac, rho.dtype), rho.shape[:1])
        return jax.vmap(lambda r, d, v: oc_update(
            r, d, v, dv=dv, move=move, xmin=xmin, xmax=xmax, damping=damping,
            n_bisect=n_bisect, l_min=l_min, l_max=l_max))(rho, dc, volfrac)
    dv = np.asarray(dv, rho.dtype) * np.ones_like(rho)
    target = volfrac * np.sum(dv)
    be = np.maximum(0.0, -dc / dv)
    lo, hi = np.clip(rho - move, xmin, xmax), np.clip(rho + move, xmin, xmax)

    def candidate(lam):
        return np.clip(rho * (be / lam) ** damping, lo, hi)

    def bisect(_, bounds):
        l1, l2 = bounds
        mid = np.sqrt(l1 * l2)
        over = np.sum(candidate(mid) * dv) > target
        return np.where(over, mid, l1), np.where(over, l2, mid)

    l1, l2 = jax.lax.fori_loop(0, n_bisect, bisect,
                               (np.asarray(l_min, rho.dtype), np.asarray(l_max, rho.dtype)))
    return candidate(np.sqrt(l1 * l2))


__all__ = ["oc_update", "simp"]
