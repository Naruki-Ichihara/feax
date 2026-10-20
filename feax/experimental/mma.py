"""MMA / GCMMA in JAX (experimental).

Pure-JAX implementation of Svanberg's *Method of Moving Asymptotes* (MMA) and
its *Globally Convergent* variant (GCMMA), following the reference MATLAB
code (``mmasub.m``, ``gcmmasub.m``, ``subsolv.m``, ``asymp.m``, ``concheck.m``,
``raaupdate.m``, ``kktcheck.m``, version September 2007) [1]_ [2]_.

Problem statement (Svanberg's standard form)::

    minimise    f0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i**2)
    subject to  f_i(x) - a_i*z - y_i <= 0,   i = 1..m
                xmin <= x <= xmax,  y >= 0,  z >= 0

With the defaults ``a0 = 1``, ``a = 0``, ``c = 1000``, ``d = 1`` this is the
usual ``min f0(x) s.t. f_i(x) <= 0`` with elastic (penalised) constraints.

.. warning::
   ``c`` is the elastic penalty on the constraint slacks ``y``.  At the
   optimum ``y_i > 0`` whenever the true multiplier of constraint ``i``
   exceeds ``c_i``, and the constraint is then *left violated*.  Scale the
   objective and the constraints so that the expected multipliers stay well
   below ``c`` (or raise ``c``).  ``f0`` of order 1 and constraints written
   as ``g/g_target - 1`` is a safe recipe.

Accuracy is limited by the interior point barrier: the final iterate is
``O(epsimin)`` away from the exact optimum, so the default ``epsimin=1e-7``
of the reference code gives roughly 8 correct digits.  Lower it for tighter
convergence.

**Everything is traced.**  Every loop in the algorithm — the interior point
barrier loop, its Newton iteration, the Newton line search, the GCMMA inner
(conservativity) loop, and the outer MMA iteration itself — is a
``lax.while_loop`` or ``lax.scan``.  No loop bound is a Python ``for``, so the
XLA graph does not grow with the iteration count and a *complete* optimisation
run compiles into a single kernel launch sequence.  The subproblem uses the
same primal-dual Newton method as ``subsolv.m``, with the ``m < n``
Schur-complement branch, so the dense linear solve is only ``(m+1)x(m+1)`` for
the typical topology-optimisation case ``m << n``.

Usage — the whole optimisation inside one ``jit``::

    from feax.experimental.mma import GCMMA

    def value_and_grad(x):
        # f0: (), df0: (n,), f: (m,), df: (m, n)     with constraints f <= 0
        return f0, df0, f, df

    opt = GCMMA(value_and_grad, xmin, xmax, n_constraints=m)
    state = opt.solve(x0, max_iter=200, kkt_tol=1e-6)   # traced end-to-end
    x_opt = state.x

``solve`` is itself ``jax.jit``-compatible (and ``jax.vmap``-compatible over
``x0`` and extra arguments), so a batch of designs can be optimised at once::

    solve = jax.jit(lambda x0: opt.solve(x0, max_iter=200))
    batch = jax.vmap(solve)(x0_batch)

Use :meth:`MMA.solve_with_history` to get per-iteration convergence data out of
the same compiled program, and :meth:`MMA.run` for a Python-level loop with a
callback (for VTU snapshots, logging, adaptive remeshing, ...).

The plain ``MMA`` class has the same interface (no inner loop, move limit
``move=0.5`` by default) and, because it never re-evaluates the functions
inside a traced loop, also works un-jitted with arbitrary Python callables via
:meth:`MMA.run`.

References
----------
.. [1] K. Svanberg, "The method of moving asymptotes — a new method for
       structural optimization", Int. J. Numer. Meth. Engng 24 (1987) 359–373.
.. [2] K. Svanberg, "A class of globally convergent optimization methods based
       on conservative convex separable approximations", SIAM J. Optim. 12
       (2002) 555–573.  Implementation notes: "MMA and GCMMA – two methods for
       nonlinear optimization" (2007).
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as np
from jax import lax

__all__ = [
    "MMA",
    "GCMMA",
    "MMAState",
    "Subproblem",
    "SubproblemSolution",
    "update_asymptotes",
    "move_bounds",
    "initial_raa",
    "build_subproblem",
    "solve_subproblem",
    "evaluate_approximation",
    "is_conservative",
    "update_raa",
    "kkt_residual",
]

Array = jax.Array
_XMAMI_EPS = 1e-5     # lower bound for xmax - xmin used in all reference routines


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Subproblem(NamedTuple):
    """MMA/GCMMA convex separable approximation around the current iterate.

    ``f0app(x) = r0 + p0/(upp-x) + q0/(x-low)``,
    ``fapp(x)  = r  + P/(upp-x)  + Q/(x-low)`` (row-wise), with the trust
    region ``alfa <= x <= beta``.
    """
    low: Array
    upp: Array
    alfa: Array
    beta: Array
    p0: Array
    q0: Array
    r0: Array
    P: Array
    Q: Array
    r: Array


class SubproblemSolution(NamedTuple):
    """Primal-dual solution of the subproblem (variables of ``subsolv.m``)."""
    x: Array
    y: Array
    z: Array
    lam: Array
    xsi: Array
    eta: Array
    mu: Array
    zet: Array
    s: Array


class MMAState(NamedTuple):
    """Optimiser state carried between outer iterations."""
    x: Array
    xold1: Array
    xold2: Array
    low: Array
    upp: Array
    raa0: Array
    raa: Array
    f0: Array        # f0(x)
    df0dx: Array     # (n,)
    fval: Array      # (m,)
    dfdx: Array      # (m, n)
    sol: SubproblemSolution   # duals of the last subproblem (for the KKT check)
    iteration: Array          # number of outer iterations performed
    inner_iterations: Array   # inner iterations used in the last outer step
    kkt_norm: Array           # ||KKT residual||_2 at x
    kkt_max: Array            # ||KKT residual||_inf at x
    f0app: Array              # last subproblem objective value at x
    fapp: Array               # last subproblem constraint values at x
    aux: object = None        # user pytree from value_and_grad_fn (has_aux)


# ---------------------------------------------------------------------------
# Building blocks (functional API)
# ---------------------------------------------------------------------------

def _xmami(xmin, xmax):
    return np.maximum(xmax - xmin, _XMAMI_EPS)


def update_asymptotes(iteration, x, xold1, xold2, xmin, xmax, low, upp, *,
                      asyinit=0.5, asyincr=1.2, asydecr=0.7,
                      asymin=0.01, asymax=10.0):
    """Moving asymptotes update (``asymp.m`` / first part of ``mmasub.m``).

    ``iteration`` is the 1-based outer iteration counter; for the first two
    iterations the asymptotes are initialised at ``x -/+ asyinit*(xmax-xmin)``.
    """
    xmami = _xmami(xmin, xmax)
    low_init = x - asyinit * xmami
    upp_init = x + asyinit * xmami

    zzz = (x - xold1) * (xold1 - xold2)
    factor = np.where(zzz > 0, asyincr, np.where(zzz < 0, asydecr, 1.0))
    low_new = x - factor * (xold1 - low)
    upp_new = x + factor * (upp - xold1)
    low_new = np.clip(low_new, x - asymax * xmami, x - asymin * xmami)
    upp_new = np.clip(upp_new, x + asymin * xmami, x + asymax * xmami)

    first = iteration < 3
    low = np.where(first, low_init, low_new)
    upp = np.where(first, upp_init, upp_new)
    return low, upp


def move_bounds(x, xmin, xmax, low, upp, *, albefa=0.1, move=None):
    """Subproblem box ``alfa <= x <= beta`` (``move=None`` → no move limit)."""
    alfa = low + albefa * (x - low)
    beta = upp - albefa * (upp - x)
    if move is not None:
        xmami = _xmami(xmin, xmax)
        alfa = np.maximum(alfa, x - move * xmami)
        beta = np.minimum(beta, x + move * xmami)
    alfa = np.maximum(alfa, xmin)
    beta = np.minimum(beta, xmax)
    return alfa, beta


def initial_raa(df0dx, dfdx, xmin, xmax, *, raa0eps=1e-6, raaeps=1e-6):
    """GCMMA initial ``raa0``, ``raa`` for an outer iteration (``asymp.m``)."""
    n = df0dx.shape[0]
    xmami = _xmami(xmin, xmax)
    raa0 = np.maximum(raa0eps, (0.1 / n) * (np.abs(df0dx) @ xmami))
    raa = np.maximum(raaeps, (0.1 / n) * (np.abs(dfdx) @ xmami))
    return raa0, raa


def build_subproblem(x, xmin, xmax, low, upp, alfa, beta,
                     f0, df0dx, fval, dfdx, raa0, raa) -> Subproblem:
    """Convex separable approximation (``gcmmasub.m`` / ``mmasub.m``).

    ``raa0`` (scalar) and ``raa`` (m,) are the curvature parameters; for
    plain MMA they are all ``1e-5``.
    """
    xmamiinv = 1.0 / _xmami(xmin, xmax)
    ux1 = upp - x
    xl1 = x - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    uxinv = 1.0 / ux1
    xlinv = 1.0 / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2
    r0 = f0 - p0 @ uxinv - q0 @ xlinv

    P = np.maximum(dfdx, 0.0)
    Q = np.maximum(-dfdx, 0.0)
    PQ = 0.001 * (P + Q) + raa[:, None] * xmamiinv[None, :]
    P = (P + PQ) * ux2[None, :]
    Q = (Q + PQ) * xl2[None, :]
    r = fval - P @ uxinv - Q @ xlinv

    return Subproblem(low, upp, alfa, beta, p0, q0, r0, P, Q, r)


def evaluate_approximation(sub: Subproblem, x):
    """Values ``(f0app, fapp)`` of the separable approximations at ``x``."""
    uxinv = 1.0 / (sub.upp - x)
    xlinv = 1.0 / (x - sub.low)
    f0app = sub.r0 + sub.p0 @ uxinv + sub.q0 @ xlinv
    fapp = sub.r + sub.P @ uxinv + sub.Q @ xlinv
    return f0app, fapp


def _pd_residual(pd: SubproblemSolution, sub: Subproblem, a0, a, c, d, epsi):
    """Perturbed KKT residual of the subproblem; returns ``(norm2, norminf)``."""
    x, y, z, lam, xsi, eta, mu, zet, s = pd
    ux1 = sub.upp - x
    xl1 = x - sub.low
    plam = sub.p0 + sub.P.T @ lam
    qlam = sub.q0 + sub.Q.T @ lam
    gvec = sub.P @ (1.0 / ux1) + sub.Q @ (1.0 / xl1)
    dpsidx = plam / (ux1 * ux1) - qlam / (xl1 * xl1)

    rex = dpsidx - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - a @ lam
    relam = gvec - a * z - y + s + sub.r        # b = -r
    rexsi = xsi * (x - sub.alfa) - epsi
    reeta = eta * (sub.beta - x) - epsi
    remu = mu * y - epsi
    rezet = zet * z - epsi
    res = lam * s - epsi

    residu = np.concatenate([
        rex, rey, np.atleast_1d(rez), relam, rexsi, reeta, remu,
        np.atleast_1d(rezet), res])
    return np.sqrt(residu @ residu), np.max(np.abs(residu))


def _pd_newton_direction(pd: SubproblemSolution, sub: Subproblem,
                         a0, a, c, d, epsi) -> SubproblemSolution:
    """One Newton direction of the perturbed KKT system (``subsolv.m``)."""
    x, y, z, lam, xsi, eta, mu, zet, s = pd
    m, n = sub.P.shape
    ux1 = sub.upp - x
    xl1 = x - sub.low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    ux3 = ux1 * ux2
    xl3 = xl1 * xl2
    uxinv1 = 1.0 / ux1
    xlinv1 = 1.0 / xl1
    uxinv2 = 1.0 / ux2
    xlinv2 = 1.0 / xl2
    xa = x - sub.alfa
    bx_ = sub.beta - x

    plam = sub.p0 + sub.P.T @ lam
    qlam = sub.q0 + sub.Q.T @ lam
    gvec = sub.P @ uxinv1 + sub.Q @ xlinv1
    GG = sub.P * uxinv2[None, :] - sub.Q * xlinv2[None, :]      # (m, n)
    dpsidx = plam / ux2 - qlam / xl2

    delx = dpsidx - epsi / xa + epsi / bx_
    dely = c + d * y - lam - epsi / y
    delz = a0 - a @ lam - epsi / z
    dellam = gvec - a * z - y + sub.r + epsi / lam

    diagx = 2.0 * (plam / ux3 + qlam / xl3) + xsi / xa + eta / bx_
    diagxinv = 1.0 / diagx
    diagy = d + mu / y
    diagyinv = 1.0 / diagy
    diaglam = s / lam
    diaglamyi = diaglam + diagyinv

    if m < n:
        blam = dellam + dely * diagyinv - GG @ (delx * diagxinv)
        bb = np.concatenate([blam, np.atleast_1d(delz)])
        Alam = np.diag(diaglamyi) + (GG * diagxinv[None, :]) @ GG.T
        AA = np.block([[Alam, a[:, None]],
                       [a[None, :], np.reshape(-zet / z, (1, 1))]])
        solut = np.linalg.solve(AA, bb)
        dlam = solut[:m]
        dz = solut[m]
        dx = -delx * diagxinv - (GG.T @ dlam) * diagxinv
    else:
        diaglamyiinv = 1.0 / diaglamyi
        dellamyi = dellam + dely * diagyinv
        Axx = np.diag(diagx) + GG.T @ (diaglamyiinv[:, None] * GG)
        azz = zet / z + a @ (a * diaglamyiinv)
        axz = -GG.T @ (a * diaglamyiinv)
        bx = delx + GG.T @ (dellamyi * diaglamyiinv)
        bz = delz - a @ (dellamyi * diaglamyiinv)
        AA = np.block([[Axx, axz[:, None]],
                       [axz[None, :], np.reshape(azz, (1, 1))]])
        bb = np.concatenate([-bx, np.atleast_1d(-bz)])
        solut = np.linalg.solve(AA, bb)
        dx = solut[:n]
        dz = solut[n]
        dlam = (GG @ dx) * diaglamyiinv - dz * (a * diaglamyiinv) \
            + dellamyi * diaglamyiinv

    dy = -dely * diagyinv + dlam * diagyinv
    dxsi = -xsi + epsi / xa - (xsi * dx) / xa
    deta = -eta + epsi / bx_ + (eta * dx) / bx_
    dmu = -mu + epsi / y - (mu * dy) / y
    dzet = -zet + epsi / z - zet * dz / z
    ds = -s + epsi / lam - (s * dlam) / lam
    return SubproblemSolution(dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds)


def solve_subproblem(sub: Subproblem, a0, a, c, d, *, epsimin=1e-7,
                     max_newton=200, max_linesearch=50) -> SubproblemSolution:
    """Primal-dual interior point solver for the subproblem (``subsolv.m``).

    Three nested ``lax.while_loop``s: barrier parameter ``epsi`` (1 → epsimin,
    factor 0.1), Newton iterations (until ``||res||_inf <= 0.9*epsi``), and a
    residual-norm halving line search.
    """
    alfa, beta = sub.alfa, sub.beta
    m = sub.P.shape[0]
    dtype = sub.p0.dtype
    one = np.asarray(1.0, dtype)

    x = 0.5 * (alfa + beta)
    y = np.ones(m, dtype)
    z = one
    lam = np.ones(m, dtype)
    xsi = np.maximum(1.0 / (x - alfa), 1.0)
    eta = np.maximum(1.0 / (beta - x), 1.0)
    mu = np.maximum(np.ones(m, dtype), 0.5 * c)
    zet = one
    s = np.ones(m, dtype)
    pd0 = SubproblemSolution(x, y, z, lam, xsi, eta, mu, zet, s)

    def residual(pd, epsi):
        return _pd_residual(pd, sub, a0, a, c, d, epsi)

    def newton_body(carry):
        ittt, pd, resnorm, resmax, epsi = carry
        dpd = _pd_newton_direction(pd, sub, a0, a, c, d, epsi)

        # Maximum step keeping strictly positive variables / inside (alfa, beta)
        xx = np.concatenate([pd.y, np.atleast_1d(pd.z), pd.lam, pd.xsi,
                             pd.eta, pd.mu, np.atleast_1d(pd.zet), pd.s])
        dxx = np.concatenate([dpd.y, np.atleast_1d(dpd.z), dpd.lam, dpd.xsi,
                              dpd.eta, dpd.mu, np.atleast_1d(dpd.zet), dpd.s])
        stmxx = np.max(-1.01 * dxx / xx)
        stmalfa = np.max(-1.01 * dpd.x / (pd.x - alfa))
        stmbeta = np.max(1.01 * dpd.x / (beta - pd.x))
        stminv = np.maximum(np.maximum(np.maximum(stmalfa, stmbeta), stmxx), 1.0)
        steg = 1.0 / stminv

        def ls_cond(c_):
            itto, _steg, _pd, resnew, _resmax = c_
            return (resnew > resnorm) & (itto < max_linesearch)

        def ls_body(c_):
            itto, steg_, _pd, _resnew, _resmax = c_
            pd_new = jax.tree_util.tree_map(lambda v, dv: v + steg_ * dv, pd, dpd)
            rn, rm = residual(pd_new, epsi)
            return itto + 1, 0.5 * steg_, pd_new, rn, rm

        _, _, pd_new, resnorm_new, resmax_new = lax.while_loop(
            ls_cond, ls_body, (0, steg, pd, 2.0 * resnorm, resmax))
        return ittt + 1, pd_new, resnorm_new, resmax_new, epsi

    def newton_cond(carry):
        ittt, _pd, _resnorm, resmax, epsi = carry
        return (resmax > 0.9 * epsi) & (ittt < max_newton)

    def epsi_body(carry):
        epsi, pd = carry
        resnorm, resmax = residual(pd, epsi)
        _, pd, _, _, _ = lax.while_loop(
            newton_cond, newton_body, (0, pd, resnorm, resmax, epsi))
        return 0.1 * epsi, pd

    def epsi_cond(carry):
        epsi, _ = carry
        return epsi > epsimin

    _, pd = lax.while_loop(epsi_cond, epsi_body, (one, pd0))
    return pd


def is_conservative(f0app, f0new, fapp, fnew, *, epsimin=1e-7):
    """``concheck.m``: are the approximations conservative at the new point?"""
    return (f0app + epsimin >= f0new) & np.all(fapp + epsimin >= fnew)


def update_raa(xnew, x, xmin, xmax, low, upp, f0new, fnew, f0app, fapp,
               raa0, raa, *, epsimin=1e-7, raacofmin=1e-12):
    """``raaupdate.m``: increase curvature of non-conservative approximations."""
    xmami = _xmami(xmin, xmax)
    xxux = (xnew - x) / (upp - xnew)
    xxxl = (xnew - x) / (xnew - low)
    ulxx = (upp - low) / xmami
    raacof = np.maximum(raacofmin, (xxux * xxxl) @ ulxx)

    raa0_new = np.minimum(1.1 * (raa0 + (f0new - f0app) / raacof), 10.0 * raa0)
    raa0 = np.where(f0new > f0app + 0.5 * epsimin, raa0_new, raa0)

    raa_new = np.minimum(1.1 * (raa + (fnew - fapp) / raacof), 10.0 * raa)
    raa = np.where(fnew > fapp + 0.5 * epsimin, raa_new, raa)
    return raa0, raa


def kkt_residual(sol: SubproblemSolution, xmin, xmax, df0dx, fval, dfdx,
                 a0, a, c, d):
    """``kktcheck.m``: KKT residual of the *original* problem at ``sol.x``
    using the subproblem multipliers.  Returns ``(norm2, norminf)``."""
    x, y, z, lam, xsi, eta, mu, zet, s = sol
    rex = df0dx + dfdx.T @ lam - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - a @ lam
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s
    residu = np.concatenate([
        rex, rey, np.atleast_1d(rez), relam, rexsi, reeta, remu,
        np.atleast_1d(rezet), res])
    return np.sqrt(residu @ residu), np.max(np.abs(residu))


# ---------------------------------------------------------------------------
# Optimiser classes
# ---------------------------------------------------------------------------

def _as_vec(v, m, dtype):
    return np.broadcast_to(np.asarray(v, dtype), (m,))


class MMA:
    """Method of Moving Asymptotes (Svanberg 1987, 2007 ``mmasub.m``).

    Parameters
    ----------
    value_and_grad_fn : callable
        ``fn(x, *args) -> (f0, df0dx, fval, dfdx)`` with shapes
        ``(), (n,), (m,), (m, n)``.  Constraints are ``fval <= 0``.
    xmin, xmax : array_like
        Variable bounds, broadcastable to ``(n,)``.
    n_constraints : int
        Number of constraints ``m`` (``fval`` may be empty when ``m = 0``).
    a0, a, c, d : float or array_like
        Svanberg's problem parameters (defaults ``1, 0, 1000, 1``).
    move : float or None
        Move limit as a fraction of ``xmax - xmin`` (``None`` = no limit).
    has_aux : bool
        If ``True``, ``value_and_grad_fn`` returns a fifth element -- any JAX
        pytree -- which is carried in ``MMAState.aux``.  Use it to get derived
        quantities (diagnostics, logged fields, intermediate solutions) out of
        the same evaluation that produced the gradients, instead of paying for
        a second one.
    asyinit, asyincr, asydecr : float
        Asymptote initialisation / expansion / contraction factors.
    epsimin : float
        Final barrier parameter of the subproblem solver.
    raa0 : float
        Fixed curvature parameter of plain MMA (``1e-5`` in ``mmasub.m``).
    """

    def __init__(self, value_and_grad_fn: Callable, xmin, xmax, *,
                 n_constraints: int, a0=1.0, a=0.0, c=1000.0, d=1.0,
                 move: Optional[float] = 0.5, asyinit=0.5, asyincr=1.2,
                 asydecr=0.7, albefa=0.1, epsimin=1e-7, raa0=1e-5,
                 max_newton=200, max_linesearch=50, dtype=None,
                 has_aux: bool = False):
        self.value_and_grad_fn = value_and_grad_fn
        self.has_aux = bool(has_aux)
        self.m = int(n_constraints)
        self.dtype = dtype or np.asarray(0.0).dtype
        self.xmin = np.asarray(xmin, self.dtype)
        self.xmax = np.asarray(xmax, self.dtype)
        self.a0 = np.asarray(a0, self.dtype)
        self.a = _as_vec(a, self.m, self.dtype)
        self.c = _as_vec(c, self.m, self.dtype)
        self.d = _as_vec(d, self.m, self.dtype)
        self.move = move
        self.asyinit, self.asyincr, self.asydecr = asyinit, asyincr, asydecr
        self.albefa = albefa
        self.epsimin = epsimin
        self.raa0_const = raa0
        self.max_newton = max_newton
        self.max_linesearch = max_linesearch

    # -- helpers -----------------------------------------------------------
    def _eval(self, x, *args):
        out = self.value_and_grad_fn(x, *args)
        if self.has_aux:
            f0, df0dx, fval, dfdx, aux = out
        else:
            f0, df0dx, fval, dfdx = out
            aux = None
        f0 = np.asarray(f0, self.dtype)
        df0dx = np.asarray(df0dx, self.dtype)
        fval = np.reshape(np.asarray(fval, self.dtype), (self.m,))
        dfdx = np.reshape(np.asarray(dfdx, self.dtype), (self.m, x.shape[0]))
        return f0, df0dx, fval, dfdx, aux

    def _bounds(self, n):
        xmin = np.broadcast_to(self.xmin, (n,))
        xmax = np.broadcast_to(self.xmax, (n,))
        return xmin, xmax

    def _solve(self, sub: Subproblem) -> SubproblemSolution:
        return solve_subproblem(sub, self.a0, self.a, self.c, self.d,
                                epsimin=self.epsimin,
                                max_newton=self.max_newton,
                                max_linesearch=self.max_linesearch)

    def _kkt(self, sol, xmin, xmax, df0dx, fval, dfdx):
        return kkt_residual(sol, xmin, xmax, df0dx, fval, dfdx,
                            self.a0, self.a, self.c, self.d)

    def _initial_raa(self, df0dx, dfdx, xmin, xmax):
        raa0 = np.asarray(self.raa0_const, self.dtype)
        return raa0, np.full((self.m,), self.raa0_const, self.dtype)

    # -- public API --------------------------------------------------------
    def init(self, x0, *args) -> MMAState:
        """Create the initial state (evaluates the functions at ``x0``)."""
        x = np.asarray(x0, self.dtype)
        n = x.shape[0]
        xmin, xmax = self._bounds(n)
        f0, df0dx, fval, dfdx, aux = self._eval(x, *args)
        raa0, raa = self._initial_raa(df0dx, dfdx, xmin, xmax)
        m = self.m
        sol = SubproblemSolution(
            x, np.zeros(m, self.dtype), np.zeros((), self.dtype),
            np.zeros(m, self.dtype), np.zeros(n, self.dtype),
            np.zeros(n, self.dtype), np.zeros(m, self.dtype),
            np.zeros((), self.dtype), np.zeros(m, self.dtype))
        inf = np.asarray(np.inf, self.dtype)
        return MMAState(x, x, x, xmin, xmax, raa0, raa, f0, df0dx, fval, dfdx,
                        sol, np.zeros((), np.int32), np.zeros((), np.int32),
                        inf, inf,
                        f0, fval, aux)

    def step(self, state: MMAState, *args) -> MMAState:
        """One outer MMA iteration."""
        n = state.x.shape[0]
        xmin, xmax = self._bounds(n)
        it = state.iteration + np.int32(1)
        low, upp = update_asymptotes(
            it, state.x, state.xold1, state.xold2, xmin, xmax,
            state.low, state.upp, asyinit=self.asyinit,
            asyincr=self.asyincr, asydecr=self.asydecr)
        alfa, beta = move_bounds(state.x, xmin, xmax, low, upp,
                                 albefa=self.albefa, move=self.move)
        raa0, raa = self._initial_raa(state.df0dx, state.dfdx, xmin, xmax)
        sub = build_subproblem(state.x, xmin, xmax, low, upp, alfa, beta,
                               state.f0, state.df0dx, state.fval, state.dfdx,
                               raa0, raa)
        sol = self._solve(sub)
        f0app, fapp = evaluate_approximation(sub, sol.x)
        f0, df0dx, fval, dfdx, aux = self._eval(sol.x, *args)
        kkt_norm, kkt_max = self._kkt(sol, xmin, xmax, df0dx, fval, dfdx)
        return MMAState(sol.x, state.x, state.xold1, low, upp, raa0, raa,
                        f0, df0dx, fval, dfdx, sol, it, np.zeros((), np.int32),
                        kkt_norm, kkt_max, f0app, fapp, aux)

    def _converged(self, state: MMAState, prev_x, prev_f0,
                   kkt_tol, xtol, ftol):
        """Traced stopping test (all tolerances are static Python floats)."""
        done = np.asarray(False)
        if kkt_tol > 0:
            done = done | (state.kkt_norm < kkt_tol)
        if xtol > 0:
            scale = np.maximum(1.0, np.max(np.abs(state.x)))
            done = done | (np.max(np.abs(state.x - prev_x)) < xtol * scale)
        if ftol > 0:
            scale = np.maximum(1.0, np.abs(state.f0))
            done = done | (np.abs(state.f0 - prev_f0) < ftol * scale)
        return done

    def solve(self, x0, *args, max_iter: int = 100, kkt_tol: float = 0.0,
              xtol: float = 0.0, ftol: float = 0.0) -> MMAState:
        """Fully traced driver — the entire optimisation is one XLA program.

        The outer loop is a ``lax.while_loop``, so the compiled graph is
        independent of ``max_iter`` and the whole solve runs without returning
        to Python.  ``solve`` composes with ``jax.jit``, ``jax.vmap`` and
        ``jax.pmap``; the user functions must therefore be JAX-traceable.

        Parameters
        ----------
        max_iter : int
            Hard cap on outer iterations (static).
        kkt_tol, xtol, ftol : float
            Stopping tests, all disabled (``0.0``) by default and identical to
            those of :meth:`run`.  They are static: passing ``0.0`` removes
            the test from the traced condition entirely.

        Returns
        -------
        MMAState
            Final state; ``state.iteration`` reports how many outer iterations
            were actually performed.
        """
        state = self.init(x0, *args)
        inf = np.full_like(state.f0, np.inf)

        def cond(carry):
            st, prev_x, prev_f0 = carry
            running = st.iteration < np.int32(max_iter)
            return running & ~self._converged(st, prev_x, prev_f0,
                                              kkt_tol, xtol, ftol)

        def body(carry):
            st, _, _ = carry
            return self.step(st, *args), st.x, st.f0

        state, _, _ = lax.while_loop(
            cond, body, (state, np.full_like(state.x, np.inf), inf))
        return state

    def solve_with_history(self, x0, *args, n_iter: int = 100
                           ) -> Tuple[MMAState, Dict[str, Array]]:
        """Traced driver with per-iteration history (``lax.scan``).

        Runs exactly ``n_iter`` outer iterations (no early exit — that is what
        ``scan`` buys) and returns the final state plus stacked arrays of
        length ``n_iter`` for ``objective``, ``constraints``, ``kkt_norm``,
        ``kkt_max`` and ``inner_iterations``.  Useful for convergence plots
        and CSV logs without breaking out of the compiled program.
        """
        state = self.init(x0, *args)

        def body(st, _):
            st = self.step(st, *args)
            return st, {
                "objective": st.f0,
                "constraints": st.fval,
                "kkt_norm": st.kkt_norm,
                "kkt_max": st.kkt_max,
                "inner_iterations": st.inner_iterations,
            }

        state, history = lax.scan(body, state, None, length=n_iter)
        return state, history

    def run(self, x0, *args, max_iter=100, kkt_tol=0.0, xtol=0.0, ftol=0.0,
            jit=True,
            callback: Optional[Callable[[MMAState], None]] = None) -> MMAState:
        """Python-loop driver, for callbacks and non-JAX callables.

        Prefer :meth:`solve` unless you need to run Python between iterations
        (VTU snapshots, logging, continuation, adaptive remeshing) or your
        ``value_and_grad_fn`` is not traceable.

        Iterates :meth:`step` until ``max_iter`` is reached or one of the
        (disabled by default) stopping tests fires:

        * ``kkt_tol`` — ``kkt_norm < kkt_tol``.  Note that ``kkt_norm`` floors
          at ``O(epsimin)``, so a tolerance below that never triggers.
        * ``xtol`` — ``max|x - x_prev| < xtol * max(1, max|x|)``.
        * ``ftol`` — ``|f0 - f0_prev| < ftol * max(1, |f0|)``.

        ``callback`` receives the state after every outer iteration.
        """
        step = jax.jit(self.step) if jit else self.step
        state = self.init(x0, *args)
        for _ in range(max_iter):
            prev_x, prev_f0 = state.x, state.f0
            state = step(state, *args)
            if callback is not None:
                callback(state)
            if kkt_tol > 0 and float(state.kkt_norm) < kkt_tol:
                break
            if xtol > 0:
                dx = float(np.max(np.abs(state.x - prev_x)))
                if dx < xtol * max(1.0, float(np.max(np.abs(state.x)))):
                    break
            if ftol > 0:
                df = float(np.abs(state.f0 - prev_f0))
                if df < ftol * max(1.0, float(np.abs(state.f0))):
                    break
        return state


class GCMMA(MMA):
    """Globally Convergent MMA (Svanberg 2002; 2007 ``gcmmasub.m`` et al.).

    Same interface as :class:`MMA`.  Each outer iteration solves the
    subproblem, evaluates the functions at the candidate and, while the
    approximations are not conservative (``concheck``), increases the
    curvature parameters (``raaupdate``) and re-solves — up to
    ``max_inner`` times.

    Parameters (in addition to :class:`MMA`)
    ----------
    value_fn : callable or None
        ``fn(x, *args) -> (f0, fval)`` used for the inner-loop evaluations.
        If ``None`` (default) the inner loop calls ``value_and_grad_fn`` and
        keeps the gradients of the accepted point, so no extra evaluation is
        needed at the end of the outer iteration.  Supply ``value_fn`` when a
        value-only evaluation is much cheaper than value-and-gradient.
    max_inner : int
        Maximum number of inner iterations (15 in the reference driver).
    raa0eps, raaeps : float
        Lower bounds of the curvature parameters.
    """

    def __init__(self, value_and_grad_fn: Callable, xmin, xmax, *,
                 n_constraints: int, value_fn: Optional[Callable] = None,
                 max_inner: int = 15, raa0eps=1e-6, raaeps=1e-6,
                 move: Optional[float] = None, **kwargs):
        super().__init__(value_and_grad_fn, xmin, xmax,
                         n_constraints=n_constraints, move=move, **kwargs)
        self.value_fn = value_fn
        self.max_inner = int(max_inner)
        self.raa0eps = raa0eps
        self.raaeps = raaeps

    def _initial_raa(self, df0dx, dfdx, xmin, xmax):
        return initial_raa(df0dx, dfdx, xmin, xmax,
                           raa0eps=self.raa0eps, raaeps=self.raaeps)

    def _eval_values(self, x, *args):
        if self.value_fn is None:
            return self._eval(x, *args)
        f0, fval = self.value_fn(x, *args)
        f0 = np.asarray(f0, self.dtype)
        fval = np.reshape(np.asarray(fval, self.dtype), (self.m,))
        df0dx = np.zeros_like(x)
        dfdx = np.zeros((self.m, x.shape[0]), self.dtype)
        return f0, df0dx, fval, dfdx, None

    def step(self, state: MMAState, *args) -> MMAState:
        """One outer GCMMA iteration (with inner conservativity loop)."""
        n = state.x.shape[0]
        xmin, xmax = self._bounds(n)
        it = state.iteration + np.int32(1)
        low, upp = update_asymptotes(
            it, state.x, state.xold1, state.xold2, xmin, xmax,
            state.low, state.upp, asyinit=self.asyinit,
            asyincr=self.asyincr, asydecr=self.asydecr)
        alfa, beta = move_bounds(state.x, xmin, xmax, low, upp,
                                 albefa=self.albefa, move=self.move)
        raa0, raa = self._initial_raa(state.df0dx, state.dfdx, xmin, xmax)

        def solve_and_eval(raa0, raa):
            sub = build_subproblem(
                state.x, xmin, xmax, low, upp, alfa, beta, state.f0,
                state.df0dx, state.fval, state.dfdx, raa0, raa)
            sol = self._solve(sub)
            f0app, fapp = evaluate_approximation(sub, sol.x)
            vals = self._eval_values(sol.x, *args)
            conserv = is_conservative(f0app, vals[0], fapp, vals[2],
                                      epsimin=self.epsimin)
            return sol, f0app, fapp, vals, conserv

        sol, f0app, fapp, vals, conserv = solve_and_eval(raa0, raa)

        if self.max_inner > 0:
            def inner_cond(carry):
                innerit, _raa0, _raa, _sol, _f0app, _fapp, _vals, conserv = carry
                return (~conserv) & (innerit < self.max_inner)

            def inner_body(carry):
                innerit, raa0, raa, sol, f0app, fapp, vals, _ = carry
                raa0, raa = update_raa(
                    sol.x, state.x, xmin, xmax, low, upp, vals[0], vals[2],
                    f0app, fapp, raa0, raa, epsimin=self.epsimin)
                sol, f0app, fapp, vals, conserv = solve_and_eval(raa0, raa)
                return (innerit + np.int32(1), raa0, raa, sol, f0app, fapp,
                        vals, conserv)

            innerit, raa0, raa, sol, f0app, fapp, vals, conserv = lax.while_loop(
                inner_cond, inner_body,
                (np.zeros((), np.int32), raa0, raa, sol, f0app, fapp,
                 vals, conserv))
        else:
            innerit = np.zeros((), np.int32)

        if self.value_fn is None:
            f0, df0dx, fval, dfdx, aux = vals
        else:
            f0, df0dx, fval, dfdx, aux = self._eval(sol.x, *args)
        kkt_norm, kkt_max = self._kkt(sol, xmin, xmax, df0dx, fval, dfdx)
        return MMAState(sol.x, state.x, state.xold1, low, upp, raa0, raa,
                        f0, df0dx, fval, dfdx, sol, it, innerit,
                        kkt_norm, kkt_max, f0app, fapp, aux)


# ---------------------------------------------------------------------------
# A batch of independent problems, evaluated together
# ---------------------------------------------------------------------------

def _select(active, new, old):
    """Per-case selection of pytree leaves with a leading batch axis."""
    def pick(n, o):
        n = np.asarray(n)
        if n.ndim == 0 or n.shape[0] != active.shape[0]:
            return n
        return np.where(active.reshape((-1,) + (1,) * (n.ndim - 1)), n, o)
    return jax.tree_util.tree_map(pick, new, old)


class BatchedGCMMA(GCMMA):
    """GCMMA over a batch of ``B`` independent problems that are *evaluated together*.

    Use it when the ``B`` objective evaluations are cheapest as one call — the
    designs of a :func:`feax.vmap` batch, whose ``B`` finite-element solves
    share one factorization. ``jax.vmap`` of :class:`GCMMA` would evaluate
    every design on its own; here every evaluation is one call on the whole
    batch, while the subproblems, asymptotes and the conservativity loop are
    per problem (vectorized, with the inner loop masked so that a problem whose
    approximation is already conservative keeps its candidate while the others
    re-solve).

    Parameters
    ----------
    value_and_grad_fn : callable
        ``fn(X, *args) -> (f0, df0dx, fval, dfdx)`` on the batch, with shapes
        ``(B,), (B, n), (B, m), (B, m, n)`` (``+ aux`` with ``has_aux``);
        constraints are ``fval <= 0`` per problem.
    value_fn : callable or None
        ``fn(X, *args) -> (f0, fval)`` on the batch, for the inner loop.
    xmin, xmax, n_constraints, ... :
        As for :class:`GCMMA`; the bounds are shared by every problem.

    The state is an :class:`MMAState` whose arrays carry a leading batch
    axis (``iteration`` is shared). :meth:`solve`, :meth:`solve_with_history`
    and :meth:`run` work unchanged; a stopping test fires when *every*
    problem passes it.

    Example — with ``feax.vmap``::

        batched = fe.vmap(solver, in_axes=..., preconditioner=fe.shared_factor())

        def value_and_grad(X, data):             # X: (B, n)
            (c, aux), dc = jax.value_and_grad(lambda X: compliances(X, data), has_aux=True)(X)
            ...                                   # one batched solve for all B
            return c / c0, dc / c0, g, dg         # (B,), (B, n), (B, m), (B, m, n)

        opt = BatchedGCMMA(value_and_grad, 0.0, 1.0, n_constraints=1)
        state = opt.solve(X0, data, max_iter=100)
    """

    # -- shapes ------------------------------------------------------------
    def _eval(self, X, *args):
        out = self.value_and_grad_fn(X, *args)
        if self.has_aux:
            f0, df0dx, fval, dfdx, aux = out
        else:
            f0, df0dx, fval, dfdx = out
            aux = None
        B, n = X.shape
        f0 = np.reshape(np.asarray(f0, self.dtype), (B,))
        df0dx = np.reshape(np.asarray(df0dx, self.dtype), (B, n))
        fval = np.reshape(np.asarray(fval, self.dtype), (B, self.m))
        dfdx = np.reshape(np.asarray(dfdx, self.dtype), (B, self.m, n))
        return f0, df0dx, fval, dfdx, aux

    def _eval_values(self, X, *args):
        if self.value_fn is None:
            return self._eval(X, *args)
        f0, fval = self.value_fn(X, *args)
        B, n = X.shape
        f0 = np.reshape(np.asarray(f0, self.dtype), (B,))
        fval = np.reshape(np.asarray(fval, self.dtype), (B, self.m))
        return (f0, np.zeros((B, n), self.dtype), fval,
                np.zeros((B, self.m, n), self.dtype), None)

    # -- per-problem building blocks, vectorized over the batch ------------
    def _v_asymptotes(self, it, x, xold1, xold2, xmin, xmax, low, upp):
        fn = lambda x, x1, x2, lo, up: update_asymptotes(
            it, x, x1, x2, xmin, xmax, lo, up, asyinit=self.asyinit,
            asyincr=self.asyincr, asydecr=self.asydecr)
        return jax.vmap(fn)(x, xold1, xold2, low, upp)

    def _v_move_bounds(self, x, xmin, xmax, low, upp):
        fn = lambda x, lo, up: move_bounds(x, xmin, xmax, lo, up, albefa=self.albefa, move=self.move)
        return jax.vmap(fn)(x, low, upp)

    def _v_initial_raa(self, df0dx, dfdx, xmin, xmax):
        return jax.vmap(lambda d0, d: self._initial_raa(d0, d, xmin, xmax))(df0dx, dfdx)

    def _v_solve(self, x, xmin, xmax, low, upp, alfa, beta, f0, df0dx, fval, dfdx, raa0, raa):
        def one(x, lo, up, al, be, f0, d0, fv, d, r0, r):
            sub = build_subproblem(x, xmin, xmax, lo, up, al, be, f0, d0, fv, d, r0, r)
            sol = self._solve(sub)
            f0app, fapp = evaluate_approximation(sub, sol.x)
            return sol, f0app, fapp
        return jax.vmap(one)(x, low, upp, alfa, beta, f0, df0dx, fval, dfdx, raa0, raa)

    def _v_kkt(self, sol, xmin, xmax, df0dx, fval, dfdx):
        return jax.vmap(lambda s, d0, fv, d: self._kkt(s, xmin, xmax, d0, fv, d))(sol, df0dx, fval, dfdx)

    # -- public API --------------------------------------------------------
    def init(self, X0, *args) -> MMAState:
        """Create the initial state (one evaluation of the whole batch)."""
        X = np.asarray(X0, self.dtype)
        if X.ndim != 2:
            raise ValueError(f"BatchedGCMMA needs a batch of designs (B, n), got shape {X.shape}.")
        B, n = X.shape
        xmin, xmax = self._bounds(n)
        f0, df0dx, fval, dfdx, aux = self._eval(X, *args)
        raa0, raa = self._v_initial_raa(df0dx, dfdx, xmin, xmax)
        m, dt = self.m, self.dtype
        sol = SubproblemSolution(
            X, np.zeros((B, m), dt), np.zeros((B,), dt), np.zeros((B, m), dt),
            np.zeros((B, n), dt), np.zeros((B, n), dt), np.zeros((B, m), dt),
            np.zeros((B,), dt), np.zeros((B, m), dt))
        inf = np.full((B,), np.inf, dt)
        low = np.broadcast_to(xmin, (B, n))
        upp = np.broadcast_to(xmax, (B, n))
        return MMAState(X, X, X, low, upp, raa0, raa, f0, df0dx, fval, dfdx,
                        sol, np.zeros((), np.int32), np.zeros((B,), np.int32),
                        inf, inf, f0, fval, aux)

    def step(self, state: MMAState, *args) -> MMAState:
        """One outer iteration of every problem, with a masked inner loop."""
        B, n = state.x.shape
        xmin, xmax = self._bounds(n)
        it = state.iteration + np.int32(1)
        low, upp = self._v_asymptotes(it, state.x, state.xold1, state.xold2, xmin, xmax,
                                      state.low, state.upp)
        alfa, beta = self._v_move_bounds(state.x, xmin, xmax, low, upp)
        raa0, raa = self._v_initial_raa(state.df0dx, state.dfdx, xmin, xmax)

        def solve_and_eval(raa0, raa):
            sol, f0app, fapp = self._v_solve(
                state.x, xmin, xmax, low, upp, alfa, beta, state.f0, state.df0dx,
                state.fval, state.dfdx, raa0, raa)
            vals = self._eval_values(sol.x, *args)                  # ONE batched evaluation
            conserv = jax.vmap(lambda a, b, c, d: is_conservative(a, b, c, d, epsimin=self.epsimin))(
                f0app, vals[0], fapp, vals[2])
            return sol, f0app, fapp, vals, conserv

        sol, f0app, fapp, vals, conserv = solve_and_eval(raa0, raa)
        innerit = np.zeros((B,), np.int32)

        if self.max_inner > 0:
            def active_of(innerit, conserv):
                return (~conserv) & (innerit < self.max_inner)

            def inner_cond(carry):
                innerit, _raa0, _raa, _sol, _f0app, _fapp, _vals, conserv = carry
                return np.any(active_of(innerit, conserv))

            def inner_body(carry):
                innerit, raa0, raa, sol, f0app, fapp, vals, conserv = carry
                active = active_of(innerit, conserv)
                raa0_n, raa_n = jax.vmap(
                    lambda xn, x, lo, up, f0n, fn, f0a, fa, r0, r: update_raa(
                        xn, x, xmin, xmax, lo, up, f0n, fn, f0a, fa, r0, r, epsimin=self.epsimin)
                )(sol.x, state.x, low, upp, vals[0], vals[2], f0app, fapp, raa0, raa)
                raa0 = np.where(active, raa0_n, raa0)
                raa = np.where(active[:, None], raa_n, raa)
                sol_n, f0app_n, fapp_n, vals_n, conserv_n = solve_and_eval(raa0, raa)
                # problems that were already conservative keep their candidate
                sol = _select(active, sol_n, sol)
                f0app = np.where(active, f0app_n, f0app)
                fapp = np.where(active[:, None], fapp_n, fapp)
                vals = _select(active, vals_n, vals)
                conserv = np.where(active, conserv_n, conserv)
                return (innerit + active.astype(np.int32), raa0, raa, sol, f0app, fapp,
                        vals, conserv)

            innerit, raa0, raa, sol, f0app, fapp, vals, conserv = lax.while_loop(
                inner_cond, inner_body,
                (innerit, raa0, raa, sol, f0app, fapp, vals, conserv))

        if self.value_fn is None:
            f0, df0dx, fval, dfdx, aux = vals
        else:
            f0, df0dx, fval, dfdx, aux = self._eval(sol.x, *args)
        kkt_norm, kkt_max = self._v_kkt(sol, xmin, xmax, df0dx, fval, dfdx)
        return MMAState(sol.x, state.x, state.xold1, low, upp, raa0, raa,
                        f0, df0dx, fval, dfdx, sol, it, innerit,
                        kkt_norm, kkt_max, f0app, fapp, aux)

    def _converged(self, state: MMAState, prev_x, prev_f0, kkt_tol, xtol, ftol):
        """Stop when every problem passes the (static) tests."""
        done = np.zeros(state.x.shape[0], bool)
        if kkt_tol > 0:
            done = done | (state.kkt_norm < kkt_tol)
        if xtol > 0:
            scale = np.maximum(1.0, np.max(np.abs(state.x), axis=1))
            done = done | (np.max(np.abs(state.x - prev_x), axis=1) < xtol * scale)
        if ftol > 0:
            scale = np.maximum(1.0, np.abs(state.f0))
            done = done | (np.abs(state.f0 - prev_f0) < ftol * scale)
        return np.all(done)

    def run(self, X0, *args, max_iter=100, kkt_tol=0.0, xtol=0.0, ftol=0.0,
            jit=True, callback: Optional[Callable[[MMAState], None]] = None) -> MMAState:
        """Python-loop driver (see :meth:`MMA.run`); stops when every problem
        passes a stopping test."""
        step = jax.jit(self.step) if jit else self.step
        state = self.init(X0, *args)
        for _ in range(max_iter):
            prev_x, prev_f0 = state.x, state.f0
            state = step(state, *args)
            if callback is not None:
                callback(state)
            if bool(self._converged(state, prev_x, prev_f0, kkt_tol, xtol, ftol)):
                break
        return state
