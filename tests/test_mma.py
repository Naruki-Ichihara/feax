"""Verification of :mod:`feax.experimental.mma` (JAX MMA / GCMMA).

The implementation is checked on four independent levels:

1. **Subproblem solver** — the primal-dual interior point solver is checked
   against the closed form of Svanberg's convex separable subproblem
   (``x(lambda)``, ``y(lambda)``), which can be written down exactly.  This
   validates both branches (``m < n`` and ``m >= n``) of the Newton step.
2. **Exact optima** — the outer iteration is checked against optima derived
   analytically (closed form or an exactly solved KKT system), including the
   ``O(epsimin)`` barrier-accuracy scaling law.
3. **NLopt** — results are cross-checked against ``nlopt.LD_MMA``
   (Svanberg's CCSA/MMA as implemented by S. G. Johnson) on the same problems
   and on randomly generated ones.
4. **Tracing** — the traced drivers (``solve``, ``solve_with_history``) are
   checked to agree with an explicit step loop, to compile to a graph whose
   size does not depend on the iteration count, and to compose with ``vmap``.

On tolerances: on every problem tested here ``nlopt.LD_MMA`` terminates at a
point that violates the constraints by ``1e-7``-ish and sits ``1e-5``-ish away
from the exact optimum, whereas this implementation terminates strictly
feasible and within ``O(epsimin)`` of the exact optimum.  NLopt's marginally
lower objective is accounted for by its constraint violation.  The NLopt
cross-checks are therefore deliberately looser than the comparisons against
the exact optima.
"""

import jax
import jax.numpy as np
import numpy as onp
import pytest

nlopt = pytest.importorskip("nlopt")

from feax.experimental.mma import (  # noqa: E402
    GCMMA,
    MMA,
    build_subproblem,
    evaluate_approximation,
    initial_raa,
    move_bounds,
    solve_subproblem,
    update_asymptotes,
)

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------
# Reference optima
#
# TOY  : min |x|^2 s.t. two ball constraints (Svanberg's GCMMA "toy" example).
#        Both constraints are active; the 5x5 KKT system was solved to machine
#        precision (residual 4.4e-16), giving lam = (0.42623975, 0.75957309).
# BEAM : Svanberg (1987) 5-variable cantilever beam.  With the single
#        constraint active the KKT system has the closed form
#        x_j = k * c_j**(1/4),  k = (sum_j c_j**(1/4))**(1/3).
# MCON : min |x - (2,2)|^2 over a polytope; the optimum is the projection of
#        (2, 2) onto x0 + x1 <= 2, i.e. exactly (1, 1).
# ---------------------------------------------------------------------------

TOY_XSTAR = onp.array([2.0175185856761395, 1.7800114373391627, 1.2375071483369766])
TOY_F0 = 8.770245902791999
BEAM_C = onp.array([61.0, 37.0, 19.0, 7.0, 1.0])
BEAM_XSTAR = (BEAM_C ** 0.25).sum() ** (1 / 3) * BEAM_C ** 0.25
BEAM_F0 = 0.0624 * BEAM_XSTAR.sum()
MCON_XSTAR = onp.array([1.0, 1.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_value_and_grad(f0_fn, f_fn):
    """Build ``x -> (f0, df0, f, df)`` from scalar/vector JAX functions."""
    def vg(x, *args):
        f0, df0 = jax.value_and_grad(f0_fn)(x, *args)
        f = f_fn(x, *args)
        df = jax.jacfwd(f_fn)(x, *args)
        return f0, df0, f, df
    return vg


def run_nlopt(vg, x0, xmin, xmax, m, maxeval=20000, xtol=1e-12):
    """Solve the same problem with NLopt's LD_MMA."""
    n = x0.shape[0]
    xmin = onp.broadcast_to(onp.asarray(xmin, float), (n,)).copy()
    xmax = onp.broadcast_to(onp.asarray(xmax, float), (n,)).copy()
    vg_j = jax.jit(vg)

    def obj(x, grad):
        f0, df0, _, _ = vg_j(np.asarray(x))
        if grad.size > 0:
            grad[:] = onp.asarray(df0)
        return float(f0)

    def con(result, x, grad):
        _, _, f, df = vg_j(np.asarray(x))
        if grad.size > 0:
            grad[:, :] = onp.asarray(df)
        result[:] = onp.asarray(f)

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_lower_bounds(xmin)
    opt.set_upper_bounds(xmax)
    opt.set_min_objective(obj)
    if m > 0:
        opt.add_inequality_mconstraint(con, 1e-12 * onp.ones(m))
    opt.set_xtol_rel(xtol)
    opt.set_maxeval(maxeval)
    x = opt.optimize(onp.asarray(x0, float))
    return x, opt.last_optimum_value()


def solve_jax(cls, vg, x0, xmin, xmax, m, n_iter=200, epsimin=1e-11, **kw):
    """Run the traced driver for a fixed number of outer iterations."""
    opt = cls(vg, xmin, xmax, n_constraints=m, epsimin=epsimin, **kw)
    return opt.solve(np.asarray(x0), max_iter=n_iter)


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

def toy_problem():
    """n = 3, m = 2."""
    def f0(x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def f(x):
        return np.stack([
            (x[0] - 5) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 - 9,
            (x[0] - 3) ** 2 + (x[1] - 4) ** 2 + (x[2] - 3) ** 2 - 9,
        ])
    return make_value_and_grad(f0, f), np.array([4.0, 3.0, 2.0]), 0.0, 5.0, 2


def beam_problem():
    """n = 5, m = 1 (single active nonlinear constraint)."""
    cvec = np.asarray(BEAM_C)

    def f0(x):
        return 0.0624 * np.sum(x)

    def f(x):
        return np.array([np.sum(cvec / x ** 3) - 1.0])
    return make_value_and_grad(f0, f), 5.0 * np.ones(5), 1.0, 10.0, 1


def compliance_like_problem(n=500, seed=0, V=0.3, lo=0.01, hi=1.0):
    """min mean(c_j / x_j) s.t. mean(x) <= V, lo <= x <= hi.

    Separable with an analytic solution ``x_j = clip(sqrt(c_j / mu), lo, hi)``
    where ``mu`` is fixed by ``mean(x) = V``.  Many bounds are active at the
    optimum, which exercises the ``alfa``/``beta`` handling.

    The objective is averaged (not summed) so that the constraint multiplier
    stays far below the elastic penalty ``c = 1000``; see the scaling warning
    in the module docstring.
    """
    rng = onp.random.default_rng(seed)
    cvec = np.asarray(rng.uniform(0.1, 1.0, n))

    def f0(x):
        return np.mean(cvec / x)

    def f(x):
        return np.array([np.mean(x) - V])

    c = onp.asarray(cvec)
    a, b = 1e-12, 1e12
    for _ in range(400):                      # bisection on log(mu)
        mu = onp.sqrt(a * b)
        xs = onp.clip(onp.sqrt(c / mu), lo, hi)
        if xs.mean() > V:
            a = mu
        else:
            b = mu
    return make_value_and_grad(f0, f), 0.5 * np.ones(n), lo, hi, 1, xs


def many_constraints_problem():
    """n = 2, m = 4 — exercises the ``m >= n`` branch of the Newton step."""
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
    b = np.array([2.0, 1.5, 1.5, 1.0])

    def f0(x):
        return (x[0] - 2) ** 2 + (x[1] - 2) ** 2

    def f(x):
        return A @ x - b
    return make_value_and_grad(f0, f), np.array([0.2, 0.3]), 0.0, 3.0, 4


def random_problem(n, m, seed):
    """Random strictly convex QP with random linear constraints.

    Scaled so the multipliers stay well below the elastic penalty.
    """
    r = onp.random.default_rng(seed)
    B = r.normal(size=(n, n)) / onp.sqrt(n)
    H = np.asarray(B @ B.T + onp.eye(n))
    g = np.asarray(r.normal(size=n) / onp.sqrt(n))
    A = np.asarray(r.normal(size=(m, n)) / onp.sqrt(n))
    b = np.asarray(r.uniform(-0.2, 0.2, m))

    def f0(x):
        return 0.5 * x @ (H @ x) + g @ x

    def f(x):
        return A @ x - b
    return make_value_and_grad(f0, f), np.zeros(n), -1.0, 1.0, m


# ---------------------------------------------------------------------------
# 1. Subproblem solver
# ---------------------------------------------------------------------------

def _random_subproblem(n, m, seed):
    r = onp.random.default_rng(seed)
    xmin = np.asarray(r.uniform(-2.0, -0.5, n))
    xmax = np.asarray(r.uniform(0.5, 2.0, n))
    x = np.asarray(r.uniform(onp.asarray(xmin) * 0.5, onp.asarray(xmax) * 0.5))
    f0 = np.asarray(r.normal())
    df0 = np.asarray(r.normal(size=n))
    fval = np.asarray(r.normal(size=m) * 0.5)
    dfdx = np.asarray(r.normal(size=(m, n)))
    low, upp = update_asymptotes(1, x, x, x, xmin, xmax, xmin, xmax)
    alfa, beta = move_bounds(x, xmin, xmax, low, upp)
    raa0, raa = initial_raa(df0, dfdx, xmin, xmax)
    return build_subproblem(x, xmin, xmax, low, upp, alfa, beta,
                            f0, df0, fval, dfdx, raa0, raa)


@pytest.mark.parametrize("n,m", [(3, 2), (5, 1), (2, 4), (8, 8), (20, 3),
                                 (4, 12), (100, 5), (30, 0)])
@pytest.mark.parametrize("a_scale", [0.0, 1.0])
def test_subproblem_matches_closed_form(n, m, a_scale):
    """The interior point solution must satisfy the exact primal map.

    For the convex separable subproblem, minimising over ``x`` for fixed duals
    ``lambda`` has the closed form

        x_j = (sqrt(plam_j) * low_j + sqrt(qlam_j) * upp_j)
              / (sqrt(plam_j) + sqrt(qlam_j)),  clipped to [alfa, beta]

    with ``plam = p0 + P' lam``, ``qlam = q0 + Q' lam``; and minimising over
    ``y`` gives ``y_i = max(0, (lam_i - c_i) / d_i)``.  This is independent of
    how the duals were obtained, so it validates the Newton solver, including
    the ``m >= n`` branch and a non-zero ``a``.
    """
    sub = _random_subproblem(n, m, seed=n * 1000 + m * 10 + int(a_scale))
    a0 = np.asarray(1.0)
    a = np.full(m, a_scale / max(m, 1))
    c = np.full(m, 1000.0)
    d = np.ones(m)
    sol = solve_subproblem(sub, a0, a, c, d, epsimin=1e-13, max_newton=500)

    plam = sub.p0 + sub.P.T @ sol.lam
    qlam = sub.q0 + sub.Q.T @ sol.lam
    sp, sq = np.sqrt(plam), np.sqrt(qlam)
    x_cf = np.clip((sp * sub.low + sq * sub.upp) / (sp + sq), sub.alfa, sub.beta)
    assert float(np.max(np.abs(x_cf - sol.x))) < 1e-9

    if m:
        y_cf = np.maximum(0.0, (sol.lam - c) / d)
        assert float(np.max(np.abs(y_cf - sol.y))) < 1e-9
    # Dual feasibility for z: a0 - a'lam - zet = 0
    assert float(np.abs(a0 - a @ sol.lam - sol.zet)) < 1e-9


@pytest.mark.parametrize("n,m", [(3, 2), (2, 4), (20, 3), (4, 12)])
@pytest.mark.parametrize("epsimin", [1e-7, 1e-10, 1e-13])
def test_subproblem_kkt_residual_reaches_barrier_level(n, m, epsimin):
    """The unperturbed KKT residual of the subproblem is driven to ``epsimin``
    and the returned point is strictly feasible."""
    from feax.experimental.mma import _pd_residual

    sub = _random_subproblem(n, m, seed=7 * n + m)
    a0, a = np.asarray(1.0), np.zeros(m)
    c, d = np.full(m, 1000.0), np.ones(m)
    sol = solve_subproblem(sub, a0, a, c, d, epsimin=epsimin, max_newton=500)

    _, resmax = _pd_residual(sol, sub, a0, a, c, d, 0.0)
    assert float(resmax) <= 10 * epsimin
    assert onp.all(onp.asarray(sol.x) > onp.asarray(sub.alfa))
    assert onp.all(onp.asarray(sol.x) < onp.asarray(sub.beta))
    _, fapp = evaluate_approximation(sub, sol.x)
    assert float(np.max(fapp - a * sol.z - sol.y)) < 1e-9


# ---------------------------------------------------------------------------
# 2. Outer iteration against exact optima
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [GCMMA, MMA])
def test_toy_matches_exact_optimum(cls):
    vg, x0, xmin, xmax, m = toy_problem()
    st = solve_jax(cls, vg, x0, xmin, xmax, m)
    onp.testing.assert_allclose(onp.asarray(st.x), TOY_XSTAR, rtol=0, atol=1e-9)
    # f0 is not stationary at a constrained optimum, so its error is
    # |grad f0| times the position error, not its square.
    onp.testing.assert_allclose(float(st.f0), TOY_F0, rtol=1e-10)
    assert onp.all(onp.asarray(st.fval) <= 1e-9)


@pytest.mark.parametrize("cls", [GCMMA, MMA])
def test_beam_matches_exact_optimum(cls):
    vg, x0, xmin, xmax, m = beam_problem()
    st = solve_jax(cls, vg, x0, xmin, xmax, m)
    onp.testing.assert_allclose(onp.asarray(st.x), BEAM_XSTAR, rtol=0, atol=1e-8)
    onp.testing.assert_allclose(float(st.f0), BEAM_F0, rtol=1e-9)
    assert float(st.fval[0]) <= 1e-9


@pytest.mark.parametrize("cls", [GCMMA, MMA])
def test_many_constraints_matches_exact_optimum(cls):
    """``m >= n``: the optimum is the exact projection point (1, 1)."""
    vg, x0, xmin, xmax, m = many_constraints_problem()
    st = solve_jax(cls, vg, x0, xmin, xmax, m)
    onp.testing.assert_allclose(onp.asarray(st.x), MCON_XSTAR, rtol=0, atol=1e-8)
    assert onp.all(onp.asarray(st.fval) <= 1e-8)


@pytest.mark.parametrize("cls", [GCMMA, MMA])
def test_compliance_like_matches_analytic(cls):
    """500 variables with many active bounds; compared to the analytic solution."""
    vg, x0, xmin, xmax, m, x_exact = compliance_like_problem(n=500)
    st = solve_jax(cls, vg, x0, xmin, xmax, m, n_iter=300)
    onp.testing.assert_allclose(onp.asarray(st.x), x_exact, rtol=0, atol=1e-7)
    assert abs(float(st.fval[0])) < 1e-9


def test_accuracy_scales_with_barrier_parameter():
    """Interior point signature: the error is ``O(epsimin)``.

    Reducing ``epsimin`` by 1e-2 must reduce the distance to the exact
    optimum by about the same factor.  This is the sharpest evidence that the
    iteration converges to the true optimum rather than to a nearby point.
    """
    vg, x0, xmin, xmax, m = toy_problem()
    errors = {}
    for eps in (1e-7, 1e-9, 1e-11):
        st = solve_jax(GCMMA, vg, x0, xmin, xmax, m, n_iter=200, epsimin=eps)
        errors[eps] = float(onp.max(onp.abs(onp.asarray(st.x) - TOY_XSTAR)))
    assert errors[1e-7] < 1e-6
    assert 30 < errors[1e-7] / errors[1e-9] < 3000
    assert 30 < errors[1e-9] / errors[1e-11] < 3000


def test_unconstrained_bound_only():
    """m = 0: bound-constrained quadratic; two bounds active, one interior."""
    target = np.array([-1.0, 0.3, 2.5])

    def vg(x):
        return np.sum((x - target) ** 2), 2 * (x - target), np.zeros(0), np.zeros((0, 3))

    st = solve_jax(GCMMA, vg, np.zeros(3), 0.0, 1.0, 0, n_iter=120)
    onp.testing.assert_allclose(onp.asarray(st.x), [0.0, 0.3, 1.0], atol=1e-9)


def test_interior_stationary_point_accuracy_is_limited():
    """Documented limitation, faithful to the reference implementation.

    When the gradient vanishes at the solution (an interior, unconstrained
    stationary point), GCMMA's curvature estimate ``raa0`` collapses to its
    floor ``raa0eps`` and ``concheck``'s absolute ``epsimin`` slack makes the
    conservativity test vacuous, so the asymptote mechanism can no longer
    drive the error down.  Convergence is still to the right point, just at
    reduced accuracy -- roughly ``1e-5`` rather than ``O(epsimin)``.
    """
    target = np.array([0.2, 0.7])

    def vg(x):
        return np.sum((x - target) ** 2), 2 * (x - target), np.zeros(0), np.zeros((0, 2))

    st = solve_jax(GCMMA, vg, np.full(2, 0.5), 0.0, 1.0, 0, n_iter=200)
    err = float(onp.max(onp.abs(onp.asarray(st.x) - onp.asarray(target))))
    assert err < 1e-5
    # Raising the curvature floor restores full accuracy.
    st = solve_jax(GCMMA, vg, np.full(2, 0.5), 0.0, 1.0, 0, n_iter=200,
                   raa0eps=1.0, raaeps=1.0)
    assert float(onp.max(onp.abs(onp.asarray(st.x) - onp.asarray(target)))) < 1e-9


# ---------------------------------------------------------------------------
# 3. Cross-check against NLopt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [GCMMA, MMA])
@pytest.mark.parametrize("name", ["toy", "beam", "compliance"])
def test_matches_nlopt(cls, name):
    """Same optimum as NLopt, reached at least as accurately.

    NLopt's LD_MMA stops ~1e-5 away from the exact optimum on the infeasible
    side, so the position/objective tolerances reflect *its* accuracy, while
    the final assertion pins down that this implementation is the more
    accurate of the two.
    """
    if name == "toy":
        vg, x0, lo, hi, m = toy_problem()
        xstar, n_iter = TOY_XSTAR, 200
    elif name == "beam":
        vg, x0, lo, hi, m = beam_problem()
        xstar, n_iter = BEAM_XSTAR, 200
    else:
        vg, x0, lo, hi, m, xstar = compliance_like_problem(n=200, seed=1)
        n_iter = 300

    st = solve_jax(cls, vg, x0, lo, hi, m, n_iter=n_iter)
    xn, fn = run_nlopt(vg, x0, lo, hi, m)

    onp.testing.assert_allclose(onp.asarray(st.x), xn, rtol=0, atol=1e-4)
    onp.testing.assert_allclose(float(st.f0), fn, rtol=1e-5)
    assert float(np.max(st.fval)) <= 1e-9, "our solution must be feasible"
    assert onp.max(onp.abs(onp.asarray(st.x) - xstar)) <= \
        onp.max(onp.abs(xn - xstar)), "we must be at least as accurate as NLopt"


@pytest.mark.parametrize("n,m,seed", [(10, 3, 0), (25, 5, 1), (6, 8, 2),
                                      (40, 1, 3), (15, 15, 4)])
def test_random_problems_match_nlopt(n, m, seed):
    """Random convex QPs: same optimum as NLopt, reached feasibly."""
    vg, x0, lo, hi, mm = random_problem(n, m, seed)
    st = solve_jax(GCMMA, vg, x0, lo, hi, mm, n_iter=300)
    xn, fn = run_nlopt(vg, x0, lo, hi, mm)
    _, _, fvn, _ = vg(np.asarray(xn))

    assert float(np.max(st.fval)) <= 1e-9, "our solution must be feasible"
    onp.testing.assert_allclose(onp.asarray(st.x), xn, rtol=0, atol=1e-4)
    onp.testing.assert_allclose(float(st.f0), fn, rtol=1e-4, atol=1e-6)
    # Any objective advantage NLopt has must be explained by infeasibility.
    assert float(np.max(st.fval)) <= max(float(np.max(fvn)), 0.0) + 1e-9


# ---------------------------------------------------------------------------
# 4. Traced (jit) drivers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [GCMMA, MMA])
def test_solve_matches_explicit_step_loop(cls):
    """``solve`` is exactly the ``lax.while_loop`` form of a manual step loop."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = cls(vg, xmin, xmax, n_constraints=m)
    st_loop = opt.init(x0)
    step = jax.jit(opt.step)
    for _ in range(25):
        st_loop = step(st_loop)
    st_solve = jax.jit(lambda z: opt.solve(z, max_iter=25))(x0)

    onp.testing.assert_allclose(onp.asarray(st_solve.x), onp.asarray(st_loop.x),
                                rtol=0, atol=1e-12)
    assert int(st_solve.iteration) == int(st_loop.iteration) == 25


@pytest.mark.parametrize("stop", [{"xtol": 1e-12}, {"ftol": 1e-14},
                                  {"kkt_tol": 1e-9}])
def test_solve_stops_early_inside_the_traced_loop(stop):
    """Stopping tests fire inside ``lax.while_loop``, without leaving XLA."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m, epsimin=1e-11)
    st = jax.jit(lambda z: opt.solve(z, max_iter=200, **stop))(x0)
    assert 0 < int(st.iteration) < 200
    # Early stopping trades a little accuracy for iterations.
    onp.testing.assert_allclose(onp.asarray(st.x), TOY_XSTAR, rtol=0, atol=1e-6)


def test_solve_graph_does_not_grow_with_max_iter():
    """The traced loop keeps the jaxpr independent of the iteration count."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m)
    def n_eqns(max_iter):
        return len(jax.make_jaxpr(lambda z: opt.solve(z, max_iter=max_iter))(x0).eqns)

    counts = [n_eqns(k) for k in (10, 1000, 100000)]
    assert counts[0] == counts[1] == counts[2]
    # The whole outer iteration is a single while_loop equation.
    jaxpr = jax.make_jaxpr(lambda z: opt.solve(z, max_iter=1000))(x0).jaxpr
    assert sum(str(e.primitive) == "while" for e in jaxpr.eqns) == 1


@pytest.mark.parametrize("cls", [GCMMA, MMA])
def test_solve_with_history(cls):
    vg, x0, xmin, xmax, m = toy_problem()
    opt = cls(vg, xmin, xmax, n_constraints=m, epsimin=1e-11)
    st, hist = jax.jit(lambda z: opt.solve_with_history(z, n_iter=40))(x0)

    assert hist["objective"].shape == (40,)
    assert hist["constraints"].shape == (40, m)
    assert hist["kkt_norm"].shape == hist["kkt_max"].shape == (40,)
    assert hist["inner_iterations"].shape == (40,)
    onp.testing.assert_allclose(float(hist["objective"][-1]), float(st.f0), rtol=0, atol=0)
    onp.testing.assert_allclose(onp.asarray(hist["constraints"][-1]),
                                onp.asarray(st.fval), rtol=0, atol=0)
    onp.testing.assert_allclose(onp.asarray(st.x), TOY_XSTAR, rtol=0, atol=1e-9)
    assert float(hist["kkt_norm"][-1]) < float(hist["kkt_norm"][0])


def test_solve_is_vmappable_over_starting_points():
    """A batch of designs optimises in one compiled call."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m, epsimin=1e-11)
    batch = np.stack([x0, np.array([1.0, 1.0, 1.0]), np.array([4.9, 0.1, 4.9])])
    sts = jax.jit(jax.vmap(lambda z: opt.solve(z, max_iter=200)))(batch)

    assert sts.x.shape == batch.shape
    for xi in onp.asarray(sts.x):
        onp.testing.assert_allclose(xi, TOY_XSTAR, rtol=0, atol=1e-8)


def test_solve_is_vmappable_over_extra_args():
    """``*args`` are batched too, so a parameter sweep is a single call."""
    def vg(x, target):
        return (np.sum((x - target) ** 2), 2 * (x - target),
                np.zeros(0), np.zeros((0, x.shape[0])))

    opt = GCMMA(vg, 0.0, 1.0, n_constraints=0, epsimin=1e-11)
    targets = np.array([[-0.5, 1.5], [2.0, -1.0], [-0.2, 1.2]])
    sts = jax.jit(jax.vmap(lambda t: opt.solve(np.full(2, 0.5), t, max_iter=120)))(targets)
    expected = onp.clip(onp.asarray(targets), 0.0, 1.0)
    onp.testing.assert_allclose(onp.asarray(sts.x), expected, rtol=0, atol=1e-8)


def test_step_is_jittable_and_matches_eager():
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m)
    st_eager, st_jit = opt.init(x0), opt.init(x0)
    step = jax.jit(opt.step)
    for _ in range(6):
        st_eager = opt.step(st_eager)
        st_jit = step(st_jit)
    onp.testing.assert_allclose(onp.asarray(st_eager.x), onp.asarray(st_jit.x),
                                rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# GCMMA-specific behaviour
# ---------------------------------------------------------------------------

def test_gcmma_inner_loop_runs_and_accepts_only_conservative_steps():
    """Each accepted outer step satisfies the conservativity test, and the
    inner loop is actually exercised on the toy problem."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m)
    step = jax.jit(opt.step)
    st = opt.init(x0)
    inner = []
    for _ in range(8):
        st = step(st)
        inner.append(int(st.inner_iterations))
        assert float(st.f0app) + opt.epsimin >= float(st.f0)
        assert onp.all(onp.asarray(st.fapp) + opt.epsimin >= onp.asarray(st.fval))
    assert max(inner) >= 1, "inner conservativity loop never triggered"


def test_gcmma_raa_stays_positive_and_finite():
    """``raa`` is reset every outer iteration and only inflated by the inner
    loop, so it stays positive and finite throughout."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m)
    _, hist = jax.jit(lambda z: opt.solve_with_history(z, n_iter=30))(x0)
    assert onp.all(onp.isfinite(onp.asarray(hist["objective"])))
    st = opt.init(x0)
    step = jax.jit(opt.step)
    for _ in range(30):
        st = step(st)
        assert float(st.raa0) > 0 and onp.isfinite(float(st.raa0))
        assert onp.all(onp.asarray(st.raa) > 0)
        assert onp.all(onp.isfinite(onp.asarray(st.raa)))


def test_gcmma_value_fn_path_equals_default():
    """A cheap value-only callback for the inner loop gives identical iterates."""
    vg, x0, xmin, xmax, m = toy_problem()

    def vals(x):
        f0, _, f, _ = vg(x)
        return f0, f

    a = GCMMA(vg, xmin, xmax, n_constraints=m)
    b = GCMMA(vg, xmin, xmax, n_constraints=m, value_fn=vals)
    sa, sb = a.init(x0), b.init(x0)
    for _ in range(10):
        sa, sb = a.step(sa), b.step(sb)
        onp.testing.assert_allclose(onp.asarray(sa.x), onp.asarray(sb.x),
                                    rtol=0, atol=1e-12)
        assert int(sa.inner_iterations) == int(sb.inner_iterations)


def test_mma_move_limit_is_respected():
    """With a move limit the first step cannot exceed ``move * (xmax - xmin)``."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = MMA(vg, xmin, xmax, n_constraints=m, move=0.1)
    st1 = opt.step(opt.init(x0))
    assert float(np.max(np.abs(st1.x - x0))) <= 0.1 * (xmax - xmin) + 1e-12


# ---------------------------------------------------------------------------
# Python-loop driver
# ---------------------------------------------------------------------------

def test_mma_with_numpy_callable_no_jit():
    """Plain MMA never traces the user function, so a NumPy callable works."""
    def vg(x):
        x = onp.asarray(x)
        f0 = 0.0624 * x.sum()
        f = onp.array([onp.sum(BEAM_C / x ** 3) - 1.0])
        df = (-3 * BEAM_C / x ** 4)[None, :]
        return f0, 0.0624 * onp.ones(5), f, df

    opt = MMA(vg, 1.0, 10.0, n_constraints=1, epsimin=1e-11)
    st = opt.run(5.0 * onp.ones(5), max_iter=200, jit=False)
    onp.testing.assert_allclose(onp.asarray(st.x), BEAM_XSTAR, rtol=0, atol=1e-8)


def test_run_matches_solve():
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m)
    st_run = opt.run(x0, max_iter=30)
    st_solve = opt.solve(x0, max_iter=30)
    onp.testing.assert_allclose(onp.asarray(st_run.x), onp.asarray(st_solve.x),
                                rtol=0, atol=1e-12)


def test_run_stopping_criteria():
    """``xtol`` / ``ftol`` / ``kkt_tol`` each stop the Python driver early."""
    vg, x0, xmin, xmax, m = toy_problem()
    opt = GCMMA(vg, xmin, xmax, n_constraints=m, epsimin=1e-11)
    for kw in ({"xtol": 1e-12}, {"ftol": 1e-14}, {"kkt_tol": 1e-9}):
        st = opt.run(x0, max_iter=200, **kw)
        assert 0 < int(st.iteration) < 200
        onp.testing.assert_allclose(onp.asarray(st.x), TOY_XSTAR, rtol=0, atol=1e-6)


def test_extra_args_are_forwarded():
    """``*args`` (continuation parameters, penalisation, ...) reach the user fn."""
    def vg(x, target):
        return (np.sum((x - target) ** 2), 2 * (x - target),
                np.zeros(0), np.zeros((0, x.shape[0])))

    opt = GCMMA(vg, 0.0, 1.0, n_constraints=0, epsimin=1e-11)
    t = np.array([-0.5, 1.5])                      # optimum clipped to (0, 1)
    st = opt.run(np.full(2, 0.5), t, max_iter=120)
    onp.testing.assert_allclose(onp.asarray(st.x), [0.0, 1.0], atol=1e-9)


def test_callback_receives_every_iteration():
    vg, x0, xmin, xmax, m = toy_problem()
    opt = MMA(vg, xmin, xmax, n_constraints=m)
    seen = []
    opt.run(x0, max_iter=7, callback=lambda s: seen.append(int(s.iteration)))
    assert seen == list(range(1, 8))
