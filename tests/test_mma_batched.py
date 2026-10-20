"""BatchedGCMMA: B independent problems evaluated together, per-problem subproblems."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import feax as fe
from feax import gene
from feax.experimental.mma import GCMMA, BatchedGCMMA


def _toy_vg(t):
    """n = 3, m = 2 toy problem of tests/test_mma.py, shifted by ``t`` per case."""
    def f0(x):
        return jnp.sum((x - t) ** 2)

    def f(x):
        return jnp.stack([(x[0] - 5) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 - 9,
                          (x[0] - 3) ** 2 + (x[1] - 4) ** 2 + (x[2] - 3) ** 2 - 9])

    def vg(x):
        f0v, df0 = jax.value_and_grad(f0)(x)
        return f0v, df0, f(x), jax.jacfwd(f)(x)
    return vg


def test_batched_gcmma_matches_per_problem_gcmma():
    B = 4
    shifts = jnp.array([0.0, 0.5, 1.0, 1.5])
    vgs = [_toy_vg(t) for t in shifts]

    def batched_vg(X):                       # one call on the whole batch
        outs = [vgs[i](X[i]) for i in range(B)]
        return tuple(jnp.stack([o[k] for o in outs]) for k in range(4))

    x0 = jnp.array([4.0, 3.0, 2.0])
    X0 = jnp.broadcast_to(x0, (B, 3))
    opt_b = BatchedGCMMA(batched_vg, 0.0, 5.0, n_constraints=2, epsimin=1e-11)
    st = jax.jit(lambda X: opt_b.solve(X, max_iter=60))(X0)
    assert st.x.shape == (B, 3) and st.inner_iterations.shape == (B,)
    for i in range(B):
        opt = GCMMA(vgs[i], 0.0, 5.0, n_constraints=2, epsimin=1e-11)
        ref = opt.solve(x0, max_iter=60)
        onp.testing.assert_allclose(onp.asarray(st.x[i]), onp.asarray(ref.x), rtol=0, atol=1e-9)
        onp.testing.assert_allclose(float(st.f0[i]), float(ref.f0), rtol=1e-9)
        assert onp.all(onp.asarray(st.fval[i]) <= 1e-8)
    # history driver and the per-problem stopping test
    st2, hist = opt_b.solve_with_history(X0, n_iter=10)
    assert hist["objective"].shape == (10, B) and hist["inner_iterations"].shape == (10, B)
    st3 = opt_b.solve(X0, max_iter=200, kkt_tol=1e-6)
    assert int(st3.iteration) < 200


class SIMPElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, t])]


def test_batched_gcmma_with_feax_vmap_topopt():
    """Three compliance minimizations with a volume constraint each, evaluated
    through one fe.vmap batch per GCMMA evaluation."""
    mesh = fe.StructuredGrid((16, 4, 4), spacing=(0.25, 0.25, 0.25)).to_mesh()
    tip = lambda p: jnp.isclose(p[0], 4.0, atol=1e-6)
    problem = SIMPElasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[tip])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    u0 = fe.zero_like_initial_guess(problem, bc)
    n = mesh.points.shape[0]
    B, vf = 3, 0.4
    unit = fe.TracedParams.create_uniform_surface_var(problem, 1.0)
    scales = jnp.array([-1e-2, -2e-2, -3e-2])
    loads = unit[None] * scales.reshape((B,) + (1,) * unit.ndim)
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    tp0 = fe.TracedParams(volume_vars=(gene.simp(jnp.full(n, vf), 1.0),), surface_vars=[(loads[0],)])
    solver = fe.create_solver(problem, bc, direct, linear=True, traced_params=tp0, traced_structure=ts)
    batched = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(0,), surface_vars=[(0,)]),
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-9, atol=1e-12, maxiter=1000))
    filt = gene.create_density_filter(mesh, radius=0.6)
    volume = gene.create_volume_fn(problem)
    comp = gene.create_compliance_fn(problem, surface_load_params=[scales])

    def compliances(X):
        rho = gene.heaviside_projection(filt(X), beta=2.0)
        U = batched(fe.TracedParams(volume_vars=(gene.simp(rho, 1.0),), surface_vars=[(loads,)]), u0, ts)
        return comp(U.dofs)

    c0 = compliances(jnp.full((B, n), vf))

    def value_and_grad(X):
        c, pull = jax.vjp(compliances, X)
        dc = jax.vmap(lambda e: pull(e)[0])(jnp.eye(B))          # (B, B, n) -> own rows
        dc = jnp.stack([dc[i, i] for i in range(B)])
        g, dg = jax.vmap(jax.value_and_grad(lambda x: volume(filt(x)) / vf - 1.0))(X)
        return c / c0, dc / c0[:, None], g[:, None], dg[:, None, :]

    opt = BatchedGCMMA(value_and_grad, 0.0, 1.0, n_constraints=1, max_inner=5)
    state = jax.jit(lambda X: opt.solve(X, max_iter=12))(jnp.full((B, n), vf))
    assert state.x.shape == (B, n)
    assert bool((state.f0 < 0.9).all()), state.f0                # every design improved
    assert float(jnp.abs(state.fval).max()) < 2e-2                # volume constraints met
