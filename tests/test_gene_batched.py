"""gene functions on a batch of designs (the designs of a feax.vmap batch)."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import feax as fe
from feax import gene


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


@pytest.fixture(scope="module")
def s():
    mesh = fe.StructuredGrid((16, 4, 4), spacing=(0.25, 0.25, 0.25)).to_mesh()
    tip = lambda p: jnp.isclose(p[0], 4.0, atol=1e-6)
    problem = SIMPElasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[tip])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    u0 = fe.zero_like_initial_guess(problem, bc)
    n = mesh.points.shape[0]
    B = 3
    key = jax.random.PRNGKey(3)
    X = jnp.clip(0.5 + 0.3 * jax.random.normal(key, (B, n)), 0.05, 1.0)
    unit = fe.TracedParams.create_uniform_surface_var(problem, 1.0)
    scales = jnp.array([-1e-2, -2e-2, -3e-2])
    return dict(mesh=mesh, problem=problem, bc=bc, ts=ts, u0=u0, n=n, B=B, X=X, unit=unit, scales=scales)


def _rel(a, b):
    return float(jnp.abs(a - b).max() / (jnp.abs(b).max() + 1e-30))


def test_density_filter_batch(s):
    filt = gene.create_density_filter(s["mesh"], radius=0.6)
    Xf = filt(s["X"])
    assert Xf.shape == s["X"].shape
    for i in range(s["B"]):
        assert _rel(Xf[i], filt(s["X"][i])) < 1e-12


def test_volume_fn_batch_and_quad_layout(s):
    vol = gene.create_volume_fn(s["problem"])
    v = vol(s["X"])
    assert v.shape == (s["B"],)
    for i in range(s["B"]):
        assert abs(float(v[i]) - float(vol(s["X"][i]))) < 1e-12
    fe0 = s["problem"].fes[0]
    rho_q = jnp.full((fe0.num_cells, fe0.num_quads), 0.3)      # quad layout, not a batch
    assert abs(float(vol(rho_q)) - 0.3) < 1e-12


def test_compliance_fn_batch_with_per_case_loads(s):
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    tp0 = fe.TracedParams(volume_vars=(gene.simp(s["X"][0], 1.0),), surface_vars=[(s["unit"] * s["scales"][0],)])
    solver = fe.create_solver(s["problem"], s["bc"], direct, linear=True, traced_params=tp0, traced_structure=s["ts"])
    E = gene.simp(s["X"], 1.0)
    loads = s["unit"][None] * s["scales"][:, None, None] if s["unit"].ndim == 2 else s["unit"][None] * s["scales"][:, None]
    U = fe.vmap(solver, in_axes=0)(fe.TracedParams(volume_vars=(E,), surface_vars=[(loads,)]), s["u0"], s["ts"])
    comp = gene.create_compliance_fn(s["problem"], surface_load_params=[s["scales"]])
    c = comp(U.dofs)
    assert c.shape == (s["B"],)
    for i in range(s["B"]):
        comp_i = gene.create_compliance_fn(s["problem"], surface_load_params=[float(s["scales"][i])])
        assert _rel(c[i], comp_i(U.dofs[i])) < 1e-12
    with pytest.raises(ValueError):
        comp(U.dofs[0])


def test_compliance_fn_tuple_surface_params(s):
    """A surface map with several parameters takes a tuple entry, each element
    scalar or batched per case."""
    class TwoParam(fe.Problem):
        def get_tensor_map(self):
            return SIMPElasticity.get_tensor_map(self)

        def get_surface_maps(self):
            return [lambda u, x, ty, tz: jnp.array([0.0, ty, tz])]

    tip = lambda p: jnp.isclose(p[0], 4.0, atol=1e-6)
    problem = TwoParam(s["mesh"], vec=3, dim=3, ele_type="HEX8", location_fns=[tip])
    sol = jax.random.normal(jax.random.PRNGKey(9), (s["B"], s["u0"].shape[0]))
    ty = jnp.array([0.1, 0.2, 0.3])
    comp = gene.create_compliance_fn(problem, surface_load_params=[(ty, 0.5)])
    c = comp(sol)
    assert c.shape == (s["B"],)
    for i in range(s["B"]):
        comp_i = gene.create_compliance_fn(problem, surface_load_params=[(float(ty[i]), 0.5)])
        assert _rel(c[i], comp_i(sol[i])) < 1e-12


def test_oc_update_batch_meets_each_volume(s):
    vol_w = jax.grad(gene.create_volume_fn(s["problem"]))(jnp.zeros(s["n"]))
    dc = -jax.random.uniform(jax.random.PRNGKey(5), s["X"].shape) - 0.1
    vf = jnp.array([0.3, 0.4, 0.5])
    # start within reach of every target (the move limit bounds the volume change)
    X0 = 0.4 + 0.05 * jax.random.normal(jax.random.PRNGKey(6), s["X"].shape)
    Xn = jax.jit(lambda X, dc: gene.oc_update(X, dc, vf, dv=vol_w, move=0.2))(X0, dc)
    assert Xn.shape == X0.shape
    assert float(jnp.abs(Xn - X0).max()) <= 0.2 + 1e-12
    assert float(Xn.min()) >= 0.0 and float(Xn.max()) <= 1.0
    for i in range(s["B"]):
        assert abs(float(Xn[i] @ vol_w) - float(vf[i]) * float(vol_w.sum())) < 1e-6 * float(vol_w.sum())
        one = gene.oc_update(X0[i], dc[i], float(vf[i]), dv=vol_w, move=0.2)
        assert _rel(Xn[i], one) < 1e-10


def test_batched_topopt_loop_decreases_compliance(s):
    """Five OC iterations on three designs at once: fe.vmap + gene only."""
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    loads = s["unit"][None] * s["scales"].reshape((s["B"],) + (1,) * s["unit"].ndim)
    tp0 = fe.TracedParams(volume_vars=(gene.simp(s["X"][0], 1.0),), surface_vars=[(loads[0],)])
    solver = fe.create_solver(s["problem"], s["bc"], direct, linear=True, traced_params=tp0, traced_structure=s["ts"])
    batched = fe.vmap(solver, in_axes=fe.TracedParams(volume_vars=(0,), surface_vars=[(0,)]),
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-9, atol=1e-12, maxiter=1000),
                      return_info=True)
    filt = gene.create_density_filter(s["mesh"], radius=0.6)
    volume = gene.create_volume_fn(s["problem"])
    comp = gene.create_compliance_fn(s["problem"], surface_load_params=[s["scales"]])
    vol_w = jax.grad(volume)(jnp.zeros(s["n"]))
    vf = 0.4

    @jax.jit
    def evaluate(X):
        def objective(X):
            rho = gene.heaviside_projection(filt(X), beta=2.0)
            U, info = batched(fe.TracedParams(volume_vars=(gene.simp(rho, 1.0),), surface_vars=[(loads,)]), s["u0"], s["ts"])
            c = comp(U.dofs)
            return jnp.sum(c), (c, info, volume(rho))
        (_, (c, info, v)), dc = jax.value_and_grad(objective, has_aux=True)(X)
        return c, dc, v, info

    X = jnp.full((s["B"], s["n"]), vf)
    c0, dc, v, info = evaluate(X)
    assert bool(info.converged.all())
    for _ in range(5):
        X = gene.oc_update(X, dc, vf, dv=vol_w, move=0.2)
        c, dc, v, info = evaluate(X)
    assert bool((c < c0).all()), (c, c0)
    assert float(jnp.abs(v - vf).max()) < 0.05
