"""feax.vmap: rhs / lhs detection, the batched solves, gradients, chunking."""
import warnings

import jax
import jax.numpy as jnp
import pytest

import feax as fe


class Elasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, -t])]


@pytest.fixture(scope="module")
def s():
    mesh = fe.mesh.box_mesh((2.0, 0.5, 0.5), mesh_size=0.25, element_type="HEX8")
    right = lambda p: jnp.isclose(p[0], 2.0, atol=1e-6)
    problem = Elasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[right])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    u0 = fe.zero_like_initial_guess(problem, bc)
    n_nodes = mesh.points.shape[0]
    N = 5
    E = jnp.exp(0.2 * jax.random.normal(jax.random.PRNGKey(1), (N, n_nodes)))
    t1 = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
    loads = jnp.stack([t1 * (1.0 + 0.2 * k) for k in range(N)])
    tp0 = fe.TracedParams(volume_vars=(E[0],), surface_vars=[(t1,)])
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    solver = fe.create_solver(problem, bc, direct, linear=True, traced_params=tp0, traced_structure=ts)
    # references, one direct solve per case
    tps_lhs = [fe.TracedParams(volume_vars=(E[i],), surface_vars=[(loads[i],)]) for i in range(N)]
    tps_rhs = [fe.TracedParams(volume_vars=(E[0],), surface_vars=[(loads[i],)]) for i in range(N)]
    ref_lhs = jnp.stack([solver(tp, u0, traced_structure=ts).dofs for tp in tps_lhs])
    ref_rhs = jnp.stack([solver(tp, u0, traced_structure=ts).dofs for tp in tps_rhs])
    loss = lambda tp: jnp.sum(solver(tp, u0, traced_structure=ts).dofs ** 2)
    g_ref = [jax.grad(loss)(tp) for tp in tps_lhs]
    return dict(problem=problem, bc=bc, ts=ts, u0=u0, solver=solver, direct=direct, tp0=tp0, N=N,
                E=E, loads=loads, t1=t1, ref_lhs=ref_lhs, ref_rhs=ref_rhs, g_ref=g_ref)


def _rel(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def test_detects_rhs_and_lhs(s):
    rhs_axes = fe.TracedParams(volume_vars=(None,), surface_vars=[(0,)])
    b = fe.vmap(s["solver"], in_axes=rhs_axes)
    assert b.kind == "rhs"
    assert fe.vmap(s["solver"], in_axes=0).kind == "lhs"
    assert fe.vmap(s["solver"], in_axes=fe.TracedParams(volume_vars=(0,), surface_vars=[(None,)])).kind == "lhs"
    assert fe.vmap(s["solver"], in_axes=rhs_axes, batch="lhs").kind == "lhs"


def test_rhs_batch_matches_per_case(s):
    axes = fe.TracedParams(volume_vars=(None,), surface_vars=[(0,)])
    batched = fe.vmap(s["solver"], in_axes=axes, return_info=True)
    tp = fe.TracedParams(volume_vars=(s["E"][0],), surface_vars=[(s["loads"],)])
    U, info = batched(tp, s["u0"], s["ts"])
    assert U.dofs.shape == (s["N"], s["u0"].shape[0])
    assert _rel(U.dofs, s["ref_rhs"]) < 1e-9
    assert int(info.iterations.max()) == 0
    with pytest.raises(ValueError):
        fe.vmap(s["solver"], in_axes=axes, preconditioner=fe.jacobi())


def test_lhs_batched_cg_matches_per_case(s):
    batched = fe.vmap(s["solver"], in_axes=0, return_info=True,
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12, maxiter=500))
    tp = fe.TracedParams(volume_vars=(s["E"],), surface_vars=[(s["loads"],)])
    U, info = batched(tp, s["u0"], s["ts"])
    assert _rel(U.dofs, s["ref_lhs"]) < 1e-7
    assert bool(info.converged.all())
    print("iterations:", info.iterations, "preconditioner:", batched.preconditioner.__name__)
    # under jit, batch axis elsewhere
    f = jax.jit(lambda E, L, ts: batched(fe.TracedParams(volume_vars=(E,), surface_vars=[(L,)]), None, ts)[0].dofs)
    assert _rel(f(s["E"], s["loads"], s["ts"]), s["ref_lhs"]) < 1e-7
    axes = fe.TracedParams(volume_vars=(1,), surface_vars=[(0,)])
    b2 = fe.vmap(s["solver"], in_axes=axes)
    tpT = fe.TracedParams(volume_vars=(s["E"].T,), surface_vars=[(s["loads"],)])
    assert _rel(b2(tpT, s["u0"], s["ts"]).dofs, s["ref_lhs"]) < 1e-7


def test_lhs_gradient_matches_per_case(s):
    batched = fe.vmap(s["solver"], in_axes=0,
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-11, atol=1e-13, maxiter=500))
    tp = fe.TracedParams(volume_vars=(s["E"],), surface_vars=[(s["loads"],)])
    g = jax.grad(lambda tp: jnp.sum(batched(tp, s["u0"], s["ts"]).dofs ** 2))(tp)
    for i in range(s["N"]):
        assert _rel(g.volume_vars[0][i], s["g_ref"][i].volume_vars[0]) < 1e-6
        assert _rel(g.surface_vars[0][0][i], s["g_ref"][i].surface_vars[0][0]) < 1e-6
    # an unbatched leaf's cotangent is the sum over the cases
    axes = fe.TracedParams(volume_vars=(0,), surface_vars=[(None,)])
    b2 = fe.vmap(s["solver"], in_axes=axes,
                 options=fe.KrylovSolverOptions(solver="cg", tol=1e-11, atol=1e-13, maxiter=500))
    tp2 = fe.TracedParams(volume_vars=(s["E"],), surface_vars=[(s["t1"],)])
    g2 = jax.grad(lambda tp: jnp.sum(b2(tp, s["u0"], s["ts"]).dofs ** 2))(tp2)
    tps = [fe.TracedParams(volume_vars=(s["E"][i],), surface_vars=[(s["t1"],)]) for i in range(s["N"])]
    loss = lambda tp: jnp.sum(s["solver"](tp, s["u0"], traced_structure=s["ts"]).dofs ** 2)
    gs = [jax.grad(loss)(tp) for tp in tps]
    assert _rel(g2.surface_vars[0][0], sum(gg.surface_vars[0][0] for gg in gs)) < 1e-6


def test_lhs_chunked_and_jacobi(s):
    batched = fe.vmap(s["solver"], in_axes=0, chunk=2, preconditioner=fe.jacobi(), return_info=True,
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12, maxiter=2000))
    tp = fe.TracedParams(volume_vars=(s["E"],), surface_vars=[(s["loads"],)])
    U, info = batched(tp, s["u0"], s["ts"])
    assert U.dofs.shape[0] == s["N"]
    assert _rel(U.dofs, s["ref_lhs"]) < 1e-7
    assert bool(info.converged.all())
    g = jax.grad(lambda tp: jnp.sum(batched(tp, s["u0"], s["ts"])[0].dofs ** 2))(tp)
    assert _rel(g.volume_vars[0][2], s["g_ref"][2].volume_vars[0]) < 1e-6
    # chunking with a shared (unbatched) leaf: only batched leaves are scanned
    axes = fe.TracedParams(volume_vars=(0,), surface_vars=[(None,)])
    b2 = fe.vmap(s["solver"], in_axes=axes, chunk=2, return_info=True,
                 options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12, maxiter=2000))
    tp2 = fe.TracedParams(volume_vars=(s["E"],), surface_vars=[(s["t1"],)])
    U2, info2 = b2(tp2, s["u0"], s["ts"])
    tps = [fe.TracedParams(volume_vars=(s["E"][i],), surface_vars=[(s["t1"],)]) for i in range(s["N"])]
    ref2 = jnp.stack([s["solver"](t, s["u0"], traced_structure=s["ts"]).dofs for t in tps])
    assert _rel(U2.dofs, ref2) < 1e-7 and bool(info2.converged.all())
    g2 = jax.grad(lambda tp: jnp.sum(b2(tp, s["u0"], s["ts"])[0].dofs ** 2))(tp2)
    loss = lambda t: jnp.sum(s["solver"](t, s["u0"], traced_structure=s["ts"]).dofs ** 2)
    assert _rel(g2.volume_vars[0][1], jax.grad(loss)(tps[1]).volume_vars[0]) < 1e-6


def test_lhs_direct_warns_without_chunk(s):
    tp = fe.TracedParams(volume_vars=(s["E"],), surface_vars=[(s["loads"],)])
    with pytest.warns(UserWarning, match="factorized"):
        batched = fe.vmap(s["solver"], in_axes=0, options=s["direct"])
    assert _rel(batched(tp, s["u0"], s["ts"]).dofs, s["ref_lhs"]) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batched = fe.vmap(s["solver"], in_axes=0, options=s["direct"], chunk=2)
    assert _rel(batched(tp, s["u0"], s["ts"]).dofs, s["ref_lhs"]) < 1e-9


def test_rejects_non_solver_and_newton(s):
    with pytest.raises(TypeError):
        fe.vmap(lambda tp: tp)
    newton = fe.create_solver(s["problem"], s["bc"], s["direct"], linear=False,
                              traced_params=s["tp0"], traced_structure=s["ts"])
    with pytest.raises(NotImplementedError):
        fe.vmap(newton)
