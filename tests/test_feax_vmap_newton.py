"""feax.vmap over the traced Newton solver: batched Newton with a shared factor."""
import jax
import jax.numpy as jnp
import pytest

import feax as fe
from feax.problem import MatrixView

E_MOD, NU = 70e3, 0.3
MU = E_MOD / (2 * (1 + NU))
KAPPA = E_MOD / (3 * (1 - 2 * NU))


class NeoHookean(fe.Problem):
    def get_tensor_map(self):
        def first_PK_stress(u_grad, *args):
            F = jnp.eye(self.dim) + u_grad
            J = jnp.linalg.det(F)
            F_inv_T = jnp.linalg.inv(F).T
            return MU * (F - F_inv_T) + KAPPA * (J - 1) * J * F_inv_T
        return first_PK_stress

    def get_surface_maps(self):
        return [lambda u, x, t: jnp.array([0.0, 0.0, t])]


@pytest.fixture(scope="module")
def s():
    mesh = fe.mesh.box_mesh((10.0, 10.0, 10.0), mesh_size=2.5)
    right = lambda p: jnp.isclose(p[0], 10.0, atol=1e-5)
    problem = NeoHookean(mesh, vec=3, dim=3, location_fns=[right], matrix_view=MatrixView.FULL)
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-5), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    u0 = fe.zero_like_initial_guess(problem, bc)
    N = 4
    t1 = fe.TracedParams.create_uniform_surface_var(problem, -200.0)
    loads = jnp.stack([t1 * (0.5 + 0.5 * k) for k in range(N)])
    tp0 = fe.TracedParams((), [(t1,)])
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    newton = fe.NewtonOptions(tol=1e-8, rel_tol=1e-10, max_iter=30)
    solver = fe.create_solver(problem, bc, direct, linear=False, newton_options=newton,
                              traced_params=tp0, traced_structure=ts)
    tps = [fe.TracedParams((), [(loads[i],)]) for i in range(N)]
    ref = jnp.stack([solver(tp, u0, traced_structure=ts).dofs for tp in tps])
    loss = lambda tp: jnp.sum(solver(tp, u0, traced_structure=ts).dofs ** 2)
    g_ref = [jax.grad(loss)(tp).surface_vars[0][0] for tp in tps]
    return dict(problem=problem, bc=bc, ts=ts, u0=u0, N=N, loads=loads, solver=solver,
                direct=direct, ref=ref, g_ref=g_ref)


def _rel(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def test_newton_batch_is_lhs_and_matches_per_case(s):
    batched = fe.vmap(s["solver"], in_axes=0, return_info=True,
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-14, maxiter=500))
    assert batched.kind == "lhs"
    tp = fe.TracedParams((), [(s["loads"],)])
    U, info = batched(tp, s["u0"], s["ts"])
    assert float(jnp.abs(s["ref"]).max()) > 1e-3          # a real (nonlinear) deformation
    assert _rel(U.dofs, s["ref"]) < 1e-6
    assert bool(info.converged.all())
    assert int(info.iterations.min()) >= 2
    # under jit as well
    f = jax.jit(lambda L, ts: batched(fe.TracedParams((), [(L,)]), None, ts)[0].dofs)
    assert _rel(f(s["loads"], s["ts"]), s["ref"]) < 1e-6


def test_newton_batch_gradient_and_chunk(s):
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-11, atol=1e-14, maxiter=500)
    for chunk in (None, 3):
        batched = fe.vmap(s["solver"], in_axes=0, options=opts, chunk=chunk)
        tp = fe.TracedParams((), [(s["loads"],)])
        assert _rel(batched(tp, s["u0"], s["ts"]).dofs, s["ref"]) < 1e-6
        g = jax.grad(lambda tp: jnp.sum(batched(tp, s["u0"], s["ts"]).dofs ** 2))(tp)
        for i in range(s["N"]):
            assert _rel(g.surface_vars[0][0][i], s["g_ref"][i]) < 1e-5


def test_newton_unconverged_cases_are_returned_with_flags(s):
    from feax.solvers.batched import BatchedNewtonInfo
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-14, maxiter=500)
    # one Newton step only: nothing converges, but every case's iterate comes back
    one_step = fe.create_solver(s["problem"], s["bc"], s["direct"], linear=False,
                                newton_options=fe.NewtonOptions(tol=1e-8, rel_tol=1e-10, max_iter=1),
                                traced_params=fe.TracedParams((), [(s["loads"][0],)]),
                                traced_structure=s["ts"])
    batched = fe.vmap(one_step, in_axes=0, options=opts, return_info=True)
    tp = fe.TracedParams((), [(s["loads"],)])
    U, info = batched(tp, s["u0"], s["ts"])
    assert isinstance(info, BatchedNewtonInfo)
    assert not bool(info.converged.any()) and not bool(info.line_search_failed.any())
    assert int(info.iterations.max()) == 1 and int(info.cg_iterations.min()) > 0
    assert float(jnp.abs(U.dofs).max()) > 0.0
    # the one-step iterate equals one Newton step of the per-case solver
    ref1 = jnp.stack([one_step(fe.TracedParams((), [(s["loads"][i],)]), s["u0"],
                               traced_structure=s["ts"]).dofs for i in range(s["N"])])
    assert _rel(U.dofs, ref1) < 1e-6
    # the converged batch reports converged flags and a tiny residual
    full = fe.vmap(s["solver"], in_axes=0, options=opts, return_info=True)
    _, info2 = full(tp, s["u0"], s["ts"])
    assert bool(info2.converged.all()) and float(info2.residual.max()) < 1e-8


def test_newton_rhs_override_is_plain_vmap(s):
    batched = fe.vmap(s["solver"], in_axes=0, batch="rhs")
    tp = fe.TracedParams((), [(s["loads"],)])
    assert batched.kind == "rhs"
    assert _rel(batched(tp, s["u0"], s["ts"]).dofs, s["ref"]) < 1e-8


def test_newton_direct_options_per_case(s):
    batched = fe.vmap(s["solver"], in_axes=0, options=s["direct"], chunk=2)
    tp = fe.TracedParams((), [(s["loads"],)])
    assert _rel(batched(tp, s["u0"], s["ts"]).dofs, s["ref"]) < 1e-8


def test_newton_host_loop_solver_is_rejected(s):
    hybrid = fe.create_solver(s["problem"], s["bc"], fe.KrylovSolverOptions(solver="cg"), linear=False,
                              traced_params=fe.TracedParams((), [(s["loads"][0],)]),
                              traced_structure=s["ts"], extra_residual_fn=lambda u: 0.0 * u)
    with pytest.raises(NotImplementedError):
        fe.vmap(hybrid, in_axes=0)
