"""The preconditioner API: one callable ``preconditioner(op) -> M`` passed to
``create_solver`` (or per call), on the linear and Newton paths."""
import jax
import jax.numpy as jnp
import numpy as onp
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
def setup():
    mesh = fe.mesh.box_mesh((2.0, 0.5, 0.5), mesh_size=0.25, element_type="HEX8")
    right = lambda p: jnp.isclose(p[0], 2.0, atol=1e-6)
    problem = Elasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[right])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    E = fe.TracedParams.create_node_var(problem, 1.0)
    t = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
    tp = fe.TracedParams(volume_vars=(E,), surface_vars=[(t,)])
    u0 = fe.zero_like_initial_guess(problem, bc)
    krylov = fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12, maxiter=5000)
    # reference: unpreconditioned CG to tight tolerance, and its gradient
    ref_solver = fe.create_solver(problem, bc, krylov, linear=True,
                                  traced_params=tp, traced_structure=ts)
    ref = ref_solver(tp, u0, traced_structure=ts).dofs
    g_ref = jax.grad(lambda tp: jnp.sum(ref_solver(tp, u0, traced_structure=ts).dofs ** 2))(tp)
    return dict(problem=problem, bc=bc, ts=ts, tp=tp, u0=u0, krylov=krylov,
                ref=ref, g_ref=g_ref.volume_vars[0])


def _rel(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def _loss(solver, s, **kw):
    return lambda tp: jnp.sum(solver(tp, s["u0"], traced_structure=s["ts"], **kw).dofs ** 2)


def test_jacobi_at_creation_and_gradient(setup):
    s = setup
    solver = fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=True,
                              traced_params=s["tp"], traced_structure=s["ts"],
                              preconditioner=fe.jacobi())
    u = solver(s["tp"], s["u0"], traced_structure=s["ts"]).dofs
    assert _rel(u, s["ref"]) < 1e-7
    g = jax.grad(_loss(solver, s))(s["tp"]).volume_vars[0]
    assert _rel(g, s["g_ref"]) < 1e-6


def test_preconditioner_per_call_under_jit(setup):
    s = setup
    solver = fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=True,
                              traced_params=s["tp"], traced_structure=s["ts"])
    f = jax.jit(lambda tp, ts: solver(tp, s["u0"], traced_structure=ts,
                                      preconditioner=fe.jacobi()).dofs)
    assert _rel(f(s["tp"], s["ts"]), s["ref"]) < 1e-7


def test_fixed_preconditioner_and_operator_diag(setup):
    s = setup
    J = fe.assembler.create_J_bc_csr_parametric(s["problem"], symmetric=True)(
        s["u0"], s["tp"], s["bc"], s["ts"])
    op = fe.LinearOperator.wrap(J, structure=s["ts"], problem=s["problem"])
    diag = op.diag
    # the slot-map diagonal agrees with the generic row == column scan
    op2 = fe.LinearOperator.wrap(J)
    assert float(jnp.abs(diag - op2.diag).max()) < 1e-12
    inv = 1.0 / diag
    solver = fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=True,
                              traced_params=s["tp"], traced_structure=s["ts"],
                              preconditioner=lambda op: (lambda r: inv * r))
    assert _rel(solver(s["tp"], s["u0"], traced_structure=s["ts"]).dofs, s["ref"]) < 1e-7


def test_amg_as_jit_argument_and_gradient(setup):
    pytest.importorskip("pyamg")
    pytest.importorskip("amjax")
    s = setup
    P = fe.amg(s["problem"], s["bc"], s["tp"], fe.AMGOptions(near_nullspace="rigid_body"),
               traced_structure=s["ts"])
    assert isinstance(P, fe.AMGPreconditioner)
    assert all(isinstance(l, jax.Array) or isinstance(l, onp.ndarray)
               for l in jax.tree_util.tree_leaves(P))
    solver = fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=True,
                              traced_params=s["tp"], traced_structure=s["ts"])
    f = jax.jit(lambda tp, ts, P: solver(tp, s["u0"], traced_structure=ts, preconditioner=P).dofs)
    assert _rel(f(s["tp"], s["ts"], P), s["ref"]) < 1e-7
    g = jax.grad(_loss(solver, s, preconditioner=P))(s["tp"]).volume_vars[0]
    assert _rel(g, s["g_ref"]) < 1e-6


def test_newton_traced_path_with_jacobi(setup):
    s = setup
    solver = fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=False,
                              traced_params=s["tp"], traced_structure=s["ts"],
                              preconditioner=fe.jacobi())
    u = jax.jit(lambda tp, ts: solver(tp, s["u0"], traced_structure=ts).dofs)(s["tp"], s["ts"])
    assert _rel(u, s["ref"]) < 1e-7


def test_newton_amg_rebuild_loop(setup):
    pytest.importorskip("pyamg")
    pytest.importorskip("amjax")
    s = setup
    P = fe.amg(s["problem"], s["bc"], s["tp"],
               fe.AMGOptions(near_nullspace="rigid_body", rebuild_every=None),
               traced_structure=s["ts"])
    solver = fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=False,
                              traced_params=s["tp"], traced_structure=s["ts"], preconditioner=P)
    assert _rel(solver(s["tp"], s["u0"], traced_structure=s["ts"]).dofs, s["ref"]) < 1e-7
    g = jax.grad(_loss(solver, s))(s["tp"]).volume_vars[0]
    assert _rel(g, s["g_ref"]) < 1e-6


def test_removed_and_invalid_settings(setup):
    s = setup
    with pytest.raises(TypeError):
        fe.KrylovSolverOptions(use_jacobi_preconditioner=True)
    with pytest.raises(TypeError):
        fe.KrylovSolverOptions(preconditioner=lambda r: r)
    with pytest.raises(TypeError, match="removed"):
        fe.AMGSolverOptions(solver="cg")
    with pytest.raises(TypeError):
        fe.create_solver(s["problem"], s["bc"], fe.AMGOptions(), linear=True, traced_params=s["tp"])
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    with pytest.raises(ValueError):
        fe.create_solver(s["problem"], s["bc"], direct, linear=True, traced_params=s["tp"],
                         preconditioner=fe.jacobi())
    with pytest.raises(TypeError):
        fe.create_solver(s["problem"], s["bc"], s["krylov"], linear=True, traced_params=s["tp"],
                         preconditioner="jacobi")
    assert not hasattr(s["ts"], "preconditioner")
    assert not hasattr(fe.TracedStructure, "with_preconditioner")
