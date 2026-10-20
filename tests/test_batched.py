"""batched_pcg and the batched preconditioners (jacobi, shared_factor)."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import feax as fe
from feax.solvers.batched import batched_pcg


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
def batch():
    """N design cases (log-normal modulus fields) with their own loads."""
    mesh = fe.mesh.box_mesh((2.0, 0.5, 0.5), mesh_size=0.25, element_type="HEX8")
    right = lambda p: jnp.isclose(p[0], 2.0, atol=1e-6)
    problem = Elasticity(mesh, vec=3, dim=3, ele_type="HEX8", location_fns=[right])
    bc = fe.DirichletBCConfig([fe.DirichletBCSpec(
        lambda p: jnp.isclose(p[0], 0.0, atol=1e-6), "all", 0.0)]).create_bc(problem)
    ts = fe.TracedStructure.from_problem(problem, free_scratch=False)
    u0 = fe.zero_like_initial_guess(problem, bc)
    n_nodes = mesh.points.shape[0]
    N = 6
    key = jax.random.PRNGKey(0)
    E = jnp.exp(0.2 * jax.random.normal(key, (N, n_nodes)))
    t = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
    loads = jnp.stack([t * (1.0 + 0.1 * k) for k in range(N)])
    J_fn = fe.assembler.create_J_bc_csr_parametric(problem, symmetric=True)
    res_fn = fe.assembler.create_res_bc_parametric(problem)
    tps = [fe.TracedParams(volume_vars=(E[i],), surface_vars=[(loads[i],)]) for i in range(N)]
    Ks = [J_fn(u0, tp, bc, ts) for tp in tps]
    values = jnp.stack([K.data for K in Ks])
    b = jnp.stack([-res_fn(u0, tp, bc, ts) for tp in tps])
    # reference: one direct solve per case
    direct = fe.DirectSolverOptions(solver="cudss" if fe.solvers.options.has_cudss() else "spsolve")
    solver = fe.create_solver(problem, bc, direct, linear=True, traced_params=tps[0], traced_structure=ts)
    ref = jnp.stack([solver(tp, u0, traced_structure=ts).dofs - u0 for tp in tps])
    return dict(problem=problem, bc=bc, ts=ts, values=values, indptr=Ks[0].indptr,
                indices=Ks[0].indices, b=b, ref=ref, N=N)


def _rel(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def test_batched_pcg_plain_cg_matches_direct(batch):
    d = batch
    X, info = batched_pcg(d["values"], d["indptr"], d["indices"], d["b"], None, tol=1e-12, maxiter=5000)
    assert _rel(X, d["ref"]) < 1e-7
    assert bool(info.converged.all())
    assert int(info.iterations.max()) > 1


def test_batched_pcg_jacobi(batch):
    d = batch
    op = fe.LinearOperator(values=d["values"], indptr=d["indptr"], indices=d["indices"],
                           structure=d["ts"], problem=d["problem"], batched=True)
    assert op.diag.shape == (d["N"], d["b"].shape[1])
    M = fe.jacobi()(op)
    X, info = batched_pcg(d["values"], d["indptr"], d["indices"], d["b"], M, tol=1e-12, maxiter=5000)
    assert _rel(X, d["ref"]) < 1e-7
    assert bool(info.converged.all())


def test_batched_pcg_shared_factor(batch):
    if not fe.solvers.options.has_cudss():
        pytest.skip("cuDSS not available")
    d = batch
    op = fe.LinearOperator(values=d["values"], indptr=d["indptr"], indices=d["indices"], batched=True)
    M = fe.shared_factor()(op)
    X, info = batched_pcg(d["values"], d["indptr"], d["indices"], d["b"], M, tol=1e-10, maxiter=200)
    assert _rel(X, d["ref"]) < 1e-7
    assert bool(info.converged.all())
    # the shared factor is a good preconditioner for a mild spread: far fewer
    # iterations than plain CG
    _, plain = batched_pcg(d["values"], d["indptr"], d["indices"], d["b"], None, tol=1e-10, maxiter=5000)
    assert int(info.iterations.max()) * 3 < int(plain.iterations.max())
    # under jit, too
    f = jax.jit(lambda V, B: batched_pcg(V, d["indptr"], d["indices"], B,
                                         fe.shared_factor()(fe.LinearOperator(values=V, indptr=d["indptr"], indices=d["indices"], batched=True)),
                                         tol=1e-10, maxiter=200)[0])
    assert _rel(f(d["values"], d["b"]), d["ref"]) < 1e-7


def test_shared_factor_single_operator_converges_in_one_iteration(batch):
    if not fe.solvers.options.has_cudss():
        pytest.skip("cuDSS not available")
    d = batch
    v = d["values"][0]
    op = fe.LinearOperator(values=v, indptr=d["indptr"], indices=d["indices"])
    M = fe.shared_factor()(op)                       # M = K⁻¹ exactly
    X, info = batched_pcg(v, d["indptr"], d["indices"], d["b"][:1], jax.vmap(M), tol=1e-10, maxiter=50)
    assert _rel(X[0], d["ref"][0]) < 1e-7
    assert int(info.iterations[0]) <= 2


def test_batched_pcg_converged_cases_stop(batch):
    d = batch
    # a zero right-hand side is converged at start and must stay zero
    b = d["b"].at[0].set(0.0)
    X, info = batched_pcg(d["values"], d["indptr"], d["indices"], b, None, tol=1e-12, maxiter=5000)
    assert float(jnp.abs(X[0]).max()) == 0.0
    assert int(info.iterations[0]) == 0
    assert _rel(X[1:], d["ref"][1:]) < 1e-7
