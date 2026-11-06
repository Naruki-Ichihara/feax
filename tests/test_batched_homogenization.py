"""feax.flat.solver.create_batched_homogenization_solver: C_hom of many unit
cells at once (shared-factor / Jacobi preconditioned CG over designs x strain
cases), checked against the single-cell solver, including gradients and the
reuse of factorizations across evaluations."""
import numpy as onp
import pytest

import jax
import jax.numpy as jnp

import feax as fe
import feax.flat as flat
from feax import gene
from feax.solvers.options import has_cudss

jax.config.update("jax_enable_x64", True)

E0, NU = 100.0, 0.3
B = 3


class _Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(self.dim) + 2 * mu * eps
        return stress


class _BoxCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size=0.25):
        return fe.mesh.box_mesh((1.0, 1.0, 1.0), mesh_size=mesh_size, element_type="HEX8")


@pytest.fixture(scope="module")
def setup():
    cell = _BoxCell(mesh_size=0.25)
    mesh = cell.mesh
    problem = _Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    P = flat.pbc.prolongation_matrix(flat.pbc.periodic_bc_3D(cell, vec=3, dim=3), mesh, vec=3)
    bc = fe.DirichletBCConfig([]).create_bc(problem)
    rho = 0.2 + 0.8 * onp.random.default_rng(0).random((B, len(mesh.cells)))
    E = jnp.asarray(E0 * (1e-3 + rho ** 3))
    nu = fe.TracedParams.create_cell_var(problem, NU)
    single = flat.solver.create_homogenization_solver(
        problem, bc, P, mesh, solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12), dim=3)
    C_ref = onp.stack([onp.asarray(single(fe.TracedParams(volume_vars=(E[b], nu))).C_hom) for b in range(B)])
    return dict(problem=problem, P=P, mesh=mesh, bc=bc, E=E, nu=nu, single=single, C_ref=C_ref,
                in_axes=fe.TracedParams(volume_vars=(0, None)))


def _batched(s, **kw):
    return flat.solver.create_batched_homogenization_solver(
        s["problem"], s["bc"], s["P"], s["mesh"], dim=3, in_axes=s["in_axes"], tol=1e-11, **kw)


def _rel(a, b):
    return float(onp.abs(onp.asarray(a) - onp.asarray(b)).max() / onp.abs(onp.asarray(b)).max())


PCS = [("jacobi", None)] + ([("shared_factor", None), ("shared_factor", 2)] if has_cudss() else [])


@pytest.mark.parametrize("pc, share", PCS)
def test_batched_matches_single_cell_solver(setup, pc, share):
    s = setup
    r = _batched(s, preconditioner=pc, share=share)(fe.TracedParams(volume_vars=(s["E"], s["nu"])))
    assert r.C_hom.shape == (B, 6, 6) and r.iterations.shape == (B, 6)
    assert bool(onp.asarray(r.converged).all())
    assert _rel(r.C_hom, s["C_ref"]) < 1e-8
    C = onp.asarray(r.C_hom)
    assert onp.allclose(C, onp.swapaxes(C, 1, 2), atol=1e-8 * onp.abs(C).max())   # symmetric


@pytest.mark.skipif(not has_cudss(), reason="needs cuDSS")
def test_own_factorization_converges_in_one_iteration(setup):
    s = setup
    r = _batched(s, share=1)(fe.TracedParams(volume_vars=(s["E"], s["nu"])))
    assert int(onp.asarray(r.iterations).max()) <= 2
    assert _rel(r.C_hom, s["C_ref"]) < 1e-8


@pytest.mark.parametrize("pc, share", PCS)
def test_gradient_matches_single_cell_solver(setup, pc, share):
    s = setup
    w = jnp.asarray(onp.random.default_rng(1).standard_normal((6, 6)))
    nu, single = s["nu"], s["single"]
    g_ref = onp.stack([onp.asarray(jax.grad(
        lambda Eb: jnp.sum(w * single(fe.TracedParams(volume_vars=(Eb, nu))).C_hom))(s["E"][b])) for b in range(B)])
    bs = _batched(s, preconditioner=pc, share=share)
    g = jax.grad(lambda Ea: jnp.sum(w[None] * bs(fe.TracedParams(volume_vars=(Ea, nu))).C_hom))(s["E"])
    assert _rel(g, g_ref) < 1e-6


@pytest.mark.skipif(not has_cudss(), reason="needs cuDSS")
def test_reused_factorizations(setup):
    """Factors from factorize() precondition later evaluations (stale is fine)."""
    s = setup
    nu = s["nu"]
    bs = _batched(s, share=2)
    tp = fe.TracedParams(volume_vars=(s["E"], nu))
    fac = bs.factorize(tp)
    assert _rel(bs(tp, factors=fac).C_hom, s["C_ref"]) < 1e-8
    E2 = s["E"] * (1 + 0.05 * jnp.asarray(onp.random.default_rng(2).standard_normal(s["E"].shape)))
    r2 = bs(fe.TracedParams(volume_vars=(E2, nu)), factors=fac)
    C2 = onp.stack([onp.asarray(s["single"](fe.TracedParams(volume_vars=(E2[b], nu))).C_hom) for b in range(B)])
    assert bool(onp.asarray(r2.converged).all()) and _rel(r2.C_hom, C2) < 1e-8
    w = jnp.ones((6, 6))
    g_fac = jax.grad(lambda Ea: jnp.sum(w[None] * bs(fe.TracedParams(volume_vars=(Ea, nu)), factors=fac).C_hom))(s["E"])
    g_new = jax.grad(lambda Ea: jnp.sum(w[None] * bs(fe.TracedParams(volume_vars=(Ea, nu))).C_hom))(s["E"])
    assert _rel(g_fac, g_new) < 1e-6


def test_signature_and_fields(setup):
    s = setup
    bs = _batched(s, preconditioner="jacobi", return_fields=True)
    tp = fe.TracedParams(volume_vars=(s["E"], s["nu"]))
    sig = bs.operator_signature(tp)
    assert sig.shape[0] == B and onp.all(onp.isfinite(onp.asarray(sig)))
    order = gene.similarity_order(onp.asarray(sig))
    assert sorted(onp.asarray(order).tolist()) == list(range(B))
    r = bs(tp)
    assert r.u_totals.shape == (B, 6, s["problem"].num_total_dofs_all_vars)


def test_fe_vmap_of_single_cell_solver(setup):
    """fe.vmap(create_homogenization_solver(...)) is the batched solver."""
    s = setup
    pc = fe.shared_factor(scaling=True) if has_cudss() else fe.jacobi()
    batched = fe.vmap(s["single"], in_axes=s["in_axes"], preconditioner=pc, share=2,
                      options=fe.KrylovSolverOptions(solver="cg", tol=1e-11))
    assert isinstance(batched, flat.solver.BatchedHomogenizationSolver)
    assert batched.preconditioner == ("shared_factor" if has_cudss() else "jacobi")
    r = batched(fe.TracedParams(volume_vars=(s["E"], s["nu"])))
    assert r.C_hom.shape == (B, 6, 6) and bool(onp.all(onp.asarray(r.converged)))
    assert _rel(r.C_hom, s["C_ref"]) < 1e-7
    with pytest.raises(ValueError):
        fe.vmap(s["single"], in_axes=s["in_axes"], batch="rhs")
    with pytest.raises(TypeError):
        fe.vmap(s["single"], in_axes=s["in_axes"], options=fe.DirectSolverOptions(solver="cudss"))
