"""Tests for feax.asd (automatic sparse differentiation via asdex) and the
solver paths it enables: assembled extra_residual_fn (direct solvers) and the
assembled reduced operator PᵀJP (direct / AMG for periodic problems)."""

import numpy as onp
import pytest
import scipy.sparse as sp

import jax
import jax.numpy as jnp

import feax as fe
from feax import asd
from feax.csr import CSRMatrix, transpose_with_maps


# ---------------------------------------------------------------------------
# Core: factories and pattern algebra
# ---------------------------------------------------------------------------

def test_sparse_jacobian_fn_matches_dense():
    def f(x):
        return jnp.sin(x[1:] * x[:-1]) + x[1:] ** 2

    x = jnp.linspace(0.3, 1.2, 30)
    jac_fn, pat = asd.sparse_jacobian_fn(f, x)
    J = jax.jit(jac_fn)(x)
    Jd = jax.jacfwd(f)(x)
    assert pat.shape == (29, 30)
    assert onp.allclose(onp.asarray(J.todense()), onp.asarray(Jd), atol=1e-12)


def test_sparse_hessian_fn_matches_dense():
    def g(x):
        return jnp.sum((x[1:] - x[:-1]) ** 2 * x[1:]) + jnp.sum(x ** 3)

    x = jnp.linspace(0.3, 1.2, 25)
    hess_fn, _ = asd.sparse_hessian_fn(g, x)
    H = jax.jit(hess_fn)(x)
    Hd = jax.hessian(g)(x)
    assert onp.allclose(onp.asarray(H.todense()), onp.asarray(Hd), atol=1e-12)


def test_operator_assembler_linear_op():
    n = 25
    A = sp.diags([onp.full(n - 1, -1.0), onp.full(n, 2.0), onp.full(n - 1, -1.0)],
                 [-1, 0, 1]).tocsr()
    Aj = jnp.asarray(A.todense())
    K = asd.operator_assembler(A)(lambda v: Aj @ v)
    assert onp.allclose(onp.asarray(K.todense()), A.todense())


def test_merge_csr_patterns_and_transpose():
    n = 20
    A = sp.diags([onp.full(n - 1, -1.0), onp.full(n, 2.0), onp.full(n - 1, -1.0)],
                 [-1, 0, 1]).tocsr()
    extra = sp.coo_matrix((onp.ones(2), ([0, n - 1], [n - 1, 0])), shape=(n, n)).tocsr()
    m = asd.merge_csr_patterns(A, extra)
    data = (jnp.zeros(m["nnz"])
            .at[m["slots_a"]].add(jnp.asarray(A.data))
            .at[m["slots_b"]].add(jnp.asarray(extra.data)))
    M = CSRMatrix(data, m["indptr"], m["indices"], m["shape"])
    ref = (A + extra).todense()
    assert onp.allclose(onp.asarray(M.todense()), ref)
    MT = transpose_with_maps(M, m["T_perm"], m["T_indptr"], m["T_indices"])
    assert onp.allclose(onp.asarray(MT.todense()), ref.T)


# ---------------------------------------------------------------------------
# FE fixtures
# ---------------------------------------------------------------------------

class _Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress


def _make_problem(dims=(4, 3, 3)):
    grid = fe.StructuredGrid(dims)
    mesh = grid.to_mesh()
    problem = _Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    E_cells = jnp.full(onp.asarray(mesh.cells).shape[0], 100.0)
    tp = fe.TracedParams(volume_vars=(E_cells,))
    return grid, mesh, problem, E_cells, tp


def test_verify_jacobian_pattern():
    _, _, problem, _, tp = _make_problem((3, 3, 3))
    report = asd.verify_jacobian_pattern(problem, tp)
    assert report["ok"]
    assert report["coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Assembled extra_residual_fn path (direct solver on merged pattern)
# ---------------------------------------------------------------------------

def test_extra_residual_direct_path():
    grid, mesh, problem, E_cells, tp = _make_problem()
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 4.0), "x", 0.05),
    ]).create_bc(problem)

    # Cubic spring between two NON-adjacent nodes (a coupling outside the mesh
    # connectivity; one endpoint is on the fixed face -> exercises BC masking)
    # + a cubic ground spring.
    i = int(grid.node_id(0, 1, 1)) * 3 + 1
    j = int(grid.node_id(4, 2, 2)) * 3 + 1
    g = int(grid.node_id(2, 1, 1)) * 3

    def extra_residual_fn(sol):
        d = sol[i] - sol[j]
        f = 5.0 * (d + d ** 3)
        r = jnp.zeros_like(sol)
        r = r.at[i].add(f).at[j].add(-f)
        return r.at[g].add(2.0 * sol[g] ** 3 + 0.5 * sol[g])

    newton = fe.NewtonOptions(tol=1e-10, max_iter=30)
    common = dict(newton_options=newton, linear=False, traced_params=tp)
    ig = fe.zero_like_initial_guess(problem, bc)

    solver_hyb = fe.create_solver(
        problem, bc,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12, atol=1e-14),
        extra_residual_fn=extra_residual_fn, **common)
    solver_dir = fe.create_solver(
        problem, bc, solver_options=fe.DirectSolverOptions(solver="spsolve"),
        extra_residual_fn=extra_residual_fn, **common)

    sol_hyb, sol_dir = solver_hyb(tp, ig), solver_dir(tp, ig)
    ref = float(jnp.max(jnp.abs(sol_hyb)))
    assert float(jnp.max(jnp.abs(sol_dir - sol_hyb))) < 1e-8 * ref

    # the extra term must actually deform the solution
    sol_plain = fe.create_solver(
        problem, bc, solver_options=fe.DirectSolverOptions(solver="spsolve"),
        **common)(tp, ig)
    assert float(jnp.max(jnp.abs(sol_dir - sol_plain))) > 1e-6

    # gradients: assembled-direct adjoint == hybrid adjoint == FD
    def make_loss(solver):
        return lambda tp_: jnp.sum(solver(tp_, ig) ** 2)

    g_dir = jax.grad(make_loss(solver_dir))(tp).volume_vars[0]
    g_hyb = jax.grad(make_loss(solver_hyb))(tp).volume_vars[0]
    scale = float(jnp.max(jnp.abs(g_hyb))) + 1e-30
    assert float(jnp.max(jnp.abs(g_dir - g_hyb))) / scale < 1e-6

    eps = 1e-5
    e0 = float(make_loss(solver_dir)(tp))
    tp_p = fe.TracedParams(volume_vars=(E_cells.at[7].add(eps),))
    fd = (float(make_loss(solver_dir)(tp_p)) - e0) / eps
    assert float(g_dir[7]) == pytest.approx(fd, rel=1e-3)


# ---------------------------------------------------------------------------
# Assembled reduced operator PᵀJP (periodic problems, direct / AMG)
# ---------------------------------------------------------------------------

def _make_periodic():
    from feax.flat.pbc import PeriodicPairing, prolongation_matrix

    L = 4
    grid, mesh, problem, _, _ = _make_problem((L, 3, 3))
    n_cells = onp.asarray(mesh.cells).shape[0]
    E_cells = jnp.asarray(1.0 + 0.5 * onp.random.default_rng(0).random(n_cells)) * 100.0
    tp = fe.TracedParams(volume_vars=(E_cells,))

    pairings = [PeriodicPairing(
        location_master=lambda p: jnp.isclose(p[0], 0.0, atol=1e-8),
        location_slave=lambda p: jnp.isclose(p[0], float(L), atol=1e-8),
        mapping=lambda p: p + jnp.array([float(L), 0.0, 0.0]),
        vec=c) for c in range(3)]
    P = prolongation_matrix(pairings, mesh, vec=3)

    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[2], 0.0), "all", 0.0),
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[2], 3.0), "z", -0.05),
    ]).create_bc(problem)
    return grid, problem, P, bc, E_cells, tp, L


def test_reduced_direct_assembled():
    grid, problem, P, bc, E_cells, tp, L = _make_periodic()
    ig = fe.zero_like_initial_guess(problem, bc)

    s_cg = fe.create_solver(problem, bc, P=P,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12, atol=1e-14))
    s_dir = fe.create_solver(problem, bc, P=P,
        solver_options=fe.DirectSolverOptions(solver="spsolve"))

    sol_cg, sol_dir = s_cg(tp, ig), s_dir(tp, ig)
    ref = float(jnp.max(jnp.abs(sol_cg)))
    assert float(jnp.max(jnp.abs(sol_dir - sol_cg))) < 1e-8 * ref

    # periodicity holds exactly on the assembled path
    i0, iL = int(grid.node_id(0, 1, 2)), int(grid.node_id(L, 1, 2))
    u = onp.asarray(sol_dir).reshape(-1, 3)
    assert onp.abs(u[i0] - u[iL]).max() < 1e-12

    # gradient vs matrix-free adjoint and FD
    def make_loss(solver):
        return lambda tp_: jnp.sum(solver(tp_, ig) ** 2)

    g_dir = jax.grad(make_loss(s_dir))(tp).volume_vars[0]
    g_cg = jax.grad(make_loss(s_cg))(tp).volume_vars[0]
    scale = float(jnp.max(jnp.abs(g_cg))) + 1e-30
    assert float(jnp.max(jnp.abs(g_dir - g_cg))) / scale < 1e-6

    eps = 1e-4
    e0 = float(make_loss(s_dir)(tp))
    tp_p = fe.TracedParams(volume_vars=(E_cells.at[5].add(eps),))
    fd = (float(make_loss(s_dir)(tp_p)) - e0) / eps
    assert float(g_dir[5]) == pytest.approx(fd, rel=5e-3)


def test_reduced_amg_assembled():
    pytest.importorskip("pyamg")
    pytest.importorskip("amjax")
    _, problem, P, bc, _, tp, _ = _make_periodic()
    ig = fe.zero_like_initial_guess(problem, bc)

    s_cg = fe.create_solver(problem, bc, P=P,
        solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12, atol=1e-14))
    s_amg = fe.create_solver(problem, bc, P=P,
        solver_options=fe.AMGSolverOptions(solver="cg", tol=1e-12, atol=1e-14))

    sol_cg, sol_amg = s_cg(tp, ig), s_amg(tp, ig)
    ref = float(jnp.max(jnp.abs(sol_cg)))
    assert float(jnp.max(jnp.abs(sol_amg - sol_cg))) < 1e-7 * ref
