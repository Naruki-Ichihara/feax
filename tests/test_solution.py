"""Tests for feax.Solution — the typed solver return value.

Solvers return a Solution by default (create_solver(..., return_solution=True)
is the default); pass return_solution=False to get a raw flat array."""

import numpy as onp
import pytest

import jax
import jax.numpy as jnp

import feax as fe
from feax.solution import Solution


class _Elast(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            nu = 0.3
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
        return stress


def _setup(dims=(4, 3, 3)):
    grid = fe.StructuredGrid(dims)
    mesh = grid.to_mesh()
    problem = _Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    E_cells = jnp.full(onp.asarray(mesh.cells).shape[0], 100.0)
    tp = fe.TracedParams(volume_vars=(E_cells,))
    bc = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], float(dims[0])), "x", 0.05),
    ]).create_bc(problem)
    return grid, mesh, problem, E_cells, tp, bc


def test_solution_wraps_and_matches_flat():
    _, mesh, problem, _, tp, bc = _setup()
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-12)
    ig = fe.zero_like_initial_guess(problem, bc)

    flat = fe.create_solver(problem, bc, solver_options=opts, linear=True)(tp, ig)
    sol = fe.create_solver(problem, bc, solver_options=opts, linear=True,
                           return_solution=True)(tp, ig)

    assert isinstance(sol, Solution)
    n_nodes = onp.asarray(mesh.points).shape[0]
    assert sol.num_fields == 1 and sol.layout == ((n_nodes, 3),)
    assert onp.allclose(onp.asarray(sol.flat), onp.asarray(flat))

    # field access replaces unflatten_fn_sol_list
    u_ref = problem.unflatten_fn_sol_list(flat)[0]
    assert onp.allclose(onp.asarray(sol.field(0)), onp.asarray(u_ref))
    # __getitem__ delegates to the flat vector (array semantics)
    assert float(sol[3]) == float(sol.flat[3])
    assert onp.allclose(onp.asarray(sol[10:20]), onp.asarray(sol.flat[10:20]))

    # node_var chaining bridge (vec>1 needs component)
    with pytest.raises(ValueError):
        sol.node_var()
    assert onp.allclose(onp.asarray(sol.node_var(component=0)),
                        onp.asarray(u_ref[:, 0]))
    # TracedParams.node_var_from_solution accepts a Solution too
    assert onp.allclose(
        onp.asarray(fe.TracedParams.node_var_from_solution(problem, sol, component=0)),
        onp.asarray(u_ref[:, 0]))


def test_solution_array_protocols_and_reuse():
    _, _, problem, _, tp, bc = _setup()
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-12)
    solver = fe.create_solver(problem, bc, solver_options=opts, linear=True,
                              return_solution=True)
    ig = fe.zero_like_initial_guess(problem, bc)
    sol = solver(tp, ig)

    # numpy protocol
    assert onp.asarray(sol).shape == sol.shape
    # jax protocol
    assert float(jnp.max(jnp.abs(sol))) == float(jnp.max(jnp.abs(sol.flat)))
    # a Solution can be fed back as the next initial_guess (unwrapped inside)
    sol2 = solver(tp, sol)
    assert isinstance(sol2, Solution)
    assert onp.allclose(onp.asarray(sol2.flat), onp.asarray(sol.flat), atol=1e-10)


def test_solution_grad_flows():
    _, _, problem, E_cells, tp, bc = _setup()
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-12)
    ig = fe.zero_like_initial_guess(problem, bc)

    solver_flat = fe.create_solver(problem, bc, solver_options=opts, linear=True)
    solver_sol = fe.create_solver(problem, bc, solver_options=opts, linear=True,
                                  return_solution=True)

    g_flat = jax.grad(lambda t: jnp.sum(solver_flat(t, ig) ** 2))(tp).volume_vars[0]
    g_sol = jax.grad(lambda t: jnp.sum(solver_sol(t, ig).flat ** 2))(tp).volume_vars[0]
    g_field = jax.grad(lambda t: jnp.sum(solver_sol(t, ig).field(0) ** 2))(tp).volume_vars[0]

    assert onp.allclose(onp.asarray(g_flat), onp.asarray(g_sol))
    assert onp.allclose(onp.asarray(g_flat), onp.asarray(g_field))


def test_solution_scatter_and_narrowband_interop():
    grid, mesh, _, _, _, _ = _setup()
    n_cells = onp.asarray(mesh.cells).shape[0]
    band = fe.NarrowBand(mesh, onp.arange(0, n_cells, 2))

    # a Solution on the band mesh scatters like a flat vector
    u_band = onp.random.default_rng(0).random(band.num_band_nodes * 3)
    sol = Solution(jnp.asarray(u_band), ((band.num_band_nodes, 3),))
    full = band.scatter_sol(sol, vec=3)
    assert onp.allclose(band.gather_sol(full, vec=3), u_band)


def test_solution_as_direct_coefficient_scalar():
    """A thermal Solution passed straight into the mechanical TracedParams."""
    grid = fe.StructuredGrid((4, 3, 3))
    mesh = grid.to_mesh()

    class Thermal(fe.Problem):
        def get_tensor_map(self):
            return lambda u_grad, k: k * u_grad

    class ThermoElastic(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, E, T):
                nu, alpha = 0.3, 1e-3
                mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
                eps = 0.5 * (u_grad + u_grad.T) - alpha * T * jnp.eye(3)
                return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
            return stress

    thermal = Thermal(mesh, vec=1, dim=3, ele_type="HEX8")
    bc_th = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 4.0), "all", 100.0),
    ]).create_bc(thermal)
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-12)
    k_cells = fe.TracedParams.create_cell_var(thermal, 1.0)

    mech = ThermoElastic(mesh, vec=3, dim=3, ele_type="HEX8")
    bc_me = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
    ]).create_bc(mech)
    E_cells = fe.TracedParams.create_cell_var(mech, 100.0)
    solver_th = fe.create_solver(thermal, bc_th, solver_options=opts, linear=True,
                                 return_solution=True)
    solver_me = fe.create_solver(mech, bc_me, solver_options=opts, linear=True)
    ig_th = fe.zero_like_initial_guess(thermal, bc_th)
    ig_me = fe.zero_like_initial_guess(mech, bc_me)

    T = solver_th(fe.TracedParams(volume_vars=(k_cells,)), ig_th)

    # direct pass == manual node_var bridge
    u_direct = solver_me(fe.TracedParams(volume_vars=(E_cells, T)), ig_me)
    u_manual = solver_me(fe.TracedParams(volume_vars=(E_cells, T.node_var())), ig_me)
    assert onp.allclose(onp.asarray(u_direct), onp.asarray(u_manual))
    assert float(jnp.max(jnp.abs(u_direct))) > 1e-3       # bar actually expands

    # end-to-end differentiability: d(mech response)/d(thermal conductivity)
    def loss(k):
        T_ = solver_th(fe.TracedParams(volume_vars=(k,)), ig_th)
        u_ = solver_me(fe.TracedParams(volume_vars=(E_cells, T_)), ig_me)
        return jnp.sum(u_ ** 2)

    g = jax.grad(loss)(k_cells)
    eps = 1e-4
    fd = (float(loss(k_cells.at[3].add(eps))) - float(loss(k_cells))) / eps
    assert float(g[3]) == pytest.approx(fd, rel=1e-3, abs=1e-12)


def test_solution_as_direct_coefficient_vector_and_errors():
    grid = fe.StructuredGrid((3, 3, 3))
    mesh = grid.to_mesh()
    problem = _Elast(mesh, vec=3, dim=3, ele_type="HEX8")
    n_nodes = onp.asarray(mesh.points).shape[0]

    # vector Solution -> (num_nodes, 3) node var, recognized by the assembler
    from feax.assembler import classify_volume_var, _VAR_NODE
    u_prev = Solution(jnp.arange(n_nodes * 3, dtype=float), ((n_nodes, 3),))
    tp = fe.TracedParams(volume_vars=(u_prev,))
    assert tp.volume_vars[0].shape == (n_nodes, 3)
    kind, fe_idx = classify_volume_var(problem, tp.volume_vars[0])
    assert kind == _VAR_NODE

    # multi-field Solutions are ambiguous
    multi = Solution(jnp.zeros(n_nodes * 4, dtype=float),
                     ((n_nodes, 3), (n_nodes, 1)))
    with pytest.raises(ValueError):
        fe.TracedParams(volume_vars=(multi,))


def test_cmg_return_solution():
    grid = fe.StructuredGrid((8, 4, 4))
    cmg = fe.NarrowBandCMG(grid, lambda ni, nj, nk, nx, ny, nz: ni == 0,
                           nu=0.3, penal=3.0, bucket=64)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    levels = cmg.build(active)
    tip = grid.nodes_where(lambda I, J, K: (I == 8) & (K == 0))
    b = cmg.load_vector(levels, tip, comp=2, value=-1.0)
    opts = fe.KrylovSolverOptions(solver="cg", tol=1e-10, maxiter=500)

    rho = jnp.full(active.size, 0.5)
    u_flat = cmg.create_solver(levels, b, solver_options=opts)(rho)
    sol = cmg.create_solver(levels, b, solver_options=opts,
                            return_solution=True)(rho)
    assert isinstance(sol, Solution)
    assert sol.layout == ((levels[0]["nnode"], 3),)
    assert onp.allclose(onp.asarray(sol.flat), onp.asarray(u_flat))
    assert sol.field(0).shape == (levels[0]["nnode"], 3)
    # compliance via the array protocol
    assert float(jnp.dot(jnp.asarray(b), sol)) == pytest.approx(
        float(jnp.dot(jnp.asarray(b), u_flat)))
