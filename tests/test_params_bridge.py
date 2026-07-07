"""Tests for the TracedParams bridges: cmg solver tp acceptance,
NarrowBand.gather_params/scatter_params, SparseDesign.traced_params/updated,
and TracedParams.node_var_from_solution (staggered chaining)."""

import numpy as onp
import pytest

import jax
import jax.numpy as jnp

import feax as fe


def _grid_mesh(dims=(4, 3, 3)):
    grid = fe.StructuredGrid(dims)
    return grid, grid.to_mesh()


# ---------------------------------------------------------------------------
# NarrowBand <-> TracedParams
# ---------------------------------------------------------------------------

def test_narrowband_gather_scatter_params():
    grid, mesh = _grid_mesh()
    n_cells = onp.asarray(mesh.cells).shape[0]
    n_nodes = onp.asarray(mesh.points).shape[0]
    band = fe.NarrowBand(mesh, onp.arange(0, n_cells, 2))

    rng = onp.random.default_rng(0)
    cell_var = jnp.asarray(rng.random(n_cells))
    node_var = jnp.asarray(rng.random(n_nodes))
    tp_full = fe.TracedParams(volume_vars=(cell_var, node_var))

    tp_band = band.gather_params(tp_full)
    assert tp_band.volume_vars[0].shape == (band.num_active_cells,)
    assert tp_band.volume_vars[1].shape == (band.num_band_nodes,)
    assert onp.allclose(tp_band.volume_vars[0], onp.asarray(cell_var)[band.active_cells])
    assert onp.allclose(tp_band.volume_vars[1], onp.asarray(node_var)[band.node_map])

    tp_back = band.scatter_params(tp_band, fill=0.0)
    assert onp.allclose(onp.asarray(tp_back.volume_vars[0])[band.active_cells],
                        tp_band.volume_vars[0])
    assert onp.asarray(tp_back.volume_vars[0]).shape == (n_cells,)

    # differentiable through the gather
    g = jax.grad(lambda tp: jnp.sum(band.gather_params(tp).volume_vars[0] ** 2))(tp_full)
    assert onp.allclose(onp.asarray(g.volume_vars[0])[band.active_cells],
                        2 * onp.asarray(cell_var)[band.active_cells])

    # surface vars are rejected (band boundary differs)
    tp_surf = fe.TracedParams(volume_vars=(cell_var,), surface_vars=[(jnp.ones(3),)])
    with pytest.raises(ValueError):
        band.gather_params(tp_surf)


# ---------------------------------------------------------------------------
# SparseDesign <-> TracedParams
# ---------------------------------------------------------------------------

def test_sparse_design_traced_params_roundtrip():
    grid, _ = _grid_mesh()
    active = grid.cells_where(lambda c: c[:, 0] < 2.0)
    sd = fe.SparseDesign.uniform(active, 0.5)

    tp = sd.traced_params(active)
    assert isinstance(tp, fe.TracedParams)
    assert tp.volume_vars[0].shape == (active.size,)
    assert onp.allclose(tp.volume_vars[0], 0.5)

    # extra vars ride along
    nu = jnp.full(active.size, 0.3)
    tp2 = sd.traced_params(active, extra_vars=(nu,))
    assert len(tp2.volume_vars) == 2

    # write-back accepts a TracedParams or a bare array
    new_vals = tp.volume_vars[0] * 2.0
    sd2 = sd.updated(active, fe.TracedParams(volume_vars=(new_vals,)))
    assert onp.allclose(sd2.gather(active), 1.0)
    sd3 = sd.updated(active, onp.full(active.size, 0.25))
    assert onp.allclose(sd3.gather(active), 0.25)
    # original store is unchanged (functional update)
    assert onp.allclose(sd.gather(active), 0.5)


# ---------------------------------------------------------------------------
# cmg solver accepts TracedParams
# ---------------------------------------------------------------------------

def test_cmg_solver_accepts_traced_params():
    grid = fe.StructuredGrid((8, 4, 4))
    fixed = lambda ni, nj, nk, nx, ny, nz: ni == 0
    cmg = fe.NarrowBandCMG(grid, fixed, nu=0.3, penal=3.0, bucket=64)
    active = onp.arange(grid.num_cells, dtype=onp.int64)
    levels = cmg.build(active)
    tip = grid.nodes_where(lambda I, J, K: (I == 8) & (K == 0))
    b = cmg.load_vector(levels, tip, comp=2, value=-1.0)
    solver = cmg.create_solver(
        levels, b, solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10,
                                                         maxiter=500))
    rho = jnp.full(active.size, 0.5)
    tp = fe.TracedParams(volume_vars=(rho,))

    u_arr = solver(rho)
    u_tp = solver(tp)
    assert onp.allclose(onp.asarray(u_arr), onp.asarray(u_tp))

    # grad w.r.t. TracedParams comes back TracedParams-shaped and matches
    g_arr = jax.grad(lambda r: jnp.dot(jnp.asarray(b), solver(r)))(rho)
    g_tp = jax.grad(lambda t: jnp.dot(jnp.asarray(b), solver(t)))(tp)
    assert onp.allclose(onp.asarray(g_arr), onp.asarray(g_tp.volume_vars[0]))

    # compliance_and_dc accepts both forms
    c1, dc1 = cmg.compliance_and_dc(levels, rho, u_arr)
    c2, dc2 = cmg.compliance_and_dc(levels, tp, u_tp)
    assert c1 == pytest.approx(c2)
    assert onp.allclose(dc1, dc2)

    # SparseDesign -> cmg in one hop
    sd = fe.SparseDesign.uniform(active, 0.5)
    u_sd = solver(sd.traced_params(active))
    assert onp.allclose(onp.asarray(u_sd), onp.asarray(u_arr))


# ---------------------------------------------------------------------------
# Staggered chaining: solution -> node var of the next problem
# ---------------------------------------------------------------------------

def test_node_var_from_solution_chains_thermal_to_mechanical():
    _, mesh = _grid_mesh((4, 3, 3))

    class Thermal(fe.Problem):
        def get_tensor_map(self):
            return lambda u_grad, k: k * u_grad

    class ThermoElastic(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, E, T_node):
                nu, alpha = 0.3, 1e-3
                mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
                eps = 0.5 * (u_grad + u_grad.T) - alpha * T_node * jnp.eye(3)
                return lam * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
            return stress

    thermal = Thermal(mesh, vec=1, dim=3, ele_type="HEX8")
    k_cells = fe.TracedParams.create_cell_var(thermal, 1.0)
    tp_th = fe.TracedParams(volume_vars=(k_cells,))
    bc_th = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 4.0), "all", 100.0),
    ]).create_bc(thermal)
    T = fe.create_solver(
        thermal, bc_th, solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12),
        linear=True)(tp_th, fe.zero_like_initial_guess(thermal, bc_th))

    # chain: flat thermal solution -> node var of the mechanical problem
    T_nodes = fe.TracedParams.node_var_from_solution(thermal, T)
    assert T_nodes.shape == (onp.asarray(mesh.points).shape[0],)
    assert float(jnp.max(T_nodes)) == pytest.approx(100.0)

    mech = ThermoElastic(mesh, vec=3, dim=3, ele_type="HEX8")
    E_cells = fe.TracedParams.create_cell_var(mech, 100.0)
    tp_me = fe.TracedParams(volume_vars=(E_cells, T_nodes))
    bc_me = fe.DirichletBCConfig([
        fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
    ]).create_bc(mech)
    u = fe.create_solver(
        mech, bc_me, solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12),
        linear=True)(tp_me, fe.zero_like_initial_guess(mech, bc_me))

    # heated bar expands in +x
    ux = onp.asarray(u).reshape(-1, 3)[:, 0]
    assert ux.max() > 1e-3

    # vec>1 requires an explicit component
    with pytest.raises(ValueError):
        fe.TracedParams.node_var_from_solution(mech, u)
    ux_nodes = fe.TracedParams.node_var_from_solution(mech, u, component=0)
    assert onp.allclose(onp.asarray(ux_nodes), ux)
