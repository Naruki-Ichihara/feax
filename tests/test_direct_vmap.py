"""Regression tests for direct sparse solvers under vmap."""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import feax as fe
from feax.problem import MatrixView
from feax.solvers.direct.spsolve import spsolve

_HAS_SKSPARSE = importlib.util.find_spec("sksparse") is not None

if _HAS_SKSPARSE:
    from feax.solvers.direct.cholmod import cholmod_solve
    from feax.solvers.direct.umfpack import umfpack_solve

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="direct sparse solvers are CPU-only",
)

_E = 70e3
_NU = 0.3
_TRACTION = 1e-3
_TOL = 1e-5
_SKSPARSE_ONLY = pytest.mark.skipif(not _HAS_SKSPARSE, reason="sksparse is not installed")
_SOLVER_PARAMS = [
    pytest.param("spsolve", id="spsolve"),
    pytest.param("umfpack", marks=_SKSPARSE_ONLY, id="umfpack"),
    pytest.param("cholmod", marks=_SKSPARSE_ONLY, id="cholmod"),
]


def _solver_entrypoints(solver_name):
    if solver_name == "spsolve":
        return spsolve, MatrixView.FULL
    if solver_name == "umfpack":
        return umfpack_solve, MatrixView.FULL
    if solver_name == "cholmod":
        return cholmod_solve, MatrixView.UPPER
    raise ValueError(f"Unsupported solver {solver_name}.")


def _spd_3x3_payload():
    values = jnp.array([4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0], dtype=jnp.float64)
    columns = jnp.array([0, 1, 0, 1, 2, 1, 2], dtype=jnp.int32)
    offsets = jnp.array([0, 2, 5, 7], dtype=jnp.int32)
    return values, columns, offsets


def _rhs_batch():
    return jnp.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, -1.0, 0.5],
            [-1.0, 0.25, 1.5],
        ],
        dtype=jnp.float64,
    )


def _values_batch(values):
    return jnp.stack([values, values * 1.1, values * 0.85], axis=0)


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_direct_vmap_shared_lhs_batched_rhs(solver_name):
    solve, _ = _solver_entrypoints(solver_name)
    values, columns, offsets = _spd_3x3_payload()
    rhs_batch = _rhs_batch()

    x_loop = jnp.stack(
        [solve(rhs, values, offsets, columns) for rhs in rhs_batch],
        axis=0,
    )
    x_vmap = jax.vmap(lambda rhs: solve(rhs, values, offsets, columns))(rhs_batch)

    assert np.allclose(np.asarray(x_vmap), np.asarray(x_loop), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_direct_vmap_batched_lhs_shared_rhs(solver_name):
    solve, _ = _solver_entrypoints(solver_name)
    values, columns, offsets = _spd_3x3_payload()
    values_batch = _values_batch(values)
    rhs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

    x_loop = jnp.stack(
        [solve(rhs, values_i, offsets, columns) for values_i in values_batch],
        axis=0,
    )
    x_vmap = jax.vmap(lambda values_i: solve(rhs, values_i, offsets, columns))(values_batch)

    assert np.allclose(np.asarray(x_vmap), np.asarray(x_loop), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_direct_vmap_batched_lhs_batched_rhs(solver_name):
    solve, _ = _solver_entrypoints(solver_name)
    values, columns, offsets = _spd_3x3_payload()
    values_batch = _values_batch(values)
    rhs_batch = _rhs_batch()

    x_loop = jnp.stack(
        [solve(rhs, values_i, offsets, columns) for rhs, values_i in zip(rhs_batch, values_batch)],
        axis=0,
    )
    x_vmap = jax.vmap(lambda rhs, values_i: solve(rhs, values_i, offsets, columns))(
        rhs_batch,
        values_batch,
    )

    assert np.allclose(np.asarray(x_vmap), np.asarray(x_loop), rtol=1e-10, atol=1e-10)


def _make_parametric_problem(simple_mesh, matrix_view, *, stiffness_from_internal, traction_from_internal):
    boundary_condition = lambda p: jnp.isclose(p[0], 10.0, _TOL)

    class ParametricLinearElasticity(fe.Problem):
        def get_tensor_map(self):
            def stress(u_grad, *volume_vars):
                youngs_modulus = _E if not stiffness_from_internal else jnp.mean(volume_vars[0])
                mu = youngs_modulus / (2 * (1 + _NU))
                lmbda = youngs_modulus * _NU / ((1 + _NU) * (1 - 2 * _NU))
                eps = 0.5 * (u_grad + u_grad.T)
                return lmbda * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps

            return stress

        def get_surface_maps(self):
            def surface_load(u, x, *surface_vars):
                traction = _TRACTION if not traction_from_internal else surface_vars[0]
                return jnp.array([0.0, 0.0, traction])

            return [surface_load]

    return ParametricLinearElasticity(
        simple_mesh,
        vec=3,
        dim=3,
        location_fns=[boundary_condition],
        matrix_view=matrix_view,
    )


def _create_direct_solver(problem, solver_name):
    left = lambda p: jnp.isclose(p[0], 0.0, _TOL)
    bc = fe.DirichletBCConfig(
        [fe.DirichletBCSpec(location=left, component="all", value=0.0)]
    ).create_bc(problem)
    return fe.create_solver(
        problem,
        bc,
        solver_options=fe.DirectSolverOptions(solver=solver_name),
        iter_num=1,
    )


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_feax_direct_vmap_shared_lhs_batched_rhs(simple_mesh, solver_name):
    _, matrix_view = _solver_entrypoints(solver_name)
    problem = _make_parametric_problem(
        simple_mesh,
        matrix_view,
        stiffness_from_internal=False,
        traction_from_internal=True,
    )
    solver = _create_direct_solver(problem, solver_name)

    traction = fe.InternalVars.create_uniform_surface_var(problem, _TRACTION)
    traction_batch = jnp.stack([traction, traction * 0.7, traction * 1.4], axis=0)

    def solve_case(traction_i):
        return solver(fe.InternalVars((), [(traction_i,)]))

    x_loop = jnp.stack([solve_case(traction_i) for traction_i in traction_batch], axis=0)
    x_vmap = jax.vmap(solve_case)(traction_batch)

    assert np.allclose(np.asarray(x_vmap), np.asarray(x_loop), rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_feax_direct_vmap_batched_lhs_shared_rhs(simple_mesh, solver_name):
    _, matrix_view = _solver_entrypoints(solver_name)
    problem = _make_parametric_problem(
        simple_mesh,
        matrix_view,
        stiffness_from_internal=True,
        traction_from_internal=False,
    )
    solver = _create_direct_solver(problem, solver_name)

    youngs_modulus = fe.InternalVars.create_node_var(problem, _E)
    youngs_modulus_batch = jnp.stack(
        [youngs_modulus, youngs_modulus * 0.8, youngs_modulus * 1.25],
        axis=0,
    )

    def solve_case(youngs_modulus_i):
        return solver(fe.InternalVars((youngs_modulus_i,), [tuple()]))

    x_loop = jnp.stack(
        [solve_case(youngs_modulus_i) for youngs_modulus_i in youngs_modulus_batch],
        axis=0,
    )
    x_vmap = jax.vmap(solve_case)(youngs_modulus_batch)

    assert np.allclose(np.asarray(x_vmap), np.asarray(x_loop), rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("solver_name", _SOLVER_PARAMS)
def test_feax_direct_vmap_batched_lhs_batched_rhs(simple_mesh, solver_name):
    _, matrix_view = _solver_entrypoints(solver_name)
    problem = _make_parametric_problem(
        simple_mesh,
        matrix_view,
        stiffness_from_internal=True,
        traction_from_internal=True,
    )
    solver = _create_direct_solver(problem, solver_name)

    youngs_modulus = fe.InternalVars.create_node_var(problem, _E)
    traction = fe.InternalVars.create_uniform_surface_var(problem, _TRACTION)
    youngs_modulus_batch = jnp.stack(
        [youngs_modulus, youngs_modulus * 1.25, youngs_modulus * 0.9],
        axis=0,
    )
    traction_batch = jnp.stack([traction, traction * 0.8, traction * 1.3], axis=0)

    def solve_case(youngs_modulus_i, traction_i):
        return solver(fe.InternalVars((youngs_modulus_i,), [(traction_i,)]))

    x_loop = jnp.stack(
        [
            solve_case(youngs_modulus_i, traction_i)
            for youngs_modulus_i, traction_i in zip(youngs_modulus_batch, traction_batch)
        ],
        axis=0,
    )
    x_vmap = jax.vmap(solve_case)(youngs_modulus_batch, traction_batch)

    assert np.allclose(np.asarray(x_vmap), np.asarray(x_loop), rtol=1e-8, atol=1e-12)
