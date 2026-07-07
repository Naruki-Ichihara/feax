"""Tests for feax.solvers.time_solver: ImplicitPipeline backward-Euler stepping,
the Solution-carry unwrap in step(), and the run() driver (history, callbacks,
TimeResult)."""

import numpy as onp
import pytest

import jax.numpy as jnp

import feax as fe
from feax.solution import Solution


class _Heat(fe.Problem):
    """Transient heat: (T - T_old)/dt + div(-k grad T) = 0 (backward Euler)."""

    dt = 0.1

    def get_tensor_map(self):
        return lambda grad_T, T_old: 1.0 * grad_T

    def get_mass_map(self):
        def mass(T, x, T_old):
            return (T - T_old) / self.dt
        return mass


def _make_pipeline():
    class HeatPipeline(fe.ImplicitPipeline):
        def build(self, mesh):
            self.problem = _Heat(mesh, vec=1, dim=3, ele_type="HEX8")
            self.n_nodes = onp.asarray(mesh.points).shape[0]
            self.x = jnp.asarray(onp.asarray(mesh.points)[:, 0])
            bc = fe.DirichletBCConfig([
                fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 0.0), "all", 0.0),
                fe.DirichletBCSpec(lambda p: jnp.isclose(p[0], 4.0), "all", 0.0),
            ]).create_bc(self.problem)
            self.solver = fe.create_solver(
                self.problem, bc,
                solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-12),
                linear=True)

        def initial_state(self):
            # smooth profile satisfying the Dirichlet BC (linear=True assumes
            # the guess sits on the constrained values; a step profile would
            # also overshoot via the consistent mass matrix)
            return 100.0 * jnp.sin(jnp.pi * self.x / 4.0)

        def update_vars(self, state, t, dt):
            return fe.TracedParams(volume_vars=(state,))

        def monitor(self, state, step, t):
            return {"T_max": float(jnp.max(state))}

    return HeatPipeline()


def test_implicit_step_unwraps_solution_carry():
    grid = fe.StructuredGrid((4, 3, 3))
    mesh = grid.to_mesh()
    pipe = _make_pipeline()
    pipe.build(mesh)
    state0 = pipe.initial_state()

    # the underlying solver returns a Solution by default ...
    tp = pipe.update_vars(state0, 0.0, _Heat.dt)
    assert isinstance(pipe.solver(tp, state0), Solution)
    # ... but step() must keep the carried state a flat vector so the carry
    # pytree structure is stable across steps
    state1 = pipe.step(state0, 0.0, _Heat.dt)
    assert not isinstance(state1, Solution)
    assert state1.shape == state0.shape
    # and a second step accepts the carried state unchanged
    state2 = pipe.step(state1, _Heat.dt, _Heat.dt)
    assert state2.shape == state0.shape

    # physics: interior cools monotonically towards the 0-BC, no overshoot
    assert float(jnp.max(state1)) < 100.0
    assert float(jnp.max(state2)) < float(jnp.max(state1))
    assert float(jnp.min(state2)) > -1e-8


def test_run_time_history_and_callbacks():
    grid = fe.StructuredGrid((4, 3, 3))
    mesh = grid.to_mesh()
    pipe = _make_pipeline()

    seen = []
    cb = fe.Callback(fn=lambda state, step, t: seen.append((step, t)), every=2)
    result = fe.run_time(pipe, mesh, fe.TimeConfig(dt=_Heat.dt, t_end=5 * _Heat.dt,
                                                   print_every=10),
                         callbacks=[cb])

    assert isinstance(result, fe.TimeResult)
    assert result.n_steps == 5
    assert result.t_final == pytest.approx(5 * _Heat.dt)
    assert not isinstance(result.final_state, Solution)

    # monitor scalars: initial record + one per step, decaying monotonically
    assert len(result.history["step"]) == result.n_steps + 1
    tmax = result.history["T_max"]
    assert tmax[0] == pytest.approx(100.0)
    assert all(b < a for a, b in zip(tmax, tmax[1:]))

    assert [step for step, _ in seen] == [2, 4]
