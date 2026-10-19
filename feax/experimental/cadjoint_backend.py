"""feax as a cadjoint FEM solver backend — experimental.

Registers ``"feax"`` in :mod:`cadjoint.fem.backends` so a cadjoint study, the
playground viewer included, solves with feax instead of jax-fem. The mesh and
the boundary conditions arrive as plain arrays over cadjoint's ABI, so nothing
here knows about SDFs or the viewer.

cadjoint resolves ``"feax"`` to this class through a lazy factory and defaults
to it, so the playground and every ``Study.solve`` use it with no configuration.
:func:`register` is only needed to register under a different name.

FORWARD SOLVES ONLY. cadjoint's ABI asks for a solve differentiable in
``points``, because its shape optimization moves the mesh. feax is built the
other way round: :class:`feax.Problem` bakes ``JxW`` and ``shape_grads`` at
construction and traces material fields instead, which is what makes its
structured-grid solvers index arithmetic. Differentiating one of these calls
w.r.t. ``points`` raises rather than returning a wrong gradient.

For shape sensitivity with feax, keep the grid fixed and let the SDF move
through the material weight instead -- see
``examples/experimental/sdf_shape_optimization.py``, which is the same geometry
source with the derivative taken somewhere feax can honestly provide it.

The two ABI conventions this relies on, both shared with feax:

* a location function may take ``(point, index)``, so a node-index set becomes
  a predicate directly;
* a boundary face belongs to a patch when ALL of its vertices do.
"""
from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as onp

import feax as fe

__all__ = ["FeaxBackend", "register"]

_ELE_TYPE = {4: "TET4", 8: "HEX8", 10: "TET10"}


@contextlib.contextmanager
def _x64():
    """x64 for the duration of one solve, as the ABI requires of a backend."""
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", was)


def _concrete(points, what):
    if isinstance(points, jax.core.Tracer):
        raise NotImplementedError(
            f"the feax backend cannot differentiate {what} w.r.t. node positions: "
            "feax fixes the mesh at Problem construction. Keep the grid fixed and "
            "move the geometry through the material field instead "
            "(examples/experimental/sdf_shape_optimization.py)."
        )
    return onp.asarray(points, dtype=onp.float64)


def _membership(node_set):
    """A 2-arg location function selecting nodes by index."""
    idx = jnp.asarray(onp.asarray(node_set, dtype=onp.int32))

    def location(point, index):      # noqa: ARG001 - called as (point, index)
        return jnp.isin(index, idx)

    return location


def _cellwise(value, num_cells):
    """A scalar or a per-element array, as a (num_cells,) volume var."""
    arr = jnp.asarray(value, dtype=jnp.float64)
    return jnp.broadcast_to(arr, (num_cells,)) if arr.ndim == 0 else arr


class FeaxBackend:
    """Forward-only cadjoint solver backend built on feax."""

    name = "feax"

    def __init__(self, solver_options=None):
        self.solver_options = solver_options or fe.DirectSolverOptions()

    # ── heat conduction ────────────────────────────────────────────────────
    def thermal(self, points, cells, bcs, *, conductivity, source,
                base_points=None):
        """Solve -div(k grad T) = q; returns per-node temperature ``(N,)``."""
        with _x64():
            pts = _concrete(points, "the thermal solve")
            cells = onp.asarray(cells)
            mesh = fe.Mesh(points=pts, cells=cells,
                           ele_type=_ELE_TYPE[cells.shape[1]])
            flux = [float(v) for v in bcs.flux_values]

            class Thermal(fe.Problem):
                def get_tensor_map(self):
                    # Every volume var is passed to every kernel, in order.
                    return lambda grad_T, k, _q: k * grad_T

                def get_mass_map(self):
                    # residual += integral(v * mass_map), so the source enters
                    # negated to give -div(k grad T) = q.
                    return lambda T, _x, _k, q: -q * jnp.ones_like(T)

                def get_surface_maps(self):
                    # Likewise for a prescribed inflow: k grad(T).n = q.
                    return [(lambda T, _x, v=v: -v * jnp.ones_like(T))
                            for v in flux]

            problem = Thermal(
                mesh, vec=1, dim=3, ele_type=mesh.ele_type,
                location_fns=[_membership(n) for n in bcs.flux_nodes])
            bc = fe.DirichletBCConfig([
                fe.DirichletBCSpec(_membership(nodes), "all", float(value))
                for nodes, value in zip(bcs.dirichlet_nodes, bcs.dirichlet_values)
            ]).create_bc(problem)

            n_cells = problem.fes[0].num_cells
            tp = fe.TracedParams(volume_vars=(_cellwise(conductivity, n_cells),
                                              _cellwise(source, n_cells)))
            # traced_params lets feax detect the matrix property (SPD here)
            # and pick the direct solver accordingly; the traced structure
            # keeps the mesh-sized arrays off the compiled executable.
            ts = fe.TracedStructure.from_problem(problem)
            solve = fe.create_solver(problem, bc,
                                     solver_options=self.solver_options,
                                     linear=True, traced_params=tp,
                                     traced_structure=ts)
            sol = solve(tp, fe.zero_like_initial_guess(problem, bc),
                        traced_structure=ts)
            return sol.field(0).reshape(-1)

    # ── linear elasticity ──────────────────────────────────────────────────
    def elastic(self, points, cells, bcs, *, youngs, poisson,
                base_points=None, body_force=None):
        """Solve small-strain linear elasticity; returns displacement ``(N, 3)``."""
        with _x64():
            pts = _concrete(points, "the elastic solve")
            cells = onp.asarray(cells)
            mesh = fe.Mesh(points=pts, cells=cells,
                           ele_type=_ELE_TYPE[cells.shape[1]])
            tractions = [onp.asarray(v, dtype=onp.float64)
                         for v in bcs.traction_vectors]
            bforce = (None if body_force is None
                      else jnp.asarray(body_force, dtype=jnp.float64))

            class Elastic(fe.Problem):
                def get_tensor_map(self):
                    def stress(u_grad, E, nu):
                        mu = E / (2.0 * (1.0 + nu))
                        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                        eps = 0.5 * (u_grad + u_grad.T)
                        return (lmbda * jnp.trace(eps) * jnp.eye(self.dim)
                                + 2.0 * mu * eps)
                    return stress

                def get_mass_map(self):
                    if bforce is None:
                        return None
                    # residual += integral(v * mass_map): a body force density
                    # enters negated, as the source does above.
                    def mass(u, x, _E, _nu):
                        f = bforce if bforce.ndim == 1 else bforce
                        return -jnp.broadcast_to(f, u.shape)
                    return mass

                def get_surface_maps(self):
                    return [(lambda u, _x, t=jnp.asarray(t): -t) for t in tractions]

            problem = Elastic(
                mesh, vec=3, dim=3, ele_type=mesh.ele_type,
                location_fns=[_membership(n) for n in bcs.traction_nodes])
            bc = fe.DirichletBCConfig([
                fe.DirichletBCSpec(_membership(nodes), "all", 0.0)
                for nodes in bcs.fixed_nodes
            ]).create_bc(problem)

            n_cells = problem.fes[0].num_cells
            tp = fe.TracedParams(volume_vars=(_cellwise(youngs, n_cells),
                                              _cellwise(poisson, n_cells)))
            # traced_params lets feax detect the matrix property (SPD here)
            # and pick the direct solver accordingly; the traced structure
            # keeps the mesh-sized arrays off the compiled executable.
            ts = fe.TracedStructure.from_problem(problem)
            solve = fe.create_solver(problem, bc,
                                     solver_options=self.solver_options,
                                     linear=True, traced_params=tp,
                                     traced_structure=ts)
            sol = solve(tp, fe.zero_like_initial_guess(problem, bc),
                        traced_structure=ts)
            return sol.field(0)


def register(name: str = "feax", **kwargs) -> None:
    """Register this backend with cadjoint under ``name``."""
    from cadjoint.fem.backends import register_backend

    register_backend(name, lambda: FeaxBackend(**kwargs))

