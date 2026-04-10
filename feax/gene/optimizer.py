"""Topology optimization driver with NLopt MMA.

Provides a ``Pipeline`` abstract class and ``run()`` function for
density-based topology optimization with optional constraints,
continuation parameters, and adaptive remeshing.

Usage:

```python
from feax.gene.optimizer import Pipeline, constraint, run

class MyPipeline(Pipeline):
    def build(self, mesh):
        ...  # set up mesh-dependent objects

    def objective(self, rho, beta=1.0):
        ...
        return compliance

    @constraint(target=0.4)
    def volume(self, rho, beta=1.0):
        ...
        return vol_frac

result = run(MyPipeline(), mesh, max_iter=300)
```
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as np
import numpy as onp

from feax.mesh import Mesh


# ---------------------------------------------------------------------------
# @constraint decorator
# ---------------------------------------------------------------------------

def constraint(target: float, type: str = 'le', tol: float = 0.0):
    """Mark a ``Pipeline`` method as an optimisation constraint.

    Parameters
    ----------
    target : float
        Constraint bound value.
    type : str
        ``'le'`` for *f(rho) <= target* (inequality, default),
        ``'eq'`` for *f(rho) == target* (equality, implemented as
        two inequality constraints: ``target - tol <= f <= target + tol``).
    tol : float
        Tolerance band for equality constraints. Only used when
        ``type='eq'``. Default ``0.0`` gives a tight equality.

    Examples:

    ```python
    @constraint(target=0.4)
    def volume(self, rho, beta=1.0):
        return volume_fn(rho)

    @constraint(target=0.3, type='eq', tol=0.001)
    def volume(self, rho, beta=1.0):
        return volume_fn(rho)
    ```
    """
    if type not in ('le', 'eq'):
        raise ValueError(f"constraint type must be 'le' or 'eq', got {type!r}")

    def decorator(fn):
        fn._constraint_meta = {'target': target, 'type': type, 'tol': tol}
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Pipeline abstract base class
# ---------------------------------------------------------------------------

class Pipeline(ABC):
    """Abstract base class for topology optimisation pipelines.

    Subclass and implement :meth:`build` and :meth:`objective`.
    Add constraints by decorating methods with :func:`constraint`.

    After each (re)mesh, ``run()`` calls ``build(mesh)`` to set up
    mesh-dependent state (solvers, filters, etc.), then JIT-compiles
    ``objective`` and all ``@constraint`` methods.
    """

    @abstractmethod
    def build(self, mesh: Mesh) -> None:
        """Create mesh-dependent objects (solvers, filters, etc.).

        Called once per mesh — re-called after each adaptive remesh.
        Store results as instance attributes for use in
        :meth:`objective`, constraint methods, and :meth:`filter`.
        """

    @abstractmethod
    def objective(self, rho, **params) -> float:
        """Objective function to minimise.

        Parameters
        ----------
        rho : ndarray
            Node-based design variables.
        **params
            Current continuation parameter values.

        Returns
        -------
        scalar
        """

    def filter(self, rho):
        """Density filter for visualisation and field transfer.

        Override to apply filtering.  The default is the identity.
        """
        return rho


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Continuation:
    """Parameter that updates periodically during optimization.

    At every ``update_every`` iterations the value changes by ``step``:

    - ``step > 0``: additive, ``value = initial + step * n``
    - ``step < 0``: additive (decreasing), ``value = initial + step * n``

    The value is always clamped between ``initial`` and ``final``.

    Examples:

    ```python
    # Heaviside beta: 1 -> 8, +1 every 50 iterations
    Continuation(initial=1.0, final=8.0, update_every=50, step=1.0)

    # SIMP penalty: 1 -> 3, +0.5 every 30 iterations
    Continuation(initial=1.0, final=3.0, update_every=30, step=0.5)
    ```
    """
    initial: float
    final: float
    update_every: int
    step: float = 1.0

    def value_at(self, iteration: int) -> float:
        """Compute parameter value at a given iteration."""
        n = max(0, iteration // self.update_every)
        v = self.initial + self.step * n
        if self.final >= self.initial:
            return min(v, self.final)
        return max(v, self.final)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive remeshing during optimization.

    Each remesh triggers ``Pipeline.build()`` re-call and JAX
    recompilation (new array shapes).

    Parameters
    ----------
    remesh : callable
        ``(old_mesh, density_filtered) -> new_mesh``.
    adapt_every : int
        Remesh every N iterations.
    n_adapts_max : int
        Maximum number of remeshes.
    transfer : callable, optional
        ``(rho_old, points_old, points_new) -> rho_new``.
        Defaults to ``gene.adaptive.interpolate_field``.

    Examples:

    ```python
    from feax.gene import adaptive

    AdaptiveConfig(
        remesh=lambda m, rho: adaptive.adaptive_mesh(
            geometry, refinement_field=adaptive.gradient_refinement(rho, m),
            old_mesh=m, h_min=0.5, h_max=2.0,
        ),
        adapt_every=100,
        n_adapts_max=4,
    )
    ```
    """
    remesh: Callable[[Mesh, onp.ndarray], Mesh]
    adapt_every: int = 40
    n_adapts_max: int = 2
    transfer: Optional[Callable[[onp.ndarray, onp.ndarray, onp.ndarray],
                                onp.ndarray]] = None


@dataclass
class OptResult:
    """Topology optimization result."""
    rho: onp.ndarray
    rho_filtered: onp.ndarray
    mesh: Mesh
    history: Dict[str, list]
    final_objective: float
    final_constraints: Dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_constraints(pipeline: Pipeline):
    """Discover methods decorated with @constraint.

    Equality constraints (``type='eq'``) are expanded into two
    inequality constraints: ``f <= target + tol`` and ``-f <= -target + tol``.
    """
    constraints = []
    for name in dir(type(pipeline)):
        attr = getattr(type(pipeline), name, None)
        if callable(attr) and hasattr(attr, '_constraint_meta'):
            meta = attr._constraint_meta
            bound_method = getattr(pipeline, name)
            if meta['type'] == 'eq':
                eq_tol = meta.get('tol', 0.0)
                # f <= target + tol
                constraints.append((name, bound_method,
                                    meta['target'] + eq_tol, 'le'))
                # -f <= -target + tol  i.e.  f >= target - tol
                constraints.append((f'{name}_lb', bound_method,
                                    meta['target'] - eq_tol, 'ge'))
            else:
                constraints.append((name, bound_method, meta['target'], meta['type']))
    return constraints


def _compile(pipeline, constraints, use_jit):
    """JIT-compile objective and constraint functions."""
    wrap = jax.jit if use_jit else lambda f: f

    obj_and_grad = wrap(jax.value_and_grad(pipeline.objective))

    compiled_constraints = []
    for name, method, target, ctype in constraints:
        fn_jit = wrap(method)
        grad_fn = wrap(jax.grad(method))
        compiled_constraints.append((name, fn_jit, grad_fn, target, ctype))

    return obj_and_grad, compiled_constraints


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    pipeline: Pipeline,
    mesh: Mesh,
    max_iter: int = 100,
    continuations: Optional[Dict[str, Continuation]] = None,
    adaptive: Optional[AdaptiveConfig] = None,
    output_dir: Optional[str] = None,
    save_every: int = 10,
    rho_init: Optional[onp.ndarray] = None,
    rho_bounds: Tuple[float, float] = (0.001, 1.0),
    jit: bool = True,
) -> OptResult:
    """Run topology optimization with NLopt MMA.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline instance defining objective, constraints, and filter.
    mesh : Mesh
        Initial finite-element mesh.
    max_iter : int
        Total iteration budget across all epochs.
    continuations : dict, optional
        ``{name: Continuation(...)}``.  Values are passed as keyword
        arguments to objective and constraint methods.
    adaptive : AdaptiveConfig, optional
        Enable adaptive remeshing.  ``None`` keeps the mesh fixed.
    output_dir : str, optional
        Write VTU snapshots and ``history.csv`` here.
    save_every : int
        Write a VTU snapshot every N iterations.
    rho_init : ndarray, optional
        Initial density field.  Default: uniform at 0.5.
    rho_bounds : (float, float)
        Lower and upper bounds for design variables.
    jit : bool
        JIT-compile objective and constraints (default ``True``).

    Returns
    -------
    OptResult
    """
    import nlopt
    from feax.gene import adaptive as adaptive_mod
    import feax as fe

    continuations = continuations or {}
    params: Dict[str, float] = {k: v.initial for k, v in continuations.items()}

    cur_mesh = mesh
    iter_count = 0
    n_adapts_done = 0

    # Discover constraints
    constraints = _collect_constraints(pipeline)

    # History: iteration, objective, plus one column per constraint
    constraint_names = [name for name, *_ in constraints]
    history: Dict[str, list] = {
        'iteration': [], 'objective': [],
        **{name: [] for name in constraint_names},
    }

    # Initial density: default to the tightest 'le' constraint target so we
    # start feasible, falling back to 0.5 when there are no constraints.
    default_init = 0.5
    for _, _, target, ctype in constraints:
        if ctype == 'le':
            default_init = min(default_init, target)
    x = (onp.full(mesh.points.shape[0], default_init)
         if rho_init is None else onp.array(rho_init))

    # -- File output ----------------------------------------------------------
    csv_file = csv_writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_file = open(
            os.path.join(output_dir, 'history.csv'), 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['iteration', 'objective'] + constraint_names
            + list(continuations))
        csv_file.flush()

    # -- Build & compile ------------------------------------------------------
    pipeline.build(cur_mesh)
    obj_and_grad, compiled_constraints = _compile(
        pipeline, constraints, use_jit=jit)

    # -- Header ---------------------------------------------------------------
    print("Starting topology optimization")
    print(f"  Design vars  : {x.shape[0]}")
    print(f"  Max iter     : {max_iter}")
    if constraints:
        for name, _, _, target, ctype in compiled_constraints:
            op = '<=' if ctype == 'le' else ('>=' if ctype == 'ge' else '==')
            print(f"  Constraint   : {name} {op} {target}")
    else:
        print("  Constraints  : none")
    for k, c in continuations.items():
        label = f"+{c.step}"
        print(f"  Continuation : {k} ({c.initial} -> {c.final}, "
              f"{label} every {c.update_every} iter)")
    if adaptive:
        print(f"  Adaptive     : every {adaptive.adapt_every} iter "
              f"(max {adaptive.n_adapts_max})")
    print(f"  JIT          : {jit}")
    print("-" * 60)

    # -- Helper: create MMA optimizer and register objective/constraints ------
    def _create_mma(n_vars):
        """Build a fresh NLopt MMA instance with objective and constraints."""
        opt = nlopt.opt(nlopt.LD_MMA, n_vars)
        opt.set_lower_bounds(rho_bounds[0])
        opt.set_upper_bounds(rho_bounds[1])

        def _objective(xx, grad):
            nonlocal iter_count
            rho = np.array(xx)

            # Convert params to JAX arrays to avoid JIT recompilation
            jax_params = {k: np.float64(v) for k, v in params.items()}
            val, g = obj_and_grad(rho, **jax_params)
            grad[:] = onp.array(g)

            iter_count += 1

            # Evaluate constraints for logging
            con_vals = {}
            for name, fn_jit, _, _, _ in compiled_constraints:
                con_vals[name] = float(fn_jit(rho, **jax_params))

            history['iteration'].append(iter_count)
            history['objective'].append(float(val))
            for name in constraint_names:
                history[name].append(con_vals.get(name, 0.0))

            con_str = '  '.join(
                f'{name}={con_vals[name]:.4f}' for name in constraint_names)
            print(f"Iter {iter_count:4d}: obj={float(val):.4e}  "
                  f"{con_str}  nodes={n_vars}")

            # VTU snapshot
            if output_dir and iter_count % save_every == 0:
                rho_f = onp.array(pipeline.filter(rho))
                fe.utils.save_sol(
                    cur_mesh,
                    os.path.join(output_dir, f'iter_{iter_count:04d}.vtu'),
                    point_infos=[('density', rho_f)],
                )

            # CSV row
            if csv_writer:
                csv_writer.writerow(
                    [iter_count, float(val)]
                    + [con_vals.get(n, 0.0) for n in constraint_names]
                    + [params[k] for k in continuations])
                csv_file.flush()

            return float(val)

        opt.set_min_objective(_objective)

        # Register constraints
        for name, fn_jit, grad_fn, target, ctype in compiled_constraints:
            def _make_constraint(fn, gfn, tgt):
                def _con(xx, grad):
                    rho = np.array(xx)
                    jax_p = {k: np.float64(v) for k, v in params.items()}
                    grad[:] = onp.array(gfn(rho, **jax_p))
                    return float(fn(rho, **jax_p)) - tgt
                return _con

            if ctype == 'le':
                con_fn = _make_constraint(fn_jit, grad_fn, target)
                opt.add_inequality_constraint(con_fn, 1e-8)
            elif ctype == 'ge':
                # f >= target  →  target - f <= 0
                def _make_ge_constraint(fn, gfn, tgt):
                    def _con(xx, grad):
                        rho = np.array(xx)
                        jax_p = {k: np.float64(v) for k, v in params.items()}
                        grad[:] = -onp.array(gfn(rho, **jax_p))
                        return -(float(fn(rho, **jax_p)) - tgt)
                    return _con
                con_fn = _make_ge_constraint(fn_jit, grad_fn, target)
                opt.add_inequality_constraint(con_fn, 1e-8)

        return opt

    # -- Compute epoch boundaries (continuation changes and remesh points) ----
    def _next_continuation_change(current_iter):
        """Find the next iteration where any continuation param changes."""
        next_change = max_iter
        for c in continuations.values():
            # Next update boundary for this continuation
            phase = current_iter // c.update_every
            boundary = (phase + 1) * c.update_every
            # Only counts if it actually changes the value
            if boundary < max_iter:
                old_val = c.value_at(current_iter)
                new_val = c.value_at(boundary)
                if old_val != new_val:
                    next_change = min(next_change, boundary)
        return next_change

    # -- Optimisation loop ----------------------------------------------------
    need_new_mma = True

    while iter_count < max_iter:
        n_vars = cur_mesh.points.shape[0]

        # Update continuation params at phase start
        for k, c in continuations.items():
            old = params[k]
            params[k] = c.value_at(iter_count)
            if params[k] != old:
                print(f"  >>> {k} = {params[k]:.4g}")

        # Create MMA when needed: initial, after remesh, or after param change
        if need_new_mma:
            opt = _create_mma(n_vars)
            need_new_mma = False

        # Budget: run until next continuation change, remesh, or end
        next_stop = max_iter
        # Next continuation parameter change
        next_cont = _next_continuation_change(iter_count)
        next_stop = min(next_stop, next_cont)
        # Next remesh point
        if adaptive and n_adapts_done < adaptive.n_adapts_max:
            next_remesh = ((iter_count // adaptive.adapt_every) + 1) * adaptive.adapt_every
            next_stop = min(next_stop, next_remesh)

        budget = next_stop - iter_count

        opt.set_maxeval(budget)

        try:
            x = opt.optimize(x)
        except nlopt.RoundoffLimited:
            print("  NLopt: roundoff limit (converged)")

        # Restart MMA if continuation params will change next iteration
        if iter_count < max_iter:
            for c in continuations.values():
                if c.value_at(iter_count) != c.value_at(iter_count - 1):
                    need_new_mma = True
                    break

        # -- Adaptive remesh --------------------------------------------------
        if (adaptive
                and n_adapts_done < adaptive.n_adapts_max
                and iter_count < max_iter
                and iter_count % adaptive.adapt_every == 0):
            rho_f = onp.array(pipeline.filter(np.array(x)))
            old_n = cur_mesh.points.shape[0]

            new_mesh = adaptive.remesh(cur_mesh, rho_f)
            if adaptive.transfer is not None:
                x = adaptive.transfer(
                    x, onp.array(cur_mesh.points), onp.array(new_mesh.points))
            else:
                x = adaptive_mod.interpolate_field(
                    x, cur_mesh, onp.array(new_mesh.points),
                    clip=rho_bounds)
            new_n = new_mesh.points.shape[0]
            print(f"  >>> Remesh: {old_n} -> {new_n} nodes")

            if output_dir:
                fe.utils.save_sol(
                    new_mesh,
                    os.path.join(
                        output_dir, f'adapt_{n_adapts_done + 1:02d}.vtu'),
                    point_infos=[('density', onp.array(x))],
                )

            cur_mesh = new_mesh
            pipeline.build(cur_mesh)
            obj_and_grad, compiled_constraints = _compile(
                pipeline, constraints, use_jit=jit)
            n_adapts_done += 1
            need_new_mma = True  # array size changed, must recreate MMA

    # -- Final summary --------------------------------------------------------
    print("-" * 60)
    rho_opt = np.array(x)
    rho_filtered = onp.array(pipeline.filter(rho_opt))
    final_obj = float(obj_and_grad(rho_opt, **params)[0])

    final_constraints = {}
    for name, fn_jit, _, _, _ in compiled_constraints:
        final_constraints[name] = float(fn_jit(rho_opt, **params))

    print(f"Final objective : {final_obj:.4e}")
    for name, val in final_constraints.items():
        print(f"Final {name:10s}: {val:.4f}")
    print(f"Final mesh      : {cur_mesh.points.shape[0]} nodes")
    if n_adapts_done:
        print(f"Adaptations     : {n_adapts_done}")

    if output_dir:
        fe.utils.save_sol(
            cur_mesh,
            os.path.join(output_dir, 'final.vtu'),
            point_infos=[('density', rho_filtered)],
        )
        print(f"Saved: {output_dir}/final.vtu")

    if csv_file:
        csv_file.close()

    return OptResult(
        rho=onp.array(x),
        rho_filtered=rho_filtered,
        mesh=cur_mesh,
        history=history,
        final_objective=final_obj,
        final_constraints=final_constraints,
    )
