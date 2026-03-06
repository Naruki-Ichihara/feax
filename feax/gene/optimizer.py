"""Topology optimization driver with NLopt MMA.

Provides a unified ``run()`` function for density-based topology
optimization with optional continuation parameters (Heaviside beta,
SIMP penalty, …) and adaptive remeshing.

Usage::

    import feax.gene as gene
    from feax.gene.optimizer import Continuation, AdaptiveConfig, run

    def my_pipeline(mesh):
        # Build mesh-dependent objects
        ...
        return {'objective': obj_fn, 'volume': vol_fn, 'filter': filter_fn}

    result = run(
        build_pipeline=my_pipeline,
        mesh=initial_mesh,
        target_volume=0.4,
        max_iter=100,
    )
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as np
import numpy as onp

from feax.mesh import Mesh


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Continuation:
    """Parameter that updates periodically during optimization.

    At every ``update_every`` iterations the value is multiplied by
    ``multiply_by``, clamped between ``initial`` and ``final``.

    .. note::

       When a continuation parameter changes, the JIT-compiled objective
       and volume functions are **not** recompiled — the new value is
       passed as a regular argument.  However, if the parameter is used
       as a **static** JAX argument (e.g. via ``static_argnums``),
       a value change will trigger JAX recompilation.

    Examples::

        # Heaviside beta: 1 → 16, doubled every 40 iterations
        Continuation(initial=1.0, final=16.0, update_every=40, multiply_by=2.0)

        # SIMP penalty: 1 → 3, +0.5 every 30 iterations
        Continuation(initial=1.0, final=3.0, update_every=30, multiply_by=1.0, add=0.5)
    """
    initial: float
    final: float
    update_every: int
    multiply_by: float = 2.0
    add: float = 0.0

    def value_at(self, iteration: int) -> float:
        """Compute parameter value at a given iteration."""
        n = iteration // self.update_every
        if self.add != 0.0:
            v = self.initial + self.add * n
        else:
            v = self.initial * (self.multiply_by ** n)
        if self.final >= self.initial:
            return min(v, self.final)
        return max(v, self.final)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive remeshing during optimization.

    .. note::

       Each remesh triggers a full ``build_pipeline`` re-call and
       JAX recompilation of the objective/volume functions, since the
       mesh (and therefore array shapes) changes.  This incurs a
       one-time compilation cost per adaptation step.

    Parameters
    ----------
    remesh : callable
        ``(old_mesh, density_filtered) -> new_mesh``.
        Any meshing backend can be used (Gmsh, custom, etc.).
    adapt_every : int
        Remesh every N iterations.
    n_adapts_max : int
        Maximum number of remeshes.
    transfer : callable, optional
        ``(rho_old, points_old, points_new) -> rho_new``.
        Defaults to ``gene.adaptive.interpolate_field``.

    Examples::

        from feax.gene import adaptive

        # Box mesh with Gmsh
        AdaptiveConfig(
            remesh=lambda m, rho: adaptive.adaptive_box_mesh(
                size=(L, W, H), refinement_field=rho, old_mesh=m,
                h_min=1.0, h_max=4.0,
            ),
            adapt_every=40,
            n_adapts_max=2,
        )

        # Custom remeshing with user-defined transfer
        AdaptiveConfig(
            remesh=my_remesh_fn,       # (old_mesh, rho_filtered) -> new_mesh
            transfer=my_transfer_fn,   # (rho_old, pts_old, pts_new) -> rho_new
        )
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
    final_volume: float


# Type alias
PipelineBuilder = Callable[[Mesh], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    build_pipeline: PipelineBuilder,
    mesh: Mesh,
    target_volume: float,
    max_iter: int = 100,
    continuations: Optional[Dict[str, Continuation]] = None,
    adaptive: Optional[AdaptiveConfig] = None,
    output_dir: Optional[str] = None,
    save_every: int = 10,
    rho_init: Optional[onp.ndarray] = None,
    rho_bounds: Tuple[float, float] = (0.001, 1.0),
) -> OptResult:
    """Run topology optimization with NLopt MMA.

    Parameters
    ----------
    build_pipeline : callable
        Factory ``(mesh) -> dict``.  Called once per mesh (re-called after
        each adaptive remesh).  Must return a dict with three keys:

        - ``'objective'``: ``(rho, **cont_params) -> scalar`` — quantity
          to minimise (e.g. compliance).
        - ``'volume'``: ``(rho, **cont_params) -> scalar`` — volume
          fraction.
        - ``'filter'``: ``(rho) -> rho_filtered`` — for visualisation
          and density transfer during remeshing.

        ``**cont_params`` are the current values of ``continuations``.
        If no continuations are specified the functions are called with
        ``rho`` only.
    mesh : Mesh
        Initial finite-element mesh.
    target_volume : float
        Volume fraction constraint (vol ≤ target_volume).
    max_iter : int
        Total iteration budget across all epochs.
    continuations : dict, optional
        ``{name: Continuation(...)}``.  Values are passed as keyword
        arguments to the pipeline's objective and volume functions.
        Continuation values are traced (not static), so updates do
        **not** trigger JAX recompilation.
    adaptive : AdaptiveConfig, optional
        Enable adaptive remeshing.  ``None`` (default) keeps the mesh
        fixed throughout.  Each remesh triggers ``build_pipeline``
        re-call and JAX recompilation (new array shapes).
    output_dir : str, optional
        Write VTU snapshots and a ``history.csv`` here.
        ``None`` disables all file output.
    save_every : int
        Write a VTU snapshot every *save_every* iterations.
    rho_init : ndarray, optional
        Initial density field.  Default: uniform at ``target_volume``.
    rho_bounds : (float, float)
        Lower and upper bounds for design variables.

    Returns
    -------
    OptResult
        Contains final density, filtered density, mesh, history dict,
        and scalar summaries.
    """
    import nlopt
    from feax.gene import adaptive as adaptive_mod
    import feax as fe

    continuations = continuations or {}
    params: Dict[str, float] = {k: v.initial for k, v in continuations.items()}

    cur_mesh = mesh
    history: Dict[str, list] = {'iteration': [], 'objective': [], 'volume': []}
    iter_count = 0
    n_adapts_done = 0

    x = (onp.full(mesh.points.shape[0], target_volume)
         if rho_init is None else onp.array(rho_init))

    # -- File output ----------------------------------------------------------
    csv_file = csv_writer = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_file = open(
            os.path.join(output_dir, 'history.csv'), 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['iteration', 'objective', 'volume'] + list(continuations))
        csv_file.flush()

    # -- Pipeline compilation -------------------------------------------------
    def _compile(m):
        pl = build_pipeline(m)
        return (
            jax.jit(jax.value_and_grad(pl['objective'])),
            jax.jit(pl['volume']),
            jax.jit(jax.grad(pl['volume'])),
            pl['filter'],
        )

    obj_and_grad, vol_jit, grad_vol, filter_fn = _compile(cur_mesh)

    # -- Header ---------------------------------------------------------------
    print("Starting topology optimization")
    print(f"  Design vars  : {x.shape[0]}")
    print(f"  Target volume: {target_volume}")
    print(f"  Max iter     : {max_iter}")
    for k, c in continuations.items():
        label = (f"×{c.multiply_by}" if c.add == 0.0
                 else f"+{c.add}")
        print(f"  Continuation : {k} ({c.initial} → {c.final}, "
              f"{label} every {c.update_every} iter)")
    if adaptive:
        print(f"  Adaptive     : every {adaptive.adapt_every} iter "
              f"(max {adaptive.n_adapts_max})")
    print("-" * 60)

    # -- Optimisation loop ----------------------------------------------------
    # Epoch length: GCD of adapt_every and all continuation update_every
    # so that both remesh and continuation updates land on epoch boundaries.
    epoch_iters = adaptive.adapt_every if adaptive else max_iter
    for c in continuations.values():
        from math import gcd
        epoch_iters = gcd(epoch_iters, c.update_every)

    while iter_count < max_iter:
        # Update continuation params at epoch start
        for k, c in continuations.items():
            old = params[k]
            params[k] = c.value_at(iter_count)
            if params[k] != old:
                print(f"  >>> {k} = {params[k]:.4g}")

        budget = min(epoch_iters, max_iter - iter_count)
        n_vars = cur_mesh.points.shape[0]

        opt = nlopt.opt(nlopt.LD_MMA, n_vars)
        opt.set_lower_bounds(rho_bounds[0])
        opt.set_upper_bounds(rho_bounds[1])

        def _objective(xx, grad):
            nonlocal iter_count
            rho = np.array(xx)

            val, g = obj_and_grad(rho, **params)
            grad[:] = onp.array(g)

            iter_count += 1
            vol = float(vol_jit(rho, **params))

            history['iteration'].append(iter_count)
            history['objective'].append(float(val))
            history['volume'].append(vol)

            print(f"Iter {iter_count:4d}: obj={float(val):.4e}  "
                  f"vol={vol:.4f}  nodes={n_vars}")

            # VTU snapshot
            if output_dir and iter_count % save_every == 0:
                rho_f = onp.array(filter_fn(rho))
                fe.utils.save_sol(
                    cur_mesh,
                    os.path.join(output_dir, f'iter_{iter_count:04d}.vtu'),
                    point_infos=[('density', rho_f)],
                )

            # CSV row
            if csv_writer:
                csv_writer.writerow(
                    [iter_count, float(val), vol]
                    + [params[k] for k in continuations])
                csv_file.flush()

            return float(val)

        def _volume_constraint(xx, grad):
            rho = np.array(xx)
            grad[:] = onp.array(grad_vol(rho, **params))
            return float(vol_jit(rho, **params)) - target_volume

        opt.set_min_objective(_objective)
        opt.add_inequality_constraint(_volume_constraint, 1e-8)
        opt.set_maxeval(budget)

        try:
            x = opt.optimize(x)
        except nlopt.RoundoffLimited:
            print("  NLopt: roundoff limit (converged)")

        # -- Adaptive remesh --------------------------------------------------
        if (adaptive
                and n_adapts_done < adaptive.n_adapts_max
                and iter_count < max_iter
                and iter_count % adaptive.adapt_every == 0):
            rho_f = onp.array(filter_fn(np.array(x)))
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
            obj_and_grad, vol_jit, grad_vol, filter_fn = _compile(cur_mesh)
            n_adapts_done += 1

    # -- Final summary --------------------------------------------------------
    print("-" * 60)
    rho_opt = np.array(x)
    rho_filtered = onp.array(filter_fn(rho_opt))
    final_obj = float(obj_and_grad(rho_opt, **params)[0])
    final_vol = float(vol_jit(rho_opt, **params))

    print(f"Final objective : {final_obj:.4e}")
    print(f"Final volume    : {final_vol:.4f}")
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
        final_volume=final_vol,
    )
