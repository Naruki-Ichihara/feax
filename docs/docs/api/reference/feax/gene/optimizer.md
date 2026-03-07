---
sidebar_label: optimizer
title: feax.gene.optimizer
---

Topology optimization driver with NLopt MMA.

Provides a unified ``run()`` function for density-based topology
optimization with optional continuation parameters (Heaviside beta,
SIMP penalty, …) and adaptive remeshing.

Usage::

    import feax.gene as gene
    from feax.gene.optimizer import Continuation, AdaptiveConfig, run

    def my_pipeline(mesh):
        # Build mesh-dependent objects
        ...
        return {`&#x27;objective&#x27;: obj_fn, &#x27;volume&#x27;: vol_fn, &#x27;filter&#x27;: filter_fn`}

    result = run(
        build_pipeline=my_pipeline,
        mesh=initial_mesh,
        target_volume=0.4,
        max_iter=100,
    )

## Continuation Objects

```python
@dataclass
class Continuation()
```

Parameter that updates periodically during optimization.

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

#### value\_at

```python
def value_at(iteration: int) -> float
```

Compute parameter value at a given iteration.

## AdaptiveConfig Objects

```python
@dataclass
class AdaptiveConfig()
```

Configuration for adaptive remeshing during optimization.

.. note::

Each remesh triggers a full ``build_pipeline`` re-call and
JAX recompilation of the objective/volume functions, since the
mesh (and therefore array shapes) changes.  This incurs a
one-time compilation cost per adaptation step.

Parameters
----------
- **remesh** (*callable*)
- **adapt_every** (*int*)
- **n_adapts_max** (*int*)
- **transfer** (*callable, optional*)
- **Examples** (*:*)


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
remesh=my_remesh_fn,       # (old_mesh, rho_filtered) -&gt; new_mesh
transfer=my_transfer_fn,   # (rho_old, pts_old, pts_new) -&gt; rho_new
)

## OptResult Objects

```python
@dataclass
class OptResult()
```

Topology optimization result.

#### run

```python
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
    rho_bounds: Tuple[float, float] = (0.001, 1.0)
) -> OptResult
```

Run topology optimization with NLopt MMA.

Parameters
----------
- **build_pipeline** (*callable*): Factory ``(mesh) -&gt; dict``.  Called once per mesh (re-called after each adaptive remesh).  Must return a dict with three keys:
- **mesh** (*Mesh*): Initial finite-element mesh.
- **target_volume** (*float*): Volume fraction constraint (vol ≤ target_volume).
- **max_iter** (*int*): Total iteration budget across all epochs.
- **continuations** (*dict, optional*): ``{`name: Continuation(...)`}``.  Values are passed as keyword arguments to the pipeline&#x27;s objective and volume functions. Continuation values are traced (not static), so updates do **not** trigger JAX recompilation.
- **adaptive** (*AdaptiveConfig, optional*): Enable adaptive remeshing.  ``None`` (default) keeps the mesh fixed throughout.  Each remesh triggers ``build_pipeline`` re-call and JAX recompilation (new array shapes).
- **output_dir** (*str, optional*): Write VTU snapshots and a ``history.csv`` here. ``None`` disables all file output.
- **save_every** (*int*): Write a VTU snapshot every *save_every* iterations.
- **rho_init** (*ndarray, optional*): Initial density field.  Default: uniform at ``target_volume``.
- **rho_bounds** (*(float, float)*): Lower and upper bounds for design variables.


Returns
-------
OptResult
    Contains final density, filtered density, mesh, history dict,
    and scalar summaries.

