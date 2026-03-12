---
sidebar_label: optimizer
title: feax.gene.optimizer
---

Topology optimization driver with NLopt MMA.

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

#### constraint

```python
def constraint(target: float, type: str = 'le')
```

Mark a ``Pipeline`` method as an optimisation constraint.

Parameters
----------
- **target** (*float*)
- **type** (*str*)


## Pipeline Objects

```python
class Pipeline(ABC)
```

Abstract base class for topology optimisation pipelines.

Subclass and implement :meth:`build` and :meth:`objective`.
Add constraints by decorating methods with :func:`constraint`.

After each (re)mesh, ``run()`` calls ``build(mesh)`` to set up
mesh-dependent state (solvers, filters, etc.), then JIT-compiles
``objective`` and all ``@constraint`` methods.

#### build

```python
@abstractmethod
def build(mesh: Mesh) -> None
```

Create mesh-dependent objects (solvers, filters, etc.).

Called once per mesh — re-called after each adaptive remesh.
Store results as instance attributes for use in
:meth:`objective`, constraint methods, and :meth:`filter`.

#### objective

```python
@abstractmethod
def objective(rho, **params) -> float
```

Objective function to minimise.

Parameters
----------
- **rho** (*ndarray*): Node-based design variables.


Returns
-------
scalar

#### filter

```python
def filter(rho)
```

Density filter for visualisation and field transfer.

Override to apply filtering.  The default is the identity.

## Continuation Objects

```python
@dataclass
class Continuation()
```

Parameter that updates periodically during optimization.

At every ``update_every`` iterations the value is multiplied by
``multiply_by`` (or incremented by ``add``), clamped between
``initial`` and ``final``.

Continuation values are passed as traced (not static) JAX
arguments, so updates do **not** trigger recompilation.

**Examples**:


```python
# Heaviside beta: 1 -&gt; 16, doubled every 40 iterations
Continuation(initial=1.0, final=16.0, update_every=40, multiply_by=2.0)

# SIMP penalty: 1 -&gt; 3, +0.5 every 30 iterations
Continuation(initial=1.0, final=3.0, update_every=30, multiply_by=1.0, add=0.5)
```

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

Each remesh triggers ``Pipeline.build()`` re-call and JAX
recompilation (new array shapes).

Parameters
----------
- **remesh** (*callable*)
- **adapt_every** (*int*)
- **n_adapts_max** (*int*)
- **transfer** (*callable, optional*)


## OptResult Objects

```python
@dataclass
class OptResult()
```

Topology optimization result.

#### run

```python
def run(pipeline: Pipeline,
        mesh: Mesh,
        max_iter: int = 100,
        continuations: Optional[Dict[str, Continuation]] = None,
        adaptive: Optional[AdaptiveConfig] = None,
        output_dir: Optional[str] = None,
        save_every: int = 10,
        rho_init: Optional[onp.ndarray] = None,
        rho_bounds: Tuple[float, float] = (0.001, 1.0),
        jit: bool = True) -> OptResult
```

Run topology optimization with NLopt MMA.

Parameters
----------
- **pipeline** (*Pipeline*): Pipeline instance defining objective, constraints, and filter.
- **mesh** (*Mesh*): Initial finite-element mesh.
- **max_iter** (*int*): Total iteration budget across all epochs.
- **continuations** (*dict, optional*): ``{`name: Continuation(...)`}``.  Values are passed as keyword arguments to objective and constraint methods.
- **adaptive** (*AdaptiveConfig, optional*): Enable adaptive remeshing.  ``None`` keeps the mesh fixed.
- **output_dir** (*str, optional*): Write VTU snapshots and ``history.csv`` here.
- **save_every** (*int*): Write a VTU snapshot every N iterations.
- **rho_init** (*ndarray, optional*): Initial density field.  Default: uniform at 0.5.
- **rho_bounds** (*(float, float)*): Lower and upper bounds for design variables.
- **jit** (*bool*): JIT-compile objective and constraints (default ``True``).


Returns
-------
OptResult

