---
sidebar_label: responses
title: feax.gene.responses
---

Response function generators for topology optimization and analysis.

#### create\_compliance\_fn

```python
def create_compliance_fn(problem, surface_load_params=None)
```

Creates a universal pure-JAX compliance function for a given problem.
Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
where u is displacement and f is traction on each loaded boundary.

**Arguments**:

- `problem` - FEAX Problem instance
- `surface_load_params` - Optional list/array of parameters for each surface map.
  If None, defaults to scalar 1.0 for each surface.
  If single value, applies to all surfaces.
  If list/array, must match number of surfaces. An
  entry is passed to the surface map as its parameter;
  a tuple entry is unpacked into several parameters
  (``surface_fn(u, x, *entry)``). An entry (or a tuple
  element) may carry a leading batch axis, (B, ...):
  the compliance of case ``i`` then uses its ``[i]``
  (one load per case, as in a ``feax.vmap`` batch).


**Returns**:

- `surface_load_params`0 - pure-JAX function (sol) -&gt; compliance; jit the
  composed objective at the call site. ``sol`` may be one flat
  solution (n,) or a batch (B, n) (a batched ``feax.Solution``&#x27;s
  ``dofs``), which gives (B,) compliances.

#### create\_dynamic\_compliance\_fn

```python
def create_dynamic_compliance_fn(problem)
```

Creates a universal pure-JAX compliance function for a given problem.
Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
where u is displacement and f is traction on each loaded boundary.

Unlike ``create_compliance_fn``, the surface load parameters are not fixed when
the function is created. They are inferred from ``surface_vars`` at runtime and
passed into each surface map when evaluating the traction field.

**Arguments**:

- `problem` - FEAX Problem instance


**Returns**:

- `compliance_fn` - pure-JAX function that takes ``sol`` and
  ``surface_vars`` and returns the compliance value.

#### create\_volume\_fn

```python
def create_volume_fn(problem)
```

Creates a pure-JAX volume fraction calculation function for a given problem.
Returns a function that computes the volume fraction of material in the domain.

Supports node-based, cell-based, and quad-based density arrays.

**Arguments**:

- `problem` - FEAX Problem instance


**Returns**:

- `volume_fn` - pure-JAX function that takes density array and returns volume fraction
- `Accepts` - (num_nodes,) node-based, (num_cells,) cell-based,
  or (num_cells, num_quads) quad-based density arrays,
  and a batch of node- or cell-based fields,
  (B, num_nodes) / (B, num_cells), which gives (B,)
  volume fractions.

