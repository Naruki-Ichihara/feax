---
sidebar_label: responses
title: feax.gene.responses
---

Response function generators for topology optimization and analysis.

#### create\_compliance\_fn

```python
def create_compliance_fn(problem, surface_load_params=None)
```

Creates a universal JIT-compiled compliance function for a given problem.
Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
where u is displacement and f is traction on each loaded boundary.

**Arguments**:

- `problem` - FEAX Problem instance
- `surface_load_params` - Optional list/array of parameters for each surface map.
  If None, defaults to scalar 1.0 for each surface.
  If single value, applies to all surfaces.
  If list/array, must match number of surfaces.


**Returns**:

- `compliance_fn` - JIT-compiled function that takes solution and returns compliance value

#### create\_dynamic\_compliance\_fn

```python
def create_dynamic_compliance_fn(problem)
```

Creates a universal JIT-compiled compliance function for a given problem.
Computes compliance (strain energy) = sum over all surfaces of integral u*f dGamma
where u is displacement and f is traction on each loaded boundary.

Unlike ``create_compliance_fn``, the surface load parameters are not fixed when
the function is created. They are inferred from ``surface_vars`` at runtime and
passed into each surface map when evaluating the traction field.

**Arguments**:

- `problem` - FEAX Problem instance


**Returns**:

- `compliance_fn` - JIT-compiled function that takes ``sol`` and
  ``surface_vars`` and returns the compliance value.

#### create\_volume\_fn

```python
def create_volume_fn(problem)
```

Creates a JIT-compiled volume fraction calculation function for a given problem.
Returns a function that computes the volume fraction of material in the domain.

Supports node-based, cell-based, and quad-based density arrays.

**Arguments**:

- `problem` - FEAX Problem instance


**Returns**:

- `volume_fn` - JIT-compiled function that takes density array and returns volume fraction
- `Accepts` - (num_nodes,) node-based, (num_cells,) cell-based,
  or (num_cells, num_quads) quad-based density arrays

