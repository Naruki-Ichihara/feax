---
sidebar_label: responses
title: feax.experimental.topopt_toolkit.responses
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

#### create\_volume\_fn

```python
def create_volume_fn(problem)
```

Creates a JIT-compiled volume fraction calculation function for a given problem.
Returns a function that computes the volume fraction of material in the domain.

**Arguments**:

- `problem` - FEAX Problem instance
  

**Returns**:

- `volume_fn` - JIT-compiled function that takes density array and returns volume fraction

