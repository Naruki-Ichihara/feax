---
sidebar_label: Overview
title: feax.gene - Topology Optimization Toolkit
---

# Topology Optimization Toolkit

Tools for density-based topology optimization with adaptive remeshing and continuation.

## Modules

### [adaptive](./adaptive.md)
Adaptive remeshing with Gmsh. Provides `adaptive_mesh()` for TET4 mesh generation with refinement fields, `interpolate_field()` for TET4 shape function-based field transfer, and `gradient_refinement()` for density-gradient-based mesh refinement.

### [optimizer](./optimizer.md)
High-level optimization driver. `run()` handles the full MMA-based optimization loop with `Continuation` parameter scheduling and `AdaptiveConfig` remeshing.

### [filters](./filters.md)
Density filtering and projection. Includes `create_density_filter()`, `create_helmholtz_filter()`, and `heaviside_projection()` for standard topology optimization pipelines.

### [responses](./responses.md)
Response functions for optimization. `create_compliance_fn()` and `create_volume_fn()` create JIT-compiled objective and constraint functions.

### [mdmm](./mdmm.md)
Modified Differential Method of Multipliers for constrained optimization with Lagrange multipliers.

## Quick Start

```python
import feax.gene as gene
from feax.gene.optimizer import Continuation, AdaptiveConfig, run

# Create pipeline factory
def build_pipeline(mesh):
    filter_fn = gene.create_density_filter(mesh, radius=3.0)
    # ... build objective, volume, filter
    return {```````````````````'objective': objective, 'volume': volume, 'filter': filter_fn```````````````````}

# Run optimization
result = run(
    build_pipeline=build_pipeline,
    mesh=mesh,
    target_volume=0.4,
    max_iter=300,
)
```
