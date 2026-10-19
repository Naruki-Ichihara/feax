---
sidebar_position: 1
title: API Reference
---

# FEAX API Reference

Complete API documentation for FEAX (Finite Element Analysis with JAX).

## Core Modules

### Problem Definition
- **[feax.problem](./reference/feax/problem.md)** - Base Problem class and problem definition
- **[feax.mesh](./reference/feax/mesh.md)** - Mesh generation and handling

### Finite Elements
- **[feax.fe](./reference/feax/fe.md)** - FiniteElement class definitions
- **[feax.basis](./reference/feax/basis.md)** - Shape functions and quadrature rules

### Assembly & Solving
- **[feax.assembler](./reference/feax/assembler.md)** - System matrix and residual assembly
- **[feax.solver](./reference/feax/solver.md)** - High-level solver creation (`create_solver`)
- **[feax.solvers](./reference/feax/solvers/index.md)** - Solver infrastructure (options, linear, Newton, AMG, reduced)
- **feax.solvers.direct** - CPU host-side direct backends: [spsolve](./reference/feax/solvers/direct/spsolve.md), [umfpack](./reference/feax/solvers/direct/umfpack.md), [cholmod](./reference/feax/solvers/direct/cholmod.md)
- **[feax.solvers.time_solver](./reference/feax/solvers/time_solver.md)** - Time stepping (`TimePipeline`, `ImplicitPipeline`, `ExplicitPipeline`, `TimeConfig`, `run`)
- **[feax.solvers.eigen](./reference/feax/solvers/eigen.md)** - Linear buckling (`create_linear_buckling_solver`; backends `"sparse"` (ARPACK, default), `"dense"`, `"matfree"`, and `"cudss"` GPU shift-invert)
- **[feax.solution](./reference/feax/solution.md)** - `Solution` return type (`.dofs`, `.field(i)`, `.node_var(component=)`)

### Boundary Conditions
- **[feax.DCboundary](./reference/feax/DCboundary.md)** - Dirichlet boundary conditions

### Internal Variables
- **[feax.traced_params](./reference/feax/traced_params.md)** - Material properties and parameters

### Utilities
- **[feax.utils](./reference/feax/utils.md)** - Utility functions (`save_sol`, `XDMFWriter`, `zero_like_initial_guess`)

## Mechanics

Constitutive models for finite element simulations:

- **[feax.mechanics.cohesive](./reference/feax/mechanics/cohesive.md)** - Cohesive zone models for fracture mechanics
- **[feax.mechanics.shell](./reference/feax/mechanics/shell.md)** - Mindlin plates and classical lamination theory
- **[feax.mechanics.tmc](./reference/feax/mechanics/tmc.md)** - Third-medium contact
- **[feax.mechanics.orientation](./reference/feax/mechanics/orientation.md)** - Fiber orientation fields and rotated stiffness
- **[feax.mechanics.layered_solid_element](./reference/feax/mechanics/layered_solid_element.md)** - Layered solid elements for laminates
- **[feax.mechanics.copv](./reference/feax/mechanics/copv.md)** - Mesh and model generation for double-wall composite pressure vessels (COPV)

## Topology Optimization (gene)

Density-based topology optimization with adaptive remeshing:

- **[feax.gene.optimizer](./reference/feax/gene/optimizer.md)** - MMA pipeline driver (`Pipeline`, `@constraint`, `run`, `Continuation`, `AdaptiveConfig`)
- **[feax.gene.adaptive](./reference/feax/gene/adaptive.md)** - Adaptive remeshing and field transfer
- **[feax.gene.filters](./reference/feax/gene/filters.md)** - Density filters and Heaviside projection
- **[feax.gene.responses](./reference/feax/gene/responses.md)** - Compliance and volume functions
- **[feax.gene.mdmm](./reference/feax/gene/mdmm.md)** - Constrained optimization (MDMM)
- **[feax.gene.length_scale](./reference/feax/gene/length_scale.md)** - Minimum length-scale constraints (geometric and morphological)
- **[feax.gene.mesh_extraction](./reference/feax/gene/mesh_extraction.md)** - Density / SDF → surface and volume mesh extraction (volume meshes need the `gmsh` extra)

## Experimental

APIs that are usable but may change without a deprecation cycle:

- **[feax.experimental.cmg](./reference/feax/experimental/cmg.md)** - Matrix-free O(band) geometric multigrid on a `StructuredGrid` band (`NarrowBandCMG`, `ScalarMaterial`, `KernelMaterial`, `build_geometry`)
- **[feax.experimental.narrowband](./reference/feax/experimental/narrowband.md)** - Moving-band OC loop and multires bootstrap (`oc_update`, `run_narrowband_oc`, `run_narrowband_multires`, `create_sensitivity_filter`)
- **[feax.experimental.mma](./reference/feax/experimental/mma.md)** - Pure-JAX MMA / GCMMA (`MMA`, `GCMMA`)

## Flat Toolkit

Lattice material analysis and homogenization:

- **[feax.flat.graph](./reference/feax/flat/graph.md)** - Graph-based density field generation
- **[feax.flat.pbc](./reference/feax/flat/pbc.md)** - Periodic boundary conditions
- **[feax.flat.solver](./reference/feax/flat/solver.md)** - Homogenization solvers
- **[feax.flat.unitcell](./reference/feax/flat/unitcell.md)** - Unit cell definitions
- **[feax.flat.utils](./reference/feax/flat/utils.md)** - Visualization utilities

## Browse All

Use the sidebar to explore all classes, functions, and modules in detail.
