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
- **[feax.solvers](./reference/feax/solvers/index.md)** - Solver infrastructure (options, linear, Newton, matrix-free, reduced)

### Boundary Conditions
- **[feax.DCboundary](./reference/feax/DCboundary.md)** - Dirichlet boundary conditions

### Internal Variables
- **[feax.internal_vars](./reference/feax/internal_vars.md)** - Material properties and parameters

### Utilities
- **[feax.utils](./reference/feax/utils.md)** - Utility functions (save_sol, etc.)

## Mechanics

Constitutive models for finite element simulations:

- **[feax.mechanics.cohesive](./reference/feax/mechanics/cohesive.md)** - Cohesive zone models for fracture mechanics

## Topology Optimization (gene)

Density-based topology optimization with adaptive remeshing:

- **[feax.gene.optimizer](./reference/feax/gene/optimizer.md)** - Optimization driver (`run`, `Continuation`, `AdaptiveConfig`)
- **[feax.gene.adaptive](./reference/feax/gene/adaptive.md)** - Adaptive remeshing and field transfer
- **[feax.gene.filters](./reference/feax/gene/filters.md)** - Density filters and Heaviside projection
- **[feax.gene.responses](./reference/feax/gene/responses.md)** - Compliance and volume functions
- **[feax.gene.mdmm](./reference/feax/gene/mdmm.md)** - Constrained optimization (MDMM)

## Flat Toolkit

Lattice material analysis and homogenization:

- **[feax.flat.graph](./reference/feax/flat/graph.md)** - Graph-based density field generation
- **[feax.flat.pbc](./reference/feax/flat/pbc.md)** - Periodic boundary conditions
- **[feax.flat.solver](./reference/feax/flat/solver.md)** - Homogenization solvers
- **[feax.flat.unitcell](./reference/feax/flat/unitcell.md)** - Unit cell definitions
- **[feax.flat.utils](./reference/feax/flat/utils.md)** - Visualization utilities

## Browse All

Use the sidebar to explore all classes, functions, and modules in detail.
