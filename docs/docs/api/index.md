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
- **[feax.solver](./reference/feax/solver.md)** - Newton and linear solvers

### Boundary Conditions
- **[feax.DCboundary](./reference/feax/DCboundary.md)** - Dirichlet boundary conditions

### Internal Variables
- **[feax.internal_vars](./reference/feax/internal_vars.md)** - Material properties and parameters

### Utilities
- **[feax.utils](./reference/feax/utils.md)** - Utility functions (save_sol, etc.)

## Flat Toolkit

### Advanced Tools
- **[feax.flat](./reference/feax/flat)** - Advanced FE analysis tools

## Experimental Features

### Symbolic DSL
- **[feax.experimental](./reference/feax/experimental)** - Symbolic interface for defining physics

## Browse All

Use the sidebar to explore all classes, functions, and modules in detail.
