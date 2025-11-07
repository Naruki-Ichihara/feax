---
sidebar_label: Overview
title: feax.flat - Flat Toolkit
---

# Flat Toolkit

Advanced tools for finite element analysis with periodic structures and homogenization.

## Modules

### [graph](./graph.md)
Graph-based density field generation for periodic structures. Create density fields from user-defined node-edge graphs for finite element analysis.

### [pbc](./pbc.md)
Periodic boundary condition implementation. Handles the construction of prolongation matrices for unit cell-based homogenization and multiscale analysis.

### [solver](./solver.md)
Solver utilities for periodic lattice structures and homogenization problems. Implements the macro term approach for prescribed macroscopic strains.

### [unitcell](./unitcell.md)
Unit cell definitions and utilities for periodic structures. Provides abstract base class and geometric operations for unit cells.

### [utils](./utils.md)
Utility functions for lattice toolkit visualization and analysis. Includes directional Young's modulus computation and stiffness sphere visualization.

## Quick Start

```python
from feax.flat.pbc import periodic_bc_3D, prolongation_matrix
from feax.flat.solver import create_homogenization_solver

# Setup periodic boundary conditions
pairings = periodic_bc_3D(unit_cell, vec=3, dim=3)
P = prolongation_matrix(pairings, mesh, vec=3)

# Create homogenization solver
solver = create_homogenization_solver(
    problem, bc, P, solver_options, mesh, dim=3
)

# Compute homogenized stiffness
C_hom = solver(internal_vars)
```
