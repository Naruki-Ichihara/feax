---
sidebar_position: 1
---

# Introduction to FEAX

**FEAX** (Finite Element Analysis with JAX) is a high-performance finite element analysis engine built on JAX. It provides a modern API for solving partial differential equations with automatic differentiation, JIT compilation, and vectorization.

## Key Features

- **Differentiable Physics**: Compute gradients through entire FE simulations for optimization, inverse problems, and machine learning
- **High Performance**: JIT compilation and vectorization through JAX for maximum computational efficiency
- **JAX Transformations**: Full support for `jit`, `grad`, `vmap`, and `pmap`

## Getting Started

New to FEAX? Start here:

1. **[Installation](getting-started/installation.md)** - Install FEAX and its dependencies
2. **[Basic Tutorials](basic/index.md)** - Learn the fundamentals with hands-on examples

## Basic Tutorials

Learn FEAX fundamentals through practical examples:

- **[Linear Elasticity](basic/linear_elasticity.md)** - Solve 3D elasticity problems with surface traction loading
- **[Hyperelasticity](basic/hyperelasticity.md)** - Nonlinear material models and Newton solver

## JAX Transformations

FEAX fully supports JAX transformations for high-performance computing:

- **[JIT Transform](basic/jit_transform.md)** - Accelerate simulations with JIT compilation
- **[Vmap Transform](basic/vmap_transform.md)** - Vectorize computations for parametric studies

## Advanced Tutorials

Explore advanced topics for specialized applications:

- **[Periodic Boundary Conditions](advanced/periodic_boundary_conditions.md)** - Apply periodic boundary conditions using prolongation matrices for unit cell analysis and homogenization
- **[Lattice Structure Homogenization](advanced/lattice_homogenization.md)** - Computational homogenization of lattice structures using FEAX's `flat` toolkit with graph-based structure definition

## FEAX Flat Toolkit

The `feax.flat` module provides specialized tools for periodic structures and computational homogenization:

- **`flat.pbc`** - Periodic boundary condition utilities
- **`flat.solver`** - Homogenization solvers
- **`flat.unitcell`** - Unit cell base classes
- **`flat.graph`** - Graph-based lattice structure generation
- **`flat.utils`** - Visualization tools (stiffness sphere, etc.)

See [Lattice Structure Homogenization](advanced/lattice_homogenization.md) for detailed usage.

## API Reference

For detailed API documentation, see the [API Reference](api/index.md) section.

## License

FEAX is licensed under the GNU General Public License v3.0.

## Acknowledgments

FEAX builds upon:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/tianjuxue/jax_fem) for inspiration and reference implementations
