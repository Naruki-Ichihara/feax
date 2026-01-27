
# FEAX 

[![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7%2B-green.svg)](https://github.com/google/jax) <img src="assets/logo.svg" alt="logo" width=150 align="right"></img>

**FEAX** (Finite Element Analysis with JAX) is a compact, high-performance finite element analysis engine built on JAX. It provides an API for solving partial differential equations on XLA.

## What is FEAX? 

FEAX combines automatic differentiation with finite element methods. It's designed for: 

- **Differentiable Physics**: Compute gradients through entire FE simulations for optimization, inverse problems, and machine learning
- **High Performance**: JIT compilation and vectorization through JAX for maximum computational efficiency 

## JAX Transformations in FEAX

<div align="center">
<img src="assets/feax_jax_transformations.svg" alt="JAX Transformations" width="800">
</div>

FEAX leverages JAX's powerful transformation system to enable:
- **Automatic Differentiation**: Compute exact gradients through finite element solvers
- **JIT Compilation**: Compile to optimized machine code for maximum performance  
- **Vectorization**: Efficiently process multiple scenarios in parallel with `vmap`
- **Parallelization**: Scale across multiple devices with `pmap`

## Installation

## feax.flat

**Flat** (Feax Lattice) is a utility for asymptotic homogenization of lattice unit cell.

## feax.gene

**Gene** (Generative design in FEAX) is a comprehensive toolkit for topology optimization and generative design. It provides efficient, JAX-native implementations of common topology optimization components.

### Key Features

- **Response Functions**: Compliance and volume fraction calculations optimized for topology optimization
- **Filtering Methods**:
  - PDE-based Helmholtz filter for smooth, physically-motivated designs
  - Distance-based density filter for efficient spatial smoothing
  - Sensitivity filter for mesh-independent gradient smoothing
- **Constrained Optimization**: MDMM (Modified Differential Multiplier Method) for handling equality and inequality constraints with automatic differentiation
- **Pure JAX Implementation**: Fully differentiable and compatible with optax optimizers


## License

FEAX is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full license text.

## Acknowledgments

FEAX builds upon the excellent work of:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/tianjuxue/jax_fem) for inspiration and reference implementations

---
<div align="center">
<img src="assets/logo.svg" alt="logo" width=150></img>
</div>

