
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

## Installation

```bash
pip install feax[cuda12]
pip install --no-build-isolation git+https://github.com/Naruki-Ichihara/spineax.git
```

> **Note**: `spineax` requires CUDA toolkit and cuDSS to be available at build time. The `--no-build-isolation` flag is required so that spineax can find JAX and NVIDIA libraries installed by `feax[cuda12]`.

## feax.flat

**Flat** (Feax Lattice) is a utility for asymptotic homogenization of lattice unit cell.

## feax.gene

**Gene** (Generative design in FEAX) is a comprehensive toolkit for topology optimization and generative design. It provides efficient, JAX-native implementations of common topology optimization components.

## License

FEAX is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full license text.

## Acknowledgments

FEAX builds upon the excellent work of:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/deepmodeling/jax-fem) for inspiration and reference implementations
- [Spineax](https://github.com/johnviljoen/spineax) for cuDSS solver implementation

---
<div align="center">
<img src="assets/logo.svg" alt="logo" width=150></img>
</div>

