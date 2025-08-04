# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Technology Stack

FEAX is a modern finite element analysis framework built on:
- **Python 3.10-3.13** with **JAX 0.7+** for automatic differentiation and JIT compilation
- **NumPy/SciPy** for numerical computing
- **JAX-FEM inspired** architecture for differentiable physics

## Essential Commands

### Development Setup
```bash
# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Test individual examples
python examples/linear_elasticity.py
python examples/hyper_elasticity_solve_fn.py
```

### Documentation
```bash
# Generate and serve API docs locally
python generate_docs.py
python docs_serve.py  # Opens browser to http://localhost:8000
```

### Docker Development
```bash
# Run with GPU support
docker-compose up
```

## Architecture Overview

**Core Design Pattern:**
FEAX separates finite element structure from material parameters for efficient optimization:

1. **Problem Class** (`feax/problem.py`) - Defines FE mesh, elements, quadrature (structure only)
2. **InternalVars Class** (`feax/internal_vars.py`) - JAX-compatible dataclass for material properties and loading parameters
3. **Separate Assembly** (`feax/assembler.py`) - Assembles system matrices/residuals from Problem + InternalVars

This separation enables efficient parameter studies and gradient-based optimization through JAX.

**Key Components:**
```
feax/
├── problem.py           # FE structure definition
├── internal_vars.py     # Parameter container (JAX dataclass)
├── assembler.py         # Matrix/residual assembly
├── solver.py            # Newton and linear solvers
├── DCboundary.py        # Dirichlet boundary conditions
├── fe.py, basis.py      # Element definitions and basis functions
└── old/                 # Legacy API (backward compatibility)
```

## JAX Configuration

The codebase enables 64-bit precision by default:
```python
jax.config.update("jax_enable_x64", True)
```

All computational functions are designed for:
- JIT compilation (`@jax.jit`)
- Vectorization (`jax.vmap` for batched operations)
- Automatic differentiation for gradients through FE simulations

## Examples Structure

Comprehensive examples in `/examples/` demonstrate:
- `linear_elasticity.py` - Basic elasticity with SIMP material interpolation
- `linear_elasticity_batch.py` - Batched processing patterns
- `hyper_elasticity_solve_fn.py` - Nonlinear materials with Newton solving
- `poisson_2d.py` - 2D Poisson equation patterns
- Benchmark files for performance testing

## Development Practices

- Use the modular Problem/InternalVars pattern for new physics
- Leverage JAX transformations (jit, vmap, grad) for performance
- Follow the kernel-based weak form approach in `assembler.py`
- Test both single and batched operations for performance
- GPU acceleration available but optional