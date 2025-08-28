# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Technology Stack

FEAX is a modern finite element analysis framework built on:
- **Python 3.9-3.13** with **JAX 0.7+** for automatic differentiation and JIT compilation
- **NumPy/SciPy** for numerical computing
- **meshio** for mesh format handling
- **fenics-basix** for finite element basis functions
- **JAX-FEM inspired** architecture for differentiable physics

## Essential Commands

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[test,docs]"
```

### Testing
```bash
# Run all tests
pytest

# Test individual examples
python examples/linear_elasticity.py
python examples/hyper_elasticity_solve_fn.py
python examples/poisson_2d.py
python examples/topopt.py

# Test vectorized operations
python examples/linear_elasticity_vmap_batch.py
python examples/linear_elasticity_vmap_density.py
```

### Docker Development
```bash
# Run with GPU support
docker-compose up

# Build and run development container
docker build -t feax-dev .
docker run --gpus all -it feax-dev
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
├── mesh.py              # Mesh handling with gmsh integration
├── fe.py                # FiniteElement class definitions
├── basis.py             # Shape functions and quadrature
├── DCboundary.py        # Dirichlet boundary conditions
├── bc_spec.py           # Modern BC specification API
├── lattice_toolkit/     # Periodic structures and unit cells
├── topopt_toolkit/      # Topology optimization tools
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

## Kernel-Based Weak Forms

FEAX uses kernel functions to construct weak forms:
- **Laplace Kernel** (`get_tensor_map`) - Gradient-based terms (elasticity, diffusion)
- **Mass Kernel** (`get_mass_map`) - Inertia/reaction terms
- **Surface Kernels** (`get_surface_maps`) - Boundary loads/tractions
- **Universal Kernels** (`get_universal_kernels`) - Custom physics

## Boundary Conditions API

Modern dataclass-based approach:
```python
from feax.bc_spec import DirichletBCConfig, DirichletBCSpec

bc_config = DirichletBCConfig([
    DirichletBCSpec(location_fn, component, value_fn)
])
bc = bc_config.create_bc(problem)
```

## Multipoint Constraints (MPC) Support

FEAX supports multipoint constraints (e.g., periodic boundary conditions) through:

**Prolongation Matrix Approach:**
- Use `feax.lattice_toolkit.pbc.prolongation_matrix()` to create P matrix
- Maps reduced (independent) to full DOF space: `u_full = P @ u_reduced`
- Automatic handling via `@reduce(P)` decorator on solver functions

**Decorator Pattern:**
```python
from feax import reduce, create_solver
from feax.lattice_toolkit.pbc import periodic_bc_3D, prolongation_matrix

# Create prolongation matrix for periodic BC
pairings = periodic_bc_3D(unitcell, vec=3, dim=3)
P = prolongation_matrix(pairings, mesh, vec=3)

# Apply MPC to any solver
@reduce(P)
def create_mpc_solver(problem, bc, **kwargs):
    return create_solver(problem, bc, **kwargs)

# Solver now works in reduced space
solver = create_mpc_solver(problem, bc, iter_num=1)
```

**Alternative API:**
- `create_solver_with_mpc(problem, bc, P_mat, ...)` for direct usage
- `create_mpc_wrapper(P_mat)` for custom transformation functions

## Examples Structure

Comprehensive examples in `/examples/` demonstrate:
- `linear_elasticity.py` - Basic elasticity with SIMP material interpolation
- `linear_elasticity_batch.py` - Batched processing patterns
- `linear_elasticity_vmap_*.py` - Vectorized operations for parameter studies
- `hyper_elasticity_solve_fn.py` - Nonlinear materials with Newton solving
- `poisson_2d.py` - 2D Poisson equation patterns
- `periodic_poisson_3d.py` - 3D Poisson with periodic BCs and @reduce(P) decorator
- `topopt.py` - Topology optimization with filtering and MMA
- `unitcell.py` - Unit cell analysis with periodic boundary conditions
- Benchmark files for performance testing

## Development Practices

- Use the modular Problem/InternalVars pattern for new physics
- Leverage JAX transformations (jit, vmap, grad) for performance
- Follow the kernel-based weak form approach in `assembler.py`
- Test both single and batched operations for performance
- GPU acceleration available but optional through Docker container
- Use existing examples as integration tests (no formal test suite yet)