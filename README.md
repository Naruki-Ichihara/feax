# FEAX: Finite Element Analysis with JAX

[![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7%2B-green.svg)](https://github.com/google/jax)

**FEAX** (Finite Element Analysis with JAX) is a compact, high-performance finite element analysis engine built on JAX. It provides an API for solving partial differential equations with automatic differentiation, JIT compilation, and GPU acceleration.

## What is FEAX?

FEAX combines the power of modern automatic differentiation with classical finite element methods. It's designed for:

- **Differentiable Physics**: Compute gradients through entire FE simulations for optimization, inverse problems, and machine learning
- **High Performance**: JIT compilation and vectorization through JAX for maximum computational efficiency
- **Modular API**: Separation between problem definition and solution parameters
- **Extensibility**: Easy to extend with custom physics, elements, and solvers

## Key Features

### Differentiable Solvers
```python
import jax
from feax import Problem, InternalVars, create_solver

# Define your physics problem
class Elasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            # Linear elasticity constitutive law
            return elastic_stress_tensor(u_grad, E)
        return stress

# Create differentiable solver
solver = create_solver(problem, boundary_conditions, options)

# Compute gradients with respect to material properties
grad_fn = jax.grad(lambda params: objective(solver(params)))
gradients = grad_fn(material_parameters)
```

### High-Performance Computing
- **JIT Compilation**: All computations compile to efficient machine code
- **GPU Acceleration**: Seamless GPU execution with JAX
- **Vectorization**: Efficient batch processing of multiple scenarios
- **Memory Efficient**: Optimized memory usage for large-scale problems

### Architecture
```python
# Separate problem definition from parameters
problem = ElasticityProblem(mesh, vec=3, dim=3)
internal_vars = InternalVars(
    volume_vars=(young_modulus, poisson_ratio),
    surface_vars=(surface_traction,)
)

# Solve with different parameter sets
solutions = jax.vmap(solver)(parameter_batch)
```

## Installation

### Requirements
- Python 3.10+
- JAX 0.7+

### Install from source
```bash
git clone https://github.com/your-repo/feax.git
cd feax
pip install -e .
```

## Quick Start

Here's a simple linear elasticity example:

```python
import jax.numpy as np
from feax import Problem, InternalVars, DirichletBC, SolverOptions
from feax import create_solver
from feax.mesh import box_mesh_gmsh, Mesh

# Define the physics
class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress_tensor(u_grad, E):
            nu = 0.3  # Poisson's ratio
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(3) + 2 * mu * epsilon
        return stress_tensor

# Create mesh
meshio_mesh = box_mesh_gmsh(10, 10, 10, 1.0, 1.0, 1.0)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Set up problem
problem = LinearElasticity(mesh, vec=3, dim=3)

# Define boundary conditions
def left_boundary(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right_boundary(point):
    return np.isclose(point[0], 1.0, atol=1e-5)

dirichlet_bc_info = [
    [left_boundary] * 3 + [right_boundary],  # boundary functions
    [0, 1, 2, 0],                           # DOF indices
    [lambda p: 0.0, lambda p: 0.0, lambda p: 0.0, lambda p: 0.1]  # values
]

bc = DirichletBC.from_bc_info(problem, dirichlet_bc_info)

# Set up material properties
E_field = InternalVars.create_uniform_volume_var(problem, 70e3)  # Young's modulus
internal_vars = InternalVars(volume_vars=(E_field,))

# Create solver
solver_options = SolverOptions(tol=1e-8, linear_solver="bicgstab")
solver = create_solver(problem, bc, solver_options)

# Solve
solution = solver(internal_vars)
print(f"Solution computed: displacement field shape {solution.shape}")

# Compute gradients
def compliance(internal_vars):
    u = solver(internal_vars)
    return np.sum(u**2)

grad_compliance = jax.grad(compliance)(internal_vars)
print(f"Gradient computed: {grad_compliance.volume_vars[0].shape}")
```

## Examples

### Linear Elasticity
- [Basic linear elasticity](examples/linear_elasticity.py): Static analysis with Dirichlet boundary conditions
- [Batch processing](examples/linear_elasticity_batch.py): Vectorized computation over parameter sets
- [Benchmarking](examples/linear_elasticity_benchmark.py): Performance comparison and scaling

### Nonlinear Problems
- [Hyperelasticity](examples/hyper_elasticity.py): Finite strain elasticity with Newton solver
- [Differentiable hyperelasticity](examples/hyperelasticity_diffrentable.py): Gradient-based optimization

### Advanced Features
- [Internal variables](examples/using_internal_vars.py): Custom material properties and loading
- [Surface loads](examples/using_internal_vars_surfaces.py): Complex boundary conditions

## API Overview

### Core Components

#### Problem Definition
```python
class MyProblem(Problem):
    def get_tensor_map(self):
        # Define constitutive relationships
        pass
    
    def get_surface_maps(self):
        # Define surface loads/tractions
        pass
```

#### Internal Variables
```python
# Material properties and loading parameters
internal_vars = InternalVars(
    volume_vars=(young_modulus, density),     # Element-wise properties
    surface_vars=(surface_traction,)          # Boundary-wise properties
)
```

#### Boundary Conditions
```python
bc = DirichletBC.from_bc_info(problem, [
    [boundary_function],  # Where to apply
    [dof_index],         # Which DOF
    [value_function]     # What value
])
```

#### Solvers
```python
# Newton solver for nonlinear problems
solution = newton_solve(J_func, res_func, initial_guess, options)

# Linear solver for linear problems
solution = linear_solve(J_func, res_func, initial_guess, options)

# Differentiable solver for optimization
solver = create_solver(problem, bc, options)
solution = solver(internal_vars)
```

## Performance

FEAX is designed for high performance:

- **Compilation**: Problems compile to efficient XLA code via JAX
- **Parallelization**: Natural parallelization over elements and quadrature points
- **Memory**: Optimized memory access patterns and minimal allocations
- **Scaling**: Efficient handling of large meshes and parameter spaces

### Benchmark Results

Performance comparison for linear elasticity problems with the following specifications:

![Linear Elasticity Solver Performance Benchmark](assets/linear_elasticity_benchmark.png)

**Problem Setup:**
- **Element Type**: 3D hexahedral elements (HEX8) 
- **Mesh Size**: 40×40×40 elements (274,625 nodes)
- **DOFs**: 823,875 degrees of freedom (3 DOFs per node)
- **Physics**: Linear elasticity with Young's modulus E = 70 GPa, Poisson's ratio ν = 0.3
- **Boundary Conditions**: Fixed left face, tension applied to right face
- **Solver**: BiCGSTAB iterative linear solver with 1e-6 tolerance

**Benchmark Details:**
- Tests batch processing of 1, 10, 50, and 100 different loading scenarios
- Compares three execution strategies: sequential loops (no JIT), JIT-compiled loops, and vectorized (vmap)
- All methods solve identical problems with varying tension loads (0.05 to 0.15)

**Machine Setup:**
- **CPU**: AMD Ryzen 9 7950X (16 cores)
- **GPU**: RTX A4000 (16GiB)
- **OS**: wsl2 on windows

The benchmark demonstrates FEAX's vectorization capabilities:
- **vmap (batched)**: Up to 4.1x speedup over sequential for loop execution
- **JIT compilation**: Significant performance improvements over Python loops
- **Scalability**: Near-linear scaling with batch size for parameter studies

## License

FEAX is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full license text.

## Acknowledgments

FEAX builds upon the excellent work of:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/tianjuxue/jax_fem) for inspiration and reference implementations
- The broader scientific computing community

---

*FEAX: Bringing modern automatic differentiation to finite element analysis.*