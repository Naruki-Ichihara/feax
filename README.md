<div align="center">
<img src="assets/logo.svg" alt="logo" width=300></img>
</div>

# FEAX

[![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7%2B-green.svg)](https://github.com/google/jax)

**FEAX** (Finite Element Analysis with JAX) is a compact, high-performance finite element analysis engine built on JAX. It provides an API for solving partial differential equations with automatic differentiation, JIT compilation, and GPU acceleration.

## What is FEAX?

FEAX combines the power of modern automatic differentiation with classical finite element methods. It's designed for:

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

### Architecture
```python
# Separate problem definition from parameters
problem = ElasticityProblem(mesh, vec=3, dim=3)
internal_vars = InternalVars(
    volume_vars=(young_modulus, density),
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
import jax
import jax.numpy as np
from feax import Problem, InternalVars, create_solver
from feax import Mesh, SolverOptions, zero_like_initial_guess
from feax import DirichletBCSpec, DirichletBCConfig
from feax.mesh import box_mesh_gmsh

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
meshio_mesh = box_mesh_gmsh(40, 20, 20, 2.0, 1.0, 1.0, data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Set up problem
problem = LinearElasticity(mesh, vec=3, dim=3)

# Define boundary conditions using new API
def left_boundary(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def right_boundary(point):
    return np.isclose(point[0], 2.0, atol=1e-5)

bc_config = DirichletBCConfig([
    # Fix left boundary completely (all components to zero)
    DirichletBCSpec(left_boundary, 0, lambda p: 0.0),
    DirichletBCSpec(left_boundary, 1, lambda p: 0.0), 
    DirichletBCSpec(left_boundary, 2, lambda p: 0.0),
    # Apply tension on right boundary
    DirichletBCSpec(right_boundary, 0, lambda p: 0.1)
])

# Create boundary conditions from config
bc = bc_config.create_bc(problem)

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
- [**linear_elasticity.py**](examples/linear_elasticity.py): Linear elasticity with SIMP-based material interpolation
- [**linear_elasticity_vmap_density.py**](examples/linear_elasticity_vmap_density.py): Vectorized density-based material optimization
- [**linear_elasticity_vmap_traction.py**](examples/linear_elasticity_vmap_traction.py): Vectorized traction loading scenarios

### Other Physics
- [**poisson_2d.py**](examples/poisson_2d.py): 2D Poisson equation solver equivalent to JAX-FEM reference

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

### Kernels for Weak Form Construction

FEAX uses kernels to construct weak forms for finite element analysis. These kernels implement the integral terms that arise from applying the Galerkin method to partial differential equations.

#### Supported Kernels

##### 1. Laplace Kernel (Diffusion/Elasticity)
The Laplace kernel handles gradient-based physics like heat conduction, diffusion, and elasticity:

$$\int_{\Omega} \sigma(\nabla u) : \nabla v \, d\Omega$$

where:
- $\sigma(\nabla u)$ is the stress/flux tensor computed from the gradient
- $v$ is the test function
- $:$ denotes tensor contraction

**Implementation**: Define `get_tensor_map()` returning a function that maps gradients to stress/flux tensors.

```python
def get_tensor_map(self):
    def stress_tensor(u_grad, material_param):
        # u_grad: (vec, dim) gradient tensor
        # Returns: (vec, dim) stress/flux tensor
        return compute_stress(u_grad, material_param)
    return stress_tensor
```

##### 2. Mass Kernel (Inertia/Reaction)
The mass kernel handles terms without derivatives, used for inertia, reaction, or body forces:

$$\int_{\Omega} m(u, x) \cdot v \, d\Omega$$

where:
- $m(u, x)$ is a mass-like term (can depend on solution and position)
- $v$ is the test function

**Implementation**: Define `get_mass_map()` returning a function for the mass term.

```python
def get_mass_map(self):
    def mass_map(u, x, density):
        # u: (vec,) solution at quadrature point
        # x: (dim,) physical coordinates
        # Returns: (vec,) mass term
        return density * acceleration_term(u)
    return mass_map
```

##### 3. Surface Kernel (Boundary Loads)
Surface kernels handle boundary integrals for surface tractions, pressures, or fluxes:

$$\int_{\Gamma} t(u, x) \cdot v \, d\Gamma$$

where:
- $t(u, x)$ is the surface traction/flux
- $\Gamma$ is the boundary surface

**Implementation**: Define `get_surface_maps()` returning a list of surface functions.

```python
def get_surface_maps(self):
    def surface_traction(u, x, traction_magnitude):
        # u: (vec,) solution at surface quadrature point
        # x: (dim,) surface coordinates
        # Returns: (vec,) traction vector
        return np.array([0., 0., traction_magnitude])
    return [surface_traction]  # List for multiple boundaries
```

##### 4. Universal Kernel (Custom Terms)
For complex physics that don't fit the above patterns, use universal kernels with full access to shape functions and quadrature data:

$$\int_{\Omega} f(u, \nabla u, x, N, \nabla N) \, d\Omega$$

**Implementation**: Define `get_universal_kernel()` for volume integrals or `get_universal_kernels_surface()` for surface integrals.

```python
def get_universal_kernel(self):
    def universal_kernel(cell_sol_flat, x, shape_grads, JxW, v_grads_JxW, *params):
        # Full access to FE data for custom weak forms
        # cell_sol_flat: flattened solution on element
        # x: quadrature points
        # shape_grads: shape function gradients
        # JxW: Jacobian times quadrature weights
        # v_grads_JxW: test function gradients times JxW
        return custom_weak_form_contribution
    return universal_kernel
```

#### Kernel Composition

The total weak form is the sum of all kernel contributions:

$$R(u) = \int_{\Omega} \left[ \sigma(\nabla u) : \nabla v + m(u) \cdot v \right] d\Omega + \sum_i \int_{\Gamma_i} t_i(u) \cdot v \, d\Gamma_i$$

FEAX automatically:
1. Evaluates each kernel at quadrature points
2. Applies quadrature weights and Jacobians
3. Assembles contributions into the global residual
4. Computes the Jacobian matrix via automatic differentiation

#### Implementation Requirements

When implementing a Problem subclass:

1. **Laplace kernel** (`get_tensor_map`): Required for gradient-based physics
2. **Mass kernel** (`get_mass_map`): Optional, for mass/reaction terms
3. **Surface kernels** (`get_surface_maps`): Optional, returns list of boundary functions
4. **Universal kernels**: Optional, for complex custom physics

The kernels receive internal variables (material properties, loads) as additional arguments, enabling parameterization and differentiation.

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
# New boundary condition API using dataclasses
bc_config = DirichletBCConfig([
    DirichletBCSpec(boundary_function, dof_index, value_function),
    # Multiple boundary conditions
])

# Legacy API still available via DirichletBC.from_bc_info
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

# Differentiable solver for optimization (recommended)
solver = create_solver(problem, bc, solver_options)
solution = solver(internal_vars)
```

## License

FEAX is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full license text.

## Acknowledgments

FEAX builds upon the excellent work of:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/tianjuxue/jax_fem) for inspiration and reference implementations

---

*FEAX: Bringing modern automatic differentiation to finite element analysis.*
