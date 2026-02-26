# Getting Started

**FEAX** (Finite Element Analysis with JAX) is a fully differentiable finite element engine. Every stage — from assembly to solve — runs on XLA and is compatible with `jax.jit`, `jax.grad`, and `jax.vmap`, enabling gradient-based optimization and machine learning directly on PDE simulations.

## Why FEAX?

- **JAX Transformations**: Solvers work seamlessly with `jax.jit`, `jax.grad`, and `jax.vmap`, and arbitrary compositions such as `jit(grad(...))`.
- **GPU Direct Solver**: Native cuDSS integration for sparse direct solves on GPU, with automatic matrix property detection (General / Symmetric / SPD).
- **End-to-End Differentiability**: Gradients flow through assembly, boundary conditions, linear/nonlinear solvers, and post-processing — enabling topology optimization, inverse problems, and physics-informed learning.
- **Neural Network Research**: The consistent differentiability makes FEAX a natural building block for coupling neural networks with PDE solvers.

## Quick Example

3D cantilever beam under traction — solve and compute gradients in a few lines:

```python
import feax as fe
import jax
import jax.numpy as np

# Mesh and material
mesh = fe.mesh.box_mesh((100, 10, 10), mesh_size=2)
E, nu = 70e3, 0.3

# Define the constitutive law
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

left  = lambda point: np.isclose(point[0], 0.,   atol=1e-5)
right = lambda point: np.isclose(point[0], 100., atol=1e-5)

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary conditions: fix the left face
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
])
bc = bc_config.create_bc(problem)

# Internal variables (surface traction magnitude)
traction = fe.InternalVars.create_uniform_surface_var(problem, 1e-3)
internal_vars = fe.InternalVars(volume_vars=(), surface_vars=[(traction,)])

# Create solver (auto-selects cuDSS on GPU, sparse direct on CPU)
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(), iter_num=1,
    internal_vars=internal_vars)
initial = fe.zero_like_initial_guess(problem, bc)

# Solve
sol = solver(internal_vars, initial)

# Differentiate through the entire solve
grad_fn = jax.grad(lambda iv: np.sum(solver(iv, initial) ** 2))
grads = grad_fn(internal_vars)
```

## Limitations

- **First-order differentiation only**: Solvers use `custom_vjp` internally, so `jax.grad` (first-order) is supported but `jax.hessian` (second-order) is not.
- **Static problems only**: No time-dependent or transient solvers. Only steady-state and quasi-static analyses are available.
- **Element order up to quadratic**: Linear (degree 1) and quadratic (degree 2) elements are supported. Cubic or higher-order elements are not.
- **Fixed mesh**: The mesh topology must remain constant throughout JAX transformations. No adaptive remeshing or h-refinement during differentiation.
- **Single machine**: No MPI or distributed computing support. Parallelism is limited to JAX's device-level parallelism (`vmap`).

## Installation

```bash
pip install feax[cuda13]
pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git
```

See the [Installation](./installation.md) page for detailed installation instructions.

## License

FEAX is licensed under the GNU General Public License v3.0.

## Acknowledgments

FEAX builds upon the excellent work of:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/deepmodeling/jax-fem) for inspiration and reference implementations
- [Spineax](https://github.com/johnviljoen/spineax) for cuDSS solver implementation
