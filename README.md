<div align="center">
<img src="assets/logo.svg" alt="logo" width=150></img>
</div>

# [FEAX](https://naruki-ichihara.github.io/feax/) 




### *A simulation node in your JAX workflow*

[**Documentation**](https://naruki-ichihara.github.io/feax/)
| [**Install guide**](https://naruki-ichihara.github.io/feax/getting-started/installation)
| [**API**](https://naruki-ichihara.github.io/feax/api/)
| [**Benchmark**](https://naruki-ichihara.github.io/feax/benchmarks/)

[![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.7%2B-green.svg)](https://github.com/google/jax) 

**FEAX** (Finite Element Analysis with JAX) is a **pure, statically-composed functional finite element engine**. The whole analysis — assembly, boundary conditions, linear/nonlinear solve, post-processing — is expressed as pure functions composed at trace time into a single static XLA program: no hidden state, no Python in the hot loop. That program is fully differentiable and composes with `jax.jit`, `jax.grad`, and `jax.vmap`, so a PDE solve becomes just another differentiable node in your JAX workflow.

## Why FEAX?

- **Pure & statically composed**: the entire FE pipeline is built from pure functions traced into one static XLA graph — deterministic, side-effect-free, and free of per-iteration Python dispatch.
- **JAX transformations**: every stage works with `jax.jit`, `jax.grad`, and `jax.vmap`, and arbitrary compositions such as `jit(grad(vmap(...)))`, giving exact gradients through the whole pipeline with no approximation.
- **Solvers**: GPU sparse direct via [cuDSS](https://docs.nvidia.com/cuda/cudss/) (automatic General / Symmetric / SPD detection), matrix-free Krylov (CG / BiCGSTAB / GMRES), and algebraic multigrid (AMG) — smoothed aggregation through [AMJax](https://github.com/vboussange/AMJax) + [PyAMG](https://github.com/pyamg/pyamg) (`pip install feax[amg]`). Optional CPU host-side direct (CHOLMOD / UMFPACK) too.
- **End-to-end differentiability**: gradients flow through assembly, boundary conditions, linear/nonlinear solvers, and post-processing — enabling topology optimization, inverse problems, and design optimization.
- **Built-in toolkits**: a `gene` topology-optimization toolkit (MMA, density filters, Heaviside continuation, adaptive remeshing) and a `flat` computational-homogenization toolkit for periodic unit cells.


## Quick Example

A 3D cantilever beam under a surface traction — solved, then differentiated. We
build it up block by block.

**1. Imports and a mesh.** A structured hex mesh of a 100×10×10 box, plus the
material constants.

```python
import feax as fe
import jax
import jax.numpy as np

mesh = fe.mesh.box_mesh((100, 10, 10), mesh_size=2)
E, nu = 70e3, 0.3
```

**2. Define the physics.** A `Problem` subclass declares the weak form through
pure callables: `get_tensor_map` returns the stress as a function of the
displacement gradient (the volume term), and `get_surface_maps` returns the
applied traction (the surface term). `traction_mag` is a *traced parameter*
supplied at solve time — that is what makes the load differentiable.

```python
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
```

**3. Instantiate the problem.** Locator functions mark the boundary faces; the
traction is applied on the right face (`location_fns=[right]`), `vec=3` is the
number of solution components (3D displacement).

```python
left  = lambda point: np.isclose(point[0], 0.,   atol=1e-5)
right = lambda point: np.isclose(point[0], 100., atol=1e-5)

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])
```

**4. Dirichlet boundary conditions.** Clamp the left face (all components zero).

```python
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
])
bc = bc_config.create_bc(problem)
```

**5. Traced parameters.** The runtime inputs you differentiate/`vmap` over —
here a uniform surface-traction magnitude. These flow in as the `traction_mag`
argument of the surface map above.

```python
traction = fe.TracedParams.create_uniform_surface_var(problem, 1e-3)
traced_params = fe.TracedParams(volume_vars=(), surface_vars=[(traction,)])
```

**6. Traced structure (recommended).** Collects the mesh-sized structural arrays
as traced leaves, so nothing mesh-sized is baked into the compiled program as a
constant — important for compile time and memory on large meshes.

```python
ts = fe.TracedStructure.from_problem(problem)
```

**7. Build the solver.** `create_solver` composes assembly + boundary conditions
+ linear solve into one differentiable function. With `DirectSolverOptions()` it
auto-selects cuDSS on GPU (sparse direct on CPU); swap in
`fe.KrylovSolverOptions()` or `fe.AMGSolverOptions()` to change the linear solver
with no other changes.

```python
solver = fe.create_solver(problem, bc,
    solver_options=fe.DirectSolverOptions(), linear=True,
    traced_params=traced_params, traced_structure=ts)
initial = fe.zero_like_initial_guess(problem, bc)
```

**8. Solve, then differentiate.** The solver is an ordinary JAX function. Call it
to get the solution; or build a scalar objective that runs the solve and hand
*that function* to `jax.grad` — the gradient flows through the *entire* pipeline
(assembly → BCs → linear solve).

```python
# Solve
sol = solver(traced_params, initial, traced_structure=ts)

# Build an objective on top of the solver, then differentiate it
def objective(params):
    u = solver(params, initial, traced_structure=ts)
    return np.sum(u ** 2)

grads = jax.grad(objective)(traced_params)
```

See [examples/](examples/) for more, including topology optimization. Installation and solver setup are covered in the [install guide](https://naruki-ichihara.github.io/feax/getting-started/installation).

## License

FEAX is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full license text.

## Acknowledgments

FEAX builds upon the excellent work of:
- [JAX](https://github.com/google/jax) for automatic differentiation and compilation
- [JAX-FEM](https://github.com/deepmodeling/jax-fem) for inspiration and reference implementations
- [Spineax](https://github.com/johnviljoen/spineax) for cuDSS solver implementation
- [AMJax](https://github.com/vboussange/AMJax) and [PyAMG](https://github.com/pyamg/pyamg) for the algebraic multigrid (AMG) solver

---

