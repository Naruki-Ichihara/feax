# Adaptive Topology Optimization

This tutorial demonstrates 3D topology optimization with adaptive remeshing using FEAX's `gene` toolkit. The mesh is refined near material interfaces and coarsened in void regions, reducing computational cost while maintaining accuracy.

## Overview

**Adaptive topology optimization** combines density-based optimization with mesh adaptivity:

1. Solve the topology optimization on the current mesh
2. Periodically remesh — refine where material gradients are large, coarsen elsewhere
3. Transfer the density field to the new mesh via TET4 shape function interpolation
4. Recompile the JAX pipeline for the new mesh and continue

The `gene.optimizer.run()` function handles this loop automatically.

## Pipeline

Each iteration evaluates the following chain:

$$
\rho_{\text{node}} \xrightarrow{\text{filter}} \tilde{\rho} \xrightarrow{\text{Heaviside}(\beta)} \bar{\rho} \xrightarrow{\text{SIMP FE}} c(\bar{\rho})
$$

- **Density filter** smooths the node-based design variable $\rho$
- **Heaviside projection** sharpens the density toward 0/1 (controlled by $\beta$)
- **SIMP** penalizes intermediate densities in the finite element solve
- **Compliance** $c$ is minimized subject to a volume constraint

## Problem Setup

### Material and Geometry

```python
import jax.numpy as np
import feax as fe
import feax.gene as gene
from feax.gene.optimizer import (
    Pipeline, constraint, Continuation, AdaptiveConfig, run,
)

E0 = 70e3          # Young's modulus
nu = 0.3            # Poisson's ratio
E_eps = E0 * 1e-6   # Ersatz stiffness for void
penalty = 3          # SIMP penalty

L, W, H = 100.0, 5.0, 20.0   # Cantilever dimensions
tol = 1e-3

left  = lambda pt: np.isclose(pt[0], 0., tol)
right = lambda pt: np.isclose(pt[0], L, tol) & (pt[2] < H / 4)
```

### Linear Elasticity with SIMP

The problem class receives the projected density $\bar{\rho}$ as a node-based internal variable:

```python
class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = E_eps + (E0 - E_eps) * rho**penalty
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, *a: np.array([0., 0., -traction_mag])]
```

## Gmsh Geometry Builder

Adaptive remeshing uses Gmsh. Define the geometry as a callable that builds the CAD model:

```python
import gmsh

def cantilever_geometry():
    """Build cantilever box geometry via Gmsh OCC."""
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()
```

This callable is passed to `gene.adaptive.adaptive_mesh()`, which handles Gmsh initialization, meshing with a size callback, and TET4 extraction.

## Pipeline Class

Subclass `Pipeline` to define the optimization problem. The `build(mesh)` method creates all mesh-dependent objects — it is called once per mesh and re-called after each remesh, triggering JAX recompilation.

The `objective` method is required (abstract). Constraints are added by decorating methods with `@constraint`.

```python
class CantileverPipeline(Pipeline):
    def build(self, mesh):
        """Create all mesh-dependent objects. Called once per mesh."""
        problem = LinearElasticity(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type, location_fns=[right])

        bc = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(
                location=left, component="all", value=0.),
        ]).create_bc(problem)

        self._initial = fe.zero_like_initial_guess(problem, bc)
        self._compliance_fn = gene.create_compliance_fn(problem)
        self._volume_fn = gene.create_volume_fn(problem)
        self._filter_fn = gene.create_density_filter(mesh, 3.0)

        sample_iv = fe.InternalVars(
            volume_vars=(fe.InternalVars.create_node_var(problem, 0.4),),
            surface_vars=())
        solver_opts = fe.DirectSolverOptions()
        self._solver = fe.create_solver(
            problem, bc, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv)

    def objective(self, rho, beta=1.0):
        rho_f = self._filter_fn(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        iv = fe.InternalVars(volume_vars=(rho_p,), surface_vars=())
        sol = self._solver(iv, self._initial)
        return self._compliance_fn(sol)

    @constraint(target=0.4)
    def volume(self, rho, beta=1.0):
        rho_f = self._filter_fn(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        return self._volume_fn(rho_p)

    def filter(self, rho):
        return self._filter_fn(rho)
```

**Key points:**
- `build(mesh)` stores mesh-dependent objects as instance attributes
- `objective` and `@constraint` methods accept continuation parameters (here `beta`) as keyword arguments
- `@constraint(target=0.4)` declares a volume fraction inequality constraint ($\text{vol} \leq 0.4$). Use `@constraint(target=x, type='eq')` for equality constraints
- Constraints are optional — pipelines without any `@constraint` methods run unconstrained
- `filter` is optional — the default is the identity function
- After remesh, `run()` calls `build()` again, then recompiles objective and constraints

## Initial Mesh

Generate a TET4 mesh with Gmsh. The coarse `h_max` keeps the initial mesh lightweight:

```python
mesh = gene.adaptive.adaptive_mesh(cantilever_geometry, h_max=1.0)
```

## Adaptive Configuration

### Continuation

`Continuation` schedules parameter updates during optimization. Here $\beta$ doubles every 20 iterations from 1 to 16:

```python
Continuation(initial=1.0, final=16.0, update_every=20, multiply_by=2.0)
```

Updates happen at epoch boundaries (synchronized with remeshing). Continuation values are passed as traced JAX arguments, so updates do **not** trigger recompilation.

### Remeshing

`AdaptiveConfig` controls adaptive remeshing:

```python
AdaptiveConfig(
    remesh=lambda m, rho: gene.adaptive.adaptive_mesh(
        cantilever_geometry,
        refinement_field=gene.adaptive.gradient_refinement(rho, m),
        old_mesh=m,
        h_min=0.2, h_max=1.0,
    ),
    adapt_every=100,
    n_adapts_max=4,
)
```

**Parameters:**
- **`remesh`** — callable `(old_mesh, filtered_density) -> new_mesh`
- **`adapt_every`** — remesh interval (iterations)
- **`n_adapts_max`** — maximum number of remeshes

### Gradient-Based Refinement

`gene.adaptive.gradient_refinement(rho, mesh)` computes the density gradient magnitude per element using TET4 shape functions, then normalizes to $[0, 1]$:

$$
g_e = \|\nabla \rho_e\| = \|J_e^{-T} \Delta\rho_e\|
$$

where $J_e$ is the element Jacobian and $\Delta\rho_e$ are the nodal density differences. Elements with large gradients (material interfaces) get small mesh sizes; uniform regions get large sizes.

### Field Transfer

After remeshing, the density field is transferred from the old mesh to the new mesh using `gene.adaptive.interpolate_field()`. This function uses TET4 barycentric coordinates (shape functions) directly — no Delaunay re-triangulation:

1. Precompute inverse Jacobians for all old-mesh elements
2. For each new node, find the containing old element via KD-tree on centroids
3. Compute barycentric weights and interpolate
4. Fall back to nearest-node for points outside the old mesh

## Running the Optimization

```python
result = run(
    pipeline=CantileverPipeline(),
    mesh=mesh,
    max_iter=500,
    continuations={
        "beta": Continuation(initial=1.0, final=16.0, update_every=20,
                             multiply_by=2.0),
    },
    adaptive=AdaptiveConfig(
        remesh=lambda m, rho: gene.adaptive.adaptive_mesh(
            cantilever_geometry,
            refinement_field=gene.adaptive.gradient_refinement(rho, m),
            old_mesh=m,
            h_min=0.2, h_max=1.0,
        ),
        adapt_every=100,
        n_adapts_max=4,
    ),
    output_dir="output_adaptive",
    save_every=2,
)
```

`run()` handles the full loop:
1. Call `pipeline.build(mesh)` and JIT-compile objective + constraints
2. At each epoch boundary, update continuation parameters
3. Run MMA (NLopt) optimizer for one epoch
4. If `iter_count % adapt_every == 0`, remesh and transfer density
5. Rebuild pipeline for new mesh and continue

Set `jit=False` to disable JIT compilation (useful for debugging).

## Visualization

The result contains convergence history:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(result.history['iteration'], result.history['objective'])
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Compliance')

axes[1].plot(result.history['iteration'], result.history['volume'])
axes[1].axhline(y=0.4, color='r', linestyle='--', label='Target')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Volume Fraction')
axes[1].legend()

plt.tight_layout()
plt.savefig("output_adaptive/history.png", dpi=150)
```

VTU files saved to `output_dir` can be opened in ParaView. Adapted meshes are saved as `adapt_01.vtu`, `adapt_02.vtu`, etc.

## Complete Code

```python
import jax.numpy as np
import numpy as onp
import gmsh
import feax as fe
import feax.gene as gene
from feax.gene.optimizer import (
    Pipeline, constraint, Continuation, AdaptiveConfig, run,
)

# Material
E0 = 70e3
nu = 0.3
E_eps = E0 * 1e-6
penalty = 3
traction_mag = 1.0

# Geometry
L, W, H = 100.0, 5.0, 20.0
tol = 1e-3
left = lambda pt: np.isclose(pt[0], 0., tol)
right = lambda pt: np.isclose(pt[0], L, tol) & (pt[2] < H / 4)

def cantilever_geometry():
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = E_eps + (E0 - E_eps) * rho**penalty
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, *a: np.array([0., 0., -traction_mag])]

class CantileverPipeline(Pipeline):
    def build(self, mesh):
        problem = LinearElasticity(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type, location_fns=[right])
        bc = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(
                location=left, component="all", value=0.),
        ]).create_bc(problem)
        self._initial = fe.zero_like_initial_guess(problem, bc)
        self._compliance_fn = gene.create_compliance_fn(problem)
        self._volume_fn = gene.create_volume_fn(problem)
        self._filter_fn = gene.create_density_filter(mesh, 3.0)

        sample_iv = fe.InternalVars(
            volume_vars=(fe.InternalVars.create_node_var(problem, 0.4),),
            surface_vars=())
        solver_opts = fe.DirectSolverOptions()
        self._solver = fe.create_solver(
            problem, bc, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv)

    def objective(self, rho, beta=1.0):
        rho_f = self._filter_fn(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        iv = fe.InternalVars(volume_vars=(rho_p,), surface_vars=())
        sol = self._solver(iv, self._initial)
        return self._compliance_fn(sol)

    @constraint(target=0.4)
    def volume(self, rho, beta=1.0):
        rho_f = self._filter_fn(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        return self._volume_fn(rho_p)

    def filter(self, rho):
        return self._filter_fn(rho)

# Initial mesh
mesh = gene.adaptive.adaptive_mesh(cantilever_geometry, h_max=1.0)

# Run
result = run(
    pipeline=CantileverPipeline(),
    mesh=mesh,
    max_iter=500,
    continuations={
        "beta": Continuation(initial=1.0, final=16.0, update_every=20,
                             multiply_by=2.0),
    },
    adaptive=AdaptiveConfig(
        remesh=lambda m, rho: gene.adaptive.adaptive_mesh(
            cantilever_geometry,
            refinement_field=gene.adaptive.gradient_refinement(rho, m),
            old_mesh=m,
            h_min=0.2, h_max=1.0,
        ),
        adapt_every=100,
        n_adapts_max=4,
    ),
    output_dir="output_adaptive",
    save_every=2,
)
```

## Summary

**Key concepts:**
- **`Pipeline`** — abstract base class for optimization pipelines (`build`, `objective`, `filter`)
- **`@constraint`** — decorator to mark methods as inequality (`type='le'`) or equality (`type='eq'`) constraints
- **`run()`** — drives the optimization loop with JIT compilation, continuation, and adaptive remeshing
- **`Continuation`** — schedules parameter updates ($\beta$, filter radius, etc.)
- **`AdaptiveConfig`** — configures remeshing interval, max remeshes, and remesh callable
- **`gene.adaptive.gradient_refinement()`** — density-gradient-based refinement field for TET4 meshes
- **`gene.adaptive.interpolate_field()`** — TET4 barycentric interpolation for field transfer

**Workflow:**
1. Define geometry as Gmsh callable
2. Subclass `Pipeline` with `build()`, `objective()`, and `@constraint` methods
3. Configure `Continuation` and `AdaptiveConfig`
4. Call `run(pipeline, mesh, ...)` — handles compilation, optimization, remeshing, and field transfer

## Further Reading

- `examples/advance/topology_optimization_adaptive.py` - Complete working example
