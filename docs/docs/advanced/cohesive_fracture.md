# Cohesive Fracture with Matrix-Free Newton Solver

This tutorial demonstrates quasi-static fracture simulation using FEAX's matrix-free Newton solver and cohesive zone model. We solve a 3D Mode I fracture problem where the total energy is composed of bulk elastic and cohesive interface contributions — all expressed as pure JAX functions and differentiated automatically.

## Overview

Traditional FE fracture solvers assemble the tangent stiffness matrix explicitly. FEAX takes a different approach:

1. Define the **total potential energy** $\Pi(\mathbf{u})$ as a sum of bulk and cohesive terms
2. The **residual** is obtained automatically: $\mathbf{r} = \nabla_\mathbf{u} \Pi$
3. The **tangent operator** is computed via JVP (Jacobian-vector product) of the residual — no sparse matrix assembly
4. An iterative Krylov solver (CG) solves the Newton linear system using only matvec operations

This is the **matrix-free Newton** path in FEAX, activated by passing `MatrixFreeOptions` as the solver options.

## Energy-Based Formulation

### Bulk Elastic Energy

The bulk elastic strain energy density for linear elasticity is:

$$
\psi(\nabla \mathbf{u}) = \frac{1}{2}\lambda\,(\text{tr}\,\boldsymbol{\varepsilon})^2 + \mu\,\boldsymbol{\varepsilon}:\boldsymbol{\varepsilon}
$$

where $\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla\mathbf{u} + \nabla\mathbf{u}^T)$ is the infinitesimal strain tensor, and $\lambda$, $\mu$ are the Lamé constants:

$$
\mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

The total bulk energy is the volume integral:

$$
\Pi_\text{bulk}(\mathbf{u}) = \int_\Omega \psi(\nabla\mathbf{u})\,d\Omega
$$

In FEAX, this is obtained via `get_energy_density()` and `create_energy_fn()`:

```python
class Elasticity3D(fe.problem.Problem):
    def get_energy_density(self):
        def psi(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(eps)**2 + mu * np.sum(eps * eps)
        return psi

elastic_energy = create_energy_fn(problem)  # ∫ ψ(∇u) dΩ
```

Note that `get_energy_density()` returns the scalar energy density $\psi$, not the stress tensor. FEAX integrates this over quadrature points to compute the total energy. The stress and tangent stiffness are never assembled explicitly — they are computed on-the-fly via JAX's automatic differentiation.

### Cohesive Interface Energy

The cohesive zone model introduces an energy contribution along the fracture interface $\Gamma_c$. For a node pair $(i^+, i^-)$ across the interface, the displacement jump is:

$$
\boldsymbol{\delta}_i = \mathbf{u}_{i^+} - \mathbf{u}_{i^-}
$$

This jump is decomposed into normal and tangential components:

$$
\delta_n = \boldsymbol{\delta} \cdot \mathbf{n}, \quad \delta_t = |\boldsymbol{\delta} - \delta_n\,\mathbf{n}|
$$

The effective opening combines both modes:

$$
\delta = \sqrt{\langle\delta_n\rangle_+^2 + \beta^2\,\delta_t^2}
$$

where $\langle\cdot\rangle_+ = \max(\cdot, 0)$ is the Macaulay bracket (no energy in compression) and $\beta$ is the mode-mixity ratio.

#### Xu–Needleman Exponential Potential

The cohesive potential per node follows the Xu–Needleman law:

$$
\phi(\delta) = \Gamma\left[1 - \left(1 + \frac{\delta}{\delta_c}\right)\exp\left(-\frac{\delta}{\delta_c}\right)\right]
$$

where $\delta_c = \Gamma / (e\,\sigma_c)$ is the characteristic opening, $\Gamma$ is the fracture energy, and $\sigma_c$ is the critical cohesive traction. The traction-separation relation is:

$$
T(\delta) = \frac{d\phi}{d\delta} = \frac{\Gamma}{\delta_c^2}\,\delta\,\exp\left(-\frac{\delta}{\delta_c}\right)
$$

with peak traction $T_\text{max} = \sigma_c$ at $\delta = \delta_c$.

#### Irreversibility

Unloading follows a secant path to prevent energy recovery:

$$
\phi_\text{unload}(\delta) = \frac{\phi(\delta_\text{max})}{\delta_\text{max}^2}\,\delta^2
$$

where $\delta_\text{max}$ is the historical maximum opening. This requires tracking $\delta_\text{max}$ as a state variable across load steps.

#### Total Cohesive Energy

The total cohesive energy is a weighted sum over interface nodes:

$$
\Pi_\text{coh}(\mathbf{u}, \boldsymbol{\delta}_\text{max}) = \sum_{i=1}^{N_\text{coh}} w_i\,\phi(\delta_i, \delta_{\text{max},i})
$$

where $w_i$ are the integration weights (lumped area per node).

```python
interface = CohesiveInterface.from_axis(
    top_nodes=coh_top, bottom_nodes=coh_bottom,
    weights=weights, normal_axis=1, vec=3, beta=0.0,
)

cohesive_energy = interface.create_energy_fn(
    exponential_potential, Gamma=Gamma, sigma_c=sigma_c,
)
```

### Total Energy and Matrix-Free Newton

The total potential energy is the sum:

$$
\Pi(\mathbf{u}, \boldsymbol{\delta}_\text{max}) = \Pi_\text{bulk}(\mathbf{u}) + \Pi_\text{coh}(\mathbf{u}, \boldsymbol{\delta}_\text{max})
$$

```python
def total_energy(u_flat, delta_max):
    return elastic_energy(u_flat) + cohesive_energy(u_flat, delta_max)
```

The Newton solver finds $\mathbf{u}$ such that $\nabla_\mathbf{u}\Pi = \mathbf{0}$. At each Newton iteration $k$:

1. **Residual**: $\mathbf{r}^{(k)} = \nabla_\mathbf{u}\Pi(\mathbf{u}^{(k)})$ — computed via `jax.grad`
2. **Newton direction**: solve $\mathbf{K}^{(k)}\,\Delta\mathbf{u} = -\mathbf{r}^{(k)}$ where $\mathbf{K} = \nabla^2_\mathbf{u}\Pi$
3. **Update**: $\mathbf{u}^{(k+1)} = \mathbf{u}^{(k)} + \Delta\mathbf{u}$

The key: step 2 never forms $\mathbf{K}$ explicitly. Instead, the CG solver only needs the matrix-vector product $\mathbf{K}\mathbf{v}$, which is computed via JVP:

$$
\mathbf{K}\mathbf{v} = \frac{\partial}{\partial\epsilon}\nabla_\mathbf{u}\Pi(\mathbf{u} + \epsilon\mathbf{v})\bigg|_{\epsilon=0}
$$

This is exactly what `jax.jvp` computes in forward-mode AD, at roughly the cost of one residual evaluation.

## Problem Setup

### Material and Geometry

```python
import jax.numpy as np
import numpy as onp
import feax as fe
from feax.mechanics.cohesive import (
    CohesiveInterface, compute_lumped_area_weights, exponential_potential,
)
from feax.solvers.matrix_free import (
    LinearSolverOptions, MatrixFreeOptions, create_energy_fn,
)

# Material parameters
E = 106e3          # Young's modulus [Pa]
nu = 0.35          # Poisson's ratio
Gamma = 15.0       # Fracture energy [J/m²]
sigma_c = 20e3     # Critical cohesive traction [Pa]

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
```

The geometry is scaled by the Griffith length:

$$
L_G = \frac{2\mu\,\Gamma}{\pi(1-\nu)\,\sigma_\infty^2}
$$

where $\sigma_\infty$ is the far-field stress. The specimen is $10\,L_G \times 2\,L_G \times L_G$ with an initial crack of length $L_G$.

### Mesh with Split Interface

The mesh consists of two half-blocks (top/bottom) separated at $y = 0$. Nodes on the interface are duplicated to allow displacement discontinuity. A pre-crack extends from $x = 0$ to $x = L_G$ (free surfaces with no cohesive traction).

```python
mesh = fe.mesh.Mesh(points=np.array(coords), cells=np.array(elements))
problem = Elasticity3D(mesh, vec=3, dim=3, ele_type='HEX8')
```

### Cohesive Interface Setup

Integration weights are computed from the quad elements on the interface surface using lumped area weighting — each quad contributes 1/4 of its area to each of its 4 nodes:

```python
weights = compute_lumped_area_weights(coh_bottom, coords, active_quads_bottom)

interface = CohesiveInterface.from_axis(
    top_nodes=coh_top, bottom_nodes=coh_bottom,
    weights=weights, normal_axis=1, vec=3, beta=0.0,
)
```

`beta=0.0` gives pure Mode I (only normal opening contributes to the effective opening).

### Boundary Conditions

Mode I loading via prescribed displacement on top/bottom faces:

```python
def make_bc(disp):
    specs = [
        fe.DCboundary.DirichletBCSpec(top_face, 'x', 0.0),
        fe.DCboundary.DirichletBCSpec(top_face, 'y', disp / 2),
        fe.DCboundary.DirichletBCSpec(top_face, 'z', 0.0),
        fe.DCboundary.DirichletBCSpec(bottom_face, 'x', 0.0),
        fe.DCboundary.DirichletBCSpec(bottom_face, 'y', -disp / 2),
        fe.DCboundary.DirichletBCSpec(bottom_face, 'z', 0.0),
        fe.DCboundary.DirichletBCSpec(left_face, 'x', 0.0),
        fe.DCboundary.DirichletBCSpec(front_face, 'z', 0.0),
        fe.DCboundary.DirichletBCSpec(back_face, 'z', 0.0),
    ]
    return fe.DCboundary.DirichletBCConfig(specs).create_bc(problem)
```

## Solver

Create the matrix-free solver by passing `MatrixFreeOptions` and the custom `total_energy` function:

```python
solver_options = MatrixFreeOptions(
    newton_tol=1e-8,
    newton_max_iter=200,
    linear_solver=LinearSolverOptions(solver='cg', atol=1e-8, maxiter=200),
    verbose=True,
)

bc0 = make_bc(0.0)
solver = fe.create_solver(
    problem, bc0,
    solver_options=solver_options,
    energy_fn=total_energy,
)
```

**Key points:**
- `MatrixFreeOptions` activates the JVP-based Newton path
- `energy_fn=total_energy` provides the custom energy (bulk + cohesive)
- The inner CG solver uses only matvec operations — no sparse matrix stored
- `verbose=True` prints Newton convergence info via `jax.debug.print` (JIT-compatible)

## Quasi-Static Loading

Each load step increments the prescribed displacement. The solver uses the previous solution as initial guess, and $\delta_\text{max}$ is updated after convergence:

```python
u_flat = np.zeros(problem.num_total_dofs_all_vars)
delta_max = np.zeros(interface.n_nodes)

for step in range(1, n_steps + 1):
    disp = applied_disp * step / n_steps

    # Apply BC values to initial guess
    bc = make_bc(disp)
    u_flat = u_flat.at[bc.bc_rows].set(bc.bc_vals)

    # Solve: delta_max is passed as extra argument
    u_flat = solver(delta_max, u_flat)

    # Update irreversibility state
    delta_current = interface.get_opening(u_flat)
    delta_max = np.maximum(delta_max, delta_current)
```

Note: `solver(delta_max, u_flat)` — the first argument corresponds to the extra arguments of `total_energy`, and the second is the initial guess.

## Post-Processing

### Reaction Force via Energy Gradient

The reaction force is extracted from the internal force vector, which is the gradient of the total energy:

$$
\mathbf{f}_\text{int} = \nabla_\mathbf{u}\Pi(\mathbf{u}, \boldsymbol{\delta}_\text{max})
$$

```python
gradient_fn = jax.grad(total_energy)
fint = gradient_fn(u_flat, delta_max)
reaction_force = float(np.sum(fint[upper_y_dofs]))
```

### Energy Decomposition

Track elastic and cohesive energy separately to monitor fracture progression:

```python
e_elastic = elastic_energy(u_flat)
e_cohesive = cohesive_energy(u_flat, delta_max)
```

When the cohesive energy reaches the total fracture work $\Gamma \cdot W$ (where $W$ is the ligament area), complete separation has occurred.

### Visualization

Save VTK files with displacement and $\delta_\text{max}$ fields:

```python
fe.utils.save_sol(
    mesh=mesh, sol_file='fracture3d.vtu',
    point_infos=[("displacement", u), ("delta_max", d_max_full[:, None])]
)
```

## Summary

**Key concepts:**
- **Energy-based formulation** — define $\Pi(\mathbf{u})$ as a scalar function; stress and tangent stiffness are derived automatically via JAX AD
- **`get_energy_density()`** — returns $\psi(\nabla\mathbf{u})$ for the bulk; `create_energy_fn()` integrates it over the domain
- **`CohesiveInterface`** — encodes interface geometry (node pairs, normals, weights) and creates the cohesive energy function
- **Matrix-free Newton** — JVP-based tangent operator avoids sparse matrix assembly; inner CG solver uses only matvec
- **Irreversibility** — $\delta_\text{max}$ state variable tracks maximum opening; updated outside the solver between load steps

**Why matrix-free?**
- Cohesive contributions couple arbitrary node pairs — hard to fit into standard FE sparse assembly
- The energy formulation naturally combines bulk and interface terms as pure JAX functions
- JVP cost ≈ 1 residual evaluation, regardless of problem size or sparsity pattern

## Further Reading

- `examples/advance/cohesive_fracture_2d.py` — 2D version with QUAD4 elements
- `examples/advance/cohesive_fracture_3d.py` — Complete 3D working example
- [API: feax.mechanics.cohesive](../api/reference/feax/mechanics/cohesive.md) — Cohesive zone models
- [API: feax.solvers.matrix_free](../api/reference/feax/solvers/matrix_free.md) — Matrix-free Newton solver
