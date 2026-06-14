# Cohesive Fracture with a Hybrid Matrix-Free Newton Solver

This tutorial demonstrates quasi-static fracture simulation using FEAX's hybrid matrix-free Newton–Krylov solver and a cohesive zone model. We solve a 3D Mode I fracture problem where the residual combines a bulk elastic contribution with a cohesive interface contribution.

## Overview

The bulk elasticity is handled by FEAX's standard residual assembly, while the cohesive interface — which couples arbitrary node pairs and does not fit the element weak form — is supplied as an **extra residual term**:

1. The **bulk residual** is assembled by FEAX from the elastic energy density (`get_energy_density` → `σ = jax.grad(ψ)`), giving the sparse bulk Jacobian.
2. The **cohesive residual** $\mathbf{r}_\text{coh} = \partial\Phi_\text{coh}/\partial\mathbf{u}$ is passed via `extra_residual_fn`.
3. At each Newton step the combined tangent applies the bulk Jacobian (sparse) plus the cohesive tangent **matrix-free** via `jax.jvp`.
4. A Krylov solver (CG) solves the Newton system using matrix–vector products.

This is the **hybrid matrix-free Newton–Krylov** path in FEAX, activated by passing `extra_residual_fn` together with `KrylovSolverOptions`.

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

In FEAX, the bulk is defined through `get_energy_density()`. When `get_tensor_map()` is not defined, the assembler automatically uses `σ = jax.grad(ψ)` for the residual and Jacobian, so the bulk needs no hand-written stress:

```python
class Elasticity3D(fe.problem.Problem):
    def get_energy_density(self):
        def psi(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(eps)**2 + mu * np.sum(eps * eps)
        return psi

# Scalar bulk energy — used for energy decomposition / post-processing only;
# the solve itself uses the residual assembled from get_energy_density.
elastic_energy = fe.create_energy_fn(problem)  # ∫ ψ(∇u) dΩ
```

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

### Cohesive Residual

The cohesive contribution enters the global residual as the gradient of the interface potential at the current history $\boldsymbol{\delta}_\text{max}$:

$$
\mathbf{r}_\text{coh}(\mathbf{u}) = \frac{\partial \Phi_\text{coh}(\mathbf{u}, \boldsymbol{\delta}_\text{max})}{\partial \mathbf{u}}
$$

```python
def cohesive_residual(u_flat, delta_max):
    return jax.grad(lambda u: cohesive_energy(u, delta_max))(u_flat)
```

The Newton solver finds $\mathbf{u}$ such that $\mathbf{r}_\text{bulk}(\mathbf{u}) + \mathbf{r}_\text{coh}(\mathbf{u}) = \mathbf{0}$. At each Newton iteration the linear system uses the combined tangent

$$
\mathbf{J}_\text{total}\,\mathbf{v} = \mathbf{J}_\text{bulk}\,\mathbf{v} + \frac{\partial}{\partial\epsilon}\,\mathbf{r}_\text{coh}(\mathbf{u} + \epsilon\mathbf{v})\bigg|_{\epsilon=0},
$$

where the bulk Jacobian $\mathbf{J}_\text{bulk}$ is assembled (and provides a Jacobi preconditioner) and the cohesive tangent is applied matrix-free via `jax.jvp` — at roughly the cost of one cohesive-residual evaluation. The CG solver needs only this combined matrix–vector product.

## Problem Setup

### Material and Geometry

```python
import jax
import jax.numpy as np
import numpy as onp
import feax as fe
from feax.mechanics.cohesive import (
    CohesiveInterface, compute_lumped_area_weights, exponential_potential,
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

Build the solver once with `extra_residual_fn` and `KrylovSolverOptions`. The cohesive history $\boldsymbol{\delta}_\text{max}$ is a quasi-static state variable updated between load steps; flow it into the extra residual through a small mutable holder so the solver need not be rebuilt:

```python
bc0 = make_bc(0.0)
history = {'delta_max': np.zeros(interface.n_nodes)}

solver = fe.create_solver(
    problem, bc0,
    solver_options=fe.KrylovSolverOptions(
        solver='cg', atol=1e-8, maxiter=200,
        use_jacobi_preconditioner=True, verbose=True,
    ),
    newton_options=fe.NewtonOptions(tol=1e-8, max_iter=200),
    extra_residual_fn=lambda u: cohesive_residual(u, history['delta_max']),
    linear=False,
)
EMPTY_IV = fe.TracedParams()  # bulk elasticity carries no internal variables
```

**Key points:**
- `extra_residual_fn` adds the cohesive residual; the bulk residual/Jacobian come from `get_energy_density`
- `KrylovSolverOptions` + `linear=False` selects the hybrid matrix-free Newton–Krylov path
- The bulk Jacobian is assembled (and gives a Jacobi preconditioner); the cohesive tangent is matrix-free
- `verbose=True` prints Newton convergence info

## Quasi-Static Loading

Each load step increments the prescribed displacement. The solver reuses the previous solution as the initial guess, and $\delta_\text{max}$ is updated after convergence:

```python
u_flat = np.zeros(problem.num_total_dofs_all_vars)
delta_max = np.zeros(interface.n_nodes)

for step in range(1, n_steps + 1):
    disp = applied_disp * step / n_steps

    # Apply BC values to initial guess, publish the current history, then solve
    bc = make_bc(disp)
    u_flat = u_flat.at[bc.bc_rows].set(bc.bc_vals)
    history['delta_max'] = delta_max
    u_flat = solver(EMPTY_IV, u_flat, bc=bc)

    # Update irreversibility state
    delta_current = interface.get_opening(u_flat)
    delta_max = np.maximum(delta_max, delta_current)
```

Note: `solver(EMPTY_IV, u_flat, bc=bc)` — the bulk has no internal variables, `u_flat` is the initial guess, and `bc=` supplies the current load step's prescribed values.

## Post-Processing

### Reaction Force via Energy Gradient

The reaction force is the internal force vector — the gradient of the total potential energy (bulk + cohesive). The scalar energies are kept purely for this post-processing:

$$
\mathbf{f}_\text{int} = \nabla_\mathbf{u}\big[\Pi_\text{bulk}(\mathbf{u}) + \Phi_\text{coh}(\mathbf{u}, \boldsymbol{\delta}_\text{max})\big]
$$

```python
def total_energy(u_flat, delta_max):
    return elastic_energy(u_flat) + cohesive_energy(u_flat, delta_max)

fint = jax.grad(total_energy)(u_flat, delta_max)
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
- **Bulk via residual assembly** — `get_energy_density()` lets FEAX assemble the bulk residual and Jacobian (`σ = jax.grad(ψ)`); no hand-written stress
- **Cohesive via `extra_residual_fn`** — the interface contribution $\partial\Phi_\text{coh}/\partial\mathbf{u}$ is added to the global residual and applied matrix-free
- **`CohesiveInterface`** — encodes interface geometry (node pairs, normals, weights) and builds the cohesive energy function
- **Hybrid Newton–Krylov** — assembled bulk Jacobian (with Jacobi preconditioner) + JVP cohesive tangent; CG solves with matvecs only
- **Irreversibility** — $\delta_\text{max}$ tracks the maximum opening; updated between load steps and flowed in through a mutable holder

**Why this split?**
- Cohesive contributions couple arbitrary node pairs — hard to fit into standard FE sparse assembly, so they enter as an extra residual
- The bulk keeps the assembled Jacobian, giving a preconditioner the pure matrix-free path would lack
- The cohesive JVP costs ≈ 1 cohesive-residual evaluation, independent of the sparsity pattern

## Further Reading

- `examples/advance/cohesive_fracture_2d.py` — 2D version with QUAD4 elements
- `examples/advance/cohesive_fracture_3d.py` — Complete 3D working example
- [Solver Guide](../getting-started/solver.md) — `extra_residual_fn` and the hybrid Newton–Krylov path
- [API: feax.mechanics.cohesive](../api/reference/feax/mechanics/cohesive.md) — Cohesive zone models
