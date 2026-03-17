---
sidebar_label: Cahn-Hilliard
---

# Cahn-Hilliard Phase Separation

This tutorial demonstrates transient phase separation using the **Cahn-Hilliard equation** with a mixed finite element formulation. We reproduce the classic FEniCS/DOLFINx Cahn-Hilliard demo using FEAX's time-stepping infrastructure (`ImplicitPipeline`).

## Overview

The Cahn-Hilliard equation is a fourth-order PDE that models spinodal decomposition — the spontaneous separation of a binary mixture into two phases. Starting from a near-uniform concentration with small random perturbations, the system evolves into distinct phase-separated domains.

Key features of this example:

1. **Mixed formulation** with two scalar fields $(c, \mu)$ to avoid $C^1$ elements
2. **Backward Euler** time integration via `ImplicitPipeline`
3. **`get_weak_form`** interface for coupled multi-field problems
4. **Natural (zero-flux) boundary conditions** — no Dirichlet BCs needed

## Mathematical Formulation

### Governing Equations

The Cahn-Hilliard equation in operator-split form consists of two coupled second-order equations:

$$
\frac{\partial c}{\partial t} - \nabla \cdot (M \nabla \mu) = 0
$$

$$
\mu - \frac{df}{dc} + \lambda \nabla^2 c = 0
$$

where $c$ is the concentration, $\mu$ is the chemical potential, $M = 1$ is the mobility, and $\lambda = 10^{-2}$ is the surface tension parameter controlling the interface width ($\sim \sqrt{\lambda}$).

### Free Energy

The double-well free energy density and its derivative are:

$$
f(c) = 100\, c^2 (1 - c)^2, \qquad \frac{df}{dc} = 200\, c(1-c)(1-2c)
$$

The two minima at $c = 0$ and $c = 1$ represent the two equilibrium phases.

### Weak Form

Multiplying by test functions $q$ (for $c$) and $v$ (for $\mu$) and integrating by parts:

**Equation 1** (concentration evolution):

$$
\int_\Omega \frac{c - c_\text{old}}{\Delta t}\, q\, d\Omega + \int_\Omega M \nabla\mu \cdot \nabla q\, d\Omega = 0
$$

**Equation 2** (chemical potential):

$$
\int_\Omega \left(\mu - \frac{df}{dc}\right) v\, d\Omega - \int_\Omega \lambda \nabla c \cdot \nabla v\, d\Omega = 0
$$

## Implementation

### Problem Definition

The `CahnHilliard` problem uses the `get_weak_form` interface, which returns mass (volume) and gradient contributions for each field at each quadrature point:

```python
class CahnHilliard(fe.problem.Problem):

    def get_weak_form(self):
        def weak_form(vals, grads, x, c_old):
            c, mu = vals[0], vals[1]
            grad_c, grad_mu = grads[0], grads[1]

            # df/dc = 200 c (1 - c)(1 - 2c)
            dfdc = 200.0 * c * (1.0 - c) * (1.0 - 2.0 * c)

            # Eq 1: ∂c/∂t = ∇·(M ∇μ)
            R_c_mass = (c - c_old) / dt
            R_c_grad = M * grad_mu

            # Eq 2: μ = df/dc - λ ∇²c
            R_mu_mass = mu - dfdc
            R_mu_grad = -lmbda * grad_c

            return ([R_c_mass, R_mu_mass],
                    [R_c_grad, R_mu_grad])

        return weak_form
```

Key points:

- **`vals`**: list of field values at the quadrature point — `vals[0]` is $c$, `vals[1]` is $\mu$
- **`grads`**: list of field gradients — `grads[0]` is $\nabla c$, `grads[1]` is $\nabla \mu$
- **`c_old`**: concentration from the previous time step, passed through `InternalVars`
- The return format is `([mass_terms], [grad_terms])` — FEAX assembles $\int R_\text{mass} \cdot v\, d\Omega + \int R_\text{grad} : \nabla v\, d\Omega$ automatically

### Mixed Formulation Setup

The problem uses two scalar fields on the same mesh — each with `vec=1`:

```python
problem = CahnHilliard(
    mesh=[mesh, mesh],
    vec=[1, 1],
    dim=2,
    ele_type=['QUAD4', 'QUAD4'],
)
```

Passing lists for `mesh`, `vec`, and `ele_type` tells FEAX to create a mixed problem with two finite element spaces that share the same mesh.

### Time-Stepping with `ImplicitPipeline`

The `ImplicitPipeline` provides a structured interface for implicit time integration. Each time step:

1. `update_vars()` packs the previous solution into `InternalVars`
2. `step()` calls `self.solver(iv, state)` to solve the nonlinear system

```python
class CahnHilliardPipeline(ImplicitPipeline):

    def build(self, mesh):
        self.mesh = mesh
        num_nodes = mesh.points.shape[0]

        self.problem = CahnHilliard(
            mesh=[mesh, mesh], vec=[1, 1], dim=2,
            ele_type=['QUAD4', 'QUAD4'],
        )

        # No Dirichlet BCs (natural zero-flux boundaries)
        bc = fe.DirichletBCConfig([]).create_bc(self.problem)

        # Random initial concentration near c = 0.63
        self._c0 = 0.63 + 0.02 * (
            0.5 - random.uniform(random.PRNGKey(42), shape=(num_nodes, 1)))

        solver_opts = fe.IterativeSolverOptions(
            solver='auto', tol=1e-10, atol=1e-10,
            maxiter=10000, use_jacobi_preconditioner=True,
        )
        newton_opts = fe.NewtonOptions(
            tol=1e-6, rel_tol=1e-8, max_iter=25, internal_jit=True,
        )
        self.solver = fe.create_solver(
            self.problem, bc,
            solver_options=solver_opts,
            newton_options=newton_opts,
            internal_vars=fe.InternalVars(volume_vars=(self._c0[:, 0],)),
        )

    def initial_state(self):
        mu0 = np.zeros((self.mesh.points.shape[0], 1))
        return jax.flatten_util.ravel_pytree([self._c0, mu0])[0]

    def update_vars(self, state, t, dt_val):
        sol_list = self.problem.unflatten_fn_sol_list(state)
        c_old = sol_list[0][:, 0]   # (num_nodes,)
        return fe.InternalVars(volume_vars=(c_old,))

    def save(self, state, step, t, output_dir):
        sol_list = self.problem.unflatten_fn_sol_list(state)
        vtk_path = os.path.join(output_dir, f'ch_{step:04d}.vtu')
        fe.utils.save_sol(
            mesh=self.mesh, sol_file=vtk_path,
            point_infos=[("c", sol_list[0]), ("mu", sol_list[1])],
        )

    def monitor(self, state, step, t):
        c = self.problem.unflatten_fn_sol_list(state)[0]
        return {'c_min': float(c.min()), 'c_max': float(c.max())}
```

### Pipeline Methods

| Method | Purpose |
|---|---|
| `build(mesh)` | Create problem, BCs, solver; called once before the time loop |
| `initial_state()` | Return the initial flat solution vector (random $c$ near 0.63, $\mu = 0$) |
| `update_vars(state, t, dt)` | Extract $c_\text{old}$ from the current state and wrap it in `InternalVars` |
| `save(state, step, t, output_dir)` | Write VTK files with $c$ and $\mu$ fields |
| `monitor(state, step, t)` | Return scalar diagnostics ($c_\text{min}$, $c_\text{max}$) for logging |

### Running the Simulation

The `run()` function drives the time loop:

```python
mesh = fe.mesh.rectangle_mesh(Nx=96, Ny=96, domain_x=1.0, domain_y=1.0)

result = run(
    CahnHilliardPipeline(),
    mesh,
    TimeConfig(dt=5e-6, t_end=50 * 5e-6, save_every=10, print_every=1),
    output_dir='data/vtk',
)
```

`TimeConfig` controls:

| Parameter | Value | Meaning |
|---|---|---|
| `dt` | $5 \times 10^{-6}$ | Time step size |
| `t_end` | $2.5 \times 10^{-4}$ | Final simulation time (50 steps) |
| `save_every` | 10 | VTK output interval |
| `print_every` | 1 | Console log interval |

## Solver Configuration

### Why Iterative Solver

The Cahn-Hilliard Jacobian is a $2 \times 2$ block system (coupling $c$ and $\mu$). It is indefinite and non-symmetric, so CG is not applicable. The auto-selection picks `bicgstab` or `gmres` based on matrix property detection:

```python
solver_opts = fe.IterativeSolverOptions(
    solver='auto',
    tol=1e-10, atol=1e-10,
    maxiter=10000,
    use_jacobi_preconditioner=True,
)
```

Jacobi preconditioning (`use_jacobi_preconditioner=True`) is important for convergence, as the diagonal scaling differs significantly between the $c$ and $\mu$ blocks.

### Newton Solver

The nonlinear system arising from backward Euler + the nonlinear $df/dc$ term is solved with adaptive Newton:

```python
newton_opts = fe.NewtonOptions(
    tol=1e-6, rel_tol=1e-8, max_iter=25, internal_jit=True,
)
```

`internal_jit=True` compiles each Newton component (residual, Jacobian, linear solve) individually, avoiding monolithic compilation overhead.

## Running the Example

```bash
python examples/advance/cahn_hilliard.py
```

VTK files are saved to `examples/advance/data/vtk/` for visualization in ParaView. The concentration field $c$ evolves from a near-uniform initial state ($c \approx 0.63$) into phase-separated domains.

## Notes

- The FEniCS demo uses a $\theta$-method (Crank-Nicolson, $\theta = 0.5$) on both diffusion and reaction terms. This example uses backward Euler ($\theta = 1$), which is unconditionally stable and gives qualitatively identical spinodal decomposition behaviour.
- Zero-flux (Neumann) boundary conditions are enforced naturally — no Dirichlet BCs are specified (`DirichletBCConfig([])`).
- The initial condition uses a fixed random seed (`PRNGKey(42)`) for reproducibility.
