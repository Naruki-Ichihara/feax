# Thermomechanical Topology Optimization

This tutorial demonstrates multi-physics topology optimization combining structural stiffness maximisation with thermal insulation, using FEAX's differentiable solvers and the `gene` optimisation toolkit.

The key technique is **coupling two physics through `InternalVars`**: the temperature field solved by the thermal problem is passed as a volume variable to the mechanical problem, where it drives thermal strain.

## Problem Statement

Design a structure that simultaneously:

1. **Maximises stiffness** under tensile loading and cryogenic thermal stress
2. **Minimises thermal conductivity** between hot and cold surfaces

This is motivated by cryogenic tank walls (e.g. liquid hydrogen at 20 K) where structural integrity and thermal insulation must be balanced.

### Boundary Conditions

| | Bottom (z = 0) | Top (z = L) |
|---|---|---|
| **Mechanics** | Fixed (all DOFs) | Surface traction in +z |
| **Thermal** | T = 20 K (cold) | T = 293 K (hot) |

### Objective

$$
\min_\rho \quad w_\text{mech} \frac{C_\text{mech}}{C_\text{mech}^0} + w_\text{therm} \frac{C_\text{therm}}{C_\text{therm}^0}
$$

where $C_\text{mech}$ is the mechanical compliance (work of external forces) and $C_\text{therm} = \int \tfrac{1}{2}\kappa|\nabla T|^2\,dV$ is the thermal compliance. Both are normalised by their reference values at full density.

## Coupling via InternalVars

The central idea is that `InternalVars` carries **any node-based or cell-based field** as extra arguments to the energy density function. This enables one-way (weak) coupling between physics without a monolithic multi-variable formulation.

### Data flow

```
rho -> filter -> Heaviside -> rho_p
                                |
                 +--------------+---------------+
                 |                               |
          Thermal solve                   (passed through)
          iv = (rho_p,)                          |
                 |                               |
              T_field                             |
                 |                               |
                 +-------> Mechanical solve <-----+
                           iv = (rho_p, T_field)
                                |
                            u_field
                                |
                 +--------------+---------------+
                 |                               |
          compliance(u)                 thermal_energy(T, rho_p)
                 |                               |
                 +----> weighted sum <-----------+
                        = objective
```

The gradient `dJ/d_rho` flows through both solver VJPs automatically via JAX's autodiff:

- **Direct path**: `rho_p` affects material stiffness and conductivity
- **Indirect path**: `rho_p -> T_field -> thermal strain -> u_field -> compliance`

### Energy Density with Temperature

The mechanical energy density receives `(u_grad, rho, T)` — the temperature `T` at each quadrature point is interpolated from the node-based thermal solution via shape functions, handled automatically by the assembler:

```python
class MechanicalProblem(fe.problem.Problem):
    def get_energy_density(self):
        def psi(u_grad, rho, T):
            E = E_eps + (E0 - E_eps) * rho ** p_mech
            mu = E / (2. * (1. + nu_val))
            lmbda = E * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
            eps = 0.5 * (u_grad + u_grad.T)
            eps_th = alpha_cte * (T - T_ref) * np.eye(3)
            eps_mech = eps - eps_th
            return 0.5 * lmbda * np.trace(eps_mech)**2 + mu * np.sum(eps_mech * eps_mech)
        return psi
```

The stress is derived automatically as $\sigma = \partial\psi / \partial(\nabla u)$, which correctly yields the thermoelastic constitutive law:

$$
\sigma = \lambda\,\mathrm{tr}(\varepsilon_\text{mech})\,I + 2\mu\,\varepsilon_\text{mech}, \qquad \varepsilon_\text{mech} = \varepsilon - \alpha(T - T_\text{ref})\,I
$$

### Passing the Temperature Field

In the `objective` method, the thermal solution is solved first, then passed to the mechanical solver as a volume variable:

```python
def objective(self, rho, beta=1.0):
    rho_p = self._phys_density(rho, beta)

    # Step 1: solve thermal
    iv_therm = fe.InternalVars(volume_vars=(rho_p,))
    sol_therm = self._solver_therm(iv_therm, self._init_therm)

    # Step 2: solve mechanics — T field passed as second volume variable
    iv_mech = fe.InternalVars(volume_vars=(rho_p, sol_therm))
    sol_mech = self._solver_mech(iv_mech, self._init_mech)

    W_mech = self._compliance_fn(sol_mech) / self._W_mech_0
    W_therm = self._energy_therm(sol_therm, iv_therm) / self._W_therm_0

    return w_mech * W_mech + w_therm * W_therm
```

Because both solvers have custom VJPs registered, `jax.grad` propagates sensitivities through the full chain — including the implicit dependence of `u` on `T` and `T` on `rho`.

## InternalVars: General Pattern for Coupled Problems

The `InternalVars` mechanism is not limited to thermomechanical coupling. The same pattern applies whenever one physics produces a field that another physics consumes:

| Coupling | Thermal var | Mechanical var | Passed via |
|---|---|---|---|
| **Thermo-mechanical** | Temperature T | Thermal strain $\alpha\Delta T$ | `volume_vars=(rho, T)` |
| **Damage-mechanical** | Damage d | Degraded stiffness $(1-d)^2 E$ | `volume_vars=(d,)` |
| **Phase-field fracture** | Phase field $\phi$ | Stress degradation | `volume_vars=(rho, phi)` |

### How interpolation works

Each entry in `volume_vars` is automatically interpolated to quadrature points by the assembler:

- **Node-based** array (shape `(num_nodes,)`): interpolated via shape functions $N_i$, so the quadrature-point value is $\sum_i N_i \cdot v_i$
- **Cell-based** array (shape `(num_cells,)`): broadcast to all quadrature points of each element

The interpolated values are passed as positional arguments to the energy density function, in order:

```python
# volume_vars = (rho, T)
# At each quadrature point, the assembler calls:
psi(u_grad_at_qp, rho_at_qp, T_at_qp)
```

### Solver setup

When a solver is created with `internal_vars` containing multiple volume variables, the Jacobian assembly and VJP correctly handle all of them:

```python
sample_iv = fe.InternalVars(
    volume_vars=(
        fe.InternalVars.create_node_var(problem, 0.5),   # rho
        fe.InternalVars.create_node_var(problem, T_ref),  # T
    ))
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(),
    iter_num=1,
    internal_vars=sample_iv)
```

## Problem Definitions

### Thermal Problem

Standard steady-state heat conduction with SIMP-interpolated conductivity:

```python
class ThermalProblem(fe.problem.Problem):
    def get_energy_density(self):
        def psi(grad_T, rho):
            kappa = kappa_eps + (kappa0 - kappa_eps) * rho ** p_therm
            return 0.5 * kappa * np.sum(grad_T * grad_T)
        return psi
```

### Solver and Filter Setup

```python
class ThermomechOpt(Pipeline):
    def build(self, mesh):
        bottom = lambda pt: np.isclose(pt[2], 0., atol=tol)
        top = lambda pt: np.isclose(pt[2], L, atol=tol)

        prob_mech = MechanicalProblem(
            mesh=mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[top])
        prob_therm = ThermalProblem(
            mesh=mesh, vec=1, dim=3, ele_type='HEX8', location_fns=[])

        bc_mech = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(
                location=bottom, component="all", value=0.),
        ]).create_bc(prob_mech)

        bc_therm = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(
                location=bottom, component="all", value=T_cold),
            fe.DCboundary.DirichletBCSpec(
                location=top, component="all", value=T_hot),
        ]).create_bc(prob_therm)

        solver_opts = fe.DirectSolverOptions()

        # Mechanical solver: 2 volume vars (rho, T)
        sample_iv_mech = fe.InternalVars(volume_vars=(
            fe.InternalVars.create_node_var(prob_mech, 0.5),
            fe.InternalVars.create_node_var(prob_mech, T_ref),
        ))
        self._solver_mech = fe.create_solver(
            prob_mech, bc_mech, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv_mech)

        # Thermal solver: 1 volume var (rho)
        sample_iv_therm = fe.InternalVars(volume_vars=(
            fe.InternalVars.create_node_var(prob_therm, 0.5),
        ))
        self._solver_therm = fe.create_solver(
            prob_therm, bc_therm, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv_therm)

        self._filter_fn = gene.create_density_filter(mesh, radius=filter_radius)
```

### Reference Energy Normalisation

To ensure balanced weighting between objectives of different magnitudes, both energies are normalised by their values at full density ($\rho = 1$):

```python
        rho_ones = np.ones(mesh.points.shape[0])

        iv_therm_ref = fe.InternalVars(volume_vars=(rho_ones,))
        sol_t_ref = self._solver_therm(iv_therm_ref, self._init_therm)

        iv_mech_ref = fe.InternalVars(volume_vars=(rho_ones, sol_t_ref))
        sol_m_ref = self._solver_mech(iv_mech_ref, self._init_mech)

        self._W_mech_0 = abs(float(self._compliance_fn(sol_m_ref)))
        self._W_therm_0 = abs(float(self._energy_therm(sol_t_ref, iv_therm_ref)))
```

## Running the Optimisation

```python
mesh = fe.mesh.box_mesh(size=L, mesh_size=L / mesh_n, element_type='HEX8')

result = run(
    pipeline=ThermomechOpt(),
    mesh=mesh,
    max_iter=300,
    continuations={
        "beta": Continuation(initial=1.0, final=8.0, update_every=50,
                             multiply_by=1.0, add=1.0),
    },
    output_dir="output_thermomech_opt",
    save_every=5,
)
```

## Summary

| Aspect | Detail |
|---|---|
| **Coupling mechanism** | `InternalVars.volume_vars` carries fields between solvers |
| **Differentiability** | Full gradient through both solver VJPs via `jax.grad` |
| **Energy density** | `psi(u_grad, rho, T)` — extra args auto-interpolated to quad points |
| **Thermal stress** | $\varepsilon_\text{th} = \alpha(T - T_\text{ref})I$ from solved T field |
| **Objectives** | Normalised compliance + thermal compliance, weighted sum |
| **Volume constraint** | Equality via two inequality constraints (`type='eq'`) |

Full source: [`examples/advance/topology_optimization_unitcell_thermomechanical.py`](https://github.com/your-repo/feax/blob/main/examples/advance/topology_optimization_unitcell_thermomechanical.py)
