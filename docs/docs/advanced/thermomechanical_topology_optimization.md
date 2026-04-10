# Thermomechanical Topology Optimization

This tutorial demonstrates multi-physics topology optimization combining structural stiffness maximisation with thermal insulation, using FEAX's differentiable solvers and the `gene` optimisation toolkit.

The key techniques are:

1. **Coupling two physics through `InternalVars`**: the temperature field solved by the thermal problem is passed as a volume variable to the mechanical problem, driving thermal strain
2. **Post-processing with `gene.extract_surface` / `gene.extract_volume_mesh`**: extracting smooth surface and volume meshes from the optimised density field for re-analysis

## Problem Statement

Design a structure separating a cryogenic fluid (liquid hydrogen, 20 K) from a room-temperature environment (293 K) that simultaneously:

1. **Maximises stiffness** under tensile loading and cryogenic thermal stress
2. **Minimises thermal conductivity** between hot and cold surfaces

### Boundary Conditions

| | Bottom (z = 0) | Top (z = L) |
|---|---|---|
| **Mechanics** | z-fixed, free in-plane; corner pins for rigid body | Tensile traction in +z |
| **Thermal** | T = 20 K (LH2) | T = 293 K (room) |

The bottom face is fixed only in z, allowing free in-plane thermal expansion/contraction. Two corner nodes are pinned to prevent rigid body translation and rotation:

- Origin (0, 0, 0): x and y fixed
- Corner (L, 0, 0): y fixed

### Objective

$$
\min_\rho \quad w_\text{mech} \frac{C_\text{mech}}{C_\text{mech}^0} + w_\text{therm} \frac{C_\text{therm}}{C_\text{therm}^0}
$$

where $C_\text{mech}$ is the mechanical compliance (external work) and $C_\text{therm} = \int \tfrac{1}{2}\kappa|\nabla T|^2\,dV$ is the thermal compliance. Both are normalised by their values at full density ($\rho = 1$).

## Coupling via InternalVars

`InternalVars` carries **any node-based or cell-based field** as extra arguments to the energy density function. This enables one-way (weak) coupling between physics without a monolithic formulation.

### Data flow

```
rho -> filter -> Heaviside -> rho_p
                                |
                 +--------------+---------------+
                 |                               |
          Thermal solve                   (passed through)
          iv = (rho_p,)                          |
                 |                               |
              T_field                            |
                 |                               |
                 +-------> Mechanical solve <----+
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

The gradient `dJ/d_rho` flows through both solver VJPs automatically via JAX's autodiff, including the indirect path `rho_p -> T_field -> thermal strain -> u_field -> compliance`.

## Problem Definitions

### Mechanical Problem with Temperature-Dependent Thermal Strain

The mechanical energy density receives `(u_grad, rho, T)`. The temperature `T` at each quadrature point is interpolated from the node-based thermal solution via shape functions, handled automatically by the assembler:

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

    def get_surface_maps(self):
        return [lambda u, x, *a: np.array([0., 0., traction_z])]
```

The stress is derived automatically as $\sigma = \partial\psi / \partial(\nabla u)$, yielding the thermoelastic constitutive law:

$$
\sigma = \lambda\,\mathrm{tr}(\varepsilon_\text{mech})\,I + 2\mu\,\varepsilon_\text{mech}, \qquad \varepsilon_\text{mech} = \varepsilon - \alpha(T - T_\text{ref})\,I
$$

### Thermal Problem

```python
class ThermalProblem(fe.problem.Problem):
    def get_energy_density(self):
        def psi(grad_T, rho):
            kappa = kappa_eps + (kappa0 - kappa_eps) * rho ** p_therm
            return 0.5 * kappa * np.sum(grad_T * grad_T)
        return psi
```

## Pipeline Setup

### Boundary Conditions

```python
bottom = lambda pt: np.isclose(pt[2], 0., atol=tol)
top = lambda pt: np.isclose(pt[2], L, atol=tol)

# Rigid body pins
origin = lambda pt: (np.isclose(pt[0], 0., atol=tol)
                     & np.isclose(pt[1], 0., atol=tol)
                     & np.isclose(pt[2], 0., atol=tol))
corner_x = lambda pt: (np.isclose(pt[0], L, atol=tol)
                       & np.isclose(pt[1], 0., atol=tol)
                       & np.isclose(pt[2], 0., atol=tol))

bc_mech = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=bottom, component="z", value=0.),
    fe.DCboundary.DirichletBCSpec(location=origin, component="x", value=0.),
    fe.DCboundary.DirichletBCSpec(location=origin, component="y", value=0.),
    fe.DCboundary.DirichletBCSpec(location=corner_x, component="y", value=0.),
]).create_bc(prob_mech)

bc_therm = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=bottom, component="all", value=T_cold),
    fe.DCboundary.DirichletBCSpec(location=top, component="all", value=T_hot),
]).create_bc(prob_therm)
```

### Solver Setup with Multiple InternalVars

```python
# Mechanical: volume_vars = (rho, T)
sample_iv_mech = fe.InternalVars(volume_vars=(
    fe.InternalVars.create_node_var(prob_mech, 0.5),
    fe.InternalVars.create_node_var(prob_mech, T_ref),
))
self._solver_mech = fe.create_solver(
    prob_mech, bc_mech, solver_options=fe.DirectSolverOptions(),
    adjoint_solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=sample_iv_mech)

# Thermal: volume_vars = (rho,)
sample_iv_therm = fe.InternalVars(volume_vars=(
    fe.InternalVars.create_node_var(prob_therm, 0.5),
))
self._solver_therm = fe.create_solver(
    prob_therm, bc_therm, solver_options=fe.DirectSolverOptions(),
    adjoint_solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=sample_iv_therm)
```

### Objective Function

```python
def objective(self, rho, beta=1.0):
    rho_p = self._phys_density(rho, beta)

    # Step 1: solve thermal
    iv_therm = fe.InternalVars(volume_vars=(rho_p,))
    sol_therm = self._solver_therm(iv_therm, self._init_therm)

    # Step 2: solve mechanics — T field as second volume variable
    iv_mech = fe.InternalVars(volume_vars=(rho_p, sol_therm))
    sol_mech = self._solver_mech(iv_mech, self._init_mech)

    W_mech = self._compliance_fn(sol_mech) / self._W_mech_0
    W_therm = self._energy_therm(sol_therm, iv_therm) / self._W_therm_0

    return w_mech * W_mech + w_therm * W_therm
```

### Volume Constraint

An equality volume constraint is implemented via two inequality constraints:

```python
@constraint(target=vol_frac, type='eq', tol=0.001)
def volume(self, rho, beta=1.0):
    return self._volume_fn(self._phys_density(rho, beta))
```

This expands internally to `volume <= target + tol` and `volume >= target - tol`, compatible with NLopt MMA.

## Running the Optimisation

```python
mesh = fe.mesh.box_mesh(size=L, mesh_size=L / mesh_n, element_type='HEX8')

result = run(
    pipeline=ThermomechOpt(),
    mesh=mesh,
    rho_init=onp.full(mesh.points.shape[0], vol_frac),
    max_iter=300,
    continuations={
        "beta": Continuation(initial=1.0, final=4.0, update_every=50, step=1.0),
    },
    output_dir="output_thermomech_opt",
    save_every=5,
)
```

### Continuation

The `Continuation` dataclass controls parameter schedules during optimisation:

```python
Continuation(initial=1.0, final=4.0, update_every=50, step=1.0)
```

| Parameter | Description |
|---|---|
| `initial` | Starting value |
| `final` | Clamped upper (or lower) bound |
| `update_every` | Iterations between updates |
| `step` | Additive increment per update: `value = initial + step * n` |

The value at iteration `i` is `initial + step * (i // update_every)`, clamped to `[initial, final]`.

## Post-Processing

### Surface Mesh Extraction

`gene.extract_surface` uses VTK's contour filter on the unstructured FE mesh for smooth iso-surfaces:

```python
surface = gene.extract_surface(result.rho_filtered, result.mesh, threshold=0.5)
surface.save("optimized.stl")
```

The result is a closed, manifold triangle mesh.

### Volume Remeshing and Re-Analysis

```python
remesh = gene.extract_volume_mesh(
    result.rho_filtered, result.mesh, threshold=0.5, mesh_size=L / mesh_n)
```

The remeshed geometry is re-solved with full density to compute stress, strain, and temperature. Note that after remeshing, the original corner nodes may not exist, so pin nodes are found dynamically:

```python
import numpy as _onp
pts_r = _onp.asarray(remesh.points)
bottom_idx = _onp.where(_onp.isclose(pts_r[:, 2], 0., atol=tol))[0]
pin1_idx = bottom_idx[_onp.argmin(pts_r[bottom_idx, 0])]  # min-x on bottom
pin2_idx = bottom_idx[_onp.argmax(pts_r[bottom_idx, 0])]  # max-x on bottom
```

Results (displacement, temperature, von Mises stress, strain tensors) are saved to VTU for visualisation in ParaView.

## InternalVars: General Pattern for Coupled Problems

The same pattern applies whenever one physics produces a field that another consumes:

| Coupling | Variable passed | Energy density signature | `volume_vars` |
|---|---|---|---|
| **Thermo-mechanical** | Temperature T | `psi(u_grad, rho, T)` | `(rho, T)` |
| **Damage-mechanical** | Damage d | `psi(u_grad, d)` | `(d,)` |
| **Phase-field fracture** | Phase field $\phi$ | `psi(u_grad, rho, phi)` | `(rho, phi)` |

Each entry in `volume_vars` is automatically interpolated to quadrature points:

- **Node-based** `(num_nodes,)`: interpolated via shape functions
- **Cell-based** `(num_cells,)`: broadcast to all quadrature points

## Summary

| Aspect | Detail |
|---|---|
| **Coupling** | `InternalVars.volume_vars` carries fields between solvers |
| **Differentiability** | Full gradient through both solver VJPs via `jax.grad` |
| **Energy density** | `psi(u_grad, rho, T)` — extra args auto-interpolated to quad points |
| **Thermal stress** | $\varepsilon_\text{th} = \alpha(T - T_\text{ref})I$ from solved T field |
| **Boundary conditions** | z-fixed bottom with free in-plane expansion; corner pins for rigid body |
| **Volume constraint** | Equality via two inequality constraints (`type='eq'`) |
| **Continuation** | `Continuation(initial, final, update_every, step)` — additive schedule |
| **Mesh extraction** | `gene.extract_surface` (PyVista/VTK contour) for smooth iso-surfaces |
| **Re-analysis** | Gmsh volume remesh + full-density FE solve with dynamic pin nodes |

Full source: [`examples/advance/topology_optimization_thermomechanical.py`](https://github.com/Naruki-Ichihara/feax/blob/main/examples/advance/topology_optimization_thermomechanical.py)
