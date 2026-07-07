# Thermo-Mechanical Coupling on a Lattice

This tutorial demonstrates a one-way coupled (staggered) thermo-mechanical forward analysis on a BCC lattice structure. A thermal conduction problem is solved first; the resulting temperature field then drives thermal expansion in a mechanical problem. The two solves are chained with `TracedParams.node_var_from_solution` — the standard feax bridge for staggered multi-physics — and the whole chain stays differentiable end to end.

The full script is [`examples/basic/thermo_mech_coupling.py`](https://github.com/Naruki-Ichihara/feax/blob/main/examples/basic/thermo_mech_coupling.py).

## Problem Description

A row of three BCC unit cells $\Omega = [0, 3] \times [0, 1] \times [0, 1]$ is embedded in a background hex mesh as a nodal density field $\rho \in \{\rho_\text{void}, 1\}$. The staggered analysis solves

$$
\nabla \cdot \big( k(\rho) \nabla T \big) = 0
\qquad \text{then} \qquad
\nabla \cdot \boldsymbol{\sigma}(\mathbf{u}, \rho, T) = \mathbf{0}
$$

with a hot left face ($T = 100$), a cold right face ($T = 0$), and the left face clamped. The mechanical strain carries the thermal expansion of the solved temperature field:

$$
\boldsymbol{\varepsilon} = \tfrac{1}{2}\big(\nabla \mathbf{u} + \nabla \mathbf{u}^T\big) - \alpha \,(T - T_\text{ref})\, \mathbf{I},
\qquad
\boldsymbol{\sigma} = \lambda\, \text{tr}(\boldsymbol{\varepsilon})\, \mathbf{I} + 2\mu\, \boldsymbol{\varepsilon}
$$

Material interpolation follows the density: $k(\rho) = k_0 \rho$ for conduction and a SIMP-style $E(\rho) = E_0 \rho^3$ for stiffness, so the void carries neither heat nor load.

## Lattice as a Nodal Density Field

The BCC lattice is defined as a graph (strut network) and rasterized onto the mesh nodes with the `flat` toolkit — no lattice-conforming mesh is needed:

```python
import numpy as onp
import jax.numpy as np
import feax as fe
import feax.flat as flat

N_CELLS, L, RADIUS, RHO_VOID = 3, 1.0, 0.15, 1e-2

mesh = fe.mesh.box_mesh((N_CELLS * L, L, L), mesh_size=0.1)

# BCC graph: cell corners + one center per cell, struts corner <-> center
corners = onp.array([[i, j, k] for i in range(N_CELLS + 1)
                     for j in (0.0, L) for k in (0.0, L)])
centers = onp.array([[i + 0.5, 0.5 * L, 0.5 * L] for i in range(N_CELLS)])
nodes = np.asarray(onp.vstack([corners, centers]))
corner_id = {(i, j, k): 4 * i + 2 * j + k
             for i in range(N_CELLS + 1) for j in (0, 1) for k in (0, 1)}
edges = np.asarray([[corner_id[(i + di, j, k)], len(corners) + i]
                    for i in range(N_CELLS)
                    for di in (0, 1) for j in (0, 1) for k in (0, 1)])

lattice_fn = flat.graph.create_lattice_function(nodes, edges, radius=RADIUS)
```

`create_lattice_density_field_nodal` evaluates the graph's signed distance at every mesh node and returns a `(num_nodes,)` density array — a nodal `TracedParams` variable that the quadrature-point material maps receive interpolated:

```python
rho = flat.graph.create_lattice_density_field_nodal(
    thermal, lattice_fn, density_solid=1.0, density_void=RHO_VOID)
```

:::note
The void floor `RHO_VOID = 1e-2` (with `PENAL = 3` giving a $10^{-6}$ stiffness contrast) keeps the operators well-conditioned. Pairing the CG solver with `use_jacobi_preconditioner=True` handles the resulting diagonal scaling.
:::

## Step 1: Thermal Conduction

The thermal problem is a scalar (`vec=1`) Poisson problem whose flux map takes the density as an extra argument:

```python
class Thermal(fe.Problem):
    def get_tensor_map(self):
        def flux(grad_T, rho):
            return K0 * rho * grad_T
        return flux

thermal = Thermal(mesh, vec=1, dim=3, ele_type="HEX8")
bc_th = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], 0.0), "all", T_HOT),
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], N_CELLS * L), "all", T_COLD),
]).create_bc(thermal)

solve_T = fe.create_solver(
    thermal, bc_th,
    solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12,
                                          use_jacobi_preconditioner=True),
    linear=True)
sol_T = solve_T(fe.TracedParams(volume_vars=(rho,)),
                fe.zero_like_initial_guess(thermal, bc_th))
```

## The Bridge: Solution → Nodal Variable

`solve_T` returns a [`Solution`](../api/reference/feax/solution.md). `TracedParams.node_var_from_solution` converts it into a `(num_nodes,)` nodal variable for the next problem:

```python
T_nodes = fe.TracedParams.node_var_from_solution(thermal, sol_T)
```

This is the staggered-coupling idiom in feax: any solved field can ride along as an input variable of a downstream problem, and because the conversion is a pure reshape, `jax.grad` flows through the whole chain (e.g. for coupled design optimization).

## Step 2: Thermo-Elasticity

The mechanical stress map receives *two* nodal variables — the density and the solved temperature — in the order they appear in `volume_vars`:

```python
class ThermoElastic(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho, T):
            E = E0 * rho ** PENAL
            mu = E / (2.0 * (1.0 + NU))
            lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
            eps = 0.5 * (u_grad + u_grad.T) - ALPHA * (T - T_REF) * np.eye(3)
            return lam * np.trace(eps) * np.eye(3) + 2.0 * mu * eps
        return stress

mech = ThermoElastic(mesh, vec=3, dim=3, ele_type="HEX8")
bc_me = fe.DirichletBCConfig([
    fe.DirichletBCSpec(lambda p: np.isclose(p[0], 0.0), "all", 0.0),
]).create_bc(mech)

tp_me = fe.TracedParams(volume_vars=(rho, T_nodes))   # the coupling
sol_u = fe.create_solver(
    mech, bc_me,
    solver_options=fe.KrylovSolverOptions(solver="cg", tol=1e-10, atol=1e-12,
                                          use_jacobi_preconditioner=True),
    linear=True)(tp_me, fe.zero_like_initial_guess(mech, bc_me))
```

## Post-Processing

`Solution.field(0)` reshapes the flat DOF vector to `(num_nodes, 3)` for output:

```python
u = onp.asarray(sol_u.field(0))
fe.utils.save_sol(mesh, "thermo_mech_coupling.vtu", point_infos=[
    ("density", onp.asarray(rho)),
    ("temperature", onp.asarray(T_nodes)),
    ("displacement", u),
])
```

Running the script prints:

```
Mesh: 3751 nodes, 3000 HEX8 cells
Lattice volume fraction: 0.329
Thermal solve: T in [0.00, 100.00]
Free-end mean axial expansion : 1.8193e-02
Max |u| (solid struts)        : 1.8395e-02
```

The linear temperature profile gives a mean thermal strain of $\alpha \, \bar{T} = 10^{-4} \times 50 = 5 \times 10^{-3}$, i.e. an analytic free-expansion estimate of $1.5 \times 10^{-2}$ over the length-3 bar — the computed $1.82 \times 10^{-2}$ is of exactly this order, with the excess coming from the lattice's compliance and Poisson effects.

## Where to Go Next

- Wrap both solves in a function of `rho` and take `jax.grad` — the chain is differentiable, which is the basis of the coupled design workflows in the advanced tutorials.
- For transient thermal loading, drive the same pair of problems with the [`ImplicitPipeline` time-stepping interface](../getting-started/time_solver.md).
- For fully periodic lattice unit cells, see [Lattice Structure Homogenization](../advanced/lattice_homogenization.md).
