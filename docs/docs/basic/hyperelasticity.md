# Nonlinear Hyperelasticity Problem

This tutorial demonstrates solving nonlinear hyperelasticity problems using FEAX. We consider a beam subjected to a torsional surface traction using a Neo-Hookean material model, showcasing Newton's method for nonlinear finite element analysis.

## Problem Description

Consider a beam $\Omega = [0, L_x] \times [0, L_y] \times [0, L_z]$ with the left face clamped and a torsional traction applied to the right face. The governing equation is:

$$
-\nabla \cdot \mathbf{P}(\mathbf{u}) = \mathbf{0} \quad \text{in } \Omega
$$

where $\mathbf{P}$ is the first Piola-Kirchhoff stress tensor. Unlike linear elasticity, the stress depends nonlinearly on the deformation gradient $\mathbf{F} = \nabla \mathbf{u} + \mathbf{I}$.

### Neo-Hookean Material Model

For hyperelastic materials, the stress derives from a strain energy density function $\psi(\mathbf{F})$:

$$
\mathbf{P} = \frac{\partial \psi}{\partial \mathbf{F}}
$$

The compressible Neo-Hookean model uses:

$$
\psi(\mathbf{F}) = \frac{\mu}{2}\left(J^{-2/3} I_1 - 3\right) + \frac{\kappa}{2}(J - 1)^2
$$

where $I_1 = \text{tr}(\mathbf{F}^T \mathbf{F})$, $J = \det(\mathbf{F})$, $\mu = E / 2(1+\nu)$, and $\kappa = E / 3(1-2\nu)$.

## Setup

```python
import feax as fe
import jax
import jax.numpy as np
import os

# Box geometry
Lx, Ly, Lz = 5., 1., 1.
mesh_size   = 0.1

# Cross-section centroid of the right face (used in torsional traction)
y_c = Ly / 2.
z_c = Lz / 2.

# Torsional traction magnitude
T = 20.
```

## Problem Definition: Energy-Based Approach

FEAX leverages JAX's automatic differentiation to compute stress from energy. Define the energy function and let JAX compute $\mathbf{P} = \partial \psi / \partial \mathbf{F}$.

The torsional traction on the right face is tangential in the yz-plane, creating a torque about the x-axis:

$$
t_y = -T(z - z_c), \quad t_z = T(y - y_c)
$$

```python
class HyperElasticityFeax(fe.problem.Problem):
    def get_tensor_map(self):
        def psi(F):
            E = 100.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)
        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress

    def get_surface_maps(self):
        def traction_map(u_grad, surface_quad_point, traction_magnitude):
            # Torsional traction about x-axis: tangential in yz-plane
            y = surface_quad_point[1]
            z = surface_quad_point[2]
            return np.array([0., -traction_magnitude * (z - z_c), traction_magnitude * (y - y_c)])
        return [traction_map]
```

`jax.grad(psi)` computes the exact stress tensor without manual derivation. FEAX then assembles the tangent stiffness automatically via `jax.jacobian`. The traction function uses `surface_quad_point` to compute the position-dependent torsional load, with the centroid `y_c`, `z_c` captured from the outer scope.

## Mesh and Boundary Conditions

```python
mesh = fe.mesh.box_mesh((Lx, Ly, Lz), mesh_size=mesh_size)

def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

# Fix left face (clamped)
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='all', value=0.)
])
```

## Solver

```python
feax_problem = HyperElasticityFeax(mesh, vec=3, dim=3, location_fns=[right])

traction_surface = fe.InternalVars.create_uniform_surface_var(feax_problem, T)
internal_vars = fe.InternalVars(
    volume_vars=[],
    surface_vars=[(traction_surface,)]
)

bc = bc_config.create_bc(feax_problem)

solver_options = fe.DirectSolverOptions()
solver = fe.create_solver(feax_problem, bc, solver_options, internal_vars=internal_vars)
```

Omitting `iter_num=1` enables Newton's method for the nonlinear problem.

## Solving

```python
def solve_fn(iv):
    sol = solver(iv, fe.zero_like_initial_guess(feax_problem, bc))
    return sol

sol = solve_fn(internal_vars)
sol_unflat = feax_problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]
```

## Visualization

```python
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_hyper_elast.vtu')

fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])
```

## Key Takeaways

1. **Energy-based formulation** with `jax.grad` eliminates manual stress derivation
2. **Automatic tangent stiffness** via JAX's automatic differentiation
3. **Newton's method** for nonlinear problems — omit `iter_num=1`
4. **Position-dependent traction** via `surface_quad_point` in `get_surface_maps`
5. **Geometry defined at top** — `y_c`, `z_c` computed from `Ly`, `Lz` and captured by the traction closure

## Further Reading

- [Linear Elasticity](./linear_elasticity.md) — linear problems with `iter_num=1`
- [JIT Transform](./jit_transform.md) — accelerate Newton iterations with `jax.jit`
- [Vectorization Transform](./vmap_transform.md) — batch parameter studies with `jax.vmap`
