---
sidebar_label: Third Medium Contact
---

# Third Medium Contact

This tutorial demonstrates frictionless contact simulation using the **third-medium method** with HuHu-LuLu Hessian-based regularization. We reproduce the FElupe ex20 benchmark: two elastic bodies approaching each other through a compliant background medium on a single mesh.

## Overview

The third-medium method avoids explicit contact detection by filling the gap between bodies with a soft artificial medium. When bodies approach each other, this medium compresses and transmits contact forces naturally through the variational formulation.

Key ingredients:

1. **Single mesh** with two material regions: stiff body and soft medium
2. **Neo-Hookean** compressible hyperelasticity for both regions, with the medium scaled down by $\gamma_0 \approx 10^{-6}$
3. **Biharmonic regularization** (HuHu-LuLu) on the medium to prevent mesh distortion
4. **Incremental loading** with non-symmetric BC elimination (`symmetric_bc=False`)

### References

- G. L. Bluhm et al., "Internal contact modeling for finite strain topology optimization", *Comput. Mech.* 67, 1099–1114 (2021).
- A. H. Frederiksen et al., "Topology optimization of self-contacting structures", *Comput. Mech.* 73, 967–981 (2023).

## Problem Setup

### Geometry and Mesh

A structured QUAD9 mesh covers the domain $[0, 1.1] \times [0, 0.5]$:

```python
mesh = fe.mesh.rectangle_mesh(
    Nx=11, Ny=5,
    domain_x=1.1, domain_y=0.5,
    ele_type='QUAD9',
)
```

Cells are classified as **body** (solid) or **medium** (background) based on their spatial location:

```python
is_medium = np.zeros(Nx_total * Ny_total)
for ix in range(Nx_total):
    for iy in range(Ny_total):
        cell_idx = ix * Ny_total + iy
        # Interior medium
        if 1 <= ix <= 9 and 1 <= iy <= 3:
            is_medium = is_medium.at[cell_idx].set(1.0)
        # Right strip medium
        if ix == 10:
            is_medium = is_medium.at[cell_idx].set(1.0)
```

### Material Parameters

Both regions use the same Neo-Hookean model but with different stiffness:

| Region | Shear modulus $\mu$ | Bulk modulus $\lambda$ |
|---|---|---|
| Body | $G = 5/14$ | $K = 5/3$ |
| Medium | $\gamma_0 G$ | $\gamma_0 K$ |

where $\gamma_0 = 5 \times 10^{-6}$ is the medium scaling factor. Cell-based material properties are assembled as:

```python
mu_cell = np.where(is_medium, G * gamma0, G)
lmbda_cell = np.where(is_medium, K * gamma0, K)
```

## Energy Formulation

### Neo-Hookean Energy Density

The compressible Neo-Hookean energy density in plane strain is:

$$
\psi(\mathbf{F}) = \frac{\mu}{2}(\text{tr}\,\mathbf{C} + 1) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2
$$

where $\mathbf{F} = \mathbf{I} + \nabla\mathbf{u}$, $\mathbf{C} = \mathbf{F}^T\mathbf{F}$, $J = \det\mathbf{F}$. The "+1" accounts for the plane-strain $F_{33} = 1$ contribution to $\text{tr}(\mathbf{C})$.

A smooth quadratic extension replaces $\ln J$ below $J_\text{min} = 10^{-4}$ to prevent NaN when Newton overshoots into element inversion — a common occurrence with the extremely soft medium ($\gamma_0 \approx 10^{-6}$):

```python
def get_energy_density(self):
    J_min = 1e-4

    def safe_lnJ(J):
        lnJ_min = np.log(J_min)
        t = (J - J_min) / J_min
        ext = lnJ_min + t - 0.5 * t ** 2
        return np.where(J > J_min, np.log(J), ext)

    def psi(u_grad, mu, lmbda, *_unused):
        F = u_grad + np.eye(2)
        C = F.T @ F
        J = np.linalg.det(F)
        lnJ = safe_lnJ(J)
        return mu / 2.0 * (np.trace(C) + 1.0) - mu * lnJ + lmbda / 2.0 * lnJ ** 2

    return psi
```

The energy density takes `mu` and `lmbda` as per-cell arguments from `InternalVars`, so the same function serves both body and medium cells.

### HuHu-LuLu Biharmonic Regularization

Without regularization, the soft medium mesh distorts severely under compression. The HuHu-LuLu regularization penalizes displacement curvature in the medium:

$$
E_\text{reg} = k_r K L^2 \int_{\Omega_\text{med}} \left( H_{ijk} H_{ijk} - \frac{1}{d} L_i L_i \right) d\Omega
$$

where $H_{ijk} = \partial^2 u_i / \partial x_j \partial x_k$ is the displacement Hessian, $L_i = \sum_j H_{ijj}$ is the displacement Laplacian, $k_r = 5 \times 10^{-6}$ is the regularization prefactor, and $d = 2$ is the spatial dimension.

This is implemented as a **universal kernel** — a low-level FEAX interface that receives raw cell-level quantities (shape function Hessians, quadrature weights) and returns the element residual vector directly:

```python
def get_universal_kernel(self):
    dim = self.dim

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads,
               cell_JxW, cell_v_grads_JxW,
               mu, lmbda, cell_shape_hess, cell_is_medium):
        cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
        cell_sol = cell_sol_list[0]
        cell_JxW_1d = cell_JxW[0]

        # Displacement Hessian at quad points
        u_hess = np.einsum('av,qaKL->qvKL', cell_sol, cell_shape_hess)

        # Laplacian
        lapl_u = np.trace(u_hess, axis1=2, axis2=3)
        shape_lapl = np.trace(cell_shape_hess, axis1=-2, axis2=-1)

        # H:::∇²v − (1/dim) L·∇²v
        term1 = np.einsum('qvKL,qaKL->qav', u_hess, cell_shape_hess)
        term2 = np.einsum('qv,qa->qav', lapl_u, shape_lapl) / dim

        integrand = (term1 - term2) * cell_JxW_1d[:, None, None]
        result = kr_coeff * cell_is_medium * np.sum(integrand, axis=0)

        return jax.flatten_util.ravel_pytree(result)[0]

    return kernel
```

Key points:

- `hess=True` in the Problem constructor enables shape function Hessian computation
- Shape Hessians (`cell_shape_hess`) are passed through `InternalVars` as volume variables
- `cell_is_medium` acts as a per-cell switch: regularization is applied only in the medium region

### Internal Variables

The volume variables bundle per-cell material parameters, shape Hessians, and the medium mask:

```python
problem = ThirdMediumContact(mesh, vec=2, dim=2, ele_type='QUAD9', hess=True)

shape_hess = problem.fes[0].shape_hessians
iv = fe.InternalVars(volume_vars=(mu_cell, lmbda_cell, shape_hess, is_medium))
```

## Boundary Conditions and Solver

### Boundary Conditions

- **Fixed**: all DOFs at $x = 0$
- **Prescribed**: vertical displacement at the point $(L, H) = (1.0, 0.5)$, ramped incrementally

```python
bc_fixed = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], 0.0, atol=1e-6),
    component='all', value=0.0,
)
bc_move = fe.DirichletBCSpec(
    location=lambda p: np.isclose(p[0], L, atol=1e-6) & np.isclose(p[1], H, atol=1e-6),
    component='y', value=0.0,
)
bc = fe.DirichletBCConfig([bc_fixed, bc_move]).create_bc(problem)
```

### Why `symmetric_bc=False`

This problem requires non-symmetric BC elimination for two reasons:

1. **Incremental loading**: the prescribed displacement changes each step, and the previous solution is reused as the initial guess. The $K_{10}$ coupling in the unsymmetric Jacobian ensures that changes in prescribed DOFs propagate correctly to interior DOFs in the Newton linearization.

2. **Large stiffness contrast**: the medium stiffness is $\sim 10^{-6}\times$ the body stiffness. Without $K_{10}$ coupling, the first Newton increment overshoots in the soft medium, causing divergence.

### Solver Configuration

```python
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.DirectSolverOptions(solver='umfpack', verbose=True),
    newton_options=fe.NewtonOptions(tol=1e-6, rel_tol=1e-8, max_iter=10, internal_jit=True),
    iter_num=None,       # adaptive Newton
    internal_vars=iv,
    symmetric_bc=False,
)
```

- **`umfpack`**: CPU direct solver with robust pivoting, necessary for the non-symmetric and ill-conditioned Jacobian arising from the $\gamma_0 \approx 10^{-6}$ stiffness contrast.
- **`iter_num=None`**: adaptive Newton with Armijo line search and automatic convergence check.
- **`internal_jit=True`**: each component (residual, Jacobian, linear solve) is JIT-compiled individually, avoiding monolithic compilation of the full Newton loop.

## Incremental Loading Loop

The prescribed displacement is ramped over 20 steps from 0 to $-0.62L$:

```python
num_steps = 20
max_disp = -0.62 * L

sol = fe.zero_like_initial_guess(problem, bc)

for step in range(1, num_steps + 1):
    disp = max_disp * step / num_steps

    # Update BC values (only prescribed DOF changes)
    new_bc_vals = bc.bc_vals.at[move_bc_pos].set(disp)
    bc_step = bc.replace_vals(new_bc_vals)

    # Solve, reusing previous solution as initial guess
    sol = solver(iv, sol, bc=bc_step)
```

Key points:

- `bc.replace_vals()` creates a new `DirichletBC` with updated values but the same DOF locations — no solver rebuild.
- The previous solution `sol` is passed as the initial guess, giving Newton a good starting point for each load increment.
- With `symmetric_bc=False`, BC values are **not** pre-applied to the initial guess. The Newton solver drives BC DOFs to their prescribed values through the modified residual.

## Running the Example

```bash
python examples/advance/third_medium_contact.py
```

Output:

```
Step        Disp      max|u|   minJ_body    minJ_med  conv
----------------------------------------------------------------------
   1     -0.0310    0.031000   9.6749e-01   9.1651e-01  OK
   2     -0.0620    0.062000   9.3467e-01   8.3483e-01  OK
   ...
  20     -0.6200    0.637660   5.0186e-01   1.5620e-02  OK
```

VTK files are saved to `examples/advance/data/vtk_tmc/` for visualization in ParaView. The output includes displacement fields, the medium mask (`is_medium`), and element quality ($\min \det \mathbf{F}$).
