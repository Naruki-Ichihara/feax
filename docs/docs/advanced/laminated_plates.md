---
sidebar_label: Laminated Plates (Shell)
---

# Laminated Plates with `feax.mechanics.shell`

The `feax.mechanics.shell` module provides a **first-order shear-deformation theory (FSDT,
"Mindlin")** plate model together with **classical lamination theory (CLT)** helpers for
building composite (multi-layer) laminates. This guide walks through the module top to
bottom: lamina stiffness → laminate assembly → thermal loads → kinematics → the weak form
and `Problem`, and finishes with a worked thermal-warping example.

All tensors are kept in **physical index form** (`C_ijkl`, `eps_ij`, …) rather than Voigt
vectors, so they compose cleanly with `jax.numpy.einsum` and stay differentiable.

## The plate model

A Mindlin plate carries **five degrees of freedom per node**, split into two FE variables
on the same 2D mesh:

| Variable | Components | `vec` | Meaning |
|---|---|---|---|
| `var0` | $(u, v, w)$ | 3 | mid-surface translations |
| `var1` | $(\theta_x, \theta_y)$ | 2 | section rotations |

The constitutive law relates the membrane strain $\varepsilon$, curvature $\kappa$, and
transverse-shear strain $\gamma$ to the stress resultants $N$, $M$, $Q$ through the
laminate stiffness matrices $A$, $B$, $D$, $G_s$:

$$
N = A : \varepsilon + B : \kappa, \qquad
M = B : \varepsilon + D : \kappa, \qquad
Q = G_s \cdot \gamma
$$

The whole module exists to produce $(A, B, D, G_s)$ — and the thermal resultants
$(N_T, M_T)$ — and to turn them into a `feax.Problem`.

## 1. Lamina stiffness

A single ply (lamina) is described by its plane-stress stiffness tensor `C_ijkl` of shape
`(2, 2, 2, 2)`, in the ply's **material axes** (fibre = local 1-axis).

```python
from feax.mechanics import shell

# Isotropic ply
C_iso = shell.isotropic_in_plane_stiffness(E=70e9, nu=0.3)

# Orthotropic ply (e.g. CFRP): fibre stiffness E1 ≫ transverse E2
C_cfrp = shell.orthotropic_in_plane_stiffness(E1=140e9, E2=10e9, G12=5e9, nu12=0.30)
```

Transverse-shear stiffness is supplied separately, as a scalar $G$ (isotropic) or a
`(2, 2)` array `diag(G13, G23)` for an orthotropic ply.

## 2. Homogeneous plate: `plate_stiffness`

For a single-layer, midplane-symmetric flat plate of thickness `h`, `plate_stiffness`
returns the thickness integrals directly — no lamination needed:

```python
A, D, G_s = shell.plate_stiffness(C_iso, h=2e-3, G_transverse=70e9 / (2 * 1.3))
```

This gives $A = h\,C$, $D = \tfrac{h^3}{12} C$, and $G_s = \kappa_s\, h\, G$ (with shear
correction $\kappa_s = 5/6$ by default). There is no coupling matrix $B$ for a homogeneous
plate.

## 3. Laminate assembly (CLT): `laminate_stiffness`

For a stack of $n$ plies at different orientations, `laminate_stiffness` performs the
classical lamination-theory thickness integrals. Layers are ordered **bottom → top**, with
the midplane at $z = 0$ (see `z_layer_coordinates`):

$$
A = \sum_k \bar{Q}_k\,(z_{k+1} - z_k), \quad
B = \tfrac12 \sum_k \bar{Q}_k\,(z_{k+1}^2 - z_k^2), \quad
D = \tfrac13 \sum_k \bar{Q}_k\,(z_{k+1}^3 - z_k^3)
$$

where $\bar{Q}_k$ is each layer's stiffness rotated by `thetas[k]` into laminate axes.

```python
import jax.numpy as jnp

# Two CFRP plies at ±45°, 0.5 mm each
C_layers     = jnp.stack([C_cfrp, C_cfrp])          # (n_layers, 2,2,2,2)
G_layers     = jnp.stack([jnp.diag(jnp.array([5e9, 3e9]))] * 2)  # (n_layers, 2, 2)
thetas       = jnp.array([-jnp.pi / 4, +jnp.pi / 4])  # radians, bottom→top
thicknesses  = jnp.array([0.5e-3, 0.5e-3])

A, B, D, G_s = shell.laminate_stiffness(C_layers, G_layers, thetas, thicknesses)
```

`C_layers` / `G_layers` may also be a single tensor (broadcast to every layer) when all
plies share the same material and only orientation differs.

:::note The coupling matrix `B`
$B$ measures **membrane–bending coupling**. It vanishes for any midplane-symmetric stack
(e.g. $[\theta / -\theta]_s$ or $[0/90]_s$) but is **non-zero for antisymmetric stacks**
like $[+45 / -45]$ — which is exactly what makes such laminates warp under in-plane or
thermal loads. Pass `B` to the resultant/weak-form helpers whenever it is non-zero.
:::

Supporting helpers:

- `z_layer_coordinates(thicknesses)` → interface $z$-coordinates, length $n+1$, midplane at 0.
- `rotate_in_plane_stiffness(C, theta)` → $\bar{C} = R\,R\,R\,R : C$ (4th-order rotation).
- `rotate_shear_stiffness(G, theta)` → rotate a transverse-shear `(2,2)` tensor.

## 4. Thermal loads: `laminate_thermal_loads`

A temperature change induces eigenstrains $\alpha\,\Delta T$ that, in a constrained
laminate, act as **pre-force / pre-moment** resultants. For a through-thickness profile
$\Delta T(z) = \texttt{dT\_avg} + z\cdot\texttt{dT\_grad}$:

$$
N_T = \sum_k (\bar{C}_k : \bar{\alpha}_k)\!\int_{z_k}^{z_{k+1}}\! \Delta T\,dz, \qquad
M_T = \sum_k (\bar{C}_k : \bar{\alpha}_k)\!\int_{z_k}^{z_{k+1}}\! z\,\Delta T\,dz
$$

```python
alpha_cfrp = jnp.diag(jnp.array([-0.5e-6, 30e-6]))   # (alpha_fibre, alpha_transverse)
alpha_layers = jnp.stack([alpha_cfrp, alpha_cfrp])

N_T, M_T = shell.laminate_thermal_loads(
    C_layers, alpha_layers, thetas, thicknesses,
    dT_avg=-150.0,   # uniform cooling (cure → room temperature)
    dT_grad=0.0,     # no through-thickness gradient
)
```

For a midplane-symmetric laminate at uniform $\Delta T$, $M_T = 0$ (no warping). For an
antisymmetric $\pm45°$ stack the diagonal terms cancel but the **shear** moment $M_{T,xy}$
adds up → **pure twist**.

Related helpers:

- `thermal_expansion_from_orientation(a2, alpha_fibre, alpha_transverse)` builds
  $\alpha = \alpha_t I + (\alpha_f - \alpha_t)\,a_2$ from an
  [orientation tensor](../api/reference/feax/mechanics/orientation.md) — useful for
  continuous fibre-orientation optimisation with thermal loading.
- `rotate_thermal_expansion(alpha, theta)` → $\bar\alpha = R\,\alpha\,R^T$.

## 5. Kinematics and resultants

These two functions evaluate the strain measures and the constitutive law at a single
quadrature point — they are the building blocks of the weak form.

```python
eps, kappa, gamma = shell.mindlin_strains(
    grad_uvw, grad_theta, theta, nonlinear="von_karman",
)
N, M, Q = shell.mindlin_resultants(
    eps, kappa, gamma, A, D, G_s, B=B, N_T=N_T, M_T=M_T,
)
```

- `mindlin_strains` returns membrane strain $\varepsilon = \operatorname{sym}(\nabla u_\text{in})$,
  curvature $\kappa = \operatorname{sym}(\nabla\theta)$, and transverse shear
  $\gamma = \nabla w + \theta$.
  - `nonlinear="linear"` (default) — small rotations.
  - `nonlinear="von_karman"` — adds the moderate-rotation membrane term
    $\tfrac12 \nabla w \otimes \nabla w$, valid up to ~10–15° rotations. This makes the
    residual **cubic in $w$**, so a Newton solve is required.
- `mindlin_resultants` applies $N = A:\varepsilon + B:\kappa$, $M = B:\varepsilon + D:\kappa$,
  $Q = G_s\gamma$, then subtracts the thermal pre-loads ($N \leftarrow N - N_T$,
  $M \leftarrow M - M_T$). All optional terms (`B`, `N_T`, `M_T`) default to `None`.

## 6. Building the problem

The quickest path is the **`make_mindlin_plate` factory**, which fixes the two-variable
layout for you:

```python
problem = shell.make_mindlin_plate(
    mesh, A, D, G_s,
    ele_type="QUAD4",
    nonlinear="von_karman",   # requires Newton (linear=False)
    B=B, N_T=N_T, M_T=M_T,
)

bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=clamped_edge, component="all", value=0.0, variable_index=0),
    fe.DirichletBCSpec(location=clamped_edge, component="all", value=0.0, variable_index=1),
]).create_bc(problem)

solver = fe.create_solver(problem, bc, linear=False)   # Newton for von Kármán
sol = solver(fe.TracedParams(volume_vars=()), fe.zero_like_initial_guess(problem, bc))
```

Under the hood `make_mindlin_plate` builds a `MindlinPlate` (a `feax.Problem` subclass)
whose `get_weak_form()` returns `mindlin_weak_form(A, D, G_s, B=, N_T=, M_T=, nonlinear=)`.
If you need a custom residual (e.g. a load-stepped thermal term, see below) you can
subclass `feax.Problem` directly and call `mindlin_strains` / `mindlin_resultants`
yourself, or wire the resultants into the weak form manually.

:::note Why two meshes in `mesh=[mesh, mesh]`?
A two-variable problem takes one mesh **per variable**. The translation field $(u,v,w)$ and
the rotation field $(\theta_x,\theta_y)$ live on the same geometry, so you pass the same
mesh twice: `mesh=[mesh, mesh]`, `vec=[3, 2]`, `ele_type=["QUAD4", "QUAD4"]`. The
`make_mindlin_plate` factory does this for you.
:::

## Worked example: thermal warping of a ±45° laminate

The example `examples/basic/laminate_thermal_warping.py` cools a 2-layer
antisymmetric $\pm45°$ CFRP laminate. Because the stack is antisymmetric, $M_{T,xx} =
M_{T,yy} = 0$ but $M_{T,xy} \neq 0$ — the plate twists into a saddle (hyperbolic
paraboloid) shape rather than bending.

The plate is long enough ($L_x = 100$ mm) that the linear-theory tip twist (~20°) leaves
the small-rotation regime, so it uses the **von Kármán** strain measure and **load-steps**
the cooling with [`ImplicitPipeline`](./cahn_hilliard.md). A pseudo-time variable
$\lambda \in [0, 1]$ scales the thermal resultants $(N_T, M_T)$, and Newton is warm-started
from the previous increment:

```python
from feax.mechanics.shell import (
    orthotropic_in_plane_stiffness, laminate_stiffness, laminate_thermal_loads,
    mindlin_strains, mindlin_resultants,
)
from feax.solvers.time_solver import ImplicitPipeline, TimeConfig, run

class RampedThermalPlate(fe.Problem):
    """Mindlin/von-Kármán plate; volume_vars[0] = load fraction λ ∈ [0,1]."""
    def custom_init(self, A, D, G_s, B, N_T_full, M_T_full, nonlinear="von_karman"):
        self.A, self.D, self.G_s, self.B = A, D, G_s, B
        self.N_T_full, self.M_T_full, self.nonlinear = N_T_full, M_T_full, nonlinear

    def get_weak_form(self):
        A, D, G_s, B = self.A, self.D, self.G_s, self.B
        N_T_full, M_T_full, nonlinear = self.N_T_full, self.M_T_full, self.nonlinear

        def weak_form(vals, grads, x, lam):
            eps, kappa, gamma = mindlin_strains(
                grads[0], grads[1], vals[1], nonlinear=nonlinear,
            )
            N, M, Q = mindlin_resultants(
                eps, kappa, gamma, A, D, G_s,
                B=B, N_T=lam * N_T_full, M_T=lam * M_T_full,   # ← scaled by load fraction
            )
            grad_w = grads[0][2, :]
            q_w = Q + N @ grad_w if nonlinear == "von_karman" else Q
            grad0 = jnp.concatenate([N, q_w[None, :]], axis=0)
            return ([jnp.zeros(3), Q], [grad0, M])
        return weak_form
```

The pipeline assembles the laminate once in `build()`, then `update_vars` sets
$\lambda = t + \Delta t$ each step so the thermal load ramps from 0 to full cooling:

```python
class ThermalRampPipeline(ImplicitPipeline):
    def build(self, mesh):
        self.mesh = mesh
        C = orthotropic_in_plane_stiffness(E1, E2, G12, NU12)
        G = jnp.diag(jnp.array([G13, G23]))
        alpha = jnp.diag(jnp.array([ALPHA_FIBRE, ALPHA_TRANS]))
        thetas, thicks = jnp.array([-jnp.pi/4, jnp.pi/4]), jnp.array([T_LAYER, T_LAYER])

        A, B, D, G_s = laminate_stiffness(jnp.stack([C, C]), jnp.stack([G, G]), thetas, thicks)
        N_T, M_T = laminate_thermal_loads(
            jnp.stack([C, C]), jnp.stack([alpha, alpha]), thetas, thicks,
            dT_avg=DELTA_T, dT_grad=0.0,
        )
        self.problem = RampedThermalPlate(
            mesh=[mesh, mesh], vec=[3, 2], dim=2, ele_type=["QUAD4", "QUAD4"],
            additional_info=(A, D, G_s, B, N_T, M_T, "von_karman"),
        )
        self.bc = fe.DirichletBCConfig([
            fe.DirichletBCSpec(location=left_edge, component="all", value=0.0, variable_index=0),
            fe.DirichletBCSpec(location=left_edge, component="all", value=0.0, variable_index=1),
        ]).create_bc(self.problem)
        self._n_nodes = mesh.points.shape[0]
        self.solver = fe.create_solver(
            self.problem, bc=self.bc,
            solver_options=fe.DirectSolverOptions(solver="umfpack"),
            newton_options=fe.NewtonOptions(tol=1e-6, rel_tol=1e-8, max_iter=30),
            linear=False,
            traced_params=fe.TracedParams(volume_vars=(jnp.zeros(self._n_nodes),)),
        )

    def initial_state(self):
        return fe.zero_like_initial_guess(self.problem, self.bc)

    def update_vars(self, state, t, dt):
        return fe.TracedParams(volume_vars=(jnp.full(self._n_nodes, t + dt),))

config = TimeConfig(dt=1.0 / N_LOAD_STEPS, t_end=1.0, save_every=5, print_every=1)
result = run(ThermalRampPipeline(), mesh, config, output_dir=str(OUTPUT_DIR))
```

:::tip Solver choice for von Kármán
Use `solver="umfpack"` (general LU) rather than a Cholesky-based direct solver. Under
twisting the geometric-stiffness term $N\cdot\nabla w$ can make the tangent **indefinite**,
and a Cholesky factorisation then produces spurious Newton steps that the line search
rejects ($\alpha \to 0$, residual stalls). More load steps (`N_LOAD_STEPS`) also widen the
convergence basin — a single shot at full cooling from the flat state typically diverges.
:::

## API summary

| Function | Purpose |
|---|---|
| `isotropic_in_plane_stiffness(E, nu)` | Plane-stress isotropic `C_ijkl` |
| `orthotropic_in_plane_stiffness(E1, E2, G12, nu12)` | Plane-stress orthotropic `C_ijkl` (material axes) |
| `plate_stiffness(C, h, G_transverse)` | `(A, D, G_s)` for a homogeneous symmetric plate |
| `z_layer_coordinates(thicknesses)` | Interface $z$-coordinates, midplane at 0 |
| `rotate_in_plane_stiffness(C, theta)` | Rotate `C_ijkl` by `theta` |
| `rotate_shear_stiffness(G, theta)` | Rotate transverse-shear `(2,2)` tensor |
| `laminate_stiffness(C_layers, G_layers, thetas, thicknesses)` | CLT `(A, B, D, G_s)` for an $n$-layer stack |
| `thermal_expansion_from_orientation(a2, alpha_f, alpha_t)` | CTE tensor from an orientation tensor |
| `rotate_thermal_expansion(alpha, theta)` | Rotate a CTE `(2,2)` tensor |
| `laminate_thermal_loads(C_layers, alpha_layers, thetas, thicknesses, dT_avg, dT_grad)` | Thermal `(N_T, M_T)` resultants |
| `mindlin_strains(grad_uvw, grad_theta, theta, nonlinear)` | `(eps, kappa, gamma)` |
| `mindlin_resultants(eps, kappa, gamma, A, D, G_s, B, N_T, M_T)` | `(N, M, Q)` |
| `mindlin_weak_form(A, D, G_s, B, N_T, M_T, nonlinear)` | Quad-point weak form for `get_weak_form()` |
| `MindlinPlate` / `make_mindlin_plate(...)` | `feax.Problem` subclass / factory |

See the [API reference](../api/reference/feax/mechanics/shell.md) for full signatures.
