# Spinodoid Metamaterials

Spinodoid metamaterials are a novel class of architected materials inspired by natural spinodal decomposition processes. They feature smooth, bi-continuous microstructures with exceptional mechanical properties and tunable anisotropy.

## Overview

Spinodoid structures approximate microstructures observed during spinodal phase separation in systems like nanoporous metal foams and polymer blends. Unlike traditional periodic lattices, spinodoids are:

- **Non-periodic and stochastic** - More resilient to fabrication defects
- **Smooth and bi-continuous** - Eliminate stress concentration points
- **Tunable anisotropic** - Tailorable directional mechanical properties

Learn more about spinodoid metamaterials at [Kumar Lab's research page](https://www.mech-mat.com/research/spinodoid-metamaterials).

## Mathematical Foundation

### Gaussian Random Field (GRF)

Spinodoid structures are generated using a Gaussian Random Field that approximates spinodal decomposition patterns:

$$
\phi(\mathbf{x}) = \sqrt{\frac{2}{N}} \sum_{i=1}^{N} \cos(\beta \mathbf{n}_i \cdot \mathbf{x} + \gamma_i)
$$

Where:
- $\mathbf{x} = (x, y, z) \in [0, L]^3$ is the spatial coordinate in the unit cell
- $\mathbf{n}_i \in \mathbb{S}^2$ are random direction vectors on the unit sphere
- $\gamma_i \sim \mathcal{U}(0, 2\pi)$ are random phase shifts
- $\beta = 2\pi/\lambda$ controls the characteristic wavelength $\lambda$
- $N$ is the number of wave components (typically 50-200)

The field $\phi(\mathbf{x})$ has zero mean: $\mathbb{E}[\phi] = 0$ and exhibits spatial correlation with a characteristic length scale determined by $\beta$.

### Anisotropic Control

Direction vectors $\mathbf{n}_i$ are constrained to create anisotropic structures with preferred orientations. Three angular parameters $(\theta_1, \theta_2, \theta_3)$ define cone angles around the principal axes (x, y, z):

**Generation method:**
- Vectors are distributed among cones around the coordinate axes
- For each active axis (where $\theta_j > 0$), approximately $N/n_{\text{active}}$ vectors are generated
- Within each cone, vectors are sampled uniformly in the cone angle $[0, \theta_j]$

**Anisotropy cases:**
- **Isotropic**: $\theta_1 = \theta_2 = \theta_3 = \pi/4$ → uniform sampling over three cones covers full sphere
- **Uniaxial**: $\theta_1 \ll \theta_2 = \theta_3 = \pi/4$ → strong alignment along x-axis
- **Orthotropic**: $\theta_1, \theta_2, \theta_3$ distinct → three independent principal directions

The constraint ensures wave directions cluster around preferred orientations, directly translating to mechanical anisotropy in the final structure.

**Note:** The raw GRF $\phi(\mathbf{x})$ does **not** enforce periodicity or symmetry. These properties are imposed by the Helmholtz filter in the next step.

### Periodicity Guarantee

For homogenization to be valid, the structure must be **strictly periodic** on the unit cell boundaries $\partial\Omega = [0,L]^3$.

**Important:** The raw GRF $\phi(\mathbf{x})$ is **not periodic** in general, since:
- Direction vectors $\mathbf{n}_i$ are randomly sampled (not constrained by periodicity)
- Phase shifts $\gamma_i$ are independent random variables
- Wave numbers $\beta n_{ij}$ do not necessarily satisfy periodicity conditions

**Periodicity is enforced by the Helmholtz filter** with periodic boundary conditions:

$$
\tilde{\rho}(\mathbf{x}) - r^2 \nabla^2 \tilde{\rho}(\mathbf{x}) = \phi(\mathbf{x}), \quad \mathbf{x} \in \Omega
$$

$$
\tilde{\rho}(\mathbf{x})|_{\partial\Omega^-} = \tilde{\rho}(\mathbf{x})|_{\partial\Omega^+}, \quad \nabla\tilde{\rho}(\mathbf{x})|_{\partial\Omega^-} = \nabla\tilde{\rho}(\mathbf{x})|_{\partial\Omega^+}
$$

where $\partial\Omega^-$ and $\partial\Omega^+$ are opposite faces of the unit cell. The solution $\tilde{\rho}$ to this PDE:

1. **Enforces strict periodicity**: $\tilde{\rho}(\mathbf{x} + L\mathbf{e}_j) = \tilde{\rho}(\mathbf{x})$ for all $\mathbf{x} \in \Omega$
2. **Enforces $C^1$ continuity**: Both $\tilde{\rho}$ and $\nabla\tilde{\rho}$ are continuous across boundaries
3. **Smooths the GRF**: Acts as a low-pass filter with length scale $r$
4. **Preserves stochastic character**: The filtered field retains the random nature of $\phi$

**Implementation via prolongation matrix:**

The periodic boundary conditions are implemented using a prolongation matrix $\mathbf{P} \in \mathbb{R}^{n_{\text{full}} \times n_{\text{indep}}}$:

$$
\tilde{\boldsymbol{\rho}}_{\text{full}} = \mathbf{P} \tilde{\boldsymbol{\rho}}_{\text{indep}}
$$

where:
- $\tilde{\boldsymbol{\rho}}_{\text{full}}$ contains all nodal DOFs (including duplicated boundary nodes)
- $\tilde{\boldsymbol{\rho}}_{\text{indep}}$ contains only independent DOFs (interior + one face per periodic pair)
- $\mathbf{P}$ maps independent DOFs to the full mesh by identifying periodic pairs

The Helmholtz equation is solved in the reduced space, automatically satisfying periodicity.

## Generation Pipeline

The spinodoid generation pipeline in FEAX consists of:

1. **GRF Generation** - Gaussian Random Field with anisotropic directions
2. **Helmholtz Filtering** - Smoothing via PDE solution: $\rho - r^2\nabla^2\rho = \phi$
3. **Heaviside Projection** - Sharp boundaries: $\rho_H = \frac{\tanh(\beta(\rho - t)) + 1}{2}$
4. **Volume Fraction Control** - Threshold $t$ computed from target volume fraction

```python
import feax.flat as flat

# Define parameters
theta1, theta2, theta3 = 0.0, 0.3, 0.5  # Anisotropy angles
N = 100  # Number of wave components
beta_grf = 15.0  # GRF wavelength parameter
radius = 0.1  # Helmholtz filter radius
target_vf = 0.5  # Target volume fraction

# Generate direction vectors
key = jax.random.PRNGKey(42)
key_n, key_g = jax.random.split(key)
n_vectors = flat.spinodoid.generate_direction_vectors(
    theta1, theta2, theta3, N, key_n
)
gamma = jax.random.uniform(key_g, shape=(N,), minval=0.0, maxval=2*np.pi)

# Evaluate GRF at cell centers
rho = flat.spinodoid.evaluate_grf_field(cell_centers, n_vectors, gamma, beta_grf)

# Apply Helmholtz filter
rho = flat.filters.helmholtz_filter(rho, mesh, radius, P, solver_opts)

# Normalize and project
rho = (rho - rho.min()) / (rho.max() - rho.min())
thresh = flat.filters.compute_volume_fraction_threshold(rho, target_vf)
rho = flat.filters.heaviside_projection(rho, beta_heaviside=10.0, threshold=thresh)
```

## Dataset Generation

FEAX provides tools for batch generation of spinodoid structures with automated homogenization:

```python
# examples/advance/spinodoid/spinodoid_generation.py
python spinodoid_generation.py
```

This script:
- Generates multiple spinodoid structures with random parameters
- Computes homogenized stiffness tensors $\mathbf{C}_{hom}$ via periodic homogenization
- Saves results to CSV with parameters and material properties
- Uses JIT-compiled `jax.jit(jax.vmap())` for efficient batch processing

**Output CSV format:**
```
seed,theta1,theta2,theta3,target_vf,mesh_size,radius,beta_heaviside,beta_grf,N,E0,nu,p,C11,...,C66
```

## Reconstruction and Export

Reconstruct spinodoids from the CSV dataset:

```bash
# VTU format (density field + stiffness sphere visualization)
python reconstruct_spinodoids.py 0 5 10

# OBJ format (closed surface mesh for 3D printing)
python reconstruct_spinodoids.py --obj 0 5 10

# Range of samples
python reconstruct_spinodoids.py --obj --range 0 100
```

**OBJ export features:**
- Marching cubes isosurface extraction at $\rho = 0.5$
- Watertight mesh with automatic boundary closure
- Configurable mesh refinement (default 2×)
- Cubic interpolation for smooth surfaces

## Homogenization

### Theory

Periodic homogenization computes the effective elastic stiffness tensor $\mathbf{C}_{\text{hom}}$ of the unit cell $\Omega = [0,L]^3$. For a heterogeneous material with local stiffness $\mathbf{C}(\mathbf{x})$, the homogenized tensor is:

$$
C_{\text{hom}}^{ijkl} = \frac{1}{|\Omega|} \int_\Omega C^{ijmn}(\mathbf{x}) \left(\varepsilon^{kl} + \frac{\partial \chi^{kl}_m}{\partial x_n}\right) d\Omega
$$

where $\boldsymbol{\chi}^{kl}$ are **characteristic displacement fields** solving the unit cell problem:

$$
\nabla \cdot \left[\mathbf{C}(\mathbf{x}) : (\boldsymbol{\varepsilon}^{kl} + \nabla \boldsymbol{\chi}^{kl})\right] = \mathbf{0}, \quad \mathbf{x} \in \Omega
$$

with periodic boundary conditions:
$$
\boldsymbol{\chi}^{kl}(\mathbf{x})|_{\partial\Omega^-} = \boldsymbol{\chi}^{kl}(\mathbf{x})|_{\partial\Omega^+}
$$

Here $\boldsymbol{\varepsilon}^{kl}$ are the 6 independent unit strain tensors:
$$
\boldsymbol{\varepsilon}^{11} = \begin{pmatrix}1&0&0\\0&0&0\\0&0&0\end{pmatrix}, \quad
\boldsymbol{\varepsilon}^{12} = \begin{pmatrix}0&0.5&0\\0.5&0&0\\0&0&0\end{pmatrix}, \quad \text{etc.}
$$

### SIMP Material Interpolation

For topology optimization, the Solid Isotropic Material with Penalization (SIMP) model relates density $\rho(\mathbf{x}) \in [0,1]$ to local stiffness:

$$
\mathbf{C}(\mathbf{x}) = \rho(\mathbf{x})^p \mathbf{C}_0
$$

where:
- $\mathbf{C}_0$ is the solid material stiffness (e.g., $E_0 = 21$ GPa, $\nu = 0.3$)
- $p \geq 1$ is the penalization parameter (typically $p = 3$)
- Higher $p$ penalizes intermediate densities, promoting 0-1 solutions

For isotropic elasticity:
$$
\mathbf{C}_0^{ijkl} = \lambda \delta_{ij}\delta_{kl} + \mu(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})
$$

with Lamé parameters:
$$
\lambda = \frac{E_0 \nu}{(1+\nu)(1-2\nu)}, \quad \mu = \frac{E_0}{2(1+\nu)}
$$

### Implementation

```python
# Create homogenization solver
compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P_3d, solver_opts, mesh, dim=3
)

# SIMP interpolation: E(ρ) = E0 * ρ^p
E_field = E0 * rho**p
E_array = fe.internal_vars.InternalVars.create_cell_var(problem, E_field)
nu_array = fe.internal_vars.InternalVars.create_cell_var(problem, 0.3)
internal_vars = fe.internal_vars.InternalVars(
    volume_vars=(E_array, nu_array),
    surface_vars=()
)

# Solve 6 unit cell problems and assemble C_hom
C_hom = compute_C_hom(internal_vars)  # 6×6 stiffness matrix (Voigt notation)
```

The solver automatically:
1. **Solves 6 periodic elasticity problems** for unit strains $\boldsymbol{\varepsilon}^{11}, \boldsymbol{\varepsilon}^{22}, \dots, \boldsymbol{\varepsilon}^{12}$
2. **Computes volume-averaged stress**: $\boldsymbol{\sigma}^{kl} = \frac{1}{|\Omega|}\int_\Omega \mathbf{C}(\mathbf{x}) : (\boldsymbol{\varepsilon}^{kl} + \nabla \boldsymbol{\chi}^{kl}) d\Omega$
3. **Assembles stiffness matrix**: $C_{\text{hom}}^{ijkl} = \sigma^{kl}_{ij}$ (using Voigt notation)

## Stiffness Visualization

### Directional Young's Modulus

The directional Young's modulus $E(\mathbf{n})$ measures stiffness in direction $\mathbf{n}$. For a uniaxial stress state $\boldsymbol{\sigma} = \sigma \mathbf{n} \otimes \mathbf{n}$, the strain in direction $\mathbf{n}$ is:

$$
\varepsilon_n = \mathbf{n}^T \boldsymbol{\varepsilon} \mathbf{n} = \mathbf{n}^T \mathbf{S} \boldsymbol{\sigma} \mathbf{n} = \sigma \mathbf{n}^T \mathbf{S} \mathbf{n}
$$

where $\mathbf{S} = \mathbf{C}_{\text{hom}}^{-1}$ is the compliance tensor. The directional Young's modulus is:

$$
E(\mathbf{n}) = \frac{\sigma}{\varepsilon_n} = \frac{1}{\mathbf{n}^T \mathbf{S} \mathbf{n}}
$$

In Voigt notation ($\mathbf{n} = (n_1, n_2, n_3)$):

$$
E(\mathbf{n}) = \frac{1}{S_{11}n_1^4 + S_{22}n_2^4 + S_{33}n_3^4 + (2S_{12} + S_{66})n_1^2 n_2^2 + (2S_{13} + S_{55})n_1^2 n_3^2 + (2S_{23} + S_{44})n_2^2 n_3^2}
$$

### Visualization

```python
# Visualize E(n) as a 3D sphere deformed by stiffness
flat.utils.visualize_stiffness_sphere(
    C_hom,
    output_file="stiffness_sphere.vtu",
    n_theta=30,
    n_phi=60
)
```

The function:
1. Samples directions $\mathbf{n}$ uniformly on the unit sphere
2. Computes $E(\mathbf{n})$ for each direction
3. Creates surface with radius $r(\mathbf{n}) = E(\mathbf{n})/E_{\max}$ (normalized)

**Interpretation:**
- **Perfect sphere** → Isotropic material ($E$ independent of direction)
- **Elongated ellipsoid** → Anisotropic with preferred stiffness directions
- **Aspect ratio** → Degree of anisotropy (max/min $E$)

## Example: Complete Workflow

```python
import jax
import jax.numpy as np
import feax as fe
import feax.flat as flat

# Setup mesh and periodic BCs
class SpinodoidUnitCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

unitcell = SpinodoidUnitCell(mesh_size=0.04)
mesh = unitcell.mesh
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=1, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)

# Generate spinodoid
cell_centers = np.mean(mesh.points[mesh.cells], axis=1)
key = jax.random.PRNGKey(42)
key_n, key_g = jax.random.split(key)

theta1, theta2, theta3 = 0.2, 0.3, 0.1
n_vectors = flat.spinodoid.generate_direction_vectors(theta1, theta2, theta3, 100, key_n)
gamma = jax.random.uniform(key_g, shape=(100,), minval=0.0, maxval=2*np.pi)

rho = flat.spinodoid.evaluate_grf_field(cell_centers, n_vectors, gamma, 15.0)
solver_opts = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg")
rho = flat.filters.helmholtz_filter(rho, mesh, 0.1, P, solver_opts)
rho = (rho - rho.min()) / (rho.max() - rho.min())

thresh = flat.filters.compute_volume_fraction_threshold(rho, 0.5)
rho = flat.filters.heaviside_projection(rho, 10.0, thresh)

# Homogenization
class ElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(3) + 2 * mu * eps
        return stress

problem = ElasticityProblem(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])
bc = fe.DCboundary.DirichletBCConfig([]).create_bc(problem)
pairings_3d = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P_3d = flat.pbc.prolongation_matrix(pairings_3d, mesh, vec=3)

compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P_3d, solver_opts, mesh, dim=3
)

# Convert nodal density to cell-based for homogenization
rho_cell = np.mean(rho[mesh.cells], axis=1)
E_field = 21e3 * rho_cell**3  # SIMP with p=3
E_array = fe.internal_vars.InternalVars.create_cell_var(problem, E_field)
nu_array = fe.internal_vars.InternalVars.create_cell_var(problem, 0.3)
internal_vars = fe.internal_vars.InternalVars(
    volume_vars=(E_array, nu_array),
    surface_vars=()
)

C_hom = compute_C_hom(internal_vars)
print(f"Homogenized stiffness tensor:\n{C_hom}")

# Visualize
flat.utils.visualize_stiffness_sphere(C_hom, "stiffness_sphere.vtu")
fe.utils.save_sol(mesh, "spinodoid.vtu", point_infos=[("density", rho.reshape(-1, 1))])
```

## Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `theta1, theta2, theta3` | Anisotropy angles | $[0, \pi/4]$ |
| `N` | Number of GRF waves | 50-200 |
| `beta_grf` | GRF wavelength parameter | 10-20 |
| `radius` | Helmholtz filter radius | 0.05-0.2 |
| `beta_heaviside` | Projection sharpness | 5-15 |
| `target_vf` | Volume fraction | 0.3-0.7 |
| `mesh_size` | FE mesh resolution | 0.02-0.1 |
| `p` | SIMP penalty | 3.0 |

## Performance

The implementation uses:
- **JIT compilation**: `jax.jit` for all core functions
- **Vectorization**: `jax.vmap` for batch processing
- **Efficient homogenization**: Solves 6 strain cases in parallel
- **Memory-efficient**: Node/cell-based internal variables

Typical performance on CPU:
- Single spinodoid generation: ~20 seconds (mesh_size=0.04)
- Batch of 10 with homogenization: ~4 minutes
- Reconstruction to OBJ: ~20 seconds per sample

## References

1. Kumar, S. et al. "Inverse-designed spinodoid metamaterials" *npj Computational Materials* (2020)
2. [Kumar Lab: Spinodoid Metamaterials](https://www.mech-mat.com/research/spinodoid-metamaterials)

## File Locations

- Generation script: `examples/advance/spinodoid/spinodoid_generation.py`
- Reconstruction script: `examples/advance/spinodoid/reconstruct_spinodoids.py`
- Core implementation: `feax/flat/spinodoid.py`
- Filters: `feax/flat/filters.py`
- Homogenization: `feax/flat/solver.py`
