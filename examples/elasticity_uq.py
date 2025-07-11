import jax
import jax.numpy as np
import jax.scipy as scipy
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
import numpy as numpy

from feax.problem import Problem
from feax.generate_mesh import box_mesh_gmsh, Mesh
from feax.boundary_conditions import apply_bc, prepare_bc_info, FixedBC, create_boundary_functions

nu = 0.3

class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad, internal_vars):
            E_val = internal_vars[0]
            mu = E_val / (2. * (1. + nu))
            lmbda = E_val * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, internal_vars):
            return np.array([0., 0., 100.])
        return [surface_map]

# Setup FEM problem
print("Setting up FEM problem...")
meshio_mesh = box_mesh_gmsh(Nx=10, Ny=2, Nz=2, Lx=10., Ly=2., Lz=2., 
                           data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
boundary_fns = create_boundary_functions(10., 2., 2.)
bc_data = prepare_bc_info(mesh, FixedBC(boundary_fns['left'], components=[0, 1, 2]), vec_dim=3)
problem = LinearElasticity(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[boundary_fns['right']])
sol_list = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))

@jax.jit
def solve(E_value):
    """Direct solve function that takes E_value as input"""
    from feax.problem import get_sparse_system
    
    # Create internal vars directly with E_value
    E_array = np.full((problem.num_cells * 8,), E_value)
    state = problem.get_functional_state(internal_vars=[E_array])
    
    A, b = get_sparse_system(state, jax.flatten_util.ravel_pytree(sol_list)[0])
    A_bc, b_bc = apply_bc(A, b, bc_data)
    # Convert sparse matrix to dense for direct solver
    A_dense = A_bc.todense()
    x_sol = scipy.linalg.solve(A_dense, b_bc)
    sol_final = state.unflatten_fn_sol_list(x_sol)
    return np.max(np.linalg.norm(sol_final[0], axis=1))

def max_displacement(E_value):
    """Function that maps E value to max displacement"""
    return solve(E_value)

# Create vmapped function for batch computation
batch_displacement = jax.jit(jax.vmap(max_displacement))

# Define NumPyro model for uncertainty quantification
def material_model():
    """NumPyro model with uncertain Young's modulus"""
    # Prior distribution for E (log-normal to ensure positive values)
    # Mean around 50 GPa, with some uncertainty
    log_E_mean = np.log(50e3)
    log_E_std = 0.3  # ~30% coefficient of variation
    
    E = numpyro.sample("E", dist.LogNormal(log_E_mean, log_E_std))
    
    # Compute displacement (deterministic given E)
    displacement = numpyro.deterministic("displacement", max_displacement(E))
    
    return displacement

# Sample from prior distribution
print("Sampling from prior distribution...")
rng_key = jax.random.PRNGKey(0)
predictive = Predictive(material_model, num_samples=500)
samples = predictive(rng_key)

E_samples = samples["E"]
displacement_samples = samples["displacement"]

print(f"Generated {len(E_samples)} samples")
print(f"E range: {np.min(E_samples):.0f} - {np.max(E_samples):.0f} Pa")
print(f"Displacement range: {np.min(displacement_samples):.4f} - {np.max(displacement_samples):.4f} m")

# Statistical analysis
E_mean = np.mean(E_samples)
E_std = np.std(E_samples)
disp_mean = np.mean(displacement_samples)
disp_std = np.std(displacement_samples)

print(f"\nStatistical Analysis:")
print(f"E: μ = {E_mean:.0f} Pa, σ = {E_std:.0f} Pa, CV = {E_std/E_mean:.2f}")
print(f"Displacement: μ = {disp_mean:.4f} m, σ = {disp_std:.4f} m, CV = {disp_std/disp_mean:.2f}")

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. E distribution
axes[0,0].hist(E_samples, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
axes[0,0].axvline(E_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {E_mean:.0f}')
axes[0,0].set_xlabel('Young\'s Modulus E [Pa]')
axes[0,0].set_ylabel('Probability Density')
axes[0,0].set_title('Prior Distribution of E')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Displacement distribution
axes[0,1].hist(displacement_samples, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
axes[0,1].axvline(disp_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {disp_mean:.4f}')
axes[0,1].set_xlabel('Max Displacement [m]')
axes[0,1].set_ylabel('Probability Density')
axes[0,1].set_title('Propagated Uncertainty in Displacement')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. E vs Displacement scatter plot
axes[1,0].scatter(E_samples, displacement_samples, alpha=0.6, s=20)
axes[1,0].set_xlabel('Young\'s Modulus E [Pa]')
axes[1,0].set_ylabel('Max Displacement [m]')
axes[1,0].set_title('E vs Displacement Relationship')
axes[1,0].grid(True, alpha=0.3)

# 4. Cumulative distribution of displacement
sorted_disp = np.sort(displacement_samples)
cdf = np.arange(1, len(sorted_disp) + 1) / len(sorted_disp)
axes[1,1].plot(sorted_disp, cdf, 'b-', linewidth=2)
axes[1,1].axvline(np.percentile(displacement_samples, 5), color='red', linestyle='--', label='5th percentile')
axes[1,1].axvline(np.percentile(displacement_samples, 95), color='red', linestyle='--', label='95th percentile')
axes[1,1].set_xlabel('Max Displacement [m]')
axes[1,1].set_ylabel('Cumulative Probability')
axes[1,1].set_title('Displacement CDF')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elasticity_uq_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Quantile analysis
percentiles = np.array([5, 25, 50, 75, 95])
disp_quantiles = np.percentile(displacement_samples, percentiles)
E_quantiles = np.percentile(E_samples, percentiles)

print(f"\nQuantile Analysis:")
print(f"Percentile | E [Pa] | Displacement [m]")
print(f"-----------|--------|------------------")
for p, e_q, d_q in zip(percentiles, E_quantiles, disp_quantiles):
    print(f"{p:8}% | {e_q:6.0f} | {d_q:15.4f}")

# Risk analysis
critical_displacement = 0.02  # 2 cm critical displacement
exceedance_prob = np.mean(displacement_samples > critical_displacement)
print(f"\nRisk Analysis:")
print(f"Probability of displacement > {critical_displacement:.3f} m: {exceedance_prob:.2%}")

print(f"\nUncertainty quantification complete! Results saved as 'elasticity_uq_analysis.png'")