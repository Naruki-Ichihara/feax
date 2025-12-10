"""
Simple Interactive Demo for Linear Elasticity Analysis
Uses matplotlib widgets for instant real-time visualization
"""
import feax as fe
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as onp

# Setup problem (done once, before UI)
print("Setting up FEA problem...")

# Material properties
elastic_moduli = 70e3
poisson_ratio = 0.3
tol = 1e-5

# Define mesh - coarse for speed
L = 100
W = 10
H = 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=3)  # Coarse mesh for instant response

# Locations
left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol)

# Define problem
E = elastic_moduli
nu = poisson_ratio

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
        return stress

    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary conditions
left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)

# Create JIT-compiled solver
solver_option = fe.solver.SolverOptions(linear_solver="bicgstab")
solver = fe.solver.create_solver(problem, bc, solver_option, iter_num=1)
initial = fe.utils.zero_like_initial_guess(problem, bc)

print("JIT-compiling solver...")

@jax.jit
def solve_forward(traction_array):
    """JIT-compiled forward solver"""
    internal_vars = fe.internal_vars.InternalVars(
        volume_vars=(),
        surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial)

# Warm up JIT compilation
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, 1.0)
_ = solve_forward(traction_array)
print("JIT compilation complete!\n")

# Extract mesh coordinates for plotting
coords = onp.array(mesh.points)
x_coords = coords[:, 0]
z_coords = coords[:, 2]

# Setup interactive plot
fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(bottom=0.15)

# Create slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(
    ax_slider,
    'Traction [MPa]',
    -5.0,
    5.0,
    valinit=1.0,
    valstep=0.1,
    color='steelblue'
)

# Initialize plots (will be updated)
scatter_original = None
scatter_deformed = None
colorbar = None
title_text = None

def update(traction_value):
    """Update function called when slider changes"""
    global scatter_original, scatter_deformed, colorbar, title_text

    # Create traction array
    traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(
        problem, traction_value
    )

    # Solve (instant with JIT!)
    sol = solve_forward(traction_array)
    sol_unflat = problem.unflatten_fn_sol_list(sol)
    displacement = onp.array(sol_unflat[0])

    # Extract displacement components
    u_x = displacement[:, 0]
    u_z = displacement[:, 2]
    u_mag = onp.sqrt(onp.sum(displacement**2, axis=1))

    # Calculate max values
    max_u_mag = onp.max(u_mag)
    max_u_z = onp.max(onp.abs(u_z))

    # Deformation scale
    scale_factor = 10 if max_u_mag > 0 else 1
    deformed_x = x_coords + scale_factor * u_x
    deformed_z = z_coords + scale_factor * u_z

    # Clear and redraw
    ax.clear()

    # Plot original shape
    ax.scatter(x_coords, z_coords, c='lightgray', s=2, alpha=0.3, label='Original')

    # Plot deformed shape
    scatter = ax.scatter(deformed_x, deformed_z, c=u_mag, cmap='jet',
                        s=6, alpha=0.9, vmin=0, vmax=max(max_u_mag, 0.01))

    ax.set_xlabel('X position [mm]', fontsize=12)
    ax.set_ylabel('Z position [mm]', fontsize=12)
    ax.set_title(f'Deformed Shape (Scale: {scale_factor}x) | Traction: {traction_value:.1f} MPa\n' +
                f'Max |u|: {max_u_mag:.4f} mm | Max u_z: {max_u_z:.4f} mm',
                fontsize=13, pad=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Update colorbar
    if colorbar is not None:
        colorbar.remove()
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label('|u| [mm]', fontsize=11)

    fig.canvas.draw_idle()

# Connect slider to update function
slider.on_changed(update)

# Initial plot
update(1.0)

print("=" * 60)
print("Interactive Demo Ready!")
print("=" * 60)
print("Use the slider to adjust traction and see instant results.")
print("Traction range: -5.0 to +5.0 MPa")
print("Close the window to exit.")
print("=" * 60)

plt.show()
