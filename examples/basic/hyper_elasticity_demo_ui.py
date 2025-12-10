"""
Interactive Demo UI for Hyperelastic Torsion Analysis
Uses JIT-compiled Newton solver for fast nonlinear computation and visualization
"""
import feax as fe
import jax
import jax.numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as onp

# Setup problem (done once, before UI)
print("Setting up Hyperelastic FEA problem...")

# Material properties
E = 100.  # Young's modulus
nu = 0.3  # Poisson's ratio
tol = 1e-5

# Define mesh - unit cube
L = 1.0
mesh = fe.mesh.box_mesh((L, L, L), mesh_size=0.05)

# Locations
left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol)


class HyperElasticityTorsion(fe.problem.Problem):
    def get_tensor_map(self):
        def psi(F):
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
        """Apply traction on the right face that creates a torque about the x-axis."""
        def torsion_traction(u, x, traction_magnitude):
            # Extract scalar from array if needed
            mag = traction_magnitude[0] if traction_magnitude.ndim > 0 else traction_magnitude

            # Center of the cross-section
            center_y = 0.5
            center_z = 0.5

            # Vector from center to current point in y-z plane
            r_y = x[1] - center_y
            r_z = x[2] - center_z

            # Distance from center axis
            r = np.sqrt(r_y**2 + r_z**2 + 1e-10)

            # Tangential direction
            t_y = -r_z / r
            t_z = r_y / r

            # Traction magnitude proportional to distance from center
            traction = mag * r

            return np.array([0., traction * t_y, traction * t_z])

        return [torsion_traction]


problem = HyperElasticityTorsion(mesh, vec=3, dim=3, location_fns=[right])

# Boundary conditions - fix left face
left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)

# Create JIT-compiled Newton solver for nonlinear problem
solver_option = fe.solver.SolverOptions(tol=1e-6, linear_solver="bicgstab")
solver = fe.solver.create_solver(problem, bc, solver_option)  # Newton solver (no iter_num=1)
initial = fe.utils.zero_like_initial_guess(problem, bc)

print("JIT-compiling Newton solver (first run may take longer for nonlinear problem)...")

@jax.jit
def solve_forward(traction_array):
    """JIT-compiled forward solver for hyperelasticity"""
    internal_vars = fe.internal_vars.InternalVars(
        volume_vars=(),
        surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial)

# Warm up JIT compilation with initial solve
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, 1.0)
_ = solve_forward(traction_array)
print("JIT compilation complete! Solver ready for fast computation.")

# Extract mesh coordinates for plotting
coords = onp.array(mesh.points)
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]


def analyze_and_plot(traction_value):
    """
    Solve hyperelastic FEA problem and create visualization

    Args:
        traction_value: Applied torque intensity

    Returns:
        matplotlib Figure object
    """
    # Create traction array
    traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(
        problem, traction_value
    )

    # Solve (fast with JIT!)
    sol = solve_forward(traction_array)
    sol_unflat = problem.unflatten_fn_sol_list(sol)
    displacement = onp.array(sol_unflat[0])

    # Extract displacement components
    u_x = displacement[:, 0]
    u_y = displacement[:, 1]
    u_z = displacement[:, 2]
    u_mag = onp.sqrt(u_x**2 + u_y**2 + u_z**2)

    # Get right face nodes only
    right_mask = onp.isclose(x_coords, L, atol=tol)
    right_y = y_coords[right_mask]
    right_z = z_coords[right_mask]
    right_u_y = u_y[right_mask]
    right_u_z = u_z[right_mask]
    right_u_mag = u_mag[right_mask]

    # Calculate max values for display
    max_u_mag = onp.max(right_u_mag) if len(right_u_mag) > 0 else 0.0

    # Estimate rotation angle
    if onp.any(right_mask):
        theta_from_y = -right_u_y / (right_z - 0.5 + 1e-10)
        valid_mask = onp.abs(right_z - 0.5) > 0.1
        if onp.any(valid_mask):
            avg_theta = onp.mean(theta_from_y[valid_mask])
            rotation_deg = onp.rad2deg(avg_theta)
        else:
            rotation_deg = 0.0
    else:
        rotation_deg = 0.0

    # Create visualization - right face only (Y-Z plane)
    fig = Figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)

    # Deformed positions (no scale factor - show actual deformation)
    deformed_y = right_y + right_u_y
    deformed_z = right_z + right_u_z

    # Original shape - outline of square
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k--', linewidth=2, label='Original', alpha=0.5)

    # Original positions
    ax.scatter(right_y, right_z, c='gray', s=30, alpha=0.3, marker='o')

    # Deformed positions with color
    scatter = ax.scatter(deformed_y, deformed_z, c=right_u_mag, cmap='jet',
                        s=50, alpha=0.9, vmin=0, edgecolors='k', linewidths=0.5)

    # Draw lines connecting original to deformed (for selected points)
    # Sample every few points to avoid clutter
    step = max(1, len(right_y) // 20)
    for i in range(0, len(right_y), step):
        ax.plot([right_y[i], deformed_y[i]], [right_z[i], deformed_z[i]],
               'b-', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Y position', fontsize=12)
    ax.set_ylabel('Z position', fontsize=12)
    ax.set_title(f'Right Face Deformation (Y-Z plane)\n'
                f'Torque: {traction_value:.1f} | Rotation: {rotation_deg:.1f}° | Max |u|: {max_u_mag:.4f}',
                fontsize=13)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    margin = 0.15
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|u|', fontsize=11)

    fig.tight_layout()

    return fig


# Create Gradio interface
with gr.Blocks(title="Hyperelastic Torsion Demo") as demo:
    gr.Markdown(
        """
        # Hyperelastic Torsion Analysis Demo

        **Nonlinear FEA simulation powered by JAX JIT compilation**

        - **Geometry**: 1×1×1 unit cube
        - **Material**: Neo-Hookean hyperelastic (E=100, nu=0.3)
        - **Loading**: Torsion (torque) applied on right face
        - **Boundary**: Left face fixed (all DOFs)
        - **Solver**: Newton-Raphson for nonlinear equilibrium

        Adjust the torque slider to see the twist deformation!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            traction_slider = gr.Slider(
                minimum=-100.0,
                maximum=100.0,
                value=50.0,
                step=5.0,
                label="Torque Intensity",
                info="Positive = counterclockwise (from +X), Negative = clockwise"
            )

            gr.Markdown(
                """
                ### Tips:
                - Try values from -100 to +100
                - Watch the twist deformation in the end view
                - Higher values cause larger nonlinear effects
                - The Newton solver handles large deformations
                """
            )

        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Analysis Results")

    # Update plot when slider changes
    traction_slider.change(
        fn=analyze_and_plot,
        inputs=traction_slider,
        outputs=plot_output
    )

    # Initial plot on load
    demo.load(
        fn=analyze_and_plot,
        inputs=traction_slider,
        outputs=plot_output
    )

# Launch the app
if __name__ == "__main__":
    print("\nLaunching Gradio demo...")
    print("The UI will open in your browser automatically.")
    print("Adjust the torque slider to see hyperelastic torsion results!")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
