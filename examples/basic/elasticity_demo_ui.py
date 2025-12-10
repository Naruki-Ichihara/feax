"""
Interactive Demo UI for Linear Elasticity Analysis
Uses JIT-compiled solver for instant computation and visualization
"""
import feax as fe
import jax
import jax.numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as onp

# Setup problem (done once, before UI)
print("Setting up FEA problem...")

# Material properties
elastic_moduli = 70e3
poisson_ratio = 0.3
tol = 1e-5

# Define mesh - smaller for faster demo
L = 100
W = 10
H = 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=1)  # Coarser mesh for speed

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

print("JIT-compiling solver (first run may take a moment)...")

@jax.jit
def solve_forward(traction_array):
    """JIT-compiled forward solver"""
    internal_vars = fe.internal_vars.InternalVars(
        volume_vars=(),
        surface_vars=[(traction_array,)]
    )
    return solver(internal_vars, initial)

# Warm up JIT compilation with initial solve
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, 1.0)
_ = solve_forward(traction_array)
print("JIT compilation complete! Solver ready for instant computation.")

# Extract mesh coordinates for plotting
coords = onp.array(mesh.points)
x_coords = coords[:, 0]
z_coords = coords[:, 2]


def analyze_and_plot(traction_value):
    """
    Solve FEA problem and create 2D visualization (side view)

    Args:
        traction_value: Applied traction in z-direction

    Returns:
        matplotlib Figure object
    """
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
    u_y = displacement[:, 1]
    u_z = displacement[:, 2]
    u_mag = onp.sqrt(u_x**2 + u_y**2 + u_z**2)

    # Calculate max values for display
    max_u_mag = onp.max(u_mag)
    max_u_z = onp.max(onp.abs(u_z))

    # Create 2D visualization
    fig = Figure(figsize=(12, 6), dpi=80)
    ax = fig.add_subplot(111)

    # Deformation scale factor
    scale_factor = 10 if max_u_mag > 0 else 1
    deformed_x = x_coords + scale_factor * u_x
    deformed_z = z_coords + scale_factor * u_z

    # Original shape (lightweight points)
    ax.scatter(x_coords, z_coords, c='lightgray', s=3, alpha=0.4, label='Original', rasterized=True)

    # Deformed shape with color
    scatter = ax.scatter(deformed_x, deformed_z, c=u_mag, cmap='jet', s=8, alpha=0.8, vmin=0, rasterized=True)

    ax.set_xlabel('X position [mm]', fontsize=11)
    ax.set_ylabel('Z position [mm]', fontsize=11)
    ax.set_title(f'Deformed Shape (Scale: {scale_factor}x) | Traction: {traction_value:.1f} MPa\n' +
                f'Max |u|: {max_u_mag:.4f} mm | Max u_z: {max_u_z:.4f} mm',
                fontsize=12, pad=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|u| [mm]', fontsize=10)

    fig.tight_layout()

    return fig


# Create Gradio interface
with gr.Blocks(title="Linear Elasticity Demo") as demo:
    gr.Markdown(
        """
        # 🔧 Linear Elasticity Analysis Demo

        **Instant FEA simulation powered by JAX JIT compilation**

        - **Geometry**: 100×10×10 mm cantilever beam
        - **Material**: Aluminum (E=70 GPa, ν=0.3)
        - **Loading**: Traction applied on right face (z-direction)
        - **Boundary**: Left face fixed (all DOFs)

        Adjust the traction slider to see results update **instantly**!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            traction_slider = gr.Slider(
                minimum=-5.0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                label="Applied Traction [MPa]",
                info="Positive = upward, Negative = downward"
            )

            gr.Markdown(
                """
                ### Tips:
                - Try values from -5 to +5 MPa
                - Watch the deformation update in real-time
                - The solver is JIT-compiled, so updates are **instant** after first run!
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
    print("Adjust the traction slider to see instant FEA results!")
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
