"""
Hyperelastic solver example with torsion loading.
Demonstrates solving nonlinear hyperelasticity problems with Neo-Hookean material model
under applied torque on the right face instead of prescribed rotation.
"""

import feax as fe
import jax
import jax.numpy as np
import os


class HyperElasticityTorsion(fe.problem.Problem):
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
        """Apply traction on the right face that creates a torque about the x-axis."""
        def torsion_traction(u, x, traction_magnitude):
            # Extract scalar from array if needed
            mag = traction_magnitude[0] if traction_magnitude.ndim > 0 else traction_magnitude

            # Center of the cross-section (y=0.5, z=0.5 for unit cube)
            center_y = 0.5
            center_z = 0.5

            # Vector from center to current point in y-z plane
            r_y = x[1] - center_y
            r_z = x[2] - center_z

            # Distance from center axis
            r = np.sqrt(r_y**2 + r_z**2 + 1e-10)  # Small offset to avoid division by zero

            # Tangential direction (perpendicular to radial direction in y-z plane)
            # For counterclockwise rotation about x-axis when looking from +x:
            # tangent = (-r_z, r_y) / r normalized
            t_y = -r_z / r
            t_z = r_y / r

            # Traction magnitude proportional to distance from center (like shear stress in torsion)
            # τ = T * r / J, where T is torque, r is distance, J is polar moment of inertia
            # Here we use a simplified linear distribution
            traction = mag * r

            # Traction vector (only y and z components, no x component for pure torsion)
            return np.array([0., traction * t_y, traction * t_z])

        return [torsion_traction]


mesh = fe.mesh.box_mesh((2, 1, 1), mesh_size=0.075)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 2., atol=1e-5)

def zero_dirichlet_val(point):
    return 0.

# Create boundary conditions using dataclass approach
# Left boundary - fix all components (clamped)
# Right boundary - free (traction will be applied via surface map)
bc_config = fe.DCboundary.DirichletBCConfig([
    fe.DCboundary.DirichletBCSpec(location=left, component='all', value=zero_dirichlet_val),
])

# Create problem with surface loading on right face
feax_problem = HyperElasticityTorsion(
    mesh,
    vec=3,
    dim=3,
    location_fns=[right]  # Surface where traction is applied
)

# Create boundary conditions
bc = bc_config.create_bc(feax_problem)

# Create internal variables with traction magnitude
# The magnitude controls the intensity of the torque
# Increase this value for larger rotations
traction_magnitude = 200.0  # Adjust this value to control the torque intensity

# Get number of boundary faces and quadrature points for the right surface
num_faces = len(feax_problem.boundary_inds_list[0])
num_quads = feax_problem.fes[0].face_shape_vals.shape[1]

# Create surface variable: traction magnitude at each face quadrature point
# Shape: (num_faces, num_quads, 1) for scalar parameter
surface_var = np.full((num_faces, num_quads, 1), traction_magnitude)

internal_vars = fe.internal_vars.InternalVars(
    volume_vars=(),
    surface_vars=[(surface_var,)]  # List of tuples, one per surface
)

# Solver with verbose output
solver_options = fe.solver.SolverOptions(tol=1e-8, linear_solver="bicgstab", verbose=True)
solver = fe.solver.create_solver(feax_problem, bc, solver_options)

def solve_fn(internal_vars):
    sol = solver(internal_vars, fe.utils.zero_like_initial_guess(feax_problem, bc))
    return sol

print("Solving hyperelasticity with torsion loading...")
print(f"Traction magnitude: {traction_magnitude}")
print(f"Number of boundary faces: {num_faces}")
sol = solve_fn(internal_vars)
sol_unflat = feax_problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]

# Print some statistics
print(f"Max displacement magnitude: {np.max(np.linalg.norm(displacement, axis=1)):.6f}")
print(f"Max x-displacement: {np.max(np.abs(displacement[:, 0])):.6f}")
print(f"Max y-displacement: {np.max(np.abs(displacement[:, 1])):.6f}")
print(f"Max z-displacement: {np.max(np.abs(displacement[:, 2])):.6f}")

# Save solution
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_hyper_elast_torsion.vtu')

fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])

print(f"Solution saved to: {vtk_path}")
