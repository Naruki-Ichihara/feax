"""
Hyperelastic solver example using energy density formulation.
Demonstrates solving nonlinear hyperelasticity problems with Neo-Hookean material model.
A torsional surface traction is applied to the right face (load-controlled twist).

This variant solves the Newton tangent systems with algebraic multigrid (AMG):
an outer GMRES preconditioned by a smoothed-aggregation hierarchy (PyAMG -> AMJax)
with the rigid-body near-null-space auto-generated from the mesh. See
``AMGSolverOptions`` below. Requires the optional ``feax[amg]`` extra.
"""

import os
import jax.numpy as np
import feax as fe
#fe.enable_x64(False)


# Box geometry
Lx, Ly, Lz = 5., 1., 1.
mesh_size   = 0.05

# Cross-section centroid of the right face (used in torsional traction)
y_c = Ly / 2.
z_c = Lz / 2.

# Torsional traction magnitude
T = 20.


class HyperElasticityFeax(fe.Problem):
    def get_energy_density(self):
        def psi(u_grad, *_):
            E = 100.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            I = np.eye(self.dim)
            F = u_grad + I
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            return (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.

        return psi

    def get_surface_maps(self):
        def traction_map(u_grad, surface_quad_point, traction_magnitude):
            # Torsional traction about x-axis: tangential in yz-plane
            y = surface_quad_point[1]
            z = surface_quad_point[2]
            return np.array([0., -traction_magnitude * (z - z_c), traction_magnitude * (y - y_c)])
        return [traction_map]


mesh = fe.mesh.box_mesh((Lx, Ly, Lz), mesh_size=mesh_size)

# Boundary locations
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

# Fix left face (clamped)
bc_config = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left, component='all', value=0.)
])

feax_problem = HyperElasticityFeax(mesh, vec=3, dim=3, location_fns=[right])

traction_surface = fe.TracedParams.create_uniform_surface_var(feax_problem, T)
traced_params = fe.TracedParams(
    volume_vars=[],
    surface_vars=[(traction_surface,)]
)

bc = bc_config.create_bc(feax_problem)

ts = fe.TracedStructure.from_problem(feax_problem)

# Algebraic multigrid solver (smoothed aggregation via PyAMG -> AMJax) used as a
# preconditioner for an outer GMRES. For this 3D elasticity (vec == dim) the
# rigid-body near-null-space is generated automatically from the mesh node
# coordinates -- this is what makes AMG effective on (hyper)elasticity. With
# rebuild_every=None (default) the hierarchy is reused across Newton steps and
# rebuilt only when the large torsional deformation makes it stale (adaptive
# lag), so the AMG setup cost is amortized over the nonlinear iteration.
# (Requires the optional `feax[amg]` extra: amjax + pyamg.)
solver_options = fe.AMGSolverOptions(
    near_nullspace="rigid_body",   # rigid body modes from mesh coords (auto for vec==dim)
    solver="gmres",                # outer Krylov (robust for the changing tangent)
    rebuild_every=None,            # adaptive lag: rebuild only when the precond goes stale
)
newton_options = fe.NewtonOptions(verbose=True)
solver = fe.create_solver(
    feax_problem,
    bc,
    solver_options,
    traced_params=traced_params,
    newton_options=newton_options,
    traced_structure=ts
)

sol = solver(traced_params, fe.zero_like_initial_guess(feax_problem, bc),
             traced_structure=ts)
displacement = sol.field(0)          # solver returns a fe.Solution

# Save solution
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_hyper_elast.vtu')

fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])
