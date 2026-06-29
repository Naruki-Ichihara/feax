import os
# feax disables XLA GPU memory preallocation by default (set
# FEAX_PREALLOCATE=1 to re-enable). No manual os.environ line needed.
import jax.numpy as np
import jax
import feax as fe

elastic_moduli = 70e3
poisson_ratio = 0.3
traction = 1e-3
tol = 1e-5

# Define mesh
L = 1000
W = 20
H = 20
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=1)

# Locations
left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol)

# Define problem
E = elastic_moduli
nu = poisson_ratio
class LinearElasticity(fe.problem.Problem):
    def get_energy_density(self):
        def psi(u_grad, *args):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return 0.5 * lmbda * np.trace(epsilon)**2 + mu * np.sum(epsilon * epsilon)
        return psi
    def get_surface_maps(self):
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]

problem = LinearElasticity(mesh, vec=3, dim=3, location_fns=[right])

# Boundary
left_fix = fe.DCboundary.DirichletBCSpec(
    location=left,
    component="all",
    value=0.
)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)

# Internal variables
traction_array = fe.TracedParams.create_uniform_surface_var(problem, traction)
traced_params = fe.TracedParams(
    volume_vars=(),
    surface_vars=[(traction_array,)]
)

initial = fe.zero_like_initial_guess(problem, bc)

# Recommended flow: build TracedStructure FIRST. It collects the mesh-sized
# structural arrays as traced leaves (so jit does not bake them into the
# executable as constants) and, by default (free_scratch=True), releases the
# large host-side index/slot scratch arrays on the problem — important on
# unified-memory devices (e.g. GB10) where host numpy shares physical RAM with
# the GPU allocator. Pass free_scratch=False if you also need get_jacobian
# (BCOO) on the same problem afterward.
ts = fe.TracedStructure.from_problem(problem)

# Solver (auto selects based on backend and matrix property). Passing traced_structure
# routes the auto-detect sample assembly through the TracedStructure path, so it works
# regardless of the create_solver/TracedStructure order and never needs the freed
# host slot maps.
solver_opts = fe.DirectSolverOptions(verbose=True)
solver = fe.create_solver(problem, bc, solver_options=solver_opts, linear=True,
                          traced_params=traced_params, traced_structure=ts)

def solve_forward(tp, ts):
    return solver(tp, initial, traced_structure=ts)
sol = solve_forward(traced_params, ts)
sol_unflat = problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]

# Save solution
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')

fe.utils.save_sol(
    mesh=mesh,
    sol_file=vtk_path,
    point_infos=[("displacement", displacement)])