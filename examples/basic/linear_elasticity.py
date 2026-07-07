"""Linear elasticity: cantilever beam with a tip traction.

Canonical feax flow with the current API:

  StructuredGrid -> to_mesh()      structured HEX8 mesh (no gmsh needed)
  Problem.get_tensor_map           stress map; material comes in as TracedParams
  DirichletBCSpec / BCConfig       boundary conditions
  TracedParams                     E (volume var) + traction (surface var)
  TracedStructure FIRST            traced structural arrays + frees host scratch
  create_solver(linear=True)       one differentiable linear solve

feax disables XLA GPU memory preallocation by default (set FEAX_PREALLOCATE=1
to re-enable) and runs in float64 (FEAX_X64=0 for float32).
"""
import os

import jax.numpy as np
import feax as fe

E0 = 70e3            # Young's modulus
nu = 0.3             # Poisson's ratio
traction = 1e-3      # tip traction in +z
tol = 1e-5

# ── Mesh ─────────────────────────────────────────────────────────────────────
# Implicit structured grid (spacing 1.0) materialized to an explicit HEX8 mesh.
L, W, H = 1200, 20, 20
mesh = fe.StructuredGrid((L, W, H)).to_mesh()

left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol)


# ── Problem ──────────────────────────────────────────────────────────────────
# Material parameters arrive through TracedParams (volume_vars, in order after
# u_grad), so the solve is differentiable w.r.t. them out of the box.
class LinearElasticity(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E):
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1. + nu) * (1. - 2. * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2. * mu * epsilon
        return stress

    def get_surface_maps(self):
        # One map per location_fn; extra args come from surface_vars.
        def surface_map(u, x, traction_mag):
            return np.array([0., 0., traction_mag])
        return [surface_map]


problem = LinearElasticity(mesh, vec=3, dim=3, ele_type='HEX8',
                           location_fns=[right])

# ── Boundary conditions ──────────────────────────────────────────────────────
bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=left, component='all', value=0.),
]).create_bc(problem)

# ── Internal variables ───────────────────────────────────────────────────────
E_nodes = fe.TracedParams.create_node_var(problem, E0)
traction_array = fe.TracedParams.create_uniform_surface_var(problem, traction)
traced_params = fe.TracedParams(volume_vars=(E_nodes,),
                                surface_vars=[(traction_array,)])

# ── TracedStructure (build FIRST) ────────────────────────────────────────────
# Collects the mesh-sized structural arrays as traced leaves (jit does not bake
# them into the executable) and releases the large host-side slot maps — this
# matters on unified-memory devices (e.g. GB10). Pass it to create_solver AND
# to every solver call.
ts = fe.TracedStructure.from_problem(problem)

# ── Solver ───────────────────────────────────────────────────────────────────
# AMG (near-null-space = rigid body modes) is the memory-lean choice for large
# elasticity; swap in fe.DirectSolverOptions() for small/medium problems.
solver = fe.create_solver(
    problem, bc,
    solver_options=fe.AMGSolverOptions(near_nullspace='rigid_body', verbose=True),
    linear=True,
    traced_params=traced_params,
    traced_structure=ts,
)

initial = fe.zero_like_initial_guess(problem, bc)
sol = solver(traced_params, initial, traced_structure=ts)
displacement = problem.unflatten_fn_sol_list(sol)[0]

print(f"tip deflection: {float(displacement[:, 2].min()):.4f}")

# The solver is differentiable end-to-end, e.g. compliance sensitivity w.r.t.
# the nodal stiffness field:
#   dc = jax.grad(lambda tp: np.sum(solver(tp, initial, traced_structure=ts)**2)
#                 )(traced_params).volume_vars[0]

# ── Save ─────────────────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
fe.utils.save_sol(
    mesh=mesh,
    sol_file=os.path.join(data_dir, 'vtk/u.vtu'),
    point_infos=[('displacement', displacement)])
