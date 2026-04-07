"""Distributed linear elasticity: solve independent problems on each GPU.

Each node solves a cantilever beam with different traction loads
simultaneously, demonstrating multi-node task parallelism.

Usage:
    python3 -m feax.distributed -c distributed.yml examples/distributed_linear_elasticity.py
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as np

from feax import distributed

try:
    distributed.initialize()

    process_id = jax.process_index()
    local_dev = jax.local_devices()[0]

    # Each node gets a different traction load
    tractions = [1e-3, 5e-3]
    traction = tractions[process_id]
    print(f"[Node {process_id}] Solving with traction = {traction}")

    # Place computation on local device
    with jax.default_device(local_dev):
        import feax as fe

        # Material
        E = 70e3
        nu = 0.3
        tol = 1e-5

        # Mesh
        L, W, H = 100, 10, 10
        mesh = fe.mesh.box_mesh((L, W, H), mesh_size=1)

        # Boundaries
        left = lambda point: np.isclose(point[0], 0., tol)
        right = lambda point: np.isclose(point[0], L, tol)

        # Problem
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

        # Dirichlet BC
        left_fix = fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
        bc = fe.DCboundary.DirichletBCConfig([left_fix]).create_bc(problem)

        # Internal variables
        traction_array = fe.InternalVars.create_uniform_surface_var(problem, traction)
        internal_vars = fe.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

        # Solve
        solver_opts = fe.DirectSolverOptions()
        solver = fe.create_solver(problem, bc, solver_options=solver_opts, iter_num=1, internal_vars=internal_vars)
        initial = fe.zero_like_initial_guess(problem, bc)
        sol = solver(internal_vars, initial)

        # Results
        displacement = problem.unflatten_fn_sol_list(sol)[0]
        max_disp = np.max(np.linalg.norm(displacement, axis=1))
        print(f"[Node {process_id}] Max displacement: {max_disp:.6f}")
        print(f"[Node {process_id}] Displacement shape: {displacement.shape}")

        # Gather all results to every process
        all_displacements = distributed.gather(displacement)

        # Save on coordinator only
        if distributed.is_coordinator():
            data_dir = os.path.join(os.path.dirname(__file__), 'data', 'vtk')
            os.makedirs(data_dir, exist_ok=True)
            for i, t in enumerate(tractions):
                vtk_path = os.path.join(data_dir, f'distributed_traction{t}.vtu')
                fe.utils.save_sol(
                    mesh=mesh,
                    sol_file=vtk_path,
                    point_infos=[("displacement", all_displacements[i])]
                )
                print(f"[Coordinator] Saved traction={t} to {vtk_path}")

finally:
    distributed.finalize()
