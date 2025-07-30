import jax
import jax.numpy as np
from feax import logger
from feax.problem import Problem as FeaxProblem
from feax.mesh import Mesh, box_mesh_gmsh
from feax.assembler import get_res
from feax.DCboundary import DirichletBC, apply_boundary_to_res, create_J_bc_updater
from feax.solver import newton_solve, SolverOptions
import time

class HyperElasticityFeax(FeaxProblem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.
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
        def first_PK_stress(u_grad, internal_vars=None):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
    
meshio_mesh = box_mesh_gmsh(15, 15, 15, 1., 1., 1., data_dir='/tmp', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)


def right(point):
    return np.isclose(point[0], 1, atol=1e-5)


def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


dirichlet_bc_info = [[left] * 3 + [right] * 3, [0, 1, 2] * 2,
                     [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] +
                     [zero_dirichlet_val] * 3]

feax_problem = HyperElasticityFeax(mesh,
                          vec=3,
                          dim=3,
                          dirichlet_bc_info=dirichlet_bc_info)

bc = DirichletBC.from_bc_info(feax_problem, dirichlet_bc_info)

initial_sol = np.zeros(feax_problem.num_total_dofs_all_vars)
initial_sol = initial_sol.at[bc.bc_rows].set(bc.bc_vals)
initial_sol_unflatten = feax_problem.unflatten_fn_sol_list(initial_sol)

J_fn = jax.jit(create_J_bc_updater(feax_problem, bc))

@jax.jit
def solve_fn(initial_sol):
    @jax.jit
    def J_bc(dofs):
        return J_fn(dofs)
    @jax.jit
    def res_bc(sol_flat):
        sol_unflat = feax_problem.unflatten_fn_sol_list(sol_flat)
        res = get_res(feax_problem, sol_unflat)
        res_flat = jax.flatten_util.ravel_pytree(res)[0]
        return apply_boundary_to_res(bc, res_flat, sol_flat)
    sol, res, rel_res, iter  = newton_solve(J_bc, res_bc, initial_sol,
                        SolverOptions(tol = 1e-6, linear_solver="bicgstab", x0_strategy="bc_aware", bc_rows=bc.bc_rows, bc_vals=bc.bc_vals))
    return sol

start = time.time()
sol = solve_fn(initial_sol)
end = time.time()

print(f"Solve took: {end-start} s.")

# Recall
start = time.time()
sol = solve_fn(initial_sol)
end = time.time()

sol_unflat = feax_problem.unflatten_fn_sol_list(sol)
displacement = sol_unflat[0]
# Save solution
from feax.utils import save_sol
save_sol(
    mesh=mesh,
    sol_file="/workspace/solution.vtk",
    point_infos=[("displacement", displacement)])

print(f"Solve took: {end-start} s.")
"""
# Test with 3 batched initial solutions
print("\n=== Testing vmapped solver with 3 batched initial solutions ===")

# Create 3 different initial solutions
# 1. Base initial solution (with BC values)
initial_sol_1 = initial_sol.copy()

# 2. Perturbed initial solution (small random perturbation)
key = jax.random.PRNGKey(42)
perturbation = 0.01 * jax.random.normal(key, shape=initial_sol.shape)
initial_sol_2 = initial_sol + perturbation
# Keep BC values unchanged
initial_sol_2 = initial_sol_2.at[bc.bc_rows].set(bc.bc_vals)

# 3. Scaled initial solution (different magnitude)
initial_sol_3 = initial_sol * 1.5
# Keep BC values unchanged
initial_sol_3 = initial_sol_3.at[bc.bc_rows].set(bc.bc_vals)

# Stack initial solutions for vmap
batched_initial_sols = np.stack([initial_sol_1, initial_sol_2, initial_sol_3])

# Solve all 3 problems in parallel
print("Solving 3 problems in parallel with different initial solutions...")
start_vmap = time.time()
batched_sols = solve_vmap_fn(batched_initial_sols)
end_vmap = time.time()

print(f"Vmap solve took: {end_vmap - start_vmap:.3f} s")

# Extract individual solutions and compute residuals
for i in range(3):
    sol_i = batched_sols[i]
    sol_unflat_i = feax_problem.unflatten_fn_sol_list(sol_i)
    res_i = get_res(feax_problem, sol_unflat_i)
    res_flat_i = jax.flatten_util.ravel_pytree(res_i)[0]
    res_bc_i = apply_boundary_to_res(bc, res_flat_i, sol_i)
    res_norm_i = np.linalg.norm(res_bc_i)
    
    print(f"\nInitial solution {i+1}:")
    print(f"  - Initial sol norm: {np.linalg.norm(batched_initial_sols[i]):.6f}")
    print(f"  - Final residual norm: {res_norm_i:.2e}")
    print(f"  - Max displacement: {np.max(np.abs(sol_unflat_i[0])):.6f}")

# Check if all solutions converged to the same result
print("\n=== Consistency check ===")
for i in range(1, 3):
    diff = np.max(np.abs(batched_sols[0] - batched_sols[i]))
    print(f"Max difference between solution 1 and solution {i+1}: {diff:.2e}")

# Save the first solution for visualization
sol_unflat = feax_problem.unflatten_fn_sol_list(batched_sols[0])
displacement = sol_unflat[0]
"""