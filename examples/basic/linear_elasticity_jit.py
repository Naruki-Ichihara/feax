import feax as fe
import jax
import jax.numpy as np
import time
import matplotlib.pyplot as plt
import os

# Problem parameters
elastic_moduli = 70e3
poisson_ratio = 0.3
traction = 1.
tol = 1e-5

# Mesh
L = 100
W = 10
H = 10
box_size = (L, W, H)
mesh = fe.mesh.box_mesh(box_size, mesh_size=1)

# Locations
left = lambda point: np.isclose(point[0], 0., tol)
right = lambda point: np.isclose(point[0], L, tol)

# Problem definition
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
left_fix = fe.DCboundary.DirichletBCSpec(location=left, component="all", value=0.)
bc_config = fe.DCboundary.DirichletBCConfig([left_fix])
bc = bc_config.create_bc(problem)

# Solver setup
solver_option = fe.solver.SolverOptions(linear_solver="cudss")
solver_no_jit = fe.solver.create_solver(problem, bc, solver_option, iter_num=1)
solver_jit = jax.jit(fe.solver.create_solver(problem, bc, solver_option, iter_num=1))

initial = fe.utils.zero_like_initial_guess(problem, bc)
traction_array = fe.internal_vars.InternalVars.create_uniform_surface_var(problem, traction)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(), surface_vars=[(traction_array,)])

# Benchmark 1: No JIT
start = time.perf_counter()
sol_no_jit = solver_no_jit(internal_vars, initial).block_until_ready()
time_no_jit = time.perf_counter() - start

# Benchmark 2: JIT with compilation overhead (first call)
start = time.perf_counter()
sol_jit_first = solver_jit(internal_vars, initial).block_until_ready()
time_jit_with_compile = time.perf_counter() - start

# Benchmark 3: JIT without compilation overhead (second call)
start = time.perf_counter()
sol_jit_second = solver_jit(internal_vars, initial).block_until_ready()
time_jit_compiled = time.perf_counter() - start

# Calculate compilation overhead and speedup
compile_overhead = time_jit_with_compile - time_jit_compiled
speedup = time_no_jit / time_jit_compiled

# Create visualization using JAX color palette
# JAX colors: https://jax.readthedocs.io/en/latest/_static/logo_250px.png
jax_blue = '#5E94D4'    # JAX blue
jax_orange = '#F26B38'  # JAX orange

fig, ax = plt.subplots(figsize=(8, 6))

# Stacked bar chart showing compilation overhead
categories = ['No JIT', 'JIT\n(1st call)', 'JIT\n(2nd call)']
execution_times = [time_no_jit, time_jit_compiled, time_jit_compiled]
compile_times = [0, compile_overhead, 0]

x_pos = np.arange(len(categories))
bars1 = ax.bar(x_pos, execution_times, color=jax_blue, alpha=0.85, label='Execution', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x_pos, compile_times, bottom=execution_times, color=jax_orange, alpha=0.85, label='Compilation', edgecolor='white', linewidth=1.5)

ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title(f'JIT Compilation Performance (Speedup: {speedup:.1f}×)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Add value labels on bars
for i, (exec_t, comp_t) in enumerate(zip(execution_times, compile_times)):
    total = exec_t + comp_t
    # Total time on top
    ax.text(i, total + 0.15, f'{total:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Execution time in middle
    if exec_t > 0.5:
        ax.text(i, exec_t/2, f'{exec_t:.2f}s', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    # Compilation time if significant
    if comp_t > 0.5:
        ax.text(i, exec_t + comp_t/2, f'+{comp_t:.2f}s', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Add speedup annotation
ax.text(0.98, 0.97, f'{speedup:.1f}× faster\nafter compilation',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor=jax_blue, alpha=0.2, edgecolor=jax_blue, linewidth=2))

plt.tight_layout()

# Save figure
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
fig_path = os.path.join(data_dir, 'jit_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')

# Save solution
sol_unflat = problem.unflatten_fn_sol_list(sol_jit_second)
displacement = sol_unflat[0]
os.makedirs(os.path.join(data_dir, 'vtk'), exist_ok=True)
vtk_path = os.path.join(data_dir, 'vtk/u_jit.vtu')
fe.utils.save_sol(mesh=mesh, sol_file=vtk_path, point_infos=[("displacement", displacement)])

print(f"No JIT: {time_no_jit:.3f}s | JIT (1st): {time_jit_with_compile:.3f}s | JIT (2nd): {time_jit_compiled:.3f}s | Speedup: {speedup:.2f}x")
print(f"Figure: {fig_path}")