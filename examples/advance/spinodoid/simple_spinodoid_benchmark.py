"""Simple script to generate and analyze a single spinodoid - for benchmarking."""
import jax
import jax.numpy as np
import feax as fe
import feax.flat as flat
import time
import matplotlib.pyplot as plt
import os

# Mesh setup - same as batch generation
class SpinodoidUnitCell(flat.unitcell.UnitCell):
    def mesh_build(self, mesh_size):
        return fe.mesh.box_mesh(size=1.0, mesh_size=mesh_size, element_type='HEX8')

unitcell = SpinodoidUnitCell(mesh_size=0.04)
mesh = unitcell.mesh
pairings = flat.pbc.periodic_bc_3D(unitcell, vec=1, dim=3)
P = flat.pbc.prolongation_matrix(pairings, mesh, vec=1)
cell_centers = np.mean(mesh.points[mesh.cells], axis=1)

# Parameters - same as batch generation
radius = 0.1
target_vf = 0.5
beta_heaviside = 10.0
beta_grf = 15.0
N = 100
theta1, theta2, theta3 = np.pi/6, np.pi/8, np.pi/10
E0 = 21e3
nu_val = 0.3
p = 3.0
solver_opts = fe.solver.SolverOptions(tol=1e-8, linear_solver="cg", verbose=False)

# Generate direction vectors and random phases
key = jax.random.PRNGKey(42)
key, key_n, key_g = jax.random.split(key, 3)
n_vectors = flat.spinodoid.generate_direction_vectors(theta1, theta2, theta3, N, key_n)
gamma = jax.random.uniform(key_g, (N,), minval=0.0, maxval=2*np.pi)

# Setup elasticity problem
class ElasticityProblem(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, E, nu):
            mu = E / (2 * (1 + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(3) + 2 * mu * eps
        return stress

problem = ElasticityProblem(mesh, vec=3, dim=3, ele_type='HEX8', location_fns=[])
bc = fe.DCboundary.DirichletBCConfig([]).create_bc(problem)
pairings_3d = flat.pbc.periodic_bc_3D(unitcell, vec=3, dim=3)
P_3d = flat.pbc.prolongation_matrix(pairings_3d, mesh, vec=3)

# Create homogenization solver
compute_C_hom = flat.solver.create_homogenization_solver(
    problem, bc, P_3d, solver_opts, mesh, dim=3
)

print("=" * 70)
print("BENCHMARK: Single Spinodoid Generation & Analysis")
print("=" * 70)
print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} cells")
print(f"Parameters: N={N}, radius={radius}, beta_grf={beta_grf}, beta_heaviside={beta_heaviside}")
print(f"Thetas: ({theta1:.4f}, {theta2:.4f}, {theta3:.4f}) rad")
print(f"Target VF: {target_vf}")
print()

# ============================================================================
# VERSION 1: WITHOUT JIT (Step-by-step timing)
# ============================================================================
print("-" * 70)
print("VERSION 1: Without JIT (step-by-step)")
print("-" * 70)

# 1. GRF evaluation
t0 = time.time()
rho = flat.spinodoid.evaluate_grf_field(cell_centers, n_vectors, gamma, beta_grf)
t1 = time.time()
print(f"1. GRF evaluation:        {(t1-t0)*1000:.2f} ms")

# 2. Helmholtz filtering
rho = flat.filters.helmholtz_filter(rho, mesh, radius, P, solver_opts)
t2 = time.time()
print(f"2. Helmholtz filter:       {(t2-t1)*1000:.2f} ms")

# 3. Normalization
rho = (rho - rho.min()) / (rho.max() - rho.min())
t3 = time.time()
print(f"3. Normalization:          {(t3-t2)*1000:.2f} ms")

# 4. Threshold computation
thresh = flat.filters.compute_volume_fraction_threshold(rho, target_vf)
t4 = time.time()
print(f"4. Threshold computation:  {(t4-t3)*1000:.2f} ms")

# 5. Heaviside projection
rho = flat.filters.heaviside_projection(rho, beta_heaviside, thresh)
t5 = time.time()
print(f"5. Heaviside projection:   {(t5-t4)*1000:.2f} ms")

print(f"\n   Total generation time:  {(t5-t0)*1000:.2f} ms")
print(f"   Final VF: {float(rho.mean()):.4f}")

# 6. Homogenization
rho_cell = np.mean(rho[mesh.cells], axis=1)
E_field = E0 * rho_cell**p
E_array = fe.internal_vars.InternalVars.create_cell_var(problem, E_field)
nu_array = fe.internal_vars.InternalVars.create_cell_var(problem, nu_val)
internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_array, nu_array), surface_vars=())

t6 = time.time()
C_hom_v1 = compute_C_hom(internal_vars)
t7 = time.time()
print(f"6. Homogenization:         {(t7-t6)*1000:.2f} ms")

total_v1 = (t7-t0)*1000
print(f"\n   >>> TOTAL TIME (v1):    {total_v1:.2f} ms <<<")

# ============================================================================
# VERSION 2: WITH JIT (Full pipeline compilation)
# ============================================================================
print("\n" + "-" * 70)
print("VERSION 2: With JIT (full pipeline)")
print("-" * 70)

# Define complete generation pipeline
def generate_spinodoid(n_vectors, gamma, target_vf):
    rho = flat.spinodoid.evaluate_grf_field(cell_centers, n_vectors, gamma, beta_grf)
    rho = flat.filters.helmholtz_filter(rho, mesh, radius, P, solver_opts)
    rho = (rho - rho.min()) / (rho.max() - rho.min())
    thresh = flat.filters.compute_volume_fraction_threshold(rho, target_vf)
    return flat.filters.heaviside_projection(rho, beta_heaviside, thresh)

# Define complete homogenization pipeline
def compute_single_C_hom(rho_nodal):
    rho_cell = np.mean(rho_nodal[mesh.cells], axis=1)
    E_field = E0 * rho_cell**p
    E_array = fe.internal_vars.InternalVars.create_cell_var(problem, E_field)
    nu_array = fe.internal_vars.InternalVars.create_cell_var(problem, nu_val)
    internal_vars = fe.internal_vars.InternalVars(volume_vars=(E_array, nu_array), surface_vars=())
    return compute_C_hom(internal_vars)

# JIT compile the pipelines
generate_spinodoid_jit = jax.jit(generate_spinodoid)
compute_C_hom_jit = jax.jit(compute_single_C_hom)

# First run (includes compilation time)
print("\nFirst run (includes JIT compilation):")
t0_jit = time.time()
rho_jit = generate_spinodoid_jit(n_vectors, gamma, target_vf)
t1_jit = time.time()
print(f"  Generation (+ compile):  {(t1_jit-t0_jit)*1000:.2f} ms")

C_hom_jit = compute_C_hom_jit(rho_jit)
t2_jit = time.time()
print(f"  Homogenization (+ compile): {(t2_jit-t1_jit)*1000:.2f} ms")
print(f"  Total (+ compile):       {(t2_jit-t0_jit)*1000:.2f} ms")

# Second run (pure execution, no compilation)
print("\nSecond run (JIT compiled, no compilation overhead):")
t3_jit = time.time()
rho_jit2 = generate_spinodoid_jit(n_vectors, gamma, target_vf)
t4_jit = time.time()
gen_time = (t4_jit-t3_jit)*1000
print(f"  Generation:              {gen_time:.2f} ms")

C_hom_jit2 = compute_C_hom_jit(rho_jit2)
t5_jit = time.time()
homo_time = (t5_jit-t4_jit)*1000
print(f"  Homogenization:          {homo_time:.2f} ms")

total_v2 = (t5_jit-t3_jit)*1000
print(f"\n   >>> TOTAL TIME (v2):    {total_v2:.2f} ms <<<")
print(f"   Final VF: {float(rho_jit2.mean()):.4f}")

# ============================================================================
# VERSION 3: WITH JIT + VMAP (Batch processing)
# ============================================================================
print("\n" + "-" * 70)
print("VERSION 3: With JIT + VMAP (batch processing)")
print("-" * 70)

# Create vmapped and jitted functions
generate_spinodoid_batch = jax.jit(jax.vmap(generate_spinodoid))
compute_C_hom_batch = jax.jit(jax.vmap(compute_single_C_hom))

batch_sizes = [5, 10, 15]
batch_results = []

for batch_size in batch_sizes:
    print(f"\n--- Batch size: {batch_size} ---")

    # Generate batch of random parameters
    key_batch = jax.random.PRNGKey(100 + batch_size)
    keys = jax.random.split(key_batch, batch_size)

    # Create batch of direction vectors and phases
    n_vecs_batch = []
    gammas_batch = []
    target_vfs = []

    for i, key in enumerate(keys):
        key_n, key_g, key_vf = jax.random.split(key, 3)
        # Use same thetas for fair comparison
        n_vecs_batch.append(flat.spinodoid.generate_direction_vectors(
            theta1, theta2, theta3, N, key_n
        ))
        gammas_batch.append(jax.random.uniform(key_g, (N,), minval=0.0, maxval=2*np.pi))
        target_vfs.append(0.5)  # Same target VF

    n_vecs_batch = np.stack(n_vecs_batch)
    gammas_batch = np.stack(gammas_batch)
    target_vfs = np.array(target_vfs)

    # First run (includes compilation)
    t0_batch = time.time()
    rho_batch = generate_spinodoid_batch(n_vecs_batch, gammas_batch, target_vfs)
    t1_batch = time.time()
    compile_gen = (t1_batch - t0_batch) * 1000
    print(f"  Generation (+ compile):     {compile_gen:.2f} ms")

    C_hom_batch_result = compute_C_hom_batch(rho_batch)
    t2_batch = time.time()
    compile_homo = (t2_batch - t1_batch) * 1000
    print(f"  Homogenization (+ compile): {compile_homo:.2f} ms")
    print(f"  Total (+ compile):          {(t2_batch - t0_batch)*1000:.2f} ms")

    # Second run (pure execution)
    t3_batch = time.time()
    rho_batch2 = generate_spinodoid_batch(n_vecs_batch, gammas_batch, target_vfs)
    t4_batch = time.time()
    exec_gen = (t4_batch - t3_batch) * 1000
    print(f"  Generation (exec only):     {exec_gen:.2f} ms")

    C_hom_batch_result2 = compute_C_hom_batch(rho_batch2)
    t5_batch = time.time()
    exec_homo = (t5_batch - t4_batch) * 1000
    print(f"  Homogenization (exec only): {exec_homo:.2f} ms")

    total_batch = (t5_batch - t3_batch) * 1000
    per_sample = total_batch / batch_size
    print(f"  Total (exec only):          {total_batch:.2f} ms")
    print(f"  Per sample:                 {per_sample:.2f} ms")

    # Calculate speedup vs v1 (without JIT)
    speedup_vs_v1 = total_v1 / per_sample
    speedup_vs_v2 = total_v2 / per_sample

    print(f"  Speedup vs v1 (no JIT):     {speedup_vs_v1:.2f}x")
    print(f"  Speedup vs v2 (JIT):        {speedup_vs_v2:.2f}x")

    batch_results.append({
        'batch_size': batch_size,
        'total_time': total_batch,
        'per_sample': per_sample,
        'gen_time': exec_gen,
        'homo_time': exec_homo,
        'speedup_vs_v1': speedup_vs_v1,
        'speedup_vs_v2': speedup_vs_v2
    })

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 70)
print(f"Single sample (no JIT):    {total_v1:>10.2f} ms")
print(f"Single sample (JIT):       {total_v2:>10.2f} ms")
print(f"Speedup (JIT vs no JIT):   {total_v1/total_v2:>10.2f}x")
print()
print("Batch processing (JIT + VMAP):")
print("-" * 70)
print(f"{'Batch Size':<12} {'Total (ms)':<12} {'Per Sample (ms)':<16} {'Speedup vs v1':<15} {'Speedup vs v2':<15}")
print("-" * 70)
for result in batch_results:
    print(f"{result['batch_size']:<12} {result['total_time']:<12.2f} {result['per_sample']:<16.2f} "
          f"{result['speedup_vs_v1']:<15.2f}x {result['speedup_vs_v2']:<15.2f}x")
print("=" * 70)

# Find best batch configuration
best_result = min(batch_results, key=lambda x: x['per_sample'])
print(f"\nBest performance: Batch size {best_result['batch_size']} "
      f"({best_result['per_sample']:.2f} ms/sample, {best_result['speedup_vs_v1']:.2f}x speedup)")

# Verify results match
print(f"\nResults verification:")
print(f"  Max difference in C_hom: {float(np.max(np.abs(C_hom_v1 - C_hom_jit2))):.2e}")
print(f"\nStiffness tensor (GPa) [JIT version]:")
print(f"C11={float(C_hom_jit2[0,0])/1e3:.2f}  C12={float(C_hom_jit2[0,1])/1e3:.2f}  C13={float(C_hom_jit2[0,2])/1e3:.2f}")
print(f"C22={float(C_hom_jit2[1,1])/1e3:.2f}  C23={float(C_hom_jit2[1,2])/1e3:.2f}")
print(f"C33={float(C_hom_jit2[2,2])/1e3:.2f}")
print(f"C44={float(C_hom_jit2[3,3])/1e3:.2f}  C55={float(C_hom_jit2[4,4])/1e3:.2f}  C66={float(C_hom_jit2[5,5])/1e3:.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("Creating performance visualization plots...")
print("=" * 70)

# Prepare data for plotting
batch_sizes_plot = [result['batch_size'] for result in batch_results]
per_sample_times = [result['per_sample'] for result in batch_results]
gen_times = [result['gen_time'] for result in batch_results]
homo_times = [result['homo_time'] for result in batch_results]
speedups_v1 = [result['speedup_vs_v1'] for result in batch_results]
speedups_v2 = [result['speedup_vs_v2'] for result in batch_results]

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Spinodoid Generation & Homogenization: Performance Benchmark',
             fontsize=16, fontweight='bold')

# Plot 1: Time per sample vs batch size
ax1 = axes[0, 0]
ax1.plot([0] + batch_sizes_plot, [total_v1] + per_sample_times, 'o-',
         linewidth=2, markersize=8, label='With JIT+VMAP')
ax1.axhline(y=total_v1, color='r', linestyle='--', linewidth=2,
            label=f'No JIT (baseline): {total_v1:.0f} ms')
ax1.axhline(y=total_v2, color='g', linestyle='--', linewidth=2,
            label=f'JIT only: {total_v2:.0f} ms')
ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time per Sample (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Time per Sample vs Batch Size', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks([0, 2, 5, 10])
ax1.set_xticklabels(['1 (no vmap)', '2', '5', '10'])

# Plot 2: Speedup vs batch size
ax2 = axes[0, 1]
ax2.plot(batch_sizes_plot, speedups_v1, 's-', linewidth=2, markersize=8,
         color='#2E86AB', label='vs No JIT')
ax2.plot(batch_sizes_plot, speedups_v2, '^-', linewidth=2, markersize=8,
         color='#A23B72', label='vs JIT only')
ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Factor vs Batch Size', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xticks(batch_sizes_plot)

# Plot 3: Breakdown of generation vs homogenization time
ax3 = axes[1, 0]
width = 0.35
x_pos = range(len(batch_sizes_plot))
ax3.bar([x - width/2 for x in x_pos], gen_times, width,
        label='Generation', color='#F18F01', alpha=0.8)
ax3.bar([x + width/2 for x in x_pos], homo_times, width,
        label='Homogenization', color='#C73E1D', alpha=0.8)
ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax3.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax3.set_title('Time Breakdown: Generation vs Homogenization', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(batch_sizes_plot)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Total execution time vs batch size
ax4 = axes[1, 1]
total_times = [result['total_time'] for result in batch_results]
ax4.plot(batch_sizes_plot, total_times, 'o-', linewidth=2, markersize=8,
         color='#6A4C93', label='Total time (batch)')
# Add reference line for sequential processing
sequential_times = [total_v2 * bs for bs in batch_sizes_plot]
ax4.plot(batch_sizes_plot, sequential_times, 's--', linewidth=2, markersize=6,
         color='orange', alpha=0.7, label='Sequential (JIT only)')
ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Time (ms)', fontsize=12, fontweight='bold')
ax4.set_title('Total Execution Time vs Batch Size', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xticks(batch_sizes_plot)

plt.tight_layout()

# Save figure
output_dir = os.path.join(os.path.dirname(__file__), "data", "plots")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "spinodoid_benchmark.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nBenchmark plot saved to: {output_path}")

# Create a summary table plot
fig2, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = [
    ['Method', 'Batch Size', 'Total Time (ms)', 'Per Sample (ms)', 'Speedup vs v1'],
    ['No JIT', '1', f'{total_v1:.2f}', f'{total_v1:.2f}', '1.00×'],
    ['JIT only', '1', f'{total_v2:.2f}', f'{total_v2:.2f}', f'{total_v1/total_v2:.2f}×'],
]

for result in batch_results:
    table_data.append([
        'JIT + VMAP',
        str(result['batch_size']),
        f"{result['total_time']:.2f}",
        f"{result['per_sample']:.2f}",
        f"{result['speedup_vs_v1']:.2f}×"
    ])

# Add best result row
table_data.append([
    '**BEST**',
    f"**{best_result['batch_size']}**",
    f"**{best_result['total_time']:.2f}**",
    f"**{best_result['per_sample']:.2f}**",
    f"**{best_result['speedup_vs_v1']:.2f}×**"
])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#2E86AB')
    cell.set_text_props(weight='bold', color='white')

# Style best result row
for i in range(5):
    cell = table[(len(table_data)-1, i)]
    cell.set_facecolor('#F18F01')
    cell.set_text_props(weight='bold')

plt.title('Performance Benchmark Summary Table',
          fontsize=16, fontweight='bold', pad=20)

table_path = os.path.join(output_dir, "spinodoid_benchmark_table.png")
plt.savefig(table_path, dpi=300, bbox_inches='tight')
print(f"Summary table saved to: {table_path}")

print("\nVisualization complete!")
print("=" * 70)
