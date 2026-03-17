"""Lightweight test: solver bc= override and vmap."""

import jax
import jax.numpy as np
import feax as fe

# 1-element mesh
mesh = fe.mesh.rectangle_mesh(Nx=1, Ny=1, domain_x=1.0, domain_y=1.0, ele_type='QUAD4')

class Elastic(fe.Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return 2.0 * eps + np.trace(eps) * np.eye(2)
        return stress

problem = Elastic(mesh, vec=2, dim=2, ele_type='QUAD4')

bc = fe.DirichletBCConfig([
    fe.DirichletBCSpec(location=lambda p: np.isclose(p[0], 0., atol=1e-6), component='all', value=0.),
    fe.DirichletBCSpec(location=lambda p: np.isclose(p[0], 1., atol=1e-6), component='x', value=0.),
]).create_bc(problem)

iv = fe.InternalVars(volume_vars=())
bc1 = bc.replace_vals(bc.bc_vals.at[-1].set(0.1))
bc2 = bc.replace_vals(bc.bc_vals.at[-1].set(0.5))
vals_batch = np.stack([bc.bc_vals.at[-1].set(0.1), bc.bc_vals.at[-1].set(0.5)])

# ── Test 1: Sequential bc= override (direct solver) ─────────────────────────
print("[Test 1] Sequential bc= override (DirectSolver)")
solver_direct = fe.create_solver(
    problem, bc, solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=iv,
)

s1 = solver_direct(iv, bc=bc1)
s2 = solver_direct(iv, bc=bc2)
print(f"  d=0.1 -> max|u|={float(np.max(np.abs(s1))):.6f}")
print(f"  d=0.5 -> max|u|={float(np.max(np.abs(s2))):.6f}")
assert float(np.max(np.abs(s2))) > float(np.max(np.abs(s1)))
print("  PASS")

# ── Test 2: vmap (iterative solver — pure JAX, no external C/CUDA) ──────────
print("\n[Test 2] jax.vmap over bc_vals (IterativeSolver/CG)")
solver_iter = fe.create_solver(
    problem, bc, solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1, internal_vars=iv,
)

s1_iter = solver_iter(iv, bc=bc1)
s2_iter = solver_iter(iv, bc=bc2)
print(f"  sequential d=0.1 -> max|u|={float(np.max(np.abs(s1_iter))):.6f}")
print(f"  sequential d=0.5 -> max|u|={float(np.max(np.abs(s2_iter))):.6f}")

try:
    sols = jax.vmap(lambda v: solver_iter(iv, bc=bc.replace_vals(v)))(vals_batch)
    err = float(np.max(np.abs(sols[0] - s1_iter))) + float(np.max(np.abs(sols[1] - s2_iter)))
    print(f"  vmap error vs sequential: {err:.2e}")
    assert err < 1e-6, f"err={err}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL (CG): {type(e).__name__}: {e}")

# ── Test 3: vmap (direct solver — cuDSS) ────────────────────────────────────
print("\n[Test 3] jax.vmap over bc_vals (DirectSolver/cuDSS)")
solver_cudss = fe.create_solver(
    problem, bc, solver_options=fe.DirectSolverOptions(),
    iter_num=1, internal_vars=iv,
)

s1_cudss = solver_cudss(iv, bc=bc1)
s2_cudss = solver_cudss(iv, bc=bc2)
print(f"  sequential d=0.1 -> max|u|={float(np.max(np.abs(s1_cudss))):.6f}")
print(f"  sequential d=0.5 -> max|u|={float(np.max(np.abs(s2_cudss))):.6f}")

try:
    sols = jax.vmap(lambda v: solver_cudss(iv, bc=bc.replace_vals(v)))(vals_batch)
    err = float(np.max(np.abs(sols[0] - s1_cudss))) + float(np.max(np.abs(sols[1] - s2_cudss)))
    print(f"  vmap error vs sequential: {err:.2e}")
    assert err < 1e-6, f"err={err}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL (cuDSS): {type(e).__name__}: {e}")

# ── Test 4: Newton fori_loop vmap (iterative solver) ─────────────────────────
print("\n[Test 4] jax.vmap over bc_vals (Newton fori_loop, IterativeSolver/CG)")
from feax.solvers.options import NewtonOptions
solver_newton = fe.create_solver(
    problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=3,
    internal_vars=iv,
    newton_options=NewtonOptions(make_jittable=True),
)

initial = fe.zero_like_initial_guess(problem, bc)
s1_newton = solver_newton(iv, initial, bc=bc1)
s2_newton = solver_newton(iv, initial, bc=bc2)
print(f"  sequential d=0.1 -> max|u|={float(np.max(np.abs(s1_newton))):.6f}")
print(f"  sequential d=0.5 -> max|u|={float(np.max(np.abs(s2_newton))):.6f}")

try:
    sols = jax.vmap(lambda v: solver_newton(iv, initial, bc=bc.replace_vals(v)))(vals_batch)
    err = float(np.max(np.abs(sols[0] - s1_newton))) + float(np.max(np.abs(sols[1] - s2_newton)))
    print(f"  vmap error vs sequential: {err:.2e}")
    assert err < 1e-6, f"err={err}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL (Newton): {type(e).__name__}: {e}")

# ── Test 5: Reduced solver vmap (P = identity, IterativeSolver/CG) ────────
print("\n[Test 5] jax.vmap over bc_vals (ReducedSolver with P, IterativeSolver/CG)")
from jax.experimental.sparse import BCOO

n_dofs = problem.num_total_dofs_all_vars
P_identity = BCOO.fromdense(np.eye(n_dofs))

solver_reduced = fe.create_solver(
    problem, bc,
    solver_options=fe.IterativeSolverOptions(solver='cg'),
    iter_num=1, P=P_identity,
)

s1_red = solver_reduced(iv, bc=bc1)
s2_red = solver_reduced(iv, bc=bc2)
print(f"  sequential d=0.1 -> max|u|={float(np.max(np.abs(s1_red))):.6f}")
print(f"  sequential d=0.5 -> max|u|={float(np.max(np.abs(s2_red))):.6f}")
assert float(np.max(np.abs(s2_red))) > float(np.max(np.abs(s1_red)))

try:
    sols = jax.vmap(lambda v: solver_reduced(iv, bc=bc.replace_vals(v)))(vals_batch)
    err = float(np.max(np.abs(sols[0] - s1_red))) + float(np.max(np.abs(sols[1] - s2_red)))
    print(f"  vmap error vs sequential: {err:.2e}")
    assert err < 1e-6, f"err={err}"
    print("  PASS")
except Exception as e:
    print(f"  FAIL (Reduced): {type(e).__name__}: {e}")

# ── Test 6: jax.grad w.r.t. bc_vals (Linear, IterativeSolver/CG) ─────────
print("\n[Test 6] jax.grad w.r.t. bc_vals (Linear, IterativeSolver/CG)")
try:
    def loss_bc(bc_vals_arg):
        bc_arg = bc.replace_vals(bc_vals_arg)
        sol = solver_iter(iv, bc=bc_arg)
        return np.sum(sol ** 2)

    grad_bc = jax.grad(loss_bc)(bc1.bc_vals)
    print(f"  grad shape: {grad_bc.shape}, max|grad|={float(np.max(np.abs(grad_bc))):.6f}")
    assert grad_bc.shape == bc1.bc_vals.shape
    assert float(np.max(np.abs(grad_bc))) > 0, "gradient is zero"

    # Finite difference check
    eps = 1e-5
    fd_grad = np.zeros_like(bc1.bc_vals)
    base_loss = loss_bc(bc1.bc_vals)
    for i in range(len(bc1.bc_vals)):
        perturbed = bc1.bc_vals.at[i].add(eps)
        fd_grad = fd_grad.at[i].set((loss_bc(perturbed) - base_loss) / eps)

    rel_err = float(np.linalg.norm(grad_bc - fd_grad) / (np.linalg.norm(fd_grad) + 1e-12))
    print(f"  finite diff relative error: {rel_err:.2e}")
    assert rel_err < 1e-3, f"rel_err={rel_err}"
    print("  PASS")
except Exception as e:
    import traceback
    print(f"  FAIL (grad bc): {type(e).__name__}: {e}")
    traceback.print_exc()

# ── Test 7: jax.grad w.r.t. bc_vals (Newton fori_loop) ───────────────────
print("\n[Test 7] jax.grad w.r.t. bc_vals (Newton fori_loop, IterativeSolver/CG)")
try:
    def loss_bc_newton(bc_vals_arg):
        bc_arg = bc.replace_vals(bc_vals_arg)
        sol = solver_newton(iv, initial, bc=bc_arg)
        return np.sum(sol ** 2)

    grad_bc_newton = jax.grad(loss_bc_newton)(bc1.bc_vals)
    print(f"  grad shape: {grad_bc_newton.shape}, max|grad|={float(np.max(np.abs(grad_bc_newton))):.6f}")
    assert float(np.max(np.abs(grad_bc_newton))) > 0, "gradient is zero"

    # Finite difference check
    fd_grad_n = np.zeros_like(bc1.bc_vals)
    base_loss_n = loss_bc_newton(bc1.bc_vals)
    for i in range(len(bc1.bc_vals)):
        perturbed = bc1.bc_vals.at[i].add(eps)
        fd_grad_n = fd_grad_n.at[i].set((loss_bc_newton(perturbed) - base_loss_n) / eps)

    rel_err_n = float(np.linalg.norm(grad_bc_newton - fd_grad_n) / (np.linalg.norm(fd_grad_n) + 1e-12))
    print(f"  finite diff relative error: {rel_err_n:.2e}")
    assert rel_err_n < 1e-4, f"rel_err={rel_err_n}"
    print("  PASS")
except Exception as e:
    import traceback
    print(f"  FAIL (grad bc newton): {type(e).__name__}: {e}")
    traceback.print_exc()
