"""
3D Topology Optimization with Adaptive Remeshing (using gene.optimizer).

Same cantilever problem as topology_optimization.py, but uses:
  - Gmsh TET4 adaptive mesh (refines near solid, coarsens in void)
  - Heaviside projection with beta continuation
  - Pipeline abstract class with @constraint decorator

Pipeline per epoch:
    rho (node) -> density filter -> heaviside(beta) -> SIMP FE -> compliance
"""

import jax.numpy as np
import numpy as onp

import gmsh

import feax as fe
import feax.gene as gene
from feax.gene.optimizer import (
    Pipeline, constraint, Continuation, AdaptiveConfig, run,
)

# ── Material ─────────────────────────────────────────────────────────────────

E0 = 70e3
nu = 0.3
E_eps = E0 * 1e-6
penalty = 3
traction_mag = 1.0
mesh_size = 0.5

# ── Geometry & BCs ───────────────────────────────────────────────────────────

L, W, H = 100.0, 5.0, 20.0
tol = 1e-3

left = lambda pt: np.isclose(pt[0], 0., tol)
right = lambda pt: np.isclose(pt[0], L, tol) & (pt[2] < H / 4)

# ── Geometry builder (Gmsh callable) ────────────────────────────────────────

def cantilever_geometry():
    """Build cantilever box geometry via Gmsh OCC."""
    gmsh.model.occ.addBox(0, 0, 0, L, W, H)
    gmsh.model.occ.synchronize()

# ── Problem class ────────────────────────────────────────────────────────────

class LinearElasticity(fe.problem.Problem):
    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = E_eps + (E0 - E_eps) * rho**penalty
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * np.trace(eps) * np.eye(self.dim) + 2 * mu * eps
        return stress

    def get_surface_maps(self):
        return [lambda u, x, *a: np.array([0., 0., -traction_mag])]


# ── Pipeline ─────────────────────────────────────────────────────────────────

class CantileverPipeline(Pipeline):
    def build(self, mesh):
        problem = LinearElasticity(
            mesh, vec=3, dim=3, ele_type=mesh.ele_type, location_fns=[right])

        bc = fe.DCboundary.DirichletBCConfig([
            fe.DCboundary.DirichletBCSpec(
                location=left, component="all", value=0.),
        ]).create_bc(problem)

        self._initial = fe.zero_like_initial_guess(problem, bc)
        self._compliance_fn = gene.create_compliance_fn(problem)
        self._volume_fn = gene.create_volume_fn(problem)
        self._filter_fn = gene.create_density_filter(mesh, 3.0)

        sample_iv = fe.InternalVars(
            volume_vars=(fe.InternalVars.create_node_var(problem, 0.4),),
            surface_vars=())
        solver_opts = fe.DirectSolverOptions()
        self._solver = fe.create_solver(
            problem, bc, solver_options=solver_opts,
            adjoint_solver_options=solver_opts,
            iter_num=1, internal_vars=sample_iv)

    def objective(self, rho, beta=1.0):
        rho_f = self._filter_fn(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        iv = fe.InternalVars(volume_vars=(rho_p,), surface_vars=())
        sol = self._solver(iv, self._initial)
        return self._compliance_fn(sol)

    @constraint(target=0.4)
    def volume(self, rho, beta=1.0):
        rho_f = self._filter_fn(rho)
        rho_p = gene.heaviside_projection(rho_f, beta=beta)
        return self._volume_fn(rho_p)

    def filter(self, rho):
        return self._filter_fn(rho)


# ── Initial mesh ─────────────────────────────────────────────────────────────

print("Generating initial TET4 mesh ...")
mesh = gene.adaptive.adaptive_mesh(cantilever_geometry, h_max=mesh_size)
print(f"  {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")

# ── Run ──────────────────────────────────────────────────────────────────────

epoch = 100  # continuation update & remesh interval

result = run(
    pipeline=CantileverPipeline(),
    mesh=mesh,
    max_iter=500,
    continuations={
        "beta": Continuation(initial=1.0, final=16.0, update_every=20,
                             multiply_by=2.0),
    },
    adaptive=AdaptiveConfig(
        remesh=lambda m, rho: gene.adaptive.adaptive_mesh(
            cantilever_geometry,
            refinement_field=gene.adaptive.gradient_refinement(rho, m),
            old_mesh=m,
            h_min=0.1, h_max=mesh_size,
        ),
        adapt_every=epoch,
        n_adapts_max=2,
    ),
    output_dir="output_adaptive",
    save_every=2,
)

# ── Plot ─────────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(result.history['iteration'], result.history['objective'],
                 'b-', linewidth=1.5)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Compliance')
    axes[0].set_title('Compliance History')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.history['iteration'], result.history['volume'],
                 'g-', linewidth=1.5)
    axes[1].axhline(y=0.4, color='r', linestyle='--', label='Target = 0.4')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Volume Fraction')
    axes[1].set_title('Volume Fraction History')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output_adaptive/history.png", dpi=150)
    plt.close()
    print("Saved: output_adaptive/history.png")
except ImportError:
    pass
