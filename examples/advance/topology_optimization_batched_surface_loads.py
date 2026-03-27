"""
Standalone 2D topology optimization example with batched surface loads.

The script optimizes ten cantilever problems in parallel on the same mesh,
each with a downward surface traction applied at a different patch along the
right edge (evenly spaced from bottom to top).

Each load case is parameterized by its own SIREN density field, while the
finite-element solve, compliance evaluation, and optimizer step are batched
with ``jax.vmap``. Material interpolation uses a SIMP-style penalty, and the
target volume fraction is enforced with an augmented-Lagrangian penalty term.

This file is intended as a compact FEAX example that shows how to:
  - define multiple Neumann load locations on one problem
  - assemble per-case surface variables for batched solves
  - optimize several density models simultaneously in a single training loop
"""

from __future__ import annotations

import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import feax as fe
import feax.gene as gene


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LX = 60.0
LY = 30.0
SCALE = 5
NX = int(LX * SCALE)
NY = int(LY * SCALE)

E0 = 70e3
E_EPS = 7.0
NU = 0.3
SIMP_PENALTY = 3.0
TRACTION_MAG = 1e2

TARGET_VOLUME_FRACTION = 0.5
CONSTRAINT_PENALTY = 20.0
NUM_ITERATIONS = 50
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 1.0
MODEL_RNG_SEED = 324

NUM_LAYERS = 3
NUM_LATENT_CHANNELS = 512
OMEGA = 23.0

OUTPUT_DIR = Path(__file__).with_name("output_batched_surface_loads")


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------

NUM_CASES = 10
X_TOL = 1e-5 * max(1.0, LX)
Y_TOL = 1e-5 * max(1.0, LY)
PATCH_HALF_BAND = 0.5 * LY / NUM_CASES + Y_TOL

CASE_NAMES = tuple(f"cantilever_{i}" for i in range(NUM_CASES))
LEFT = lambda pt: jnp.isclose(pt[0], 0.0, atol=X_TOL)

# 10 load patches evenly spaced along the right edge
PATCH_CENTERS = [LY * (i + 0.5) / NUM_CASES for i in range(NUM_CASES)]

def _make_right_patch(center):
    def loc_fn(pt):
        return jnp.isclose(pt[0], LX, atol=X_TOL) & (jnp.abs(pt[1] - center) <= PATCH_HALF_BAND)
    return loc_fn

LOAD_LOCATIONS = tuple(_make_right_patch(c) for c in PATCH_CENTERS)


# -----------------------------------------------------------------------------
# SIREN model https://www.vincentsitzmann.com/siren/
# -----------------------------------------------------------------------------

def siren_weight_init(key, shape, omega: float, *, first_layer: bool):
    fan_in = shape[-2]
    limit = 1.0 / fan_in if first_layer else math.sqrt(6.0 / fan_in) / omega
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


def siren_bias_init(key, shape):
    fan_in = shape[-1]
    limit = math.sqrt(1.0 / fan_in)
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


class SIREN(eqx.Module):
    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]
    omega: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_channels_in: int,
        num_channels_out: int,
        num_layers: int,
        num_latent_channels: int,
        omega: float,
        key,
    ):
        self.omega = omega

        channels = (
            num_channels_in,
            *[num_latent_channels] * (num_layers - 1),
            num_channels_out,
        )
        keys = jax.random.split(key, 2 * (len(channels) - 1))
        weight_keys = keys[: len(channels) - 1]
        bias_keys = keys[len(channels) - 1 :]

        weights = []
        biases = []
        for index, (in_channels, out_channels, weight_key, bias_key) in enumerate(
            zip(channels[:-1], channels[1:], weight_keys, bias_keys)
        ):
            weights.append(
                siren_weight_init(
                    weight_key,
                    (in_channels, out_channels),
                    omega,
                    first_layer=(index == 0),
                )
            )
            biases.append(siren_bias_init(bias_key, (out_channels,)))

        self.weights = tuple(weights)
        self.biases = tuple(biases)

    def __call__(self, x):
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = jnp.sin(self.omega * (x @ weight + bias))
        return x @ self.weights[-1] + self.biases[-1]

def predict_densities(models, coords):
    logits = jax.vmap(lambda model: model(coords))(models)
    return jax.nn.sigmoid(jnp.squeeze(logits, axis=-1))


# -----------------------------------------------------------------------------
# FE problem setup
# -----------------------------------------------------------------------------

class LinearElasticityProblem(fe.Problem):
    def custom_init(self, E0: float, E_eps: float, nu: float, p: float):
        self.E0 = E0
        self.E_eps = E_eps
        self.nu = nu
        self.p = p

    def get_tensor_map(self):
        def stress(u_grad, rho):
            E = (self.E0 - self.E_eps) * rho**self.p + self.E_eps
            mu = E / (2.0 * (1.0 + self.nu))
            lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            strain = 0.5 * (u_grad + u_grad.T)
            return lam * jnp.trace(strain) * jnp.eye(self.dim) + 2.0 * mu * strain

        return stress

    def get_surface_maps(self):
        def surface_map(u, x, load):
            traction = jnp.zeros(self.dim)
            return traction.at[-1].set(load)

        return [surface_map for _ in range(max(1, len(self.location_fns)))]


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------

def create_model_batch():
    keys = jax.random.split(jax.random.PRNGKey(MODEL_RNG_SEED), len(CASE_NAMES))
    models = [
        SIREN(
            num_channels_in=2,
            num_channels_out=1,
            num_layers=NUM_LAYERS,
            num_latent_channels=NUM_LATENT_CHANNELS,
            omega=OMEGA,
            key=key,
        )
        for key in keys
    ]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *models)


def print_iteration(iteration, total_loss, aux):
    compliance_text = " | ".join(
        f"{case_name}: {float(value):.6f}"
        for case_name, value in zip(CASE_NAMES, aux["compliances"])
    )
    volume_text = " | ".join(
        f"{case_name}: {float(value):.6f}"
        for case_name, value in zip(CASE_NAMES, aux["volume_fractions"])
    )
    print(
        f"Iteration {iteration:03d} | total loss = {float(total_loss):.6f}\n"
        f"  compliance      [{compliance_text}]\n"
        f"  volume fraction [{volume_text}]"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    mesh = fe.mesh.rectangle_mesh(
        Nx=NX,
        Ny=NY,
        domain_x=LX,
        domain_y=LY,
        ele_type="QUAD4",
    )

    problem = LinearElasticityProblem(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type="QUAD4",
        location_fns=LOAD_LOCATIONS,
        additional_info=(E0, E_EPS, NU, SIMP_PENALTY),
    )

    bc = fe.DirichletBCConfig(
        [fe.DirichletBCSpec(location=LEFT, component="all", value=0.0)]
    ).create_bc(problem)

    solver_options = fe.DirectSolverOptions(
        solver="cudss",
    )

    initial_guess = fe.zero_like_initial_guess(problem, bc)
    compliance_fn = gene.create_dynamic_compliance_fn(problem)
    volume_fraction_fn = gene.create_volume_fn(problem)

    # Create traction variable for each load patch
    tractions = [
        fe.InternalVars.create_uniform_surface_var(
            problem, TRACTION_MAG, surface_index=i,
        )
        for i in range(NUM_CASES)
    ]

    # Create sample internal_vars for cuDSS pre-warming (must happen before vmap)
    sample_rho = jnp.full(problem.num_cells, TARGET_VOLUME_FRACTION)
    sample_surface_vars = tuple((t,) for t in tractions)
    sample_internal_vars = fe.InternalVars(
        volume_vars=(sample_rho,),
        surface_vars=sample_surface_vars,
    )
    solver = fe.create_solver(
        problem,
        bc=bc,
        solver_options=solver_options,
        adjoint_solver_options=solver_options,
        iter_num=1,
        internal_vars=sample_internal_vars,
    )

    # For each case, activate only its own load patch (zero out others)
    per_case_surface_vars = []
    for case_idx in range(NUM_CASES):
        sv = tuple(
            (tractions[i],) if i == case_idx else (jnp.zeros_like(tractions[i]),)
            for i in range(NUM_CASES)
        )
        per_case_surface_vars.append(sv)
    batched_surface_vars = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *per_case_surface_vars,
    )


    # Normalize centroids to [-1, 1] before feeding them to the SIREN.
    # The SIREN paper analyzes the initialization for inputs in this range and
    # uses normalized coordinate domains in several experiments.
    points = jnp.asarray(mesh.points)[:, :2]
    cells = jnp.asarray(mesh.cells, dtype=jnp.int32)
    cell_centroids = points[cells].mean(axis=1)
    coord_min = cell_centroids.min(axis=0)
    coord_max = cell_centroids.max(axis=0)
    coord_range = jnp.where(coord_max > coord_min, coord_max - coord_min, 1.0)
    coords = 2.0 * (cell_centroids - coord_min) / coord_range - 1.0

    def solve_forward(rho, surface_vars):
        internal_vars = fe.InternalVars(volume_vars=(rho,), surface_vars=surface_vars)
        solution = solver(internal_vars, initial_guess)
        return compliance_fn(solution, surface_vars)

    @eqx.filter_jit
    @eqx.filter_value_and_grad(has_aux=True)
    def batched_loss(models, coords, target_volume_fractions, lams, penalties, batched_surface_vars):
        densities = predict_densities(models, coords)
        compliances = jax.vmap(solve_forward)(densities, batched_surface_vars)
        volume_fractions = jax.vmap(volume_fraction_fn)(densities)

        volume_errors = volume_fractions - target_volume_fractions
        violations = jnp.maximum(volume_errors, 0.0)
        losses = compliances + lams * violations + 0.5 * penalties * violations**2

        aux = {
            "compliances": compliances,
            "volume_fractions": volume_fractions,
            "violations": violations,
            "losses": losses,
        }
        return jnp.sum(losses), aux

    models = create_model_batch()
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adabelief(LEARNING_RATE),
    )
    opt_states = jax.vmap(lambda model: optimizer.init(eqx.filter(model, eqx.is_array)))(
        models
    )

    target_volume_fractions = jnp.full((len(CASE_NAMES),), TARGET_VOLUME_FRACTION)
    penalties = jnp.full((len(CASE_NAMES),), CONSTRAINT_PENALTY)
    lams = jnp.zeros_like(target_volume_fractions)

    for iteration in range(NUM_ITERATIONS):
        (total_loss, aux), grads = batched_loss(
            models,
            coords,
            target_volume_fractions,
            lams,
            penalties,
            batched_surface_vars,
        )
        updates, opt_states = jax.vmap(optimizer.update)(
            grads,
            opt_states,
            value=aux["losses"],
        )
        models = jax.vmap(eqx.apply_updates)(models, updates)
        lams = lams + penalties * aux["violations"]
        print_iteration(iteration, total_loss, aux)

    final_densities = predict_densities(models, coords)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case_index, case_name in enumerate(CASE_NAMES):
        fe.utils.save_sol(
            mesh,
            str(OUTPUT_DIR / f"density_{case_index + 1}_{case_name}.vtu"),
            cell_infos=[("density", final_densities[case_index])],
        )
    print(f"\nSaved densities to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
