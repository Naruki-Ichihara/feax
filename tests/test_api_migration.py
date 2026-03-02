"""Regression tests for solver API migration."""

import jax.numpy as jnp
import pytest

import feax as fe
import feax.flat as flat
import feax.gene as gene


def test_solver_options_constructor_raises():
    """Legacy SolverOptions should fail fast with migration guidance."""
    with pytest.raises(RuntimeError, match="SolverOptions has been removed"):
        fe.SolverOptions()


def test_gene_helmholtz_filter_default_solver_options(simple_mesh):
    """Default gene Helmholtz filter should work without legacy SolverOptions."""
    filter_fn = gene.filters.create_helmholtz_filter(simple_mesh, radius=0.1)
    rho_source = jnp.ones(simple_mesh.points.shape[0])
    rho_filtered = filter_fn(rho_source)

    assert rho_filtered.shape == rho_source.shape
    assert jnp.all(jnp.isfinite(rho_filtered))


def test_flat_helmholtz_filter_default_solver_options(simple_mesh):
    """Default flat Helmholtz filter should work without legacy SolverOptions."""
    filter_fn = flat.filters.create_helmholtz_filter(simple_mesh, radius=0.1)
    rho_source = jnp.ones(simple_mesh.points.shape[0])
    rho_filtered = filter_fn(rho_source)

    assert rho_filtered.shape == rho_source.shape
    assert jnp.all(jnp.isfinite(rho_filtered))
