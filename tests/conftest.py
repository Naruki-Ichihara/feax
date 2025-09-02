"""Pytest configuration and fixtures."""
import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_jax():
    """Configure JAX for testing."""
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_disable_jit", True)  # Disable JIT for easier debugging