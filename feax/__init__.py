"""feax - Finite Element Analysis with JAX"""
import jax
jax.config.update("jax_enable_x64", True)
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)