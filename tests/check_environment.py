#!/usr/bin/env python
"""Check test environment without pytest."""

import jax
import jax.numpy as jnp


def has_gpu():
    """Check if GPU is available."""
    try:
        devices = jax.devices('gpu')
        return len(devices) > 0
    except:
        return False


def has_cudss():
    """Check if cuDSS backend is available."""
    try:
        import feax
        from feax.solver import CUDSSOptions
        return has_gpu()
    except ImportError:
        return False


def get_backend():
    """Get current JAX backend."""
    try:
        return jax.default_backend()
    except:
        return "unknown"


def main():
    """Print environment information."""
    print("\n" + "="*60)
    print("Test Environment Information")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {get_backend()}")
    print(f"GPU available: {has_gpu()}")
    print(f"cuDSS available: {has_cudss()}")

    if has_gpu():
        devices = jax.devices('gpu')
        print(f"GPU devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  GPU {i}: {device}")

    print("="*60)
    print("\nEnvironment is ready for testing!")


if __name__ == "__main__":
    main()
