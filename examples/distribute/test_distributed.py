"""Test JAX distributed multi-node GPU recognition.

Usage (single node):
    docker exec feax python3 examples/test_distributed.py

Usage (distributed via launcher):
    ./jax-distributed-run.sh examples/test_distributed.py
"""

import jax
import jax.numpy as jnp

from feax import distributed

try:
    distributed.initialize()

    devices = jax.devices()
    local_devices = jax.local_devices()

    print(f"Process ID:      {jax.process_index()}")
    print(f"Total processes: {jax.process_count()}")
    print(f"Total devices:   {jax.device_count()}")
    print(f"Local devices:   {jax.local_device_count()}")
    print(f"All devices:     {devices}")
    print(f"Local devices:   {local_devices}")
    print()

    # Simple computation on each device
    for dev in local_devices:
        x = jax.device_put(jnp.ones(1000), dev)
        y = jnp.sum(x ** 2)
        print(f"  Device {dev}: sum(ones^2) = {y} (OK)")

    print()

    # --- Cross-device sharded computation ---
    if jax.device_count() >= 2:
        from jax.sharding import NamedSharding, Mesh, PartitionSpec as P

        mesh = Mesh(jax.devices(), axis_names=("devices",))
        sharding = NamedSharding(mesh, P("devices"))

        n = 2048
        a = jax.device_put(jnp.arange(n, dtype=jnp.float32), sharding)
        b = jax.device_put(jnp.ones(n, dtype=jnp.float32) * 2.0, sharding)

        @jax.jit
        def compute(a, b):
            return jnp.dot(a, b)

        result = compute(a, b)
        expected = jnp.sum(jnp.arange(n, dtype=jnp.float32) * 2.0)

        print(f"  Sharded dot product across {jax.device_count()} GPUs:")
        print(f"    a = [0, 1, 2, ..., {n-1}]  (sharded)")
        print(f"    b = [2, 2, 2, ..., 2]      (sharded)")
        print(f"    dot(a, b) = {result}")
        print(f"    expected  = {expected}")
        print(f"    match: {jnp.allclose(result, expected)}")
        print()

    print(f"Distributed test PASSED: {len(devices)} GPU(s) visible")

finally:
    distributed.finalize()
