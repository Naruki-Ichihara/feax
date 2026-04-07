"""JAX distributed multi-node utilities for FEAX.

Provides helpers for initializing and finalizing JAX distributed
runtime across multiple nodes, and a CLI launcher for running scripts
across a cluster.

Usage as module:
    from feax import distributed
    try:
        distributed.initialize()
        ...
    finally:
        distributed.finalize()

Usage as CLI launcher:
    python3 -m feax.distributed --config distributed.yml script.py [args...]
"""

import os
import subprocess
import sys

import jax


def initialize():
    """Initialize JAX distributed runtime from environment variables.

    Reads JAX_COORD_ADDR, JAX_NUM_PROCESSES, and JAX_PROCESS_ID from
    the environment. If JAX_COORD_ADDR is not set, does nothing
    (single-node mode).

    Returns:
        True if distributed mode was initialized, False otherwise.
    """
    coord = os.environ.get("JAX_COORD_ADDR")
    if coord:
        jax.distributed.initialize(
            coordinator_address=f"{coord}:1234",
            num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
            process_id=int(os.environ["JAX_PROCESS_ID"]),
        )
        return True
    return False


def finalize():
    """Shut down JAX distributed runtime cleanly.

    Calls jax.distributed.shutdown() to release coordination service
    resources and prevent stale process errors on the next run.
    Safe to call even if distributed mode was not initialized.
    """
    try:
        jax.distributed.shutdown()
    except Exception:
        pass


def gather(local_data):
    """Gather local arrays from all processes to every process.

    Each process contributes its local array, and all processes receive
    the full collection stacked along a new leading axis.

    Args:
        local_data: A JAX array, numpy array, or pytree of arrays.
            All processes must provide the same structure and shapes.

    Returns:
        The gathered result with a new leading axis of size num_processes.
        For example, if each process has shape (N, 3), the result is
        (num_processes, N, 3).

    Example:
        # Each node solves independently
        displacement = solve(...)  # shape (12221, 3)

        # Gather all results
        all_displacements = distributed.gather(displacement)
        # shape (2, 12221, 3) on all processes

        # Save on coordinator only
        if jax.process_index() == 0:
            for i in range(len(all_displacements)):
                save(all_displacements[i])
    """
    from jax.experimental.multihost_utils import process_allgather
    return process_allgather(local_data, tiled=False)


def is_coordinator():
    """Return True if this is the coordinator process (process_id=0)."""
    return jax.process_index() == 0


def _load_config(config_path):
    """Load distributed cluster configuration from YAML file.

    Expected format:
        coordinator: 192.168.100.10
        port: 1234  # optional, default 1234
        nodes:
          - host: 192.168.100.11
            ssh_key: /root/.ssh/id_ed25519_shared
            container: feax
    """
    try:
        import yaml
    except ImportError:
        print("PyYAML is required: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def _launch(config_path, script, script_args):
    """Launch a script across all nodes defined in the config."""
    config = _load_config(config_path)
    coordinator = config["coordinator"]
    port = config.get("port", 1234)
    nodes = config.get("nodes", [])
    num_processes = 1 + len(nodes)

    # Start workers on remote nodes in background
    workers = []
    for i, node in enumerate(nodes, start=1):
        host = node["host"]
        ssh_key = node.get("ssh_key", "")
        container = node.get("container", "feax")

        ssh_cmd = ["ssh"]
        if ssh_key:
            ssh_cmd += ["-i", ssh_key]
        ssh_cmd += ["-o", "StrictHostKeyChecking=no", f"admin@{host}"]

        docker_cmd = (
            f"docker exec -i"
            f" -e JAX_COORD_ADDR={coordinator}"
            f" -e JAX_COORD_PORT={port}"
            f" -e JAX_NUM_PROCESSES={num_processes}"
            f" -e JAX_PROCESS_ID={i}"
            f" {container} python3 -"
        )
        ssh_cmd.append(docker_cmd)

        with open(script) as f:
            proc = subprocess.Popen(ssh_cmd, stdin=f, stdout=sys.stdout, stderr=sys.stderr)
        workers.append(proc)

    # Run coordinator locally (process_id=0)
    env = os.environ.copy()
    env["JAX_COORD_ADDR"] = coordinator
    env["JAX_COORD_PORT"] = str(port)
    env["JAX_NUM_PROCESSES"] = str(num_processes)
    env["JAX_PROCESS_ID"] = "0"

    cmd = [sys.executable, script] + list(script_args)
    result = subprocess.run(cmd, env=env)

    # Wait for workers
    for proc in workers:
        proc.wait()

    sys.exit(result.returncode)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m feax.distributed",
        description="Launch a FEAX script across a distributed JAX cluster.",
    )
    parser.add_argument(
        "--config", "-c",
        default="distributed.yml",
        help="Path to cluster config YAML (default: distributed.yml)",
    )
    parser.add_argument("script", help="Python script to run")
    parser.add_argument("args", nargs="*", help="Arguments for the script")

    args = parser.parse_args()
    _launch(args.config, args.script, args.args)
