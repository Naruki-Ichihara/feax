---
sidebar_label: distributed
title: feax.distributed
---

JAX distributed multi-node utilities for FEAX.

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

#### initialize

```python
def initialize()
```

Initialize JAX distributed runtime from environment variables.

Reads JAX_COORD_ADDR, JAX_NUM_PROCESSES, and JAX_PROCESS_ID from
the environment. If JAX_COORD_ADDR is not set, does nothing
(single-node mode).

**Returns**:

  True if distributed mode was initialized, False otherwise.

#### finalize

```python
def finalize()
```

Shut down JAX distributed runtime cleanly.

Calls jax.distributed.shutdown() to release coordination service
resources and prevent stale process errors on the next run.
Safe to call even if distributed mode was not initialized.

#### gather

```python
def gather(local_data)
```

Gather local arrays from all processes to every process.

Each process contributes its local array, and all processes receive
the full collection stacked along a new leading axis.

**Arguments**:

- `local_data` - A JAX array, numpy array, or pytree of arrays.
  All processes must provide the same structure and shapes.


**Returns**:

  The gathered result with a new leading axis of size num_processes.
  For example, if each process has shape (N, 3), the result is
  (num_processes, N, 3).


**Example**:

# Each node solves independently
displacement = solve(...)  # shape (12221, 3)
# Gather all results
all_displacements = distributed.gather(displacement)
# shape (2, 12221, 3) on all processes
# Save on coordinator only
if jax.process_index() == 0:
for i in range(len(all_displacements)):
save(all_displacements[i])

#### is\_coordinator

```python
def is_coordinator()
```

Return True if this is the coordinator process (process_id=0).

