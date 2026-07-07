---
sidebar_label: feax
title: feax
---

#### enable\_x64

```python
def enable_x64(flag: bool = True) -> None
```

Switch JAX between float64 (``flag=True``) and float32 (``flag=False``).

.. warning::
    JAX&#x27;s x64 setting is global and only affects arrays created
    *after* it is set.  Call this immediately after ``import feax``
    and before constructing any meshes / problems / arrays — arrays
    created earlier keep their original dtype.  For a guaranteed-clean
    run prefer the ``FEAX_X64`` environment variable instead.

Examples
--------
```python
>>> import feax
>>> feax.enable_x64(False)        # run the rest of the script in float32
>>> import feax as fe             # (re-import is a no-op; flag persists)
```

#### x64\_enabled

```python
def x64_enabled() -> bool
```

Return ``True`` if JAX is currently in float64 (double-precision) mode.

#### enable\_preallocate

```python
def enable_preallocate(flag: bool = True) -> None
```

Enable (``flag=True``) or disable (``flag=False``) XLA&#x27;s GPU memory
preallocation by setting ``XLA_PYTHON_CLIENT_PREALLOCATE``.

feax disables preallocation by default; call this to turn the ~75% upfront
grab back on (e.g. to reduce fragmentation for a fixed-shape workload).

.. warning::
    XLA reads ``XLA_PYTHON_CLIENT_PREALLOCATE`` only when the GPU backend
    initializes (the first device op). Call this — or set
    ``FEAX_PREALLOCATE`` / ``XLA_PYTHON_CLIENT_PREALLOCATE`` — *before* any
    JAX array is created; afterwards it has no effect. For a guaranteed-clean
    run prefer the ``FEAX_PREALLOCATE`` environment variable.

Examples
--------
```python
>>> import feax
>>> feax.enable_preallocate(True)   # re-enable XLA&#x27;s 75% preallocation
```

#### preallocate\_enabled

```python
def preallocate_enabled() -> bool
```

Return ``True`` if XLA GPU memory preallocation is currently enabled.

