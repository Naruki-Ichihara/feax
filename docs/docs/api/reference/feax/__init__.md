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

