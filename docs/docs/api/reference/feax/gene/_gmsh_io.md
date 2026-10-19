---
sidebar_label: _gmsh_io
title: feax.gene._gmsh_io
---

Read the current Gmsh model&#x27;s 3D volume mesh into a FEAX Mesh (no meshio).

Gmsh&#x27;s Python API returns element connectivity in Gmsh&#x27;s own node ordering,
which for quadratic elements differs from the meshio/VTK ordering that FEAX
stores cells in. The per-type permutations below map Gmsh-API order -&gt; meshio
order (derived empirically against meshio&#x27;s Gmsh reader).

#### gmsh\_volume\_to\_mesh

```python
def gmsh_volume_to_mesh(gmsh, element_type: str) -> Mesh
```

Convert the active Gmsh model&#x27;s volume mesh to a FEAX Mesh.

Call this while a Gmsh model is initialized and meshed (before
``gmsh.finalize()``), instead of writing a ``.msh`` and reading it back
with meshio.

