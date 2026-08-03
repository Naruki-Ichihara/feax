"""Read the current Gmsh model's 3D volume mesh into a FEAX Mesh (no meshio).

Gmsh's Python API returns element connectivity in Gmsh's own node ordering,
which for quadratic elements differs from the meshio/VTK ordering that FEAX
stores cells in. The per-type permutations below map Gmsh-API order -> meshio
order (derived empirically against meshio's Gmsh reader).
"""

import numpy as onp

from feax.mesh import Mesh

# element type -> (gmsh element type id, nodes per element)
_GMSH_VOL_ID = {
    'TET4':  (4, 4),
    'TET10': (11, 10),
    'HEX8':  (5, 8),
}

# Gmsh-API node order -> meshio/FEAX node order.
_GMSH_TO_MESHIO = {
    'TET4':  [0, 1, 2, 3],
    'TET10': [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
    'HEX8':  list(range(8)),
}


def gmsh_volume_to_mesh(gmsh, element_type: str) -> Mesh:
    """Convert the active Gmsh model's volume mesh to a FEAX Mesh.

    Call this while a Gmsh model is initialized and meshed (before
    ``gmsh.finalize()``), instead of writing a ``.msh`` and reading it back
    with meshio.
    """
    if element_type not in _GMSH_VOL_ID:
        raise ValueError(
            f"gmsh_volume_to_mesh: unsupported element_type '{element_type}'. "
            f"Supported: {sorted(_GMSH_VOL_ID)}"
        )
    gmsh_id, npe = _GMSH_VOL_ID[element_type]

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = onp.asarray(node_coords, dtype=onp.float64).reshape(-1, 3)

    # Gmsh node tags can be sparse/1-based; build a dense 0-based remap.
    max_tag = int(onp.max(node_tags))
    remap = onp.full(max_tag + 1, -1, dtype=onp.int64)
    remap[onp.asarray(node_tags, dtype=onp.int64)] = onp.arange(len(node_tags))

    etypes, _, enodes = gmsh.model.mesh.getElements(3, -1)
    cells = None
    for etype, enode in zip(etypes, enodes):
        if int(etype) == gmsh_id:
            cells = remap[onp.asarray(enode, dtype=onp.int64).reshape(-1, npe)]
            break
    if cells is None:
        raise RuntimeError(
            f"Gmsh model contains no {element_type} (id {gmsh_id}) volume "
            f"elements; got types {[int(e) for e in etypes]}"
        )

    # Reorder Gmsh -> meshio, then compact to the nodes actually used.
    cells = cells[:, _GMSH_TO_MESHIO[element_type]]
    used = onp.unique(cells)
    nmap = onp.full(points.shape[0], -1, dtype=onp.int64)
    nmap[used] = onp.arange(len(used))
    return Mesh(points[used], nmap[cells], ele_type=element_type)
