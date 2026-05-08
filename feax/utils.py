"""
Utility functions for FEAX finite element analysis framework.

This module provides utility functions for file I/O, solution initialization,
and data processing operations commonly needed in finite element analysis.
"""

import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import jax
import jax.numpy as np
import meshio
import numpy as onp

from feax.DCboundary import DirichletBC
from feax.mesh import Mesh, get_meshio_cell_type

if TYPE_CHECKING:
    from feax.problem import Problem



def save_sol(
    mesh: Mesh,
    sol_file: str,
    cell_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
    point_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
) -> None:
    """Save mesh and solution data to VTK format.

    Args:
        mesh: feax mesh object containing nodes and elements
        sol_file: Output file path for VTK file
        cell_infos: List of (name, data) tuples for cell-based data.
            Data shape should be (n_elements, ...) where ... can be:
            - () or (1,) for scalar data
            - (n,) for vector data
            - (3, 3) for tensor data (will be flattened to (9,))
        point_infos: List of (name, data) tuples for point-based data.
            Data shape should be (n_nodes, ...)

    Raises:
        ValueError: If neither cell_infos nor point_infos is provided.
    """
    if cell_infos is None and point_infos is None:
        raise ValueError("At least one of cell_infos or point_infos must be provided.")

    # Get meshio cell type from mesh element type
    # We need to infer element type from the mesh structure
    n_nodes_per_element = mesh.cells.shape[1]
    n_dim = mesh.points.shape[1]  # 2D or 3D

    if n_nodes_per_element == 3:
        element_type = 'TRI3'
    elif n_nodes_per_element == 4:
        element_type = 'QUAD4' if n_dim == 2 else 'TET4'
    elif n_nodes_per_element == 6:
        element_type = 'TRI6'
    elif n_nodes_per_element == 8:
        element_type = 'HEX8'
    elif n_nodes_per_element == 10:
        element_type = 'TET10'
    elif n_nodes_per_element == 20:
        element_type = 'HEX20'
    else:
        raise ValueError(f"Unsupported element type with {n_nodes_per_element} nodes per element")

    cell_type = get_meshio_cell_type(element_type)

    # Create output directory if needed
    sol_dir = os.path.dirname(sol_file)
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)

    # Convert JAX arrays to numpy for meshio
    nodes_np = onp.array(mesh.points)
    elements_np = onp.array(mesh.cells)

    # Create meshio mesh
    out_mesh = meshio.Mesh(
        points=nodes_np,
        cells={cell_type: elements_np}
    )

    # Process cell data
    if cell_infos is not None:
        out_mesh.cell_data = {}
        for name, data in cell_infos:
            # Convert to numpy if it's a JAX array
            data = onp.array(data, dtype=onp.float32)

            # Validate shape
            if data.shape[0] != mesh.cells.shape[0]:
                raise ValueError(
                    f"Cell data '{name}' has wrong shape: got {data.shape}, "
                    f"expected first dimension = {mesh.cells.shape[0]}"
                )

            # Handle different data dimensions
            if data.ndim == 3:
                # Tensor (n_elements, 3, 3) -> flatten to (n_elements, 9)
                data = data.reshape(mesh.cells.shape[0], -1)
            elif data.ndim == 2:
                # Vector (n_elements, n) is OK as is
                pass
            else:
                # Scalar (n_elements,) -> (n_elements, 1)
                data = data.reshape(mesh.cells.shape[0], 1)

            out_mesh.cell_data[name] = [data]

    # Process point data
    if point_infos is not None:
        out_mesh.point_data = {}
        for name, data in point_infos:
            # Convert to numpy if it's a JAX array
            data = onp.array(data, dtype=onp.float32)

            # Validate shape
            if data.shape[0] != mesh.points.shape[0]:
                raise ValueError(
                    f"Point data '{name}' has wrong shape: got {data.shape}, "
                    f"expected first dimension = {mesh.points.shape[0]}"
                )

            out_mesh.point_data[name] = data

    # Write the mesh
    out_mesh.write(sol_file)


def _normalize_field(name: str, data, expected_n: int):
    """Normalize a (name, data) field for meshio XDMF output.

    Converts JAX arrays to NumPy float32, validates the leading
    dimension, and reshapes to one of the layouts that meshio's XDMF
    writer recognises:

    * ``(n,)`` or ``(n, 1)``      — Scalar
    * ``(n, 2)`` or ``(n, 3)``    — Vector
    * ``(n, 6)``                  — Tensor6 (symmetric 3×3, upper)
    * ``(n, 9)`` or ``(n, 3, 3)`` — Tensor (full 3×3)

    Convenience reshapes:

    * ``(n, 2, 2)`` is **zero-padded** to ``(n, 3, 3)`` so 2-D tensor
      fields are written as ``Tensor`` and ParaView treats them
      uniformly across 2-D and 3-D problems.
    """
    arr = onp.array(data, dtype=onp.float32)
    if arr.shape[0] != expected_n:
        raise ValueError(
            f"Field '{name}' has wrong shape: got {arr.shape}, "
            f"expected first dimension = {expected_n}"
        )
    if arr.ndim == 3 and arr.shape[1:] == (2, 2):
        padded = onp.zeros((arr.shape[0], 3, 3), dtype=arr.dtype)
        padded[:, :2, :2] = arr
        arr = padded
    elif arr.ndim >= 3 and arr.shape[1:] != (3, 3):
        raise ValueError(
            f"Field '{name}': unsupported tensor shape {arr.shape}.  "
            "XDMF accepts (n,), (n,k) with k in {1,2,3,6,9}, or "
            "(n,2,2) / (n,3,3)."
        )
    return arr


_XDMF_TOPOLOGY_TYPE = {
    'TRI3': 'Triangle',
    'TRI6': 'Triangle_6',
    'QUAD4': 'Quadrilateral',
    'TET4': 'Tetrahedron',
    'TET10': 'Tetrahedron_10',
    'HEX8': 'Hexahedron',
    'HEX20': 'Hexahedron_20',
}

_XDMF_ATTRIBUTE_TYPE = {
    1: 'Scalar',
    2: 'Vector',
    3: 'Vector',
    6: 'Tensor6',
    9: 'Tensor',
}


class XDMFWriter:
    """XDMF + HDF5 time-series writer.

    Streaming writer for optimisation / load-stepping runs where the
    mesh is fixed but many frames are saved.  The mesh is stored
    **once** in the companion HDF5 file; each call to
    :meth:`write_iteration` appends only the new field values.
    Compared to writing N separate ``.vtu`` files with :func:`save_sol`,
    this:

    * eliminates mesh duplication (≫ 90% disk savings on dense meshes),
    * is faster to write (one streaming HDF5 file vs. N independent files),
    * lets ParaView load the entire history with one "Open" action and
      scrub through iterations using the Time toolbar.

    The XML is written so each frame's ``<Grid>`` self-contains its
    ``<Topology>`` and ``<Geometry>`` references (pointing to the same
    HDF5 datasets), rather than relying on ``xpointer`` cross-references
    — this is the format ParaView's Xdmf3 reader (and most other XDMF
    consumers) load most reliably as a true time series.

    Use as a context manager:

    .. code-block:: python

        with XDMFWriter(mesh, OUTPUT_DIR / "history.xdmf") as w:
            for k in range(n_iters):
                w.write_iteration(
                    k,
                    point_infos=[("density", rho_k), ("director", d_k)],
                )

    Caveats
    -------
    * Field **names** and **shapes** must be consistent across
      iterations.
    * ``time`` must be monotonically increasing.
    * The ``.xdmf`` file is small (XML); the bulk lives in a sibling
      ``.h5`` file with the same base name.

    Parameters
    ----------
    mesh : feax.Mesh
        The mesh on which all fields live.  Written once.
    filename : str or os.PathLike
        Output path; should end in ``.xdmf``.  A companion ``.h5`` file
        is created automatically.
    """

    def __init__(self, mesh: Mesh, filename) -> None:
        filename = str(filename)
        if not filename.endswith(".xdmf"):
            raise ValueError(
                f"XDMFWriter expects a .xdmf path, got {filename!r}"
            )

        n_nodes_per_element = mesh.cells.shape[1]
        n_dim = mesh.points.shape[1]
        if n_nodes_per_element == 3:
            element_type = 'TRI3'
        elif n_nodes_per_element == 4:
            element_type = 'QUAD4' if n_dim == 2 else 'TET4'
        elif n_nodes_per_element == 6:
            element_type = 'TRI6'
        elif n_nodes_per_element == 8:
            element_type = 'HEX8'
        elif n_nodes_per_element == 10:
            element_type = 'TET10'
        elif n_nodes_per_element == 20:
            element_type = 'HEX20'
        else:
            raise ValueError(
                f"Unsupported element type with {n_nodes_per_element} "
                f"nodes per element"
            )
        if element_type not in _XDMF_TOPOLOGY_TYPE:
            raise ValueError(
                f"Element type {element_type!r} not yet supported in XDMF writer."
            )
        self._element_type = element_type
        self._xdmf_topology_type = _XDMF_TOPOLOGY_TYPE[element_type]

        sol_dir = os.path.dirname(filename)
        if sol_dir:
            os.makedirs(sol_dir, exist_ok=True)

        self._xdmf_filename = filename
        self._h5_filename = os.path.splitext(filename)[0] + ".h5"
        self._h5_basename = os.path.basename(self._h5_filename)
        self._n_points = mesh.points.shape[0]
        self._n_cells = mesh.cells.shape[0]
        self._n_nodes_per_element = n_nodes_per_element
        self._n_dim = n_dim
        self._points = onp.asarray(mesh.points, dtype=onp.float64)
        self._cells = onp.asarray(mesh.cells, dtype=onp.int32)

        # State populated in __enter__ / write_iteration.
        self._h5 = None
        self._frames = None  # list of (time, dict point_data, dict cell_data, h5 prefix)
        self._frame_id = 0
        self._field_signature = None  # validate consistency across frames

    # -- context management --------------------------------------------------

    def __enter__(self) -> "XDMFWriter":
        import h5py
        # Truncate the .h5 file and write the mesh once.
        self._h5 = h5py.File(self._h5_filename, "w")
        self._h5.create_dataset("mesh/points", data=self._points)
        self._h5.create_dataset("mesh/cells", data=self._cells)
        self._frames = []
        self._frame_id = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Flush + close the HDF5 file, then write the .xdmf XML index.
        try:
            if self._h5 is not None:
                self._h5.close()
                self._h5 = None
            self._write_xdmf_xml()
        except Exception:
            # Don't mask the original exception if there was one.
            if exc_type is None:
                raise

    # -- public API ----------------------------------------------------------

    def write_iteration(
        self,
        time: float,
        cell_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
        point_infos: Optional[List[Tuple[str, Union[np.ndarray, 'jax.Array']]]] = None,
    ) -> None:
        """Append one frame of field data.

        Parameters
        ----------
        time : float or int
            Frame time / iteration index.  Must be monotonically
            increasing across calls.
        cell_infos : list of (name, data), optional
            Cell-centred fields, shape ``(n_cells, ...)``.
        point_infos : list of (name, data), optional
            Node-centred fields, shape ``(n_points, ...)``.
        """
        if self._h5 is None:
            raise RuntimeError(
                "XDMFWriter must be used as a context manager: "
                "wrap calls in `with XDMFWriter(...) as w:`"
            )

        frame_id = self._frame_id
        prefix = f"frame_{frame_id:06d}"
        self._h5.create_group(prefix)

        # Validate field signature is consistent across frames so the
        # resulting XDMF is a well-formed time series.
        signature_keys = (
            tuple(name for name, _ in (point_infos or [])),
            tuple(name for name, _ in (cell_infos or [])),
        )
        if self._field_signature is None:
            self._field_signature = signature_keys
        elif self._field_signature != signature_keys:
            raise ValueError(
                f"Field set changed between frames: "
                f"first frame had {self._field_signature}, frame {frame_id} "
                f"has {signature_keys}.  XDMF time series requires every "
                "frame to have the same field names."
            )

        point_records = []
        if point_infos:
            for name, data in point_infos:
                arr = _normalize_field(name, data, self._n_points)
                self._h5.create_dataset(f"{prefix}/point/{name}", data=arr)
                point_records.append((name, arr.shape))
        cell_records = []
        if cell_infos:
            for name, data in cell_infos:
                arr = _normalize_field(name, data, self._n_cells)
                self._h5.create_dataset(f"{prefix}/cell/{name}", data=arr)
                cell_records.append((name, arr.shape))

        self._frames.append((float(time), prefix, point_records, cell_records))
        self._frame_id += 1

    # -- XML emission --------------------------------------------------------

    def _attribute_type(self, shape):
        """Return ('Scalar' | 'Vector' | 'Tensor' | 'Tensor6', dims_str)."""
        if len(shape) == 1:
            return "Scalar", f"{shape[0]} 1"
        if len(shape) == 2:
            ncomp = shape[1]
            if ncomp not in _XDMF_ATTRIBUTE_TYPE:
                raise ValueError(
                    f"XDMF: cannot map shape {shape} to an attribute type."
                )
            return _XDMF_ATTRIBUTE_TYPE[ncomp], f"{shape[0]} {ncomp}"
        if len(shape) == 3 and shape[1:] == (3, 3):
            return "Tensor", f"{shape[0]} 9"
        raise ValueError(f"XDMF: unsupported field shape {shape}.")

    def _write_xdmf_xml(self):
        import xml.etree.ElementTree as ET

        h5 = self._h5_basename
        topology_type = self._xdmf_topology_type
        n_pts = self._n_points
        n_cells = self._n_cells
        nnpe = self._n_nodes_per_element
        ndim = self._n_dim

        xdmf = ET.Element("Xdmf", Version="3.0")
        domain = ET.SubElement(xdmf, "Domain")
        collection = ET.SubElement(
            domain, "Grid", Name="TimeSeries",
            GridType="Collection", CollectionType="Temporal",
        )

        for time, prefix, point_records, cell_records in self._frames:
            grid = ET.SubElement(
                collection, "Grid", Name=prefix, GridType="Uniform"
            )
            ET.SubElement(grid, "Time", Value=repr(float(time)))

            topo = ET.SubElement(
                grid, "Topology",
                TopologyType=topology_type, NumberOfElements=str(n_cells),
            )
            ET.SubElement(
                topo, "DataItem",
                Dimensions=f"{n_cells} {nnpe}", NumberType="Int",
                Format="HDF",
            ).text = f"{h5}:/mesh/cells"

            geom_type = "XY" if ndim == 2 else "XYZ"
            geom = ET.SubElement(grid, "Geometry", GeometryType=geom_type)
            ET.SubElement(
                geom, "DataItem",
                Dimensions=f"{n_pts} {ndim}", NumberType="Float",
                Precision="8", Format="HDF",
            ).text = f"{h5}:/mesh/points"

            for name, shape in point_records:
                attr_type, dims = self._attribute_type(shape)
                attr = ET.SubElement(
                    grid, "Attribute",
                    Name=name, AttributeType=attr_type, Center="Node",
                )
                ET.SubElement(
                    attr, "DataItem",
                    Dimensions=dims, NumberType="Float",
                    Precision="4", Format="HDF",
                ).text = f"{h5}:/{prefix}/point/{name}"

            for name, shape in cell_records:
                attr_type, dims = self._attribute_type(shape)
                attr = ET.SubElement(
                    grid, "Attribute",
                    Name=name, AttributeType=attr_type, Center="Cell",
                )
                ET.SubElement(
                    attr, "DataItem",
                    Dimensions=dims, NumberType="Float",
                    Precision="4", Format="HDF",
                ).text = f"{h5}:/{prefix}/cell/{name}"

        ET.indent(xdmf, space="  ")
        tree = ET.ElementTree(xdmf)
        tree.write(self._xdmf_filename, xml_declaration=True, encoding="utf-8")


def zero_like_initial_guess(problem: 'Problem', bc: DirichletBC) -> np.ndarray:
    """Create a zero initial guess with boundary condition values set.
    
    This is the standard initial guess for FE problems: zeros everywhere
    except at Dirichlet boundary condition locations where the prescribed
    values are set.
    
    Parameters
    ----------
    problem : Problem
        The FE problem instance containing DOF information
    bc : DirichletBC
        Boundary conditions with rows and values to set
        
    Returns
    -------
    initial_guess : jax.numpy.ndarray
        Initial guess vector of shape (num_total_dofs,) with zeros
        everywhere except BC locations which have prescribed values
        
    Examples
    --------
    >>> from feax.utils import zero_like_initial_guess
    >>> initial_guess = zero_like_initial_guess(problem, bc)
    >>> solution = solver(internal_vars, initial_guess)
    
    For time-dependent problems:
    >>> # First timestep
    >>> solution = solver(internal_vars_t0, zero_like_initial_guess(problem, bc))
    >>> # Subsequent timesteps use previous solution
    >>> for t in timesteps[1:]:
    >>>     solution = solver(internal_vars_t, solution)
    """
    initial_guess = np.zeros(problem.num_total_dofs_all_vars)
    initial_guess = initial_guess.at[bc.bc_rows].set(bc.bc_vals)
    return initial_guess
