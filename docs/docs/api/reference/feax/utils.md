---
sidebar_label: utils
title: feax.utils
---

Utility functions for FEAX finite element analysis framework.

This module provides utility functions for file I/O, solution initialization,
and data processing operations commonly needed in finite element analysis.

#### save\_sol

```python
def save_sol(
    mesh: Mesh,
    sol_file: str,
    cell_infos: Optional[List[Tuple[str, Union[np.ndarray,
                                               'jax.Array']]]] = None,
    point_infos: Optional[List[Tuple[str, Union[np.ndarray,
                                                'jax.Array']]]] = None
) -> None
```

Save mesh and solution data to VTK format.

**Arguments**:

- `mesh` - feax mesh object containing nodes and elements
- `sol_file` - Output file path for VTK file
- `cell_infos` - List of (name, data) tuples for cell-based data.
  Data shape should be (n_elements, ...) where ... can be:
  - () or (1,) for scalar data
  - (n,) for vector data
  - (3, 3) for tensor data (will be flattened to (9,))
- `point_infos` - List of (name, data) tuples for point-based data.
  Data shape should be (n_nodes, ...)


**Raises**:

- `ValueError` - If neither cell_infos nor point_infos is provided.

## XDMFWriter Objects

```python
class XDMFWriter()
```

XDMF + HDF5 time-series writer.

Streaming writer for optimisation / load-stepping runs where the
mesh is fixed but many frames are saved.  The mesh is stored
**once** in the companion HDF5 file; each call to
:meth:`write_iteration` appends only the new field values.
Compared to writing N separate ``.vtu`` files with :func:`save_sol`,
this:

* eliminates mesh duplication (≫ 90% disk savings on dense meshes),
* is faster to write (one streaming HDF5 file vs. N independent files),
* lets ParaView load the entire history with one &quot;Open&quot; action and
  scrub through iterations using the Time toolbar.

The XML is written so each frame&#x27;s ``&lt;Grid&gt;`` self-contains its
``&lt;Topology&gt;`` and ``&lt;Geometry&gt;`` references (pointing to the same
HDF5 datasets), rather than relying on ``xpointer`` cross-references
— this is the format ParaView&#x27;s Xdmf3 reader (and most other XDMF
consumers) load most reliably as a true time series.

Use as a context manager:

.. code-block:: python

    with XDMFWriter(mesh, OUTPUT_DIR / &quot;history.xdmf&quot;) as w:
        for k in range(n_iters):
            w.write_iteration(
                k,
                point_infos=[(&quot;density&quot;, rho_k), (&quot;director&quot;, d_k)],
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
- **mesh** (*feax.Mesh*): The mesh on which all fields live.  Written once.
- **filename** (*str or os.PathLike*): Output path; should end in ``.xdmf``.  A companion ``.h5`` file is created automatically.


#### write\_iteration

```python
def write_iteration(
    time: float,
    cell_infos: Optional[List[Tuple[str, Union[np.ndarray,
                                               'jax.Array']]]] = None,
    point_infos: Optional[List[Tuple[str, Union[np.ndarray,
                                                'jax.Array']]]] = None
) -> None
```

Append one frame of field data.

Parameters
----------
- **time** (*float or int*): Frame time / iteration index.  Must be monotonically increasing across calls.
- **cell_infos** (*list of (name, data), optional*): Cell-centred fields, shape ``(n_cells, ...)``.
- **point_infos** (*list of (name, data), optional*): Node-centred fields, shape ``(n_points, ...)``.


#### zero\_like\_initial\_guess

```python
def zero_like_initial_guess(problem: 'Problem', bc: DirichletBC) -> np.ndarray
```

Create a zero initial guess with boundary condition values set.

This is the standard initial guess for FE problems: zeros everywhere
except at Dirichlet boundary condition locations where the prescribed
values are set.

Parameters
----------
- **problem** (*Problem*): The FE problem instance containing DOF information
- **bc** (*DirichletBC*): Boundary conditions with rows and values to set


Returns
-------
- **initial_guess** (*jax.numpy.ndarray*): Initial guess vector of shape (num_total_dofs,) with zeros everywhere except BC locations which have prescribed values


Examples
--------
```python
>>> from feax.utils import zero_like_initial_guess
>>> initial_guess = zero_like_initial_guess(problem, bc)
>>> solution = solver(internal_vars, initial_guess)
```
For time-dependent problems:
```python
>>> # First timestep
>>> solution = solver(internal_vars_t0, zero_like_initial_guess(problem, bc))
>>> # Subsequent timesteps use previous solution
>>> for t in timesteps[1:]:
>>>     solution = solver(internal_vars_t, solution)
```

