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

