---
sidebar_label: problem
title: feax.problem
---

Problem class with modular design separating FE structure from material parameters.

This module provides the core Problem class that defines finite element problem
structure independent of material parameters, enabling efficient optimization
and parameter studies through JAX transformations.

## Problem Objects

```python
@dataclass

@jax.tree_util.register_pytree_node_class
class Problem()
```

Finite element problem definition.

This class defines the finite element problem structure.

Parameters
----------
- **mesh** (*Union[Mesh, List[Mesh]]*): Finite element mesh(es). Single mesh for single-variable problems, list of meshes for multi-variable problems
- **vec** (*Union[int, List[int]]*): Number of vector components per variable. Single int for single-variable, list of ints for multi-variable problems
- **dim** (*int*): Spatial dimension of the problem (2D or 3D)
- **ele_type** (*Union[str, List[str]], optional*): Element type identifier(s). Default &#x27;HEX8&#x27;
- **gauss_order** (*Union[int, List[int]], optional*): Gaussian quadrature order(s). Default determined by element type
- **location_fns** (*Optional[List[Callable]], optional*): Functions defining boundary locations for surface integrals
- **additional_info** (*Tuple[Any, ...], optional*): Additional problem-specific information passed to custom_init()


Attributes
----------
- **num_vars** (*int*): Number of variables in the problem
- **fes** (*List[FiniteElement]*): Finite element objects for each variable
- **num_cells** (*int*): Total number of elements
- **num_total_dofs_all_vars** (*int*): Total degrees of freedom across all variables
- **unflatten_fn_sol_list** (*Callable*): Function to unflatten solution vector to per-variable arrays


Notes
-----
Subclasses should implement:
- get_tensor_map(): Returns function for gradient-based physics 
- get_mass_map(): Returns function for mass/reaction terms (optional)
- get_surface_maps(): Returns functions for surface loads (optional)
- custom_init(): Additional initialization if needed (optional)

#### \_\_post\_init\_\_

```python
def __post_init__() -> None
```

Initialize all state data for the finite element problem.

This method handles the conversion of single variables to lists for
uniform processing, creates finite element objects, computes assembly
indices, and pre-computes geometric data for efficient assembly.

The initialization process:
1. Normalizes input parameters to list format
2. Creates FiniteElement objects for each variable  
3. Computes sparse matrix assembly indices (I, J)
4. Pre-computes shape functions and Jacobian data
5. Sets up boundary condition data structures
6. Calls custom_init() for problem-specific setup

#### custom\_init

```python
def custom_init(*args: Any) -> None
```

Custom initialization for problem-specific setup.

Subclasses should override this method to perform additional
initialization using the additional_info parameters.

Parameters
----------
*args : Any
    Arguments passed from additional_info tuple

#### get\_tensor\_map

```python
def get_tensor_map() -> Callable
```

Get tensor map function for gradient-based physics.

This method must be implemented by subclasses to define the constitutive
relationship between gradients and stress/flux tensors.

Returns
-------
TensorMap
    Function that maps gradients to stress/flux tensors
    Signature: (u_grad: Array, *internal_vars) -&gt; stress_tensor: Array

Raises
------
NotImplementedError
    If not implemented by subclass

Examples
--------
For linear elasticity:
&gt;&gt;&gt; def tensor_map(u_grad, E, nu):
...     # Compute stress from displacement gradient
...     return stress_tensor

#### get\_surface\_maps

```python
def get_surface_maps() -> List[Callable]
```

Get surface map functions for boundary loads.

Override this method to define surface tractions, pressures, or fluxes
applied to boundaries identified by location_fns.

Returns
-------
List[SurfaceMap]
    List of functions for surface loads. Each function has signature:
    (u: Array, x: Array, *internal_vars) -&gt; traction: Array

Notes
-----
The number of surface maps should match the number of location_fns
provided to the Problem constructor.

#### get\_mass\_map

```python
def get_mass_map() -> Optional[Callable]
```

Get mass map function for inertia/reaction terms.

Override this method to define mass matrix contributions or reaction terms
that don&#x27;t involve gradients (e.g., inertia, damping, reactions).

Returns
-------
Optional[MassMap]
    Function for mass/reaction terms with signature:
    (u: Array, x: Array, *internal_vars) -&gt; mass_term: Array
    Returns None if no mass terms are present

#### tree\_flatten

```python
def tree_flatten() -> Tuple[Tuple, dict]
```

Flatten Problem object for JAX pytree registration.

Since Problem objects contain only static structure information
(no JAX arrays), all data goes into the static part.

Parameters
----------
- **self** (*Problem*): Problem object to flatten


Returns
-------
Tuple[Tuple, dict]
    (dynamic_data, static_data) where dynamic_data is empty
    and static_data contains all Problem fields

#### tree\_unflatten

```python
@classmethod
def tree_unflatten(cls, static: dict, _dynamic: Tuple) -> 'Problem'
```

Reconstruct Problem object from flattened parts.

Parameters
----------
- **static** (*dict*): Static data containing Problem constructor arguments
- **_dynamic** (*Tuple*): Dynamic data (empty for Problem objects, unused)


Returns
-------
Problem
    Reconstructed Problem instance

