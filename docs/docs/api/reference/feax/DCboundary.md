---
sidebar_label: DCboundary
title: feax.DCboundary
---

Dirichlet boundary condition implementation for FEAX finite element framework.

This module provides the core DirichletBC class for boundary condition
application and dataclass-based BC specification classes for type-safe definition.

Key Classes:
- DirichletBC: JAX-compatible BC class with apply methods
- DirichletBCSpec: Dataclass for specifying individual boundary conditions
- DirichletBCConfig: Container for multiple BC specifications with convenience methods

## DirichletBC Objects

```python
@dataclass(frozen=True)
class DirichletBC()
```

JAX-compatible dataclass for Dirichlet boundary conditions.

This class pre-computes and stores all BC information as static JAX arrays,
making it suitable for JIT compilation.

#### bc\_rows

All boundary condition row indices

#### bc\_mask

Boolean mask for BC rows (size: total_dofs)

#### bc\_vals

Boundary condition values for each BC row

#### from\_specs

```python
@staticmethod
def from_specs(problem: 'Problem',
               specs: List['DirichletBCSpec']) -> 'DirichletBC'
```

Create DirichletBC directly from a list of DirichletBCSpec objects.

This is a convenient factory method that creates a DirichletBC without
needing to create an intermediate DirichletBCConfig object.

Parameters
----------
- **problem** (*Problem*): The finite element problem instance
- **specs** (*List[DirichletBCSpec]*): List of boundary condition specifications


Returns
-------
DirichletBC
    The compiled boundary condition object

Examples
--------
```python
>>> bc = DirichletBC.from_specs(problem, [
...     DirichletBCSpec(left_boundary, &#x27;all&#x27;, 0.0),
...     DirichletBCSpec(right_boundary, &#x27;x&#x27;, 0.1)
... ])
```

#### apply\_boundary\_to\_J

```python
def apply_boundary_to_J(bc: DirichletBC, J: BCOO) -> BCOO
```

Apply Dirichlet boundary conditions to Jacobian matrix J using row elimination.

This function modifies the Jacobian matrix to enforce Dirichlet boundary conditions
by zeroing out all entries in boundary condition rows and setting diagonal entries
to 1.0 for those rows. This transforms the system to enforce u[bc_dof] = bc_val.

The algorithm:
1. Zero out all entries in BC rows (both on-diagonal and off-diagonal)
2. Set diagonal entries to 1.0 for all BC rows
3. Handle potential duplicates by concatenation (JAX sparse solvers handle this)

Parameters
----------
- **bc** (*DirichletBC*): Pre-computed boundary condition information containing: - bc_rows: DOF indices where BCs are applied - bc_mask: Boolean mask for fast BC row identification - bc_vals: Prescribed values (not used in Jacobian modification) - total_dofs: Total number of DOFs in the system
- **J** (*jax.experimental.sparse.BCOO*): The sparse Jacobian matrix in BCOO format with shape (total_dofs, total_dofs)


Returns
-------
- **J_bc** (*jax.experimental.sparse.BCOO*): The Jacobian matrix with boundary conditions applied, same shape as input


Notes
-----
This function is JAX-JIT compatible and designed for efficient use in Newton solvers.
The returned matrix may have duplicate entries (original zeros + new diagonal ones),
but JAX sparse solvers handle this correctly by summing duplicates.

#### apply\_boundary\_to\_res

```python
def apply_boundary_to_res(bc: DirichletBC,
                          res_vec: np.ndarray,
                          sol_vec: np.ndarray,
                          scale: float = 1.0) -> np.ndarray
```

Apply Dirichlet boundary conditions to residual vector using row elimination.

This function modifies the residual vector to enforce Dirichlet boundary conditions
by setting residual entries at BC DOFs to: res[bc_dof] = sol[bc_dof] - bc_val * scale
This ensures that the Newton step will drive the solution towards the prescribed values.

The modified residual enforces the constraint that after the Newton update:
sol_new[bc_dof] = bc_val * scale

Parameters
----------
- **bc** (*DirichletBC*): Pre-computed boundary condition information containing: - bc_rows: DOF indices where BCs are applied   - bc_vals: Prescribed values at boundary DOFs - bc_mask: Boolean mask (not used in this function) - total_dofs: Total number of DOFs (for validation)
- **res_vec** (*np.ndarray*): The residual vector (flattened) with shape (total_dofs,)
- **sol_vec** (*np.ndarray*): The current solution vector (flattened) with shape (total_dofs,)
- **scale** (*float, optional*): Scaling factor for boundary condition values, by default 1.0 Useful for ramping up BCs or unit conversion


Returns
-------
np.ndarray
    The residual vector with boundary conditions applied, same shape as input

Notes
-----
This function is JAX-JIT compatible and creates a copy of the input residual
to avoid modifying the original array. The boundary condition enforcement
follows the standard penalty method approach for constraint enforcement
in Newton-Raphson solvers.

## DirichletBCSpec Objects

```python
@dataclass
class DirichletBCSpec()
```

Specification for a single Dirichlet boundary condition.

This dataclass provides a clear, type-safe way to specify boundary conditions.

Parameters
----------
- **location** (*Callable[[np.ndarray], bool]*): Function that takes a point (x, y, z) and returns True if the point is on the boundary where this BC should be applied
- **component** (*Union[int, str]*): Which component to constrain: - For scalar problems: must be 0 or &#x27;all&#x27; - For vector problems: 0=&#x27;x&#x27;, 1=&#x27;y&#x27;, 2=&#x27;z&#x27;, or &#x27;all&#x27; for all components
- **value** (*Union[float, Callable[[np.ndarray], float]]*): The prescribed value, either: - A constant float value - A function that takes a point and returns the value at that point
- **variable_index** (*Optional[int]*): For multi-variable problems, specifies which variable this BC applies to. If None (default), applies to all variables (backward compatible behavior). For single-variable problems, this parameter is ignored.


Examples
--------
```python
>>> # Fix left boundary in x-direction to zero
>>> bc1 = DirichletBCSpec(
...     location=lambda pt: np.isclose(pt[0], 0.0),
...     component=&#x27;x&#x27;,  # or component=0
...     value=0.0
... )
```

#### \_\_post\_init\_\_

```python
def __post_init__() -> None
```

Validate and normalize the component specification.

1. Convert string component names (&#x27;x&#x27;, &#x27;y&#x27;, &#x27;z&#x27;, &#x27;all&#x27;) to integers
2. Validate integer component indices are non-negative
3. Convert constant values to functions for uniform interface

Raises
------
ValueError
    If component string is invalid or integer component is negative

## DirichletBCConfig Objects

```python
@dataclass
class DirichletBCConfig()
```

Configuration for all Dirichlet boundary conditions in a problem.

This dataclass holds a collection of DirichletBCSpec objects and provides
methods to convert to the format expected by DirichletBC.from_bc_info.

Parameters
----------
- **specs** (*List[DirichletBCSpec]*): List of boundary condition specifications


Examples
--------
```python
>>> # Create BC configuration for elasticity problem
>>> bc_config = DirichletBCConfig([
...     DirichletBCSpec(
...         location=lambda pt: np.isclose(pt[0], 0.0),
...         component=&#x27;all&#x27;,
...         value=0.0
...     ),
...     DirichletBCSpec(
...         location=lambda pt: np.isclose(pt[0], 1.0),
...         component=&#x27;x&#x27;,
...         value=0.1
...     )
... ])
>>>
>>> # Create DirichletBC from config
>>> bc = bc_config.create_bc(problem)
```

#### add

```python
def add(location: Callable[[np.ndarray], bool], component: Union[int, str],
        value: Union[float, Callable[[np.ndarray],
                                     float]]) -> 'DirichletBCConfig'
```

Add a boundary condition specification to the configuration.

This method allows for fluent-style chaining when building BC configurations.

Parameters
----------
- **location** (*Callable[[np.ndarray], bool]*): Function that takes a point coordinate array and returns True if  the point is on the boundary where this BC should be applied
- **component** (*Union[int, str]*): Which component to constrain: - For scalar problems: 0 or &#x27;all&#x27; - For vector problems: 0/&#x27;x&#x27;, 1/&#x27;y&#x27;, 2/&#x27;z&#x27;, or &#x27;all&#x27;
- **value** (*Union[float, Callable[[np.ndarray], float]]*): The prescribed value, either a constant or a spatial function


Returns
-------
- **self** (*DirichletBCConfig*): Returns self for method chaining, allowing: config.add(...).add(...)


Examples
--------
```python
>>> config = DirichletBCConfig()
>>> config.add(lambda pt: np.isclose(pt[0], 0.0), &#x27;all&#x27;, 0.0)
>>> config.add(lambda pt: np.isclose(pt[0], 1.0), &#x27;x&#x27;, 0.1)
```

#### create\_bc

```python
def create_bc(problem: 'Problem') -> 'DirichletBC'
```

Create a DirichletBC object from this configuration.

This method directly processes the BC specification without intermediate format conversion.

Parameters
----------
- **problem** (*Problem*): The finite element problem instance containing mesh information and vector dimension specifications


Returns
-------
- **bc** (*DirichletBC*): The compiled boundary condition object ready for use in solvers


Notes
-----
This method automatically detects the vector dimension from the problem
and handles both single-variable and multi-variable problems.

#### dirichlet\_bc\_config

```python
def dirichlet_bc_config(*specs: DirichletBCSpec) -> DirichletBCConfig
```

Convenience function to create a DirichletBCConfig from multiple specs.

This function provides a concise way to create BC configurations without
explicitly instantiating the DirichletBCConfig class.

Parameters
----------
*specs : DirichletBCSpec
    Variable number of boundary condition specifications

Returns
-------
- **config** (*DirichletBCConfig*): The BC configuration containing all provided specifications


Examples
--------
```python
>>> # Create BC config with multiple specifications
>>> config = dirichlet_bc_config(
...     DirichletBCSpec(left_boundary, &#x27;all&#x27;, 0.0),
...     DirichletBCSpec(right_boundary, &#x27;x&#x27;, 0.1),
...     DirichletBCSpec(top_boundary, &#x27;y&#x27;, lambda pt: 0.01 * pt[0])
... )
>>> bc = config.create_bc(problem)
```
See Also
--------
DirichletBCConfig : The main configuration class
DirichletBCSpec : Individual BC specification

