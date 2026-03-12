---
sidebar_label: problem
title: feax.problem
---

Problem class with modular design separating FE structure from material parameters.

This module provides the core Problem class that defines finite element problem
structure independent of material parameters, enabling efficient optimization
and parameter studies through JAX transformations.

## MatrixView Objects

```python
class MatrixView(Enum)
```

Matrix storage format for sparse assembly.

Controls which entries are stored in the assembled matrix:
- FULL: Store all entries (default, backward compatible)
- UPPER: Store only upper triangular entries (j &gt;= i)
- LOWER: Store only lower triangular entries (j &lt;= i)

For symmetric problems, UPPER or LOWER reduces memory by ~50%
and enables optimized solvers like Cholesky factorization.

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
- **matrix_view** (*Union[MatrixView, str], optional*): Matrix storage format: &#x27;FULL&#x27; (default), &#x27;UPPER&#x27;, or &#x27;LOWER&#x27;. Use UPPER for symmetric problems to reduce memory by ~50%.
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
def get_tensor_map() -> Optional[Callable]
```

Get tensor map function for gradient-based physics.

Override this method to define the constitutive relationship between
gradients and stress/flux tensors directly.

Alternatively, override :meth:`get_energy_density` to define a scalar
energy density — the stress tensor will be derived automatically via
``jax.grad``.

Returns
-------
Optional[Callable]
    Function that maps gradients to stress/flux tensors.
    Signature: ``(u_grad, *internal_vars) -&gt; stress_tensor``
    Returns ``None`` if not defined (default).

Examples
--------
For linear elasticity:
```python
def get_tensor_map(self):
    def stress(u_grad):
        eps = 0.5 * (u_grad + u_grad.T)
        return lmbda * jnp.trace(eps) * jnp.eye(3) + 2 * mu * eps
    return stress
```

#### get\_energy\_density

```python
def get_energy_density() -> Optional[Callable]
```

Get energy density function for gradient-based physics.

Override this method to define the strain energy density as a scalar
function of the displacement gradient. The stress tensor is derived
automatically via ``jax.grad``:

```python
σ = ∂ψ/∂(∇u)
```

This is an alternative to :meth:`get_tensor_map`. If both are defined,
``get_tensor_map`` takes precedence.

Returns
-------
Optional[Callable]
    Scalar energy density function.
    Signature: ``(u_grad, *internal_vars) -&gt; scalar``
    Returns ``None`` if not defined (default).

Examples
--------
For Neo-Hookean hyperelasticity:
```python
def get_energy_density(self):
    def psi(F):
        C = F.T @ F
        I1 = jnp.trace(C)
        J = jnp.linalg.det(F)
        return mu/2 * (I1 - 3) - mu * jnp.log(J) + lmbda/2 * jnp.log(J)**2
    return psi
```

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

#### get\_weak\_form

```python
def get_weak_form() -> Optional[Callable]
```

Get weak form function for multi-variable problems.

Override this method to define coupled physics at a single quadrature
point. The framework automatically handles solution interpolation,
gradient computation, and integration. This is the recommended
interface for multi-variable problems.

The function is automatically ``jax.vmap``-ed over quadrature points.

Returns
-------
Optional[Callable]
    Weak form function with signature:

    ```python
    (vals, grads, x, *internal_vars) -&gt; (mass_terms, grad_terms)
    ```

    where:

    - ``vals[i]``: solution of variable *i*, shape ``(vec_i,)``
    - ``grads[i]``: gradient of variable *i*, shape ``(vec_i, dim)``
    - ``x``: physical coordinate, shape ``(dim,)``
    - ``mass_terms[i]``: residual integrated as ``∫ · v dΩ``, shape ``(vec_i,)``
    - ``grad_terms[i]``: residual integrated as ``∫ · ∇v dΩ``, shape ``(vec_i, dim)``

    Returns ``None`` if not defined (default).

Examples
--------
Cahn-Hilliard with mixed (c, μ) formulation:
```python
def get_weak_form(self):
    def weak_form(vals, grads, x, c_old):
        c, mu = vals[0], vals[1]
        grad_c, grad_mu = grads[0], grads[1]
        return ([(c - c_old) / dt, mu - (c**3 - c)],
                [M * grad_mu, -kappa * grad_c])
    return weak_form
```

#### get\_surface\_weak\_forms

```python
def get_surface_weak_forms() -> List[Callable]
```

Get surface weak form functions for multi-variable boundary loads.

Override this method to define surface tractions/fluxes at a single
surface quadrature point. The framework handles solution interpolation
and integration automatically. This is the recommended interface for
multi-variable problems with boundary conditions.

The function is automatically ``jax.vmap``-ed over surface quadrature
points.

Returns
-------
List[Callable]
    List of surface weak form functions, one per boundary (matching
    ``location_fns``). Each function has signature:

    ```python
    (vals, x, *internal_vars) -&gt; tractions
    ```

    where:

    - ``vals[i]``: solution of variable *i*, shape ``(vec_i,)``
    - ``x``: physical coordinate, shape ``(dim,)``
    - ``tractions[i]``: surface load integrated as ``∫ t_i · v_i dΓ``,
      shape ``(vec_i,)``

Examples
--------
Pressure BC on a Stokes problem (u: vec=2, p: vec=1):
```python
def get_surface_weak_forms(self):
    def inlet_pressure(vals, x):
        return [np.array([p_in, 0.]), np.zeros(1)]
    return [inlet_pressure]
```

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

