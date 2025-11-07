---
sidebar_label: filter
title: feax.experimental.topopt_toolkit.filter
---

Helmholtz filter for topology optimization using FEAX framework.

Simple implementation using InternalVars pattern.

## HelmholtzProblem Objects

```python
class HelmholtzProblem(Problem)
```

Helmholtz equation problem for design variable filtering.

#### get\_tensor\_map

```python
def get_tensor_map()
```

Get the diffusion tensor mapping for the Helmholtz equation.

#### get\_mass\_map

```python
def get_mass_map()
```

Get the mass term mapping for the Helmholtz equation.

#### create\_helmholtz\_filter

```python
def create_helmholtz_filter(base_problem: Problem,
                            radius: float = 0.05) -> Callable
```

Create a Helmholtz filter function for quadrature point design variables.

**Arguments**:

- `base_problem` - Base FE problem defining mesh and element structure
- `radius` - Filter radius controlling smoothing length scale
  

**Returns**:

  Filter function that applies Helmholtz filtering to design variables at quadrature points

#### create\_helmholtz\_transform

```python
def create_helmholtz_transform(problem: Problem,
                               key: str,
                               radius: float = 0.05)
```

Create a simplified Helmholtz filtering transformation for optax.

This creates an optax-compatible gradient transformation that applies
Helmholtz filtering to specific design variable updates with no conditional overhead.

**Arguments**:

- `problem` - Base FE problem defining mesh and element structure
- `key` - Key name for the design variable to filter (e.g., &#x27;rho&#x27;, &#x27;density&#x27;)
- `radius` - Filter radius controlling smoothing length scale
  

**Returns**:

  Optax gradient transformation that applies Helmholtz filtering
  

**Example**:

  &gt;&gt;&gt; filter_transform = create_helmholtz_transform(problem, &#x27;rho&#x27;, radius=0.05)
  &gt;&gt;&gt; optimizer = optax.chain(
  ...     optax.adam(0.01),
  ...     filter_transform
  ... )

#### create\_box\_projection\_transform

```python
def create_box_projection_transform(key: str,
                                    lower: float = 0.0,
                                    upper: float = 1.0)
```

Create a box projection transformation for optax.

This creates an optax-compatible gradient transformation that modifies updates
to ensure parameters stay within specified bounds after the update is applied.

The transform clips the update such that param + update stays within [lower, upper].

**Arguments**:

- `key` - Key name for the design variable to project (e.g., &#x27;rho&#x27;, &#x27;density&#x27;)
- `lower` - Lower bound for projection
- `upper` - Upper bound for projection
  

**Returns**:

  Optax gradient transformation that clips updates to maintain bounds
  

**Example**:

  &gt;&gt;&gt; box_transform = create_box_projection_transform(&#x27;rho&#x27;, lower=0.0, upper=1.0)
  &gt;&gt;&gt; optimizer = optax.chain(
  ...     optax.adam(0.01),
  ...     create_helmholtz_transform(problem, &#x27;rho&#x27;, radius=0.05),
  ...     box_transform
  ... )

#### create\_sigmoid\_transform

```python
def create_sigmoid_transform(key: str, scale: float = 5.0)
```

Create a sigmoid transformation for design variables.

This transformation maintains an unconstrained variable internally and
applies sigmoid to map it to [0, 1]. The gradients are automatically
adjusted through the chain rule.

The sigmoid function used is: rho = 1 / (1 + exp(-scale * x))
where x is the unconstrained variable.

**Arguments**:

- `key` - Key name for the design variable (e.g., &#x27;rho&#x27;)
- `scale` - Scaling factor for sigmoid steepness (higher = steeper transition)
  

**Returns**:

  Optax gradient transformation that handles sigmoid reparameterization
  

**Example**:

  &gt;&gt;&gt; sigmoid_transform = create_sigmoid_transform(&#x27;rho&#x27;, scale=5.0)
  &gt;&gt;&gt; optimizer = optax.chain(
  ...     optax.adam(0.01),
  ...     mdmm.optax_prepare_update(),
  ...     sigmoid_transform,
  ...     filter_transform
  ... )

