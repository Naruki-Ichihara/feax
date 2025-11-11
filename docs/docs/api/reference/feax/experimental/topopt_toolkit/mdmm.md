---
sidebar_label: mdmm
title: feax.experimental.topopt_toolkit.mdmm
---

The Modified Differential Multiplier Method (MDMM) for JAX.

## LagrangeMultiplier Objects

```python
class LagrangeMultiplier(NamedTuple)
```

Marks the Lagrange multipliers as such in the gradient and update so
the MDMM gradient descent ascent update can be prepared from the gradient
descent update.

#### prepare\_update

```python
def prepare_update(tree)
```

Prepares an MDMM gradient descent ascent update from a gradient descent
update.

**Arguments**:

  A pytree containing the original gradient descent update.


**Returns**:

  A pytree containing the gradient descent ascent update.

#### optax\_prepare\_update

```python
def optax_prepare_update()
```

A gradient transformation for Optax that prepares an MDMM gradient
descent ascent update from a normal gradient descent update.

It should be used like this with a base optimizer:
optimizer = optax.chain(
optax.sgd(1e-3),
mdmm_jax.optax_prepare_update(),
)

**Returns**:

  An Optax gradient transformation that converts a gradient descent update
  into a gradient descent ascent update.

## Constraint Objects

```python
class Constraint(NamedTuple)
```

A pair of pure functions implementing a constraint.

**Attributes**:

- `init` - A pure function which, when called with an example instance of
  the arguments to the constraint functions, returns a pytree
  containing the constraint&#x27;s learnable parameters.
- `loss` - A pure function which, when called with the the learnable
  parameters returned by init() followed by the arguments to the
  constraint functions, returns the loss value for the constraint.

#### eq

```python
def eq(fun, damping=1., weight=1., reduction=jnp.sum)
```

Represents an equality constraint, g(x) = 0.

**Arguments**:

- `fun` - The constraint function, a differentiable function of your
  parameters which should output zero when satisfied and smoothly
  increasingly far from zero values for increasing levels of
  constraint violation.
- `damping` - Sets the damping (oscillation reduction) strength.
- `weight` - Weights the loss from the constraint relative to the primary
  loss function&#x27;s value.
- `reduction` - The function that is used to aggregate the constraints
  if the constraint function outputs more than one element.


**Returns**:

  An (init_fn, loss_fn) constraint tuple for the equality constraint.

#### ineq

```python
def ineq(fun, damping=1., weight=1., reduction=jnp.sum)
```

Represents an inequality constraint, h(x) &gt;= 0, which uses a slack
variable internally to convert it to an equality constraint.

**Arguments**:

- `fun` - The constraint function, a differentiable function of your
  parameters which should output greater than or equal to zero when
  satisfied and smoothly increasingly negative values for increasing
  levels of constraint violation.
- `damping` - Sets the damping (oscillation reduction) strength.
- `weight` - Weights the loss from the constraint relative to the primary
  loss function&#x27;s value.
- `reduction` - The function that is used to aggregate the constraints
  if the constraint function outputs more than one element.


**Returns**:

  An (init_fn, loss_fn) constraint tuple for the inequality constraint.

#### combine

```python
def combine(*args)
```

Combines multiple constraint tuples into a single constraint tuple.

**Arguments**:

- `*args` - A series of constraint (init_fn, loss_fn) tuples.


**Returns**:

  A single (init_fn, loss_fn) tuple that wraps the input constraints.

