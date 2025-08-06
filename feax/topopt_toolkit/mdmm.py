"""The Modified Differential Multiplier Method (MDMM) for JAX."""

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax


class LagrangeMultiplier(NamedTuple):
    """Marks the Lagrange multipliers as such in the gradient and update so
    the MDMM gradient descent ascent update can be prepared from the gradient
    descent update."""
    value: Any


def prepare_update(tree):
    """Prepares an MDMM gradient descent ascent update from a gradient descent
    update.

    Args:
        A pytree containing the original gradient descent update.

    Returns:
        A pytree containing the gradient descent ascent update.
    """
    pred = lambda x: isinstance(x, LagrangeMultiplier)
    return jax.tree_map(lambda x: LagrangeMultiplier(-x.value) if pred(x) else x, tree, is_leaf=pred)


def optax_prepare_update():
    """A gradient transformation for Optax that prepares an MDMM gradient
    descent ascent update from a normal gradient descent update.

    It should be used like this with a base optimizer:
        optimizer = optax.chain(
            optax.sgd(1e-3),
            mdmm_jax.optax_prepare_update(),
        )

    Returns:
        An Optax gradient transformation that converts a gradient descent update
        into a gradient descent ascent update.
    """
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        return prepare_update(updates), state

    return optax.GradientTransformation(init_fn, update_fn)


class Constraint(NamedTuple):
    """A pair of pure functions implementing a constraint.

    Attributes:
        init: A pure function which, when called with an example instance of
            the arguments to the constraint functions, returns a pytree
            containing the constraint's learnable parameters.
        loss: A pure function which, when called with the the learnable
            parameters returned by init() followed by the arguments to the
            constraint functions, returns the loss value for the constraint.
    """
    init: Callable
    loss: Callable


def eq(fun, damping=1., weight=1., reduction=jnp.sum):
    """Represents an equality constraint, g(x) = 0.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output zero when satisfied and smoothly
            increasingly far from zero values for increasing levels of
            constraint violation.
        damping: Sets the damping (oscillation reduction) strength.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the equality constraint.
    """

    def init_fn(*args, **kwargs):
        return {'lambda': LagrangeMultiplier(jnp.zeros_like(fun(*args, **kwargs)))}

    def loss_fn(params, *args, **kwargs):
        inf = fun(*args, **kwargs)
        return weight * reduction(params['lambda'].value * inf + damping * inf ** 2 / 2), inf

    return Constraint(init_fn, loss_fn)


def ineq(fun, damping=1., weight=1., reduction=jnp.sum):
    """Represents an inequality constraint, h(x) >= 0, which uses a slack
    variable internally to convert it to an equality constraint.

    Args:
        fun: The constraint function, a differentiable function of your
            parameters which should output greater than or equal to zero when
            satisfied and smoothly increasingly negative values for increasing
            levels of constraint violation.
        damping: Sets the damping (oscillation reduction) strength.
        weight: Weights the loss from the constraint relative to the primary
            loss function's value.
        reduction: The function that is used to aggregate the constraints
            if the constraint function outputs more than one element.

    Returns:
        An (init_fn, loss_fn) constraint tuple for the inequality constraint.
    """

    def init_fn(*args, **kwargs):
        out = fun(*args, **kwargs)
        return {'lambda': LagrangeMultiplier(jnp.zeros_like(out)),
                'slack': jax.nn.relu(out) ** 0.5}

    def loss_fn(params, *args, **kwargs):
        inf = fun(*args, **kwargs) - params['slack'] ** 2
        return weight * reduction(params['lambda'].value * inf + damping * inf ** 2 / 2), inf

    return Constraint(init_fn, loss_fn)


def combine(*args):
    """Combines multiple constraint tuples into a single constraint tuple.

    Args:
        *args: A series of constraint (init_fn, loss_fn) tuples.

    Returns:
        A single (init_fn, loss_fn) tuple that wraps the input constraints.
    """
    init_fns, loss_fns = zip(*args)

    def init_fn(*args, **kwargs):
        return tuple(fn(*args, **kwargs) for fn in init_fns)

    def loss_fn(params, *args, **kwargs):
        outs = [fn(p, *args, **kwargs) for p, fn in zip(params, loss_fns)]
        return sum(x[0] for x in outs), tuple(x[1] for x in outs)

    return Constraint(init_fn, loss_fn)