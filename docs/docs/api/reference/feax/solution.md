---
sidebar_label: solution
title: feax.solution
---

Solution — a typed carrier for solved DOF vectors.

feax solvers natively return a flat DOF vector (``jax.Array``). ``Solution``
wraps that vector together with the problem&#x27;s DOF layout, so downstream code
does not need the ``Problem`` to interpret it:

* ``sol.field(i)`` — variable ``i`` as ``(num_nodes, vec)``
  (replaces ``problem.unflatten_fn_sol_list(sol)[i]``).
* ``sol.node_var(component=..)`` — chaining bridge: the solved field as a
  ``(num_nodes,)`` node variable for the NEXT solve&#x27;s ``TracedParams``.
* array protocols (``__jax_array__`` / ``__array__``) — a ``Solution`` is
  accepted anywhere a flat array is (``jnp.dot(b, sol)``,
  ``band.scatter_sol(sol, vec)``, ...), so it composes with existing code.
* direct coefficient passing — a ``Solution`` placed in
  ``TracedParams(volume_vars=(..., sol))`` is converted to its node-based
  field automatically (``(num_nodes,)`` scalar or ``(num_nodes, vec)``
  vector), so one solve&#x27;s result feeds the next solver without manual
  unflatten/reshape::

      T   = solver_thermal(tp_th)                       # Solution
      u   = solver_mech(fe.TracedParams(volume_vars=(E_cells, T)))

It is a JAX pytree (the DOF vector is the single leaf; the layout is static),
so it passes through ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` transparently.
Batched (vmapped) solutions keep the batch axes in front; ``field`` /
``node_var`` reshape only the trailing DOF axis.

Opt in via ``fe.create_solver(..., return_solution=True)`` (same for
``NarrowBandCMG.create_solver``), or wrap manually with
``Solution.from_problem(problem, dofs)``.

## Solution Objects

```python
@jax.tree_util.register_pytree_node_class
class Solution()
```

Flat DOF vector + static layout ``((num_nodes, vec), ...)`` per variable.

#### from\_problem

```python
@classmethod
def from_problem(cls, problem, dofs) -> "Solution"
```

Wrap a flat solution using ``problem``&#x27;s DOF layout.

#### flat

```python
@property
def flat()
```

The raw flat DOF vector.

#### field

```python
def field(var_index: int = 0)
```

Variable ``var_index`` as ``(num_nodes, vec)`` (batch axes kept in
front for vmapped solutions).

#### \_\_getitem\_\_

```python
def __getitem__(key)
```

Indexing delegates to the flat DOF vector (array semantics: batch
lanes of a vmapped solve, DOF slices, boolean masks, ...). Use
:meth:`field` for typed per-variable access.

#### node\_var

```python
def node_var(component: Optional[int] = None, var_index: int = 0)
```

The solved field as a ``(num_nodes,)`` node variable — directly
usable in the next solve&#x27;s ``TracedParams`` (staggered chaining).
``component`` is required for vector fields (``vec &gt; 1``).

