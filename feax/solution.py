"""Solution — a typed carrier for solved DOF vectors.

feax solvers natively return a flat DOF vector (``jax.Array``). ``Solution``
wraps that vector together with the problem's DOF layout, so downstream code
does not need the ``Problem`` to interpret it:

* ``sol.field(i)`` — variable ``i`` as ``(num_nodes, vec)``
  (replaces ``problem.unflatten_fn_sol_list(sol)[i]``).
* ``sol.node_var(component=..)`` — chaining bridge: the solved field as a
  ``(num_nodes,)`` node variable for the NEXT solve's ``TracedParams``.
* array protocols (``__jax_array__`` / ``__array__``) — a ``Solution`` is
  accepted anywhere a flat array is (``jnp.dot(b, sol)``,
  ``band.scatter_sol(sol, vec)``, ...), so it composes with existing code.
* direct coefficient passing — a ``Solution`` placed in
  ``TracedParams(volume_vars=(..., sol))`` is converted to its node-based
  field automatically (``(num_nodes,)`` scalar or ``(num_nodes, vec)``
  vector), so one solve's result feeds the next solver without manual
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
"""

from typing import Optional, Tuple

import numpy as onp

import jax
import jax.numpy as np


@jax.tree_util.register_pytree_node_class
class Solution:
    """Flat DOF vector + static layout ``((num_nodes, vec), ...)`` per variable."""

    def __init__(self, dofs, layout: Tuple[Tuple[int, int], ...]):
        self.dofs = dofs
        self.layout = tuple((int(n), int(v)) for n, v in layout)

    @classmethod
    def from_problem(cls, problem, dofs) -> "Solution":
        """Wrap a flat solution using ``problem``'s DOF layout."""
        layout = tuple((fe.num_total_nodes, int(v))
                       for fe, v in zip(problem.fes, problem.vec))
        return cls(dofs, layout)

    # --- pytree -------------------------------------------------------------
    def tree_flatten(self):
        return (self.dofs,), self.layout

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], aux)

    # --- array protocols: behaves as the flat DOF vector ---------------------
    def __jax_array__(self):
        return self.dofs

    def __array__(self, dtype=None):
        a = onp.asarray(self.dofs)
        return a.astype(dtype) if dtype is not None else a

    @property
    def flat(self):
        """The raw flat DOF vector."""
        return self.dofs

    @property
    def shape(self):
        return self.dofs.shape

    @property
    def dtype(self):
        return self.dofs.dtype

    def block_until_ready(self) -> "Solution":
        jax.block_until_ready(self.dofs)
        return self

    # --- layout accessors -----------------------------------------------------
    # --- arithmetic: delegates to the flat vector, returns plain arrays ------
    def _other(self, other):
        return other.dofs if isinstance(other, Solution) else other

    def __add__(self, o):
        return self.dofs + self._other(o)

    def __radd__(self, o):
        return self._other(o) + self.dofs

    def __sub__(self, o):
        return self.dofs - self._other(o)

    def __rsub__(self, o):
        return self._other(o) - self.dofs

    def __mul__(self, o):
        return self.dofs * self._other(o)

    def __rmul__(self, o):
        return self._other(o) * self.dofs

    def __truediv__(self, o):
        return self.dofs / self._other(o)

    def __rtruediv__(self, o):
        return self._other(o) / self.dofs

    def __pow__(self, o):
        return self.dofs ** self._other(o)

    def __neg__(self):
        return -self.dofs

    def __matmul__(self, o):
        return self.dofs @ self._other(o)

    def __rmatmul__(self, o):
        return self._other(o) @ self.dofs

    # --- layout accessors -----------------------------------------------------
    @property
    def num_fields(self) -> int:
        return len(self.layout)

    def _offset(self, var_index: int) -> Tuple[int, int, int]:
        if not isinstance(var_index, (int, onp.integer)):
            raise TypeError(
                f"field() takes an int (got {type(var_index).__name__})")
        sizes = [n * v for n, v in self.layout]
        if not 0 <= var_index < len(sizes):
            raise IndexError(f"var_index {var_index} out of range "
                             f"({len(sizes)} field(s))")
        start = sum(sizes[:var_index])
        return start, *self.layout[var_index]

    def field(self, var_index: int = 0):
        """Variable ``var_index`` as ``(num_nodes, vec)`` (batch axes kept in
        front for vmapped solutions)."""
        start, n, v = self._offset(var_index)
        seg = self.dofs[..., start:start + n * v]
        return seg.reshape(*seg.shape[:-1], n, v)

    def __getitem__(self, key):
        """Indexing delegates to the flat DOF vector (array semantics: batch
        lanes of a vmapped solve, DOF slices, boolean masks, ...). Use
        :meth:`field` for typed per-variable access."""
        return self.dofs[key]

    def node_var(self, component: Optional[int] = None, var_index: int = 0):
        """The solved field as a ``(num_nodes,)`` node variable — directly
        usable in the next solve's ``TracedParams`` (staggered chaining).
        ``component`` is required for vector fields (``vec > 1``)."""
        u = self.field(var_index)
        vec = self.layout[var_index][1]
        if vec == 1:
            return u[..., 0]
        if component is None:
            raise ValueError(f"field has vec={vec} components; pass component=")
        return u[..., component]

    def __repr__(self):
        fields = ", ".join(f"({n}, {v})" for n, v in self.layout)
        return f"Solution(dofs={self.dofs.shape}, fields=[{fields}])"
