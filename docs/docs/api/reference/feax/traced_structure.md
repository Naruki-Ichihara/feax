---
sidebar_label: traced_structure
title: feax.traced_structure
---

Runtime container for the mesh-sized *structural* arrays of a Problem.

Why this exists
---------------
``Problem`` is registered as a pytree with **no dynamic leaves** — every array
it holds (quadrature geometry, CSR slot maps, residual scatter maps, ...) is
static structure. When a solver function built from a Problem is traced under
``jax.jit``, all of those arrays are captured as **closure constants**: they
get baked into the compiled executable, XLA constant-folds whole subgraphs of
them into giant literals (e.g. an nnz-sized boolean mask for the Dirichlet
row/column elimination), and each new mesh shape leaves another copy pinned in
the global compilation cache until ``jax.clear_caches()``.

``TracedStructure`` is the runtime-argument counterpart, mirroring how
:class:`feax.traced_params.TracedParams` carries material parameters. It holds
the same arrays as **pytree leaves**, so a solve function with signature

    solver(traced_params, initial_guess, traced_structure=ts)

receives them as traced arguments: nothing mesh-sized is baked into the
executable, no structural constant folding happens at compile time, and one
compiled executable can be reused across problems that share shapes.

Field names intentionally match the corresponding ``Problem`` attributes so
assembly code can read from either source (``src = ts if ts is not None else
problem``).

Usage
-----
&gt;&gt;&gt; ts = feax.TracedStructure.from_problem(problem)
&gt;&gt;&gt; solver = feax.create_solver(problem, bc, solver_options=opts, linear=True,
...                             traced_params=tp)
&gt;&gt;&gt; sol = solver(tp, initial, traced_structure=ts)          # eager
&gt;&gt;&gt; jit_solve = jax.jit(lambda tp, ts: solver(tp, initial, traced_structure=ts))
&gt;&gt;&gt; sol = jit_solve(tp, ts)                            # nothing baked

## TracedStructure Objects

```python
@jax.tree_util.register_pytree_node_class

@dataclass(frozen=True)
class TracedStructure()
```

Pytree of the structural arrays consumed by the assembly/solve path.

All fields are leaves (arrays or tuples of arrays). Static metadata
(sizes, batching, element type, kernels) stays on the ``Problem``, which
remains a closure constant — it contributes nothing mesh-sized to the
trace once the arrays come from here.

#### from\_problem

```python
@classmethod
def from_problem(cls, problem, free_scratch: bool = True) -> "TracedStructure"
```

Collect the structural arrays of ``problem`` into a pytree.

Parameters
----------
- **free_scratch** (*bool, default True*): After collecting the structural arrays (including the slot sorts, which become device leaves here), release the large host-side scratch arrays on ``problem`` via :meth:`feax.problem.Problem.free_assembly_scratch` with ``drop_no_ts_maps=True``. Building a TracedStructure signals commitment to the TracedStructure (``traced_structure=ts``) assembly path, which no longer needs those host arrays — so freeing them by default removes the largest static memory cost (critical on unified-memory devices such as GB10). Set ``free_scratch=False`` if you will *also* call ``get_jacobian`` / a ``traced_structure=None`` assembly on the same ``problem`` afterward (e.g. a linear-buckling ``K`` / ``K_g`` build), which still reads those arrays.

