"""Runtime container for the mesh-sized *structural* arrays of a Problem.

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
>>> ts = feax.TracedStructure.from_problem(problem)
>>> solver = feax.create_solver(problem, bc, solver_options=opts, linear=True,
...                             traced_params=tp)
>>> sol = solver(tp, initial, traced_structure=ts)          # eager
>>> jit_solve = jax.jit(lambda tp, ts: solver(tp, initial, traced_structure=ts))
>>> sol = jit_solve(tp, ts)                            # nothing baked
"""

from dataclasses import dataclass, fields
from typing import Tuple

import jax
import jax.numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TracedStructure:
    """Pytree of the structural arrays consumed by the assembly/solve path.

    All fields are leaves (arrays or tuples of arrays). Static metadata
    (sizes, batching, element type, kernels) stays on the ``Problem``, which
    remains a closure constant — it contributes nothing mesh-sized to the
    trace once the arrays come from here.
    """

    # --- element connectivity (per variable) ---
    cells_list: Tuple

    # --- volume quadrature geometry ---
    physical_quad_points: np.ndarray
    shape_grads: np.ndarray
    JxW: np.ndarray
    v_grads_JxW: np.ndarray

    # --- residual scatter map ---
    res_perm: np.ndarray
    res_sorted_dofs: np.ndarray

    # --- fused residual+J path (Newton): volume residual-DOF sort + face maps ---
    res_vol_perm: np.ndarray
    res_vol_sorted_dofs: np.ndarray
    res_face_perm: np.ndarray
    res_face_sorted_dofs: np.ndarray

    # --- CSR structure / slot maps ---
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    csr_row_of_slot: np.ndarray
    csr_diag_slots: np.ndarray
    csr_T_perm: np.ndarray
    csr_T_indptr: np.ndarray
    csr_T_indices: np.ndarray

    # --- precomputed volume->CSR slot sort (see assembler._vol_slot_sort) ---
    csr_vol_perm: np.ndarray
    csr_vol_sorted_slots: np.ndarray

    # --- surface (per registered boundary, tuples of arrays) ---
    boundary_inds_list: Tuple
    physical_surface_quad_points: Tuple
    selected_face_shape_vals: Tuple
    selected_face_shape_grads: Tuple
    nanson_scale: Tuple
    csr_face_perm: np.ndarray
    csr_face_sorted_slots: np.ndarray

    @classmethod
    def from_problem(cls, problem, free_scratch: bool = True) -> "TracedStructure":
        """Collect the structural arrays of ``problem`` into a pytree.

        Parameters
        ----------
        free_scratch : bool, default True
            After collecting the structural arrays (including the slot sorts,
            which become device leaves here), release the large host-side
            scratch arrays on ``problem`` via
            :meth:`feax.problem.Problem.free_assembly_scratch` with
            ``drop_no_ts_maps=True``. Building a TracedStructure signals commitment to
            the TracedStructure (``traced_structure=ts``) assembly path, which no longer
            needs those host arrays — so freeing them by default removes the
            largest static memory cost (critical on unified-memory devices such
            as GB10). Set ``free_scratch=False`` if you will *also* call
            ``get_jacobian`` / a ``traced_structure=None`` assembly on the same
            ``problem`` afterward (e.g. a linear-buckling ``K`` / ``K_g`` build),
            which still reads those arrays.
        """
        from feax.assembler import _res_vol_slot_sort, _vol_slot_sort

        vol_perm, vol_sorted = _vol_slot_sort(problem)
        res_vol_perm, res_vol_sorted = _res_vol_slot_sort(problem)

        def arr(x):
            return np.asarray(x)

        result = cls(
            cells_list=tuple(arr(c) for c in problem.cells_list),
            physical_quad_points=arr(problem.physical_quad_points),
            shape_grads=arr(problem.shape_grads),
            JxW=arr(problem.JxW),
            v_grads_JxW=arr(problem.v_grads_JxW),
            res_perm=arr(problem.res_perm),
            res_sorted_dofs=arr(problem.res_sorted_dofs),
            res_vol_perm=arr(res_vol_perm),
            res_vol_sorted_dofs=arr(res_vol_sorted),
            res_face_perm=arr(problem.res_face_perm),
            res_face_sorted_dofs=arr(problem.res_face_sorted_dofs),
            csr_indptr=arr(problem.csr_indptr),
            csr_indices=arr(problem.csr_indices),
            csr_row_of_slot=arr(problem.csr_row_of_slot),
            csr_diag_slots=arr(problem.csr_diag_slots),
            csr_T_perm=arr(problem.csr_T_perm),
            csr_T_indptr=arr(problem.csr_T_indptr),
            csr_T_indices=arr(problem.csr_T_indices),
            csr_vol_perm=arr(vol_perm),
            csr_vol_sorted_slots=arr(vol_sorted),
            boundary_inds_list=tuple(arr(b) for b in problem.boundary_inds_list),
            physical_surface_quad_points=tuple(
                arr(x) for x in problem.physical_surface_quad_points),
            selected_face_shape_vals=tuple(
                arr(x) for x in problem.selected_face_shape_vals),
            selected_face_shape_grads=tuple(
                arr(x) for x in problem.selected_face_shape_grads),
            nanson_scale=tuple(arr(x) for x in problem.nanson_scale),
            csr_face_perm=arr(problem.csr_face_perm),
            csr_face_sorted_slots=arr(problem.csr_face_sorted_slots),
        )

        # The slot sorts above are now device leaves of ``result``; the host
        # scratch arrays they (and the no-TracedStructure path) read are no longer
        # needed for the TracedStructure assembly path. Release them by default.
        if free_scratch:
            problem.free_assembly_scratch(drop_no_ts_maps=True)

        return result

    def tree_flatten(self):
        leaves = tuple(getattr(self, f.name) for f in fields(self))
        return leaves, None

    @classmethod
    def tree_unflatten(cls, _aux, leaves):
        return cls(*leaves)
