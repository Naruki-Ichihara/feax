"""
Problem class with modular design separating FE structure from material parameters.

This module provides the core Problem class that defines finite element problem
structure independent of material parameters, enabling efficient optimization
and parameter studies through JAX transformations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.flatten_util
import jax.numpy as np
import numpy as onp

from feax.fe import FiniteElement
from feax.mesh import Mesh


class MatrixView(Enum):
    """Matrix storage format for sparse assembly.

    Controls which entries are stored in the assembled matrix:
    - FULL: Store all entries (default, backward compatible)
    - UPPER: Store only upper triangular entries (j >= i)
    - LOWER: Store only lower triangular entries (j <= i)

    For symmetric problems, UPPER or LOWER reduces memory by ~50%
    and enables optimized solvers like Cholesky factorization.
    """
    FULL = 0
    UPPER = 1
    LOWER = 2

@dataclass
@jax.tree_util.register_pytree_node_class
class Problem:
    """Finite element problem definition.
    
    This class defines the finite element problem structure.
    
    Parameters
    ----------
    mesh : Union[Mesh, List[Mesh]]
        Finite element mesh(es). Single mesh for single-variable problems,
        list of meshes for multi-variable problems
    vec : Union[int, List[int]] 
        Number of vector components per variable. Single int for single-variable,
        list of ints for multi-variable problems
    dim : int
        Spatial dimension of the problem (2D or 3D)
    ele_type : Union[str, List[str]], optional
        Element type identifier(s). Default 'HEX8'
    gauss_order : Union[int, List[int]], optional
        Gaussian quadrature order(s). Default determined by element type
    location_fns : Optional[List[Callable]], optional
        Functions defining boundary locations for surface integrals
    matrix_view : Union[MatrixView, str], optional
        Matrix storage format: 'FULL' (default), 'UPPER', or 'LOWER'.
        Use UPPER for symmetric problems to reduce memory by ~50%.
    additional_info : Tuple[Any, ...], optional
        Additional problem-specific information passed to custom_init()

    Attributes
    ----------
    num_vars : int
        Number of variables in the problem
    fes : List[FiniteElement] 
        Finite element objects for each variable
    num_cells : int
        Total number of elements
    num_total_dofs_all_vars : int
        Total degrees of freedom across all variables
    I, J : np.ndarray
        Sparse matrix indices for assembly
    unflatten_fn_sol_list : Callable
        Function to unflatten solution vector to per-variable arrays
        
    Notes
    -----
    Subclasses should implement:
    - get_tensor_map(): Returns function for gradient-based physics 
    - get_mass_map(): Returns function for mass/reaction terms (optional)
    - get_surface_maps(): Returns functions for surface loads (optional)
    - custom_init(): Additional initialization if needed (optional)
    """
    mesh: Union[Mesh, List[Mesh]]
    vec: Union[int, List[int]]
    dim: int
    ele_type: Union[str, List[str]] = 'HEX8'
    gauss_order: Optional[Union[int, List[int]]] = None
    location_fns: Optional[List[Callable]] = None
    matrix_view: Union[MatrixView, str] = MatrixView.FULL
    additional_info: Tuple[Any, ...] = ()
    hess: bool = False

    def __post_init__(self) -> None:
        """Initialize all state data for the finite element problem.

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
        """
        # Convert matrix_view string to enum if needed
        if isinstance(self.matrix_view, str):
            try:
                object.__setattr__(self, 'matrix_view', MatrixView[self.matrix_view.upper()])
            except KeyError:
                raise ValueError(f"Invalid matrix_view: {self.matrix_view}. Must be 'FULL', 'UPPER', or 'LOWER'")

        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.gauss_order = [self.gauss_order]

        self.num_vars = len(self.mesh)

        # Build FiniteElement objects with deduplication.
        # When two variables share the same mesh object, vec, ele_type, and
        # gauss_order the expensive JAX operations (shape_grads, linalg.inv)
        # are identical.  Reuse the first FE instead of recomputing.
        _fe_cache: dict = {}
        fes = []
        for i in range(self.num_vars):
            go_i = (self.gauss_order[i]
                    if type(self.gauss_order) == type([])
                    else self.gauss_order)
            cache_key = (id(self.mesh[i]), self.vec[i], self.ele_type[i], go_i, self.hess)
            if cache_key not in _fe_cache:
                _fe_cache[cache_key] = FiniteElement(
                    mesh=self.mesh[i],
                    vec=self.vec[i],
                    dim=self.dim,
                    ele_type=self.ele_type[i],
                    gauss_order=go_i,
                    hess=self.hess,
                )
            fes.append(_fe_cache[cache_key])
        self.fes = fes

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

        self.offset = [0]
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def _build_inds(cells_list_local):
            """Build per-element global DOF index array using pure numpy.

            Avoids JAX JIT compilation overhead for index arithmetic, which
            has no gradient and does not need to be traced.

            Returns ndarray of shape (num_elems, total_dofs_per_elem).
            """
            parts = []
            for i, cells_i in enumerate(cells_list_local):
                c = onp.asarray(cells_i)             # (E, nn)
                vec_i = self.fes[i].vec
                # c * vec_i expands node index to DOF base; arange adds component offset
                dof = (vec_i * c[:, :, None]         # (E, nn, 1)
                       + onp.arange(vec_i)[None, None, :]  # (1, 1, vec)
                       + self.offset[i])              # scalar
                parts.append(dof.reshape(len(c), -1))  # (E, nn*vec)
            # DOF indices fit in int32 (num_total_dofs < 2^31); the COO index
            # arrays built from these are the largest structures in feax, so
            # int32 halves their memory and bandwidth.
            return onp.concatenate(parts, axis=1).astype(onp.int32)  # (E, total_dofs)

        # (num_cells, total_dofs_per_elem). I / J are nnz_raw-sized and used
        # only for structure setup and the BCOO path — kept on the HOST as
        # numpy int32 (never as device arrays).
        inds = _build_inds(self.cells_list)
        ndof = inds.shape[1]
        self.I = onp.repeat(inds[:, :, None], ndof, axis=2).reshape(-1)
        self.J = onp.repeat(inds[:, None, :], ndof, axis=1).reshape(-1)

        # Residual scatter: the global DOF of each per-element residual entry,
        # in the same volume-then-boundary order get_res produces values. Lets
        # the residual be assembled with one sorted segment_sum (deterministic)
        # instead of per-region scatter-adds.
        _res_dofs = [onp.asarray(inds).reshape(-1)]

        # Note: I and J are kept as FULL for assembly
        # Filtering to UPPER/LOWER happens in get_J() after computing values

        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list]
            inds_face = _build_inds(cells_list_face)
            nf = inds_face.shape[1]
            I_face = onp.repeat(inds_face[:, :, None], nf, axis=2).reshape(-1)
            J_face = onp.repeat(inds_face[:, None, :], nf, axis=1).reshape(-1)

            self.I = onp.hstack((self.I, I_face))
            self.J = onp.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)
            _res_dofs.append(inds_face.reshape(-1))

        self._res_scatter_dofs_np = onp.concatenate(_res_dofs)  # (total residual entries,)

        # Pre-compute filtering indices for UPPER/LOWER views (for JIT compatibility)
        # Store integer indices instead of boolean mask to enable JIT compilation
        if self.matrix_view == MatrixView.UPPER:
            mask = self.J >= self.I
            self.filter_indices = onp.where(mask)[0].astype(onp.int32)
            self.I_filtered = self.I[mask]
            self.J_filtered = self.J[mask]
        elif self.matrix_view == MatrixView.LOWER:
            mask = self.J <= self.I
            self.filter_indices = onp.where(mask)[0].astype(onp.int32)
            self.I_filtered = self.I[mask]
            self.J_filtered = self.J[mask]
        else:  # FULL
            self.filter_indices = None
            self.I_filtered = self.I
            self.J_filtered = self.J

        # Concatenate cell connectivity for all variables along the node axis.
        # Pure numpy avoids a JAX vmap JIT compilation for a trivial concat.
        self.cells_flat = np.array(
            onp.concatenate([onp.asarray(c) for c in self.cells_list], axis=1)
        )  # (num_cells, total_nodes_per_elem)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)

        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
        self.num_total_dofs_all_vars = len(dumb_dofs)

        # Precompute the COO-value -> CSR-slot mapping for fast, sort-free,
        # deterministic Jacobian assembly. Requires num_total_dofs_all_vars and
        # the filtered index pairs (set above).
        self._build_csr_assembly_structure()

        # All variables must share the same volume quadrature (the assembler
        # interpolates every variable's solution and internal vars onto one set
        # of quad points, and stacks per-variable JxW). Element *type* (and node
        # count) may differ per variable, but num_quads must match. Checked
        # before the stacks below so the failure message is clear.
        _nq0 = self.fes[0].num_quads
        for i, fe in enumerate(self.fes):
            if fe.num_quads != _nq0:
                raise ValueError(
                    f"All FE variables must share the same number of volume "
                    f"quadrature points; variable 0 has {_nq0} but variable {i} "
                    f"has {fe.num_quads}. Use a matching gauss_order across "
                    f"variables.")

        self.num_nodes_cumsum = np.cumsum(np.array([0] + [fe.num_nodes for fe in self.fes]))
        # (num_cells, num_vars, num_quads)
        self.JxW = np.transpose(np.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2))
        # (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = np.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points()

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for fe in self.fes:
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(boundary_inds)
                selected_face_shape_vals = fe.face_shape_vals[boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(boundary_inds)

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # TODO: assert all vars face quad points be the same
            self.physical_surface_quad_points.append(physical_surface_quad_points)

        # Initialize without internal_vars - kernels will be created separately
        self.custom_init(*self.additional_info)

        # Validate multi-variable problem setup
        self._validate_multi_variable_setup()

    def _build_csr_assembly_structure(self) -> None:
        """Precompute the COO-value → CSR-slot mapping for assembly.

        The sparsity *pattern* of the assembled Jacobian — the filtered index
        pairs ``(I_filtered, J_filtered)`` — is fixed for the life of the
        Problem; only the *values* change between solves. Computing the
        deduplicated CSR structure once, in numpy, lets every subsequent
        assembly turn the raw per-element value array ``V`` into the cuDSS CSR
        value array with a single sorted segment-sum::

            V_f      = V if filter_indices is None else V[filter_indices]
            csr_data = segment_sum(V_f[csr_perm], csr_seg_ids,
                                   num_segments=csr_nse, indices_are_sorted=True)

        This eliminates the per-solve ``BCOO.sum_duplicates`` sort and the
        intermediate BCOO entirely, and — unlike scatter-add — is deterministic
        (no atomics).

        Attributes set
        --------------
        csr_nse : int
            Number of unique (row, col) entries (CSR ``nnz``).
        csr_indptr : ndarray (num_dofs + 1,), int32
            CSR row pointer.
        csr_indices : ndarray (csr_nse,), int32
            CSR column indices (within-row sorted by column).
        csr_perm : ndarray (nnz_filtered,), int
            Permutation that sorts the filtered raw entries by (row, col).
        csr_seg_ids : ndarray (nnz_filtered,), int
            Segment id (target CSR slot) of each *sorted* raw entry; sorted
            ascending, so ``indices_are_sorted=True`` is valid.
        csr_slot : ndarray (nnz_filtered,), int
            Target CSR slot of each *raw* (unsorted) filtered entry — the
            scatter-add form ``zeros(nse).at[csr_slot].add(V_f)``.

        Notes
        -----
        The deduplicated ordering (lexicographic by row then col) matches
        ``BCSR.from_bcoo(BCOO(...).sum_duplicates())``, so the produced CSR is
        byte-compatible with what a BCOO→CSR conversion would yield.
        """
        rows = onp.asarray(self.I_filtered).astype(onp.int64)
        cols = onp.asarray(self.J_filtered).astype(onp.int64)
        n = int(self.num_total_dofs_all_vars)
        nnz_raw = rows.shape[0]

        # Encode each (row, col) as a single key so a 1-D sort gives the
        # lexicographic (row, col) order used by CSR.
        keys = rows * n + cols
        order = onp.argsort(keys, kind='stable')         # raw position -> sorted rank
        sorted_keys = keys[order]

        # Mark the first occurrence of each unique key; cumsum gives the CSR
        # slot of every sorted entry.
        is_new = onp.ones(nnz_raw, dtype=bool)
        if nnz_raw > 1:
            is_new[1:] = sorted_keys[1:] != sorted_keys[:-1]
        seg_ids_sorted = (onp.cumsum(is_new) - 1).astype(onp.int64)
        nse = int(seg_ids_sorted[-1]) + 1 if nnz_raw > 0 else 0

        # Unique (row, col) -> CSR indptr / indices.
        unique_keys = sorted_keys[is_new]
        csr_rows = (unique_keys // n).astype(onp.int32)   # row of each CSR slot
        csr_cols = (unique_keys % n).astype(onp.int32)
        indptr = onp.zeros(n + 1, dtype=onp.int64)
        onp.add.at(indptr, csr_rows + 1, 1)
        indptr = onp.cumsum(indptr).astype(onp.int32)

        # Raw (unsorted) entry -> CSR slot, for the scatter-add form.
        slot = onp.empty(nnz_raw, dtype=onp.int64)
        slot[order] = seg_ids_sorted

        # Diagonal CSR slot of each row (for BC enforcement and diagonal/Jacobi
        # extraction). FEM always has a (d, d) entry for every dof; rows without
        # one keep the sentinel -1.
        diag_slot = onp.full(n, -1, dtype=onp.int64)
        is_diag = csr_rows == csr_cols
        diag_positions = onp.where(is_diag)[0]
        diag_slot[csr_rows[diag_positions]] = diag_positions

        # Unfiltered COO entry -> CSR slot, or `nse` (a discard bucket) for
        # entries dropped by the UPPER/LOWER filter. This is the per-entry map
        # used by the memory-efficient per-batch CSR assembly: each element's
        # local Jacobian entries scatter directly to their CSR slots, so the
        # full element-Jacobian array is never materialized.
        rows_all = onp.asarray(self.I).astype(onp.int64)
        cols_all = onp.asarray(self.J).astype(onp.int64)
        keys_all = rows_all * n + cols_all
        if nse > 0:
            pos_all = onp.searchsorted(unique_keys, keys_all)
            pos_safe = onp.minimum(pos_all, nse - 1)
            matched = unique_keys[pos_safe] == keys_all
            full_slot = onp.where(matched, pos_all, nse).astype(onp.int64)
        else:
            full_slot = onp.zeros(keys_all.shape[0], dtype=onp.int64)
        # Volume block: the first num_cells * ndof_vol^2 entries of (I, J) are
        # the volume element Jacobians (ndof_vol = total dofs per element),
        # assembled volume-first in Problem.__post_init__.
        ndof_vol = int(sum(fe.num_nodes * fe.vec for fe in self.fes))
        vol_len = self.num_cells * ndof_vol * ndof_vol

        # Transpose structure (static): the CSR layout of A^T, plus a permutation
        # mapping this matrix's slots to the transposed order. Lets the adjoint
        # solve assemble J^T in CSR form as ``data[csr_T_perm]`` with structure
        # ``(csr_T_indptr, csr_T_indices)`` — no per-solve sort. Sort slots by
        # (col, row) = transpose lexicographic order.
        t_keys = csr_cols.astype(onp.int64) * n + csr_rows.astype(onp.int64)
        t_order = onp.argsort(t_keys, kind='stable')
        t_rows = csr_cols[t_order]            # transposed row = original col
        t_cols = csr_rows[t_order]            # transposed col = original row
        t_indptr = onp.zeros(n + 1, dtype=onp.int64)
        onp.add.at(t_indptr, t_rows.astype(onp.int64) + 1, 1)
        t_indptr = onp.cumsum(t_indptr).astype(onp.int32)
        # Structural symmetry: pattern equals its transpose (then J^T reuses the
        # forward structure and a symmetric matrix needs no transpose at all).
        structurally_symmetric = bool(
            onp.array_equal(csr_cols, t_cols) and onp.array_equal(indptr, t_indptr))

        # Index dtype policy: every index/permutation value here is bounded by
        # nnz_raw (< 2^31), so int32 is exact and halves memory + bandwidth —
        # these are the largest static arrays in feax. nnz_raw-sized maps that
        # the solve path never touches (csr_perm / csr_seg_ids / csr_slot /
        # csr_full_slot / csr_volume_slots / res_volume_dofs) stay on the HOST
        # as numpy; only nse- and residual-sized maps live on the device.
        self.csr_nse = nse
        self.csr_indptr = np.array(indptr)
        self.csr_indices = np.array(csr_cols)
        self.csr_row_of_slot = np.array(csr_rows)         # (nse,) row index per slot
        self.csr_perm = order.astype(onp.int32)           # host: reference path only
        self.csr_seg_ids = seg_ids_sorted.astype(onp.int32)   # host: reference path only
        self.csr_slot = slot.astype(onp.int32)            # host: scatter-add form only
        self.csr_diag_slots = np.array(diag_slot.astype(onp.int32))  # (num_dofs,) slot of (d,d)
        self.csr_T_perm = np.array(t_order.astype(onp.int32))  # (nse,) slot -> transposed slot
        self.csr_T_indptr = np.array(t_indptr)
        self.csr_T_indices = np.array(t_cols.astype(onp.int32))
        self.csr_structurally_symmetric = structurally_symmetric
        # Per-batch CSR assembly maps (memory-efficient path).
        self.csr_full_slot = full_slot.astype(onp.int32)  # host: (len(I),) unfiltered -> slot|nse
        self.csr_vol_len = vol_len                        # length of the volume block in (I, J)
        self.csr_ndof_vol = ndof_vol                      # total dofs per element
        # (num_cells, ndof_vol^2): each cell's local entries -> CSR slot|nse.
        # Host-only: consumed by the slot-sort precompute (assembler._vol_slot_sort).
        self.csr_volume_slots = self.csr_full_slot[:vol_len].reshape(
            self.num_cells, ndof_vol * ndof_vol)
        # Surface block: a static slot-sorted permutation so the per-solve face
        # segment_sum is deterministic (ascending segment ids).
        face_slots_np = full_slot[vol_len:]
        face_perm = onp.argsort(face_slots_np, kind='stable')
        self.csr_face_perm = np.array(face_perm.astype(onp.int32))
        self.csr_face_sorted_slots = np.array(face_slots_np[face_perm].astype(onp.int32))

        # Residual scatter: sorted DOF permutation for the deterministic
        # single-pass residual segment_sum (volume + all boundaries combined).
        res_dofs = self._res_scatter_dofs_np
        res_perm = onp.argsort(res_dofs, kind='stable')
        self.res_perm = np.array(res_perm.astype(onp.int32))
        self.res_sorted_dofs = np.array(res_dofs[res_perm].astype(onp.int32))

        # Split for the fused residual+Jacobian path (#5): the volume residual
        # is accumulated in the assembly scan (per-cell DOF block), boundaries
        # added separately. ``num_total_dofs_all_vars`` is the discard DOF for
        # padded cells. Host-only (consumed by assembler._res_vol_slot_sort).
        res_vol_len = self.num_cells * ndof_vol
        self.res_volume_dofs = res_dofs[:res_vol_len].reshape(
            self.num_cells, ndof_vol).astype(onp.int32)   # (num_cells, ndof)
        res_face_dofs = res_dofs[res_vol_len:]
        res_face_perm = onp.argsort(res_face_dofs, kind='stable')
        self.res_face_perm = np.array(res_face_perm.astype(onp.int32))
        self.res_face_sorted_dofs = np.array(res_face_dofs[res_face_perm].astype(onp.int32))

    def custom_init(self, *args: Any) -> None:
        """Custom initialization for problem-specific setup.

        Subclasses should override this method to perform additional
        initialization using the additional_info parameters.

        Parameters
        ----------
        *args : Any
            Arguments passed from additional_info tuple
        """
        pass

    def _validate_multi_variable_setup(self) -> None:
        """Validate that multi-variable problems are set up correctly.

        Checks that multi-variable problems (num_vars > 1) use universal kernels
        instead of the single-variable laplace/mass/surface kernels.

        Raises
        ------
        ValueError
            If multi-variable problem incorrectly uses single-variable kernels
        """
        if self.num_vars <= 1:
            return  # Single-variable - all kernels allowed

        # Check for problematic kernel definitions (call methods to check return values)
        has_tensor_map = self.get_tensor_map() is not None
        has_energy_density = self.get_energy_density() is not None
        has_mass_map = self.get_mass_map() is not None
        has_surface_maps = len(self.get_surface_maps()) > 0
        has_universal = hasattr(self, 'get_universal_kernel') and callable(self.get_universal_kernel)
        has_weak_form = self.get_weak_form() is not None

        # For multi-var, must have universal kernel or weak_form
        if not has_universal and not has_weak_form:
            raise ValueError(
                f"Multi-variable problem (num_vars={self.num_vars}) requires get_universal_kernel() "
                f"or get_weak_form() implementation. "
                f"The standard laplace/mass/surface kernels only support single-variable problems."
            )

        # Warn if single-variable kernels are defined (they will be ignored)
        single_var_methods = []
        if has_tensor_map:
            single_var_methods.append("get_tensor_map()")
        if has_energy_density:
            single_var_methods.append("get_energy_density()")
        if has_mass_map:
            single_var_methods.append("get_mass_map()")
        if has_surface_maps:
            single_var_methods.append("get_surface_maps()")

        if single_var_methods:
            import warnings as warn_module
            warn_module.warn(
                f"Multi-variable problem (num_vars={self.num_vars}) defines {', '.join(single_var_methods)} "
                f"which will be IGNORED. Only get_universal_kernel() is used for multi-variable problems. "
                f"Remove these methods or move their logic into get_universal_kernel().",
                UserWarning,
                stacklevel=3
            )

    def get_tensor_map(self) -> Optional[Callable]:
        """Get tensor map function for gradient-based physics.

        Override this method to define the constitutive relationship between
        gradients and stress/flux tensors directly.

        Alternatively, override :meth:`get_energy_density` to define a scalar
        energy density — the stress tensor will be derived automatically via
        ``jax.grad``.

        Returns
        -------
        Optional[Callable]
            Function that maps gradients to stress/flux tensors.
            Signature: ``(u_grad, *internal_vars) -> stress_tensor``
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
        """
        return None

    def get_energy_density(self) -> Optional[Callable]:
        """Get energy density function for gradient-based physics.

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
            Signature: ``(u_grad, *internal_vars) -> scalar``
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
        """
        return None

    def get_surface_maps(self) -> List[Callable]:
        """Get surface map functions for boundary loads.
        
        Override this method to define surface tractions, pressures, or fluxes
        applied to boundaries identified by location_fns.
        
        Returns
        -------
        List[SurfaceMap]
            List of functions for surface loads. Each function has signature:
            (u: Array, x: Array, *internal_vars) -> traction: Array
            
        Notes
        -----
        The number of surface maps should match the number of location_fns
        provided to the Problem constructor.
        """
        return []

    def get_mass_map(self) -> Optional[Callable]:
        """Get mass map function for inertia/reaction terms.

        Override this method to define mass matrix contributions or reaction terms
        that don't involve gradients (e.g., inertia, damping, reactions).

        Returns
        -------
        Optional[MassMap]
            Function for mass/reaction terms with signature:
            (u: Array, x: Array, *internal_vars) -> mass_term: Array
            Returns None if no mass terms are present
        """
        return None

    def get_weak_form(self) -> Optional[Callable]:
        """Get weak form function for multi-variable problems.

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
            (vals, grads, x, *internal_vars) -> (mass_terms, grad_terms)
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
        """
        return None

    def get_surface_weak_forms(self) -> List[Callable]:
        """Get surface weak form functions for multi-variable boundary loads.

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
            (vals, x, *internal_vars) -> tractions
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
        """
        return []

    def tree_flatten(self) -> Tuple[Tuple, dict]:
        """Flatten Problem object for JAX pytree registration.

        Since Problem objects contain only static structure information
        (no JAX arrays), all data goes into the static part.

        Parameters
        ----------
        self : Problem
            Problem object to flatten

        Returns
        -------
        Tuple[Tuple, dict]
            (dynamic_data, static_data) where dynamic_data is empty
            and static_data contains all Problem fields
        """
        # No dynamic parts - everything is static structure
        dynamic = ()

        # All data is static geometric/structural information
        static = {
            'mesh': self.mesh,
            'vec': self.vec,
            'dim': self.dim,
            'ele_type': self.ele_type,
            'gauss_order': self.gauss_order,
            'location_fns': self.location_fns,
            'matrix_view': self.matrix_view,
            'additional_info': self.additional_info,
            'hess': self.hess,
        }
        return dynamic, static

    @classmethod
    def tree_unflatten(cls, static: dict, _dynamic: Tuple) -> 'Problem':
        """Reconstruct Problem object from flattened parts.

        Parameters
        ----------
        static : dict
            Static data containing Problem constructor arguments
        _dynamic : Tuple
            Dynamic data (empty for Problem objects, unused)

        Returns
        -------
        Problem
            Reconstructed Problem instance
        """
        # Create instance with original constructor parameters
        instance = cls(
            mesh=static['mesh'],
            vec=static['vec'],
            dim=static['dim'],
            ele_type=static['ele_type'],
            gauss_order=static['gauss_order'],
            location_fns=static['location_fns'],
            matrix_view=static.get('matrix_view', MatrixView.FULL),  # Default for backward compatibility
            additional_info=static['additional_info'],
            hess=static.get('hess', False),
        )

        return instance
