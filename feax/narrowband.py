"""Narrow-band solves on an active subset of a domain.

A *narrow band* is a user-chosen subset of the domain's cells. feax solves only
on the sub-mesh those cells span — the reduced system has DOFs only for the
nodes the active cells touch. The band works on both domain representations:

* an unstructured :class:`feax.Mesh` (any element type), or
* an implicit :class:`feax.StructuredGrid` (the full grid is never built —
  only the O(band) sub-mesh is materialized).

The band is defined by the user from any field or policy (a density, a
level-set, an error indicator, ...): the mechanism only consumes
``active_cells`` (integer cell indices). If the excluded cells are void
(stiffness → 0), the band solve reproduces the full solve to O(void
stiffness); in general it is the solve on the sub-domain the active cells form,
with a free surface on the band boundary.

Everything downstream is stock feax: build a ``Problem`` on ``band.mesh``
(coordinate-based ``location_fns`` transfer unchanged), assemble, and solve
with the usual differentiable ``create_solver``; ``jax.grad`` flows through
via feax's existing implicit-diff adjoint.

Usage
-----
>>> active = np.nonzero(rho > 1e-2)[0]                 # any user policy
>>> band = fe.NarrowBand(mesh, active)                 # Mesh or StructuredGrid
>>> sub_problem = MyProblem(band.mesh, vec=3, dim=3, location_fns=[...])
>>> ...                                                # bc, tp, ts, solver as usual
>>> sol_band = solver(tp_band, initial_band, traced_structure=ts_band)
>>> u_full = band.scatter_sol(sol_band, vec=3)         # back to the full domain
"""

from dataclasses import dataclass

import numpy as onp

from .mesh import Mesh
from .spgrid import StructuredGrid


def _node_adjacent_cells(cells, num_nodes, seed_cells):
    """Cells sharing at least one node with any seed cell (1-ring dilation)."""
    seed_nodes = onp.zeros(num_nodes, bool)
    seed_nodes[onp.unique(cells[seed_cells])] = True
    touches = seed_nodes[cells].any(axis=1)
    return onp.nonzero(touches)[0]


@dataclass(frozen=True)
class NarrowBand:
    """An active-cell subset of a domain and the maps to/from the full domain.

    ``domain`` is either an unstructured :class:`feax.Mesh` or an implicit
    :class:`feax.StructuredGrid`; both yield the same kind of band.

    Attributes
    ----------
    mesh : Mesh
        The sub-mesh spanned by the active cells (renumbered nodes). Build a
        ``Problem`` on this directly.
    active_cells : ndarray, (n_active,)
        Global indices of the active cells (band cell -> global cell).
    node_map : ndarray, (n_band_nodes,)
        Band node index -> global node index.
    num_full_cells, num_full_nodes : int
        Sizes of the parent domain (for scatter/gather).
    """

    mesh: Mesh
    active_cells: onp.ndarray
    node_map: onp.ndarray
    num_full_cells: int
    num_full_nodes: int

    # --- construction --------------------------------------------------------
    def __init__(self, domain, active_cells):
        active = onp.asarray(active_cells, onp.int64).ravel()
        if isinstance(domain, StructuredGrid):
            cells = domain.cell_to_nodes(active)              # (na, 8) global nodes
            used = onp.unique(cells)                          # band node -> global
            sub_points = domain.node_coords(used)
            n_cells, n_nodes = domain.num_cells, domain.num_nodes
        else:
            cells_full = onp.asarray(domain.cells)
            points_full = onp.asarray(domain.points)
            cells = cells_full[active]                        # (na, npe) global nodes
            used = onp.unique(cells)
            sub_points = points_full[used]
            n_cells, n_nodes = cells_full.shape[0], points_full.shape[0]

        # all entries of `cells` are in the sorted `used` -> exact positions
        sub = Mesh(sub_points, onp.searchsorted(used, cells),
                   ele_type=domain.ele_type)

        object.__setattr__(self, "mesh", sub)
        object.__setattr__(self, "active_cells", active)
        object.__setattr__(self, "node_map", used)
        object.__setattr__(self, "num_full_cells", int(n_cells))
        object.__setattr__(self, "num_full_nodes", int(n_nodes))

    @classmethod
    def from_structured_grid(cls, grid, active_cells):
        """Alias for ``NarrowBand(grid, active_cells)`` (kept for clarity)."""
        return cls(grid, active_cells)

    @classmethod
    def from_field(cls, mesh, field, threshold, *, margin=0, keep_cells=None,
                   cell_field=True):
        """Build a band from a scalar field on an unstructured mesh via
        ``field > threshold`` (+ dilation). For a :class:`StructuredGrid` use
        :meth:`StructuredGrid.cells_where` / :meth:`SparseDesign.band_cells`.

        ``field``      : per-cell (``cell_field=True``) or per-node scalar array.
        ``margin``     : node-adjacency dilation rings added around the seed
                         (``margin=1`` is the usual narrow-band safety layer).
        ``keep_cells`` : extra cell indices always kept active (e.g. near loads /
                         supports so the band never disconnects the load path).
        """
        cells_full = onp.asarray(mesh.cells)
        num_nodes = onp.asarray(mesh.points).shape[0]
        field = onp.asarray(field).ravel()
        if cell_field:
            seed = field > threshold                            # (num_cells,)
        else:
            seed = (field[cells_full] > threshold).any(axis=1)  # node field -> cell
        active = onp.nonzero(seed)[0]
        for _ in range(int(margin)):
            active = _node_adjacent_cells(cells_full, num_nodes, active)
        if keep_cells is not None:
            active = onp.union1d(active, onp.asarray(keep_cells).ravel())
        return cls(mesh, active)

    # --- scatter / gather between band and full domain ------------------------
    @property
    def num_active_cells(self):
        return int(self.active_cells.size)

    @property
    def num_band_nodes(self):
        return int(self.node_map.size)

    def gather_cells(self, full_cell_field):
        """Full per-cell array -> band per-cell array (active cells only)."""
        return onp.asarray(full_cell_field).reshape(self.num_full_cells, -1
                                                    ).squeeze()[self.active_cells]

    def scatter_cells(self, band_cell_field, fill=0.0):
        """Band per-cell array -> full per-cell array (inactive filled)."""
        out = onp.full((self.num_full_cells,), fill, float)
        out[self.active_cells] = onp.asarray(band_cell_field).ravel()
        return out

    def scatter_sol(self, sol_band, vec):
        """Band solution (flat, band_node*vec+comp) -> full flat solution vector.

        Inactive DOFs are left zero. Accepts a numpy or JAX array.
        """
        sb = onp.asarray(sol_band).reshape(self.num_band_nodes, vec)
        out = onp.zeros((self.num_full_nodes, vec))
        out[self.node_map] = sb
        return out.reshape(-1)

    def gather_sol(self, sol_full, vec):
        """Full flat solution vector -> band flat solution (active nodes only)."""
        sf = onp.asarray(sol_full).reshape(self.num_full_nodes, vec)
        return sf[self.node_map].reshape(-1)

    # --- TracedParams bridge --------------------------------------------------
    def gather_params(self, tp_full):
        """Full-domain :class:`feax.TracedParams` -> band TracedParams.

        Each volume var is gathered by its leading axis: ``(num_full_cells, ...)``
        by ``active_cells``, ``(num_full_nodes, ...)`` by ``node_map`` (cells take
        precedence if the counts coincide). ``surface_vars`` must be empty — the
        band's boundary faces differ from the full mesh's; rebuild surface vars
        on the band ``Problem`` instead. Differentiable (pure gathers).
        """
        from .traced_params import TracedParams
        if tp_full.surface_vars:
            raise ValueError(
                "gather_params only maps volume_vars; surface_vars must be "
                "rebuilt on the band Problem (the boundary faces differ).")
        gathered = []
        for v in tp_full.volume_vars:
            if v.shape[0] == self.num_full_cells:
                gathered.append(v[self.active_cells])
            elif v.shape[0] == self.num_full_nodes:
                gathered.append(v[self.node_map])
            else:
                raise ValueError(
                    f"volume var with leading dim {v.shape[0]} matches neither "
                    f"num_full_cells ({self.num_full_cells}) nor num_full_nodes "
                    f"({self.num_full_nodes})")
        return TracedParams(volume_vars=tuple(gathered))

    def scatter_params(self, tp_band, fill=0.0):
        """Band TracedParams -> full-domain TracedParams (inactive entries
        ``fill``). Inverse of :meth:`gather_params`; same shape dispatch."""
        import jax.numpy as jnp
        from .traced_params import TracedParams
        scattered = []
        for v in tp_band.volume_vars:
            if v.shape[0] == self.num_active_cells:
                n, idx = self.num_full_cells, self.active_cells
            elif v.shape[0] == self.num_band_nodes:
                n, idx = self.num_full_nodes, self.node_map
            else:
                raise ValueError(
                    f"volume var with leading dim {v.shape[0]} matches neither "
                    f"num_active_cells ({self.num_active_cells}) nor "
                    f"num_band_nodes ({self.num_band_nodes})")
            out = jnp.full((n,) + v.shape[1:], fill, dtype=v.dtype)
            scattered.append(out.at[idx].set(v))
        return TracedParams(volume_vars=tuple(scattered))


class SupersetBand:
    """Fixed-superset band manager for moving bands (e.g. topology optimization).

    A *superset* is a NarrowBand that contains the current active band plus
    ``margin`` rings of dilation cells. While the superset is held fixed, the
    active band may move freely inside it: the design lives in ``traced_params``
    on the superset (inactive-but-in-superset cells are just low-stiffness), so
    every solve has the same shapes and reuses one compiled feax solver with any
    ``solver_options``. Only when the active band migrates within ``guard``
    rings of the superset boundary is the superset re-extracted (one recompile,
    amortized).

    This class owns only the superset lifecycle + full<->superset maps; the user
    builds the feax ``Problem``/solver on ``band.mesh`` when told to re-extract::

        mgr = SupersetBand(mesh, margin=2)
        for it in range(n_iter):
            active = onp.nonzero(rho > threshold)[0]          # any policy
            if mgr.needs_reextract(active):
                band = mgr.reextract(active)                  # recompile point
                solver, run = build_solver(band.mesh)         # user physics/solver
            sol = run(mgr.map_cells(rho))                     # warm solve
            ...                                               # update rho
    """

    def __init__(self, mesh, *, margin=2, guard=1):
        self.mesh = mesh
        self.margin = int(margin)
        self.guard = int(guard)
        self.band = None
        self._superset_mask = None
        self._num_cells = onp.asarray(mesh.cells).shape[0]
        self._num_nodes = onp.asarray(mesh.points).shape[0]

    def reextract(self, active_cells):
        """(Re)build the superset = dilate(active_cells, margin). Returns the band.

        Call when :meth:`needs_reextract` is True — this is the (rare) recompile
        point since the sub-mesh shapes change.
        """
        active = onp.asarray(active_cells).astype(onp.int64).ravel()
        superset = active
        for _ in range(self.margin):
            superset = _node_adjacent_cells(onp.asarray(self.mesh.cells),
                                            self._num_nodes, superset)
        self.band = NarrowBand(self.mesh, superset)
        mask = onp.zeros(self._num_cells, bool)
        mask[self.band.active_cells] = True
        self._superset_mask = mask
        return self.band

    def needs_reextract(self, active_cells):
        """True if no superset yet, or the active band (dilated by ``guard``) is
        not fully contained in the current superset (it reached the margin)."""
        if self.band is None:
            return True
        active = onp.asarray(active_cells).astype(onp.int64).ravel()
        guarded = active
        for _ in range(self.guard):
            guarded = _node_adjacent_cells(onp.asarray(self.mesh.cells),
                                           self._num_nodes, guarded)
        return not self._superset_mask[guarded].all()

    # --- full <-> superset maps (thin wrappers over the current band) --------
    def map_cells(self, full_cell_field):
        """Full per-cell field -> superset per-cell field (for traced_params)."""
        return self.band.gather_cells(full_cell_field)

    def scatter_cells(self, band_cell_field, fill=0.0):
        return self.band.scatter_cells(band_cell_field, fill=fill)

    def scatter_sol(self, sol_band, vec):
        return self.band.scatter_sol(sol_band, vec)
