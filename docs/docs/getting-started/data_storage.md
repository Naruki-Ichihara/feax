# Data Storage Formats

feax keeps every array in a small set of explicit storage formats. Knowing them makes the whole library legible: solvers, `TracedParams`, and post-processing all move data between these few layouts, and most "shape errors" are just a mismatch between two of them.

| Format | What it stores | Memory |
|---|---|---|
| [`Mesh`](#mesh--explicit-unstructured) | explicit unstructured mesh (`points` + `cells`) | O(nodes + cells) |
| [`StructuredGrid`](#structuredgrid--the-implicit-voxel-grid) | implicit uniform voxel grid (index arithmetic) | **O(1)** |
| [`SparseDesign`](#sparsedesign--sparse-per-cell-values) | per-cell values on active cells only | O(stored cells) |
| [Flat DOF vector / `Solution`](#the-flat-dof-vector-and-solution) | solution fields | O(dofs) |
| [`CSRMatrix` + `MatrixView`](#csrmatrix--assembled-sparse-operators) | assembled sparse operators | O(nnz) |
| [`TracedParams` variables](#tracedparams-variable-layouts) | material / load fields by shape convention | O(field) |

## `Mesh` — explicit unstructured

The classic representation: node coordinates and element connectivity.

```python
mesh = fe.mesh.box_mesh((3.0, 1.0, 1.0), mesh_size=0.1)
mesh.points    # (num_nodes, dim)  float coordinates
mesh.cells     # (num_cells, nodes_per_elem)  int connectivity
mesh.ele_type  # "HEX8", "TET4", "QUAD4", ...
```

Everything downstream (shape functions, assembly, BC location functions) is derived from these two arrays. The mesh topology is static — it must not change inside `jax.jit` / `jax.grad` traces.

## `StructuredGrid` — the implicit voxel grid

A uniform HEX8 grid never needs `points`/`cells` arrays: connectivity and coordinates are index arithmetic. `fe.StructuredGrid` stores only `(nx, ny, nz)`, `spacing`, and `origin`, so a giga-cell background domain costs **O(1) memory**.

```python
grid = fe.StructuredGrid((128, 64, 64), spacing=(1.0, 1.0, 1.0))

# cell e(cx,cy,cz) = (cx*ny + cy)*nz + cz
# node n(i,j,k)    = (i*(ny+1) + j)*(nz+1) + k
grid.cell_id(3, 2, 1), grid.node_id(0, 0, 0)
grid.cell_to_nodes(cell_ids)     # (n, 8) global node ids, feax HEX8 order
grid.node_coords(node_ids)       # coordinates on the fly
grid.cells_where(pred)           # active cells from a centroid predicate
grid.nodes_where(pred)           # node ids by grid index (loads / BCs)
```

Materialize only what you touch: `grid.to_mesh(cell_ids)` builds an explicit `Mesh` of a cell subset, and `fe.NarrowBand(grid, active_cells)` adds the band ↔ full-domain index maps. `StructuredGrid.fit(points, h=...)` embeds any point cloud (e.g. an unstructured mesh) in an enclosing grid, and `fe.voxelize_mesh(grid, mesh)` marks the cells a mesh occupies.

## `SparseDesign` — sparse per-cell values

The companion to `StructuredGrid` for extreme resolution: per-cell values (a density design, a material tag) stored **only on the cells that carry them**, keyed by global cell id — about 12 bytes per stored cell (`int32` id + `float64` value, kept sorted for `searchsorted` lookup) instead of a dense O(num_cells) array.

```python
sd = fe.SparseDesign.uniform(active_cells, 0.5)
sd.gather(query_ids, default=0.0)      # ids not in the store -> default
sd2 = sd.update(ids, vals)             # functional merge (new overwrites old)
band = sd.band_cells(grid, threshold=1e-2, margin=2)   # dilate({rho > thr})
tp = sd.traced_params(band)            # -> TracedParams for any feax solver
```

## The flat DOF vector and `Solution`

Solvers work on a single flat DOF vector in **node-major order**: DOF `node * vec + component`. For multi-variable problems the variables are concatenated in order. Solvers return this vector wrapped in a [`Solution`](../api/reference/feax/solution.md), which remembers the `(num_nodes, vec)` layout of each variable:

```python
sol = solver(traced_params, initial)   # Solution (behaves like the flat array)
sol.dofs                # the raw flat vector
sol.field(0)            # variable 0 as (num_nodes, vec)
sol.node_var()          # scalar field as (num_nodes,)  (vec == 1)
sol.node_var(component=0)              # one component of a vector field
```

`Solution` supports arithmetic, indexing, and `np.asarray`, so existing flat-vector code keeps working; pass `return_solution=False` to `create_solver` to get the bare array.

## `CSRMatrix` — assembled sparse operators

Assembled Jacobians are stored as `feax.csr.CSRMatrix` — standard compressed sparse row with a **fixed, precomputed structure**:

```python
J.data      # (nnz,)  nonzero values (row-major, sorted by column within a row)
J.indptr    # (num_rows + 1,)  row r owns slots indptr[r] : indptr[r+1]
J.indices   # (nnz,)  column of each slot
J.shape     # (num_rows, num_cols), static
```

`CSRMatrix` is a JAX pytree: `data` is traced while the structure stays constant, which is what lets solvers jit/vmap over the *values* of a matrix whose *pattern* never changes. Structure-dependent operations (transpose, pattern merges) are precomputed once as index maps and applied as pure gathers (`transpose_with_maps`, `feax.asd.merge_csr_patterns` — see the [Sparse AD tutorial](../advanced/sparse_ad.md)).

### `MatrixView` — symmetric storage

`Problem(matrix_view=...)` controls which entries of a symmetric operator are stored:

| View | Stored entries | Typical nnz |
|---|---|---|
| `MatrixView.FULL` (default) | all | 1× |
| `MatrixView.UPPER` | row ≤ col | ≈ 0.55× |
| `MatrixView.LOWER` | row ≥ col | ≈ 0.55× |

`UPPER`/`LOWER` nearly halve assembly memory and pair with cuDSS's symmetric/SPD factorizations; matrix-free Krylov paths reconstruct the symmetric action automatically.

## `TracedParams` variable layouts

Material and load fields ride in `TracedParams`, and feax classifies each **volume variable by its global shape** — this is a storage convention, resolved statically at trace time:

| Shape | Kind | Delivered to your material map as |
|---|---|---|
| scalar or `(num_cells,)` | cell | one value per cell, broadcast to its quadrature points |
| `(num_nodes,)` / `(num_nodes, k)` | node | interpolated at each quadrature point via the shape functions |
| `(num_cells, num_quads)` | quad | passed through per quadrature point |

```python
E_cells = fe.TracedParams.create_cell_var(problem, 70e3)     # (num_cells,)
T_nodes = fe.TracedParams.node_var_from_solution(thermal, sol_T)  # (num_nodes,)
tp = fe.TracedParams(volume_vars=(E_cells, T_nodes))
# -> def stress(u_grad, E, T): ...   (arguments in volume_vars order)
```

Surface variables (`surface_vars`) are grouped per boundary in `location_fns` order; see the [Solver Guide](./solver.md) for how `TracedParams` flows through solves and gradients.

## Choosing a domain representation

- **Unstructured geometry, moderate size** → `Mesh` everywhere; this is the default feax workflow.
- **Voxel domains or very high resolution** → `StructuredGrid` (+ `SparseDesign` for the design field), materializing sub-meshes only through `NarrowBand`; solve with cuDSS or the [GMG solver](./solver.md#geometric-multigrid-gmg--narrowbandcmg).
- **Both**: `StructuredGrid.fit` + `voxelize_mesh` embed an unstructured mesh into a grid, which is how narrow-band methods host arbitrary geometries — see the [Narrow-Band & Giga-Voxel tutorial](../advanced/narrowband_topology_optimization.md).
