# feax benchmarks

## `bench_linear_elasticity.py`

Scaling benchmark for a linear-elasticity **cantilever** across a DOF sweep, in
five execution modes (`eager`, `jit`, `grad`, `vmap-lhs`, `vmap-rhs`). Results
are appended to a CSV tagged with the auto-detected device name, so runs across
machines / solvers accumulate in one file.

**Scale-invariant by design.** The cantilever's physical dimensions (`--dims`,
default `8 2 2`) are held fixed; only the mesh is refined to hit each target
DOF. Every DOF point is therefore the *same physical problem* at a different
resolution, and the per-DOF numbers are directly comparable.

### Usage

```bash
# default quick sweep (krylov CG, 20k..300k DOF), all five modes
python bench_linear_elasticity.py

# explicit DOF targets and the direct (cuDSS) solver, forward + adjoint only
python bench_linear_elasticity.py --dofs 20000 100000 300000 --solver direct --modes jit grad

# load-batched vmap throughput only, batch of 16, single precision
python bench_linear_elasticity.py --dofs 50000 --modes vmap-rhs --batch 16 --no-x64

# label the run and choose the output file
python bench_linear_elasticity.py --output runs/gb10.csv --tag gb10-nightly
```

The full sweeps used for the docs dashboard (50k → 1M DOF, with a host-memory
watchdog) go through the launcher:

```bash
bash run_guarded.sh direct jit          # solver: direct | amg | krylov | cpu-direct
bash run_guarded.sh amg grad            # mode:   eager | jit | grad | vmap-lhs | vmap-rhs
bash run_guarded.sh krylov vmap-lhs --tol 1e-6   # extra flags are forwarded
```

`run_cudss_guarded.sh`, `run_krylov_guarded.sh` and `run_cpu_direct.sh` are kept
as one-line wrappers around it.

Key arguments (`--help` for the full list):

| Arg | Meaning |
|---|---|
| `--dofs N ...` | Target DOF counts; the mesh is refined to the closest achievable count. Default `20000 60000 150000 300000`; `run_guarded.sh` passes its own sweep. |
| `--dims LX LY LZ` | Fixed physical cantilever dimensions (scale-invariant). |
| `--solver` | `krylov` (matrix-free CG, memory-lean — default), `direct` (cuDSS / host sparse), `amg` (rigid-body near-null-space). |
| `--modes` | Subset of `eager jit grad vmap-lhs vmap-rhs` (default: all). |
| `--batch` | `vmap` batch size (independent cases). |
| `--repeats` / `--warmup` | Timed repeats / untimed warmups per point. |
| `--tol` / `--maxiter` | Krylov convergence controls. |
| `--x64` / `--no-x64` | float64 (default) / float32. |
| `--cpu` | Force the JAX CPU backend (the direct solver becomes a host sparse solver). |
| `--output` / `--tag` | CSV path / free-form label column. |

### Modes

| Mode | What runs | Notes |
|---|---|---|
| `eager` | `solver(tp, u0, traced_structure=ts)` op-by-op | no compile |
| `jit` | `jax.jit` of the same call | |
| `grad` | `jax.jit(jax.value_and_grad(J))`, `J = ½‖u‖²`, w.r.t. the nodal Young's modulus field | forward + adjoint solve; `grad / jit` at one DOF is the adjoint overhead |
| `vmap-lhs` | `jax.jit(jax.vmap(...))` over the **material** `E` | `A` varies per case → `direct` factorizes `B` times |
| `vmap-rhs` | the same over the **load** (traction) | `A` fixed → `direct` factors once and multi-RHS solves (`reuse_factorization=True`); iterative solvers run per-RHS iterations, kept for parity |

All five modes apply to all three solvers. With `linear=True`, `AMGSolverOptions`
is lowered inside `create_solver` to a matrix-free Krylov solve preconditioned by a
fixed V-cycle (the hierarchy is built once from a sample Jacobian), so the solve
itself jits, vmaps and differentiates like the others.

### What is measured

Per `(dof, mode)`: `compile_s` (for the compiled modes: the ahead-of-time
`jit(f).lower(...).compile()` wall time, trace + XLA compile with **no execution**
in it; the timed repeats call that compiled executable directly), `mean_s` /
`std_s` / `min_s` over `--repeats` timed calls (each `block_until_ready`-synced),
`throughput_dofps` (`dof * batch / mean_s`), and `amortized_s` (`mean_s / batch`).
A build or solve that raises (e.g. OOM at high DOF) is recorded with
`mean_s = nan` and the exception name in `note`, so one failure does not abort
the sweep.

The `device` column comes from `jax.devices()[0].device_kind` (with an
`nvidia-smi` fallback); override it with `--device-name`.

### Where results go

* `bench.csv` next to the script — the local append target. It is git-ignored.
* `results/*.csv` — tracked result sets. Move the rows worth keeping there
  (`results/dashboard_2026-07.csv` holds the runs the docs dashboard shipped with;
  their `compile_s` still includes one execution, see `note`).
* `sync_docs_dashboard.py` merges `bench.csv` + `results/*.csv` into
  `docs/static/benchmarks/dashboard.html` (data rows and the mode list), and
  `plot_bench.py` draws a static two-panel PNG from any CSVs.

### Comparing `direct` vs `amg` fairly

All solvers solve the *same* assembled `A u = b`, but the timed region is not
identical work. Each solver hoists whatever it can precompute *before* seeing the
matrix values into `create_solver` (untimed) and times the rest — but the
"precomputable" boundary differs by method, so this is **not** an arbitrary tilt:

- **`amg`** builds its whole multigrid hierarchy once at setup (its main cost) and
  reuses that fixed V-cycle as values change; each timed solve is just matrix-free
  `A@v` + the preconditioner. AMG's dominant cost is value-independent, so it
  legitimately belongs at setup.
- **`direct`** can only do symbolic analysis at setup; the numeric factorization
  (its main cost) is inseparable from the matrix *values* and recurs every timed
  solve. `reuse_factorization=True` shares factors forward↔adjoint and powers
  `vmap-rhs` (factor-once / solve-many) — it does **not** cache across independent
  solves.

So the `jit`/`eager` single-solve columns model *"many systems, same pattern,
changing values"* (sweeps / topopt), where `amg` has a real edge; `grad` adds the
adjoint, where `direct` reuses its factorization and `amg` its hierarchy; the
`vmap-rhs` column models *"one matrix, many RHS,"* where `direct` amortizes its
factorization and wins. Also note accuracy is not equalized: `direct` is exact to
round-off while `amg`/`krylov` stop at `tol` (default `1e-8`).
