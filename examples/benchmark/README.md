# feax benchmarks

## `bench_linear_elasticity.py`

Scaling benchmark for a linear-elasticity **cantilever** across a DOF sweep, in
three execution modes (`eager`, `jit`, `vmap`). Results are appended to a CSV
tagged with the auto-detected device name, so runs across machines/solvers
accumulate in one file.

**Scale-invariant by design.** The cantilever's physical dimensions (`--dims`,
default `8 2 2`) are held fixed; only the mesh is refined to hit each target
DOF. Every DOF point is therefore the *same physical problem* at a different
resolution, and the per-DOF numbers are directly comparable.

### Usage

```bash
# default sweep (krylov CG), all three modes
python bench_linear_elasticity.py

# explicit DOF targets and a direct (cuDSS) solver
python bench_linear_elasticity.py --dofs 20000 100000 300000 --solver direct

# vmap throughput only, batch of 16, single precision
python bench_linear_elasticity.py --dofs 50000 --modes vmap --batch 16 --no-x64

# label the run and choose the output file
python bench_linear_elasticity.py --output runs/gb10.csv --tag gb10-nightly
```

Key arguments (`--help` for the full list):

| Arg | Meaning |
|---|---|
| `--dofs N ...` | Target DOF counts; the mesh is refined to the closest achievable count. |
| `--dims LX LY LZ` | Fixed physical cantilever dimensions (scale-invariant). |
| `--solver` | `krylov` (matrix-free CG, memory-lean — default), `direct` (cuDSS/host), `amg` (rigid-body near-null-space; `vmap` is skipped). |
| `--modes` | Subset of `eager jit vmap`. |
| `--batch` | `vmap` batch size (independent stiffness cases). |
| `--repeats` / `--warmup` | Timed repeats / untimed warmups per point. |
| `--tol` / `--maxiter` | Krylov convergence controls. |
| `--x64` / `--no-x64` | float64 (default) / float32. |
| `--output` / `--tag` | CSV path / free-form label column. |

### What is measured

Per `(dof, mode)`: `compile_s` (jit/vmap first-call compile, reported
separately from steady state), `mean_s` / `std_s` / `min_s` over `--repeats`
timed calls (each `block_until_ready`-synced), `throughput_dofps`
(`dof * batch / mean_s`), and `amortized_s` (`mean_s / batch`). A build or solve
that raises (e.g. OOM at high DOF) is recorded with `mean_s = nan` and the
exception name in `note`, so one failure does not abort the sweep.

The `device` column comes from `jax.devices()[0].device_kind` (with an
`nvidia-smi` fallback); override it with `--device-name`.
