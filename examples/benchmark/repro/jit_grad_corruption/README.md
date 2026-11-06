# Reproducers: jit+grad corruption of the batched reduced operator, and the fan-in assembly

Scripts behind the investigation of 2026-11-05 (feax 0.9.0 working tree, GB10 / sm_12.1a,
driver 580.178, jax 0.11.0.dev). Nothing here is part of the library; run from this directory.

## The bug

A compiled gradient program (`jax.jit(jax.grad(...))`) that assembles the batched
periodic-homogenization operator (3 lattices, 375 reduced DOF) **inside** the solver's
`custom_vjp` intermittently returns a wrong gradient. The forward CG hits `maxiter`
(2000 instead of 28 iterations), so the mean operator used by `fe.shared_factor` was
corrupted at program start. Mostly the first execution of a process fails, but later
executions failed too. Eager execution is always correct.

| Script | What it shows |
|---|---|
| `inside_assembly.py <slots>-<pins>` | The failing structure. `scatter-scatter` is the original; `gather-mask` replaces every scatter in the operator assembly with gathers/masks. Both fail. `NRUNS`, `INV_AOT=1` (ahead-of-time compile), `INV_SIMPLIFIED_CONSTS=1` are optional. |
| `inside_assembly_gather.py gatherP\|gatherAll` | Same program with the prolongation products `P @ v`, `Pᵀ @ y` as gathers (no BCOO scatter-add). Still fails. |
| `library_path.py <nruns>` | The shipped structure (`create_batched_homogenization_solver`: operator values and rhs assembled **outside** the `custom_vjp`). 40/40 runs correct. |
| `user_paths.py homog\|vmap` | Single-cell direct homogenization gradient and the `fe.vmap` solver path under jit. Correct. |
| `bench_assembly.py <n>` | Residual and CSR Jacobian assembly time on an n³ HEX8 box; set `FEAX_ASSEMBLY_SCATTER=atomic\|fanin\|scan`. |
| `check_assembly_equal.py HEX8\|TET4` | Assembly results per scatter mode (saved as `asm_<mode>_<ele>.npz`, compared against other modes found in `OUT_DIR`). |

Typical commands:

```bash
for i in 1 2 3; do python inside_assembly.py scatter-scatter; done        # first run usually wrong
XLA_FLAGS=--xla_gpu_deterministic_ops=true python inside_assembly.py scatter-scatter   # rarer, not gone (1/16)
python library_path.py 10                                                 # library path, always correct
FEAX_ASSEMBLY_SCATTER=atomic python bench_assembly.py 30; FEAX_ASSEMBLY_SCATTER=fanin python bench_assembly.py 30
FEAX_ASSEMBLY_SCATTER=atomic python check_assembly_equal.py HEX8; FEAX_ASSEMBLY_SCATTER=fanin python check_assembly_equal.py HEX8
```

## Findings

- On XLA:GPU a sorted `segment_sum` with non-unique ids still compiles to `atomicAdd`
  (checked in the PTX). feax's assembly was therefore atomic regardless of vmap nesting.
- `feax.assembler.segment_sum_fanin` (gather fan-in times, write with unique indices)
  removes every atomic from the assembly. Results agree with the atomic path to 1e-16 and
  are bitwise reproducible. CSR Jacobian: 25.7 → 19.7 ms at 89k DOF, 201 → 153 ms at 681k DOF.
- The corruption persists with the atomic-free assembly and with gathers for `P`, so the
  race is in the kernels XLA generates for the gradient program, outside feax.
  `--xla_gpu_deterministic_ops=true` reduces it (1 failure in 16 first runs) but does not
  remove it.
- `BatchedHomogenizationSolver.__call__` re-evaluates once when CG did not converge and
  raises if it still fails (skipped under `jax.grad`).

Observed outputs (grad rel err / forward CG iterations per run):

```
scatter-scatter  eager its 28 | jit: 2e-01/2000  1e-15/28  1e-15/28  4e-01/2000
gatherAll:       eager its 28 | runs: 8e-01/2000  8e-01/2000  8e-16/28
library batched homogenization grad runs: 2e-12/30 2e-12/30 3e-12/30 ... (40/40 correct)
```
