#!/usr/bin/env bash
#
# Linear-elasticity scaling benchmark driver.
#
#   * eager / jit : DOF sweep up to 1,000,000 DOF
#   * vmap        : DOF sweep up to   500,000 DOF, batch = 10
#
# Both phases append to the SAME CSV (results_1M.csv next to this script), so
# the vmap rows land alongside the eager/jit rows for every shared DOF point.
#
# Solver: krylov (matrix-free CG) — memory scales, safe to 1M DOF on GB10.
# (The cuDSS direct path is NOT used here; its jit compile can OOM the host.)
#
# Usage:  bash run_bench_1M.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_linear_elasticity.py"
OUT="$HERE/results_1M.csv"

# Fresh CSV so a re-run is self-contained (header written once by the script).
rm -f "$OUT"

# ---- Phase 1: vmap (+eager/jit) up to 500k DOF, batch 10 -------------------
python "$SCRIPT" \
    --solver krylov \
    --modes eager jit vmap \
    --batch 10 \
    --dofs 50000 100000 200000 350000 500000 \
    --output "$OUT" \
    --tag "sweep<=500k_vmapB10"

# ---- Phase 2: eager/jit only, 500k -> 1M DOF ------------------------------
python "$SCRIPT" \
    --solver krylov \
    --modes eager jit \
    --dofs 750000 1000000 \
    --output "$OUT" \
    --tag "sweep>500k_novmap"

echo
echo "Done. Combined results in: $OUT"
