#!/usr/bin/env bash
#
# Krylov (matrix-free CG) scaling benchmark — the iterative counterpart to
# run_cudss_guarded.sh, for "single-shot 1M-DOF in how many seconds" and to put
# FEAX on the same footing as an iterative-CG benchmark (e.g. torchSLA's CG
# numbers) rather than direct-vs-iterative apples-to-oranges.
#
# CG is matrix-free (a residual-JVP matvec, no assembled matrix, no
# factorization) so memory scales ~linearly and the host-crash risk that dogs the
# cuDSS path is minimal. The same MemAvailable watchdog is kept purely as a guard
# against a compile-time constant-folding spike.
#
# NOTE: vmap-b (load-batched) has NO factor-reuse benefit here — CG has no
# factorization, so each RHS runs its own CG iterations. The mode is kept for
# parity with run_cudss_guarded.sh. Convergence is controlled by CG tol/maxiter;
# override by forwarding flags, e.g.  run_krylov_guarded.sh jit --tol 1e-6.
#
# Usage:
#   bash run_krylov_guarded.sh eager                 # eager, sweep to 1,000,000
#   bash run_krylov_guarded.sh jit                   # jit,   sweep to 1,000,000
#   bash run_krylov_guarded.sh vmap-a                # material-batched, <=500k, b=10
#   bash run_krylov_guarded.sh vmap-b                # load-batched,     <=500k, b=10
#   bash run_krylov_guarded.sh jit --tol 1e-6 --maxiter 5000   # match a CG tol
#
# Results append to results_krylov_1M.csv (overlay with results_direct_1M.csv:
#   python plot_bench.py results_direct_1M.csv results_krylov_1M.csv).
#
set -uo pipefail

MODE="${1:?usage: run_krylov_guarded.sh <eager|jit|vmap-a|vmap-b> [extra bench flags]}"
shift                                       # remaining args forwarded to the bench

case "$MODE" in
    vmap-*)
        export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=constant_folding}"
        RESERVE_GIB="${RESERVE_GIB:-15}"
        POLL_S="${POLL_S:-0.1}"
        echo "[guard] $MODE: XLA_FLAGS=$XLA_FLAGS"
        ;;
    *)
        RESERVE_GIB="${RESERVE_GIB:-12}"    # CG is light; small reserve is plenty
        POLL_S="${POLL_S:-0.25}"
        ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_linear_elasticity.py"
OUT="$HERE/results_krylov_1M.csv"
RESERVE_KB=$(( RESERVE_GIB * 1024 * 1024 ))

# vmap-*: cap DOF at 500k, batch 10.  eager/jit: full sweep to 1M.
case "$MODE" in
    vmap-*)
        DOFS=(50000 100000 200000 350000 500000)
        BATCH=10
        ;;
    *)
        DOFS=(50000 100000 200000 350000 500000 750000 1000000)
        BATCH=8
        ;;
esac

echo "[guard] mode=$MODE  solver=krylov  reserve=${RESERVE_GIB}GiB  dofs=${DOFS[*]}  batch=$BATCH  out=$OUT"

# -u => unbuffered python stdout so progress lines appear live.
stdbuf -oL -eL python -u "$SCRIPT" \
    --solver krylov \
    --modes "$MODE" \
    --batch "$BATCH" \
    --dofs "${DOFS[@]}" \
    --output "$OUT" \
    --tag "krylov_${MODE}" \
    "$@" &
PY_PID=$!

# ---- watchdog ----
while kill -0 "$PY_PID" 2>/dev/null; do
    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt "$RESERVE_KB" ]; then
        echo ""
        echo "[guard] !!! MemAvailable $((avail_kb/1024/1024))GiB < ${RESERVE_GIB}GiB -> KILLING solver (pid $PY_PID) to protect host"
        kill -9 "$PY_PID" 2>/dev/null
        pkill -9 -P "$PY_PID" 2>/dev/null
        wait "$PY_PID" 2>/dev/null
        echo "[guard] solver killed. Host protected. Partial CSV (rows flushed per DOF) in $OUT"
        exit 137
    fi
    sleep "$POLL_S"
done

wait "$PY_PID"
rc=$?
echo "[guard] solver exited rc=$rc"
exit "$rc"
