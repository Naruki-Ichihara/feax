#!/usr/bin/env bash
#
# cuDSS (direct-solver) scaling benchmark with a HOST-MEMORY WATCHDOG.
#
# On this GB10 (Grace-Blackwell UNIFIED memory) a cuDSS jit compile at large DOF
# can spike host RAM and freeze the whole OS (not a catchable Python OOM).
# cgroup limits are unavailable here (cgroupfs is read-only in this container),
# so instead a watchdog polls /proc/meminfo every 0.25 s and SIGKILLs the solver
# process the instant free host memory drops below --reserve GiB. That sacrifices
# the run but keeps the host alive.
#
# cuDSS vmap is memory-safe: under jax.vmap a batched matrix is factorized one
# element at a time via jax.lax.map (spineax custom_vmap), so peak memory stays
# at ~one factorization — NOT batch-multiplied. vmap is capped at 500k DOF,
# batch 10 (per the original spec); eager/jit sweep to 1,000,000 DOF.
#
# Usage:
#   bash run_cudss_guarded.sh eager        # eager mode, DOF sweep to 1,000,000
#   bash run_cudss_guarded.sh jit          # jit mode  (the dangerous one)
#   bash run_cudss_guarded.sh vmap         # vmap, <=500k DOF, batch 10
#   RESERVE_GIB=15 bash run_cudss_guarded.sh eager
#
# Results append to results_direct_1M.csv next to this script.
#
set -uo pipefail

MODE="${1:?usage: run_cudss_guarded.sh <eager|jit|vmap>}"

# vmap's jit compile constant-folds a huge gather (pred[~39M]) whose host-side
# evaluation spikes RAM fast enough to trip the watchdog. Disabling the XLA
# constant_folding pass leaves that gather as a runtime op (no compile-time host
# evaluation), and we watch tighter because the spike is sub-0.25s.
if [ "$MODE" = "vmap" ]; then
    export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=constant_folding}"
    RESERVE_GIB="${RESERVE_GIB:-20}"
    POLL_S="${POLL_S:-0.1}"
    echo "[guard] vmap: XLA_FLAGS=$XLA_FLAGS"
else
    RESERVE_GIB="${RESERVE_GIB:-15}"      # kill solver if MemAvailable < this
    POLL_S="${POLL_S:-0.25}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_linear_elasticity.py"
OUT="$HERE/results_direct_1M.csv"
RESERVE_KB=$(( RESERVE_GIB * 1024 * 1024 ))

# vmap: cap DOF at 500k, batch 10.  eager/jit: full sweep to 1M, batch irrelevant.
if [ "$MODE" = "vmap" ]; then
    DOFS=(50000 100000 200000 350000 500000)
    BATCH=10
else
    DOFS=(50000 100000 200000 350000 500000 750000 1000000)
    BATCH=8
fi

echo "[guard] mode=$MODE  reserve=${RESERVE_GIB}GiB  dofs=${DOFS[*]}  batch=$BATCH  out=$OUT"

# -u => unbuffered python stdout so progress lines appear live.
stdbuf -oL -eL python -u "$SCRIPT" \
    --solver direct \
    --modes "$MODE" \
    --batch "$BATCH" \
    --dofs "${DOFS[@]}" \
    --output "$OUT" \
    --tag "cudss_${MODE}" &
PY_PID=$!

# ---- watchdog ----
while kill -0 "$PY_PID" 2>/dev/null; do
    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt "$RESERVE_KB" ]; then
        echo ""
        echo "[guard] !!! MemAvailable $((avail_kb/1024/1024))GiB < ${RESERVE_GIB}GiB -> KILLING solver (pid $PY_PID) to protect host"
        kill -9 "$PY_PID" 2>/dev/null
        # kill any child python too
        pkill -9 -P "$PY_PID" 2>/dev/null
        wait "$PY_PID" 2>/dev/null
        echo "[guard] solver killed. Host protected. Partial CSV (if any phase finished) is in $OUT"
        exit 137
    fi
    sleep "$POLL_S"
done

wait "$PY_PID"
rc=$?
echo "[guard] solver exited rc=$rc"
exit "$rc"
