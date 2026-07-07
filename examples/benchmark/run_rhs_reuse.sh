#!/usr/bin/env bash
#
# Factor-once/solve-many (RHS-batched) benchmark, host-memory guarded.
#
# Same watchdog as run_cudss_guarded.sh: polls /proc/meminfo and SIGKILLs the
# solver if free host RAM drops below --reserve GiB, so a jit(vmap) compile spike
# can't freeze the GB10. XLA constant_folding is disabled (the spike source).
#
# Usage:
#   bash run_rhs_reuse.sh                     # full sweep, batch 10
#   RESERVE_GIB=25 bash run_rhs_reuse.sh --dofs 500000
#
# Extra args are forwarded to bench_rhs_reuse.py. Results append to
# results_rhs_reuse.csv next to this script.
#
set -uo pipefail

RESERVE_GIB="${RESERVE_GIB:-20}"
POLL_S="${POLL_S:-0.1}"
export XLA_FLAGS="${XLA_FLAGS:---xla_disable_hlo_passes=constant_folding}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_rhs_reuse.py"
RESERVE_KB=$(( RESERVE_GIB * 1024 * 1024 ))

echo "[guard] reserve=${RESERVE_GIB}GiB  XLA_FLAGS=$XLA_FLAGS  args: $*"

stdbuf -oL -eL python -u "$SCRIPT" "$@" &
PY_PID=$!

while kill -0 "$PY_PID" 2>/dev/null; do
    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt "$RESERVE_KB" ]; then
        echo ""
        echo "[guard] !!! MemAvailable $((avail_kb/1024/1024))GiB < ${RESERVE_GIB}GiB -> KILLING solver (pid $PY_PID)"
        kill -9 "$PY_PID" 2>/dev/null
        pkill -9 -P "$PY_PID" 2>/dev/null
        wait "$PY_PID" 2>/dev/null
        echo "[guard] solver killed. Host protected. Partial CSV kept (rows are flushed per DOF)."
        exit 137
    fi
    sleep "$POLL_S"
done

wait "$PY_PID"; rc=$?
echo "[guard] solver exited rc=$rc"
exit "$rc"
