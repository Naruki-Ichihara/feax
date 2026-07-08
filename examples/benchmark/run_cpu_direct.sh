#!/usr/bin/env bash
#
# CPU direct-solver benchmark (host sparse: spsolve / umfpack / cholmod — NOT cuDSS).
#
# Forces the JAX CPU backend (--cpu) with --solver direct, which on CPU resolves
# to a host sparse direct solver. A MemAvailable watchdog guards against a large
# host factorization exhausting RAM. CPU direct factorization of 3-D elasticity
# is costly, so the DOF sweep is capped well below the GPU runs.
#
# NOTE: there is no cuDSS on CPU, so vmap-rhs has NO factor-once benefit here
# (it performs the same as vmap-lhs).
#
# Usage:
#   bash run_cpu_direct.sh eager        # host direct, DOF sweep to ~300k
#   bash run_cpu_direct.sh jit
#   bash run_cpu_direct.sh vmap-lhs     # material-batched, <=150k, batch 6
#   bash run_cpu_direct.sh vmap-rhs     # load-batched, <=150k, batch 6 (no reuse on CPU)
#   RESERVE_GIB=8 bash run_cpu_direct.sh jit --dofs 50000 100000   # override sweep
#
# Extra flags after the mode are forwarded to bench_linear_elasticity.py (a
# forwarded --dofs / --output overrides the defaults below). Results append to
# the shared bench.csv next to this script.
#
set -uo pipefail

MODE="${1:?usage: run_cpu_direct.sh <eager|jit|vmap-lhs|vmap-rhs> [extra flags]}"
shift

RESERVE_GIB="${RESERVE_GIB:-10}"
POLL_S="${POLL_S:-0.25}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_linear_elasticity.py"
OUT="$HERE/bench.csv"
RESERVE_KB=$(( RESERVE_GIB * 1024 * 1024 ))

# CPU-appropriate DOF caps (host direct factorization is expensive).
case "$MODE" in
    vmap-*)
        DOFS=(20000 60000 150000)
        BATCH=6
        ;;
    *)
        DOFS=(20000 60000 150000 300000)
        BATCH=8
        ;;
esac

echo "[guard] mode=$MODE  solver=direct(cpu)  reserve=${RESERVE_GIB}GiB  dofs=${DOFS[*]}  batch=$BATCH  out=$OUT"

# -u => unbuffered python stdout so progress lines appear live.
stdbuf -oL -eL python -u "$SCRIPT" \
    --cpu \
    --solver direct \
    --modes "$MODE" \
    --batch "$BATCH" \
    --dofs "${DOFS[@]}" \
    --output "$OUT" \
    --tag "cpu_direct_${MODE}" \
    "$@" &
PY_PID=$!

# ---- host-memory watchdog ----
while kill -0 "$PY_PID" 2>/dev/null; do
    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt "$RESERVE_KB" ]; then
        echo ""
        echo "[guard] !!! MemAvailable $((avail_kb/1024/1024))GiB < ${RESERVE_GIB}GiB -> KILLING solver (pid $PY_PID)"
        kill -9 "$PY_PID" 2>/dev/null
        pkill -9 -P "$PY_PID" 2>/dev/null
        wait "$PY_PID" 2>/dev/null
        echo "[guard] solver killed. Host protected. Partial CSV kept (rows flushed per DOF)."
        exit 137
    fi
    sleep "$POLL_S"
done

wait "$PY_PID"; rc=$?
echo "[guard] solver exited rc=$rc"
exit "$rc"
