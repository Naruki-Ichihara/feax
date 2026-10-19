#!/usr/bin/env bash
#
# Guarded benchmark launcher: one DOF sweep of bench_linear_elasticity.py for a
# given solver and execution mode, under a HOST-MEMORY WATCHDOG.
#
# On unified-memory machines (e.g. GB10, Grace-Blackwell) a cuDSS compile at
# large DOF can spike host RAM and freeze the OS — not a catchable Python OOM.
# cgroup limits are not always available, so a watchdog polls /proc/meminfo and
# SIGKILLs the benchmark the instant MemAvailable drops below RESERVE_GIB. That
# sacrifices the run but keeps the host alive; rows already finished are safe
# (the benchmark fsyncs each row).
#
# Usage:
#   bash run_guarded.sh <solver> <mode> [extra bench flags]
#     solver: direct | amg | krylov | cpu-direct
#     mode:   eager | jit | grad | vmap-lhs | vmap-rhs
#
#   bash run_guarded.sh direct jit
#   bash run_guarded.sh amg grad
#   bash run_guarded.sh krylov vmap-lhs --tol 1e-6 --maxiter 5000
#   RESERVE_GIB=20 bash run_guarded.sh direct vmap-rhs
#
# Sweeps: eager / jit / grad go to 1,000,000 DOF (batch 8 is irrelevant there);
# the vmap modes are capped at 500k DOF with batch 10. cpu-direct uses smaller
# caps (host factorization of 3-D elasticity is expensive). A forwarded --dofs /
# --batch / --output overrides these. Results append to bench.csv next to this
# script; move the rows you want to keep into results/ and run
# sync_docs_dashboard.py.
#
set -uo pipefail

SOLVER="${1:?usage: run_guarded.sh <direct|amg|krylov|cpu-direct> <eager|jit|grad|vmap-lhs|vmap-rhs> [flags]}"
MODE="${2:?usage: run_guarded.sh <solver> <eager|jit|grad|vmap-lhs|vmap-rhs> [flags]}"
shift 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/bench_linear_elasticity.py"
OUT="$HERE/bench.csv"
POLL_S="${POLL_S:-0.25}"
EXTRA=()

case "$SOLVER" in
    direct)      BENCH_SOLVER=direct; RESERVE_DEFAULT=15 ;;
    amg)         BENCH_SOLVER=amg;    RESERVE_DEFAULT=12 ;;
    krylov)      BENCH_SOLVER=krylov; RESERVE_DEFAULT=12 ;;
    cpu-direct)  BENCH_SOLVER=direct; RESERVE_DEFAULT=10; EXTRA+=(--cpu) ;;
    *) echo "unknown solver '$SOLVER' (direct|amg|krylov|cpu-direct)"; exit 2 ;;
esac

case "$MODE" in
    vmap-*)
        # A jit(vmap) compile spikes host memory faster than eager/jit; watch tighter.
        POLL_S="${POLL_S:-0.1}"
        if [ "$SOLVER" = cpu-direct ]; then DOFS=(20000 60000 150000); BATCH=6
        else DOFS=(50000 100000 200000 350000 500000); BATCH=10; fi
        [ "$SOLVER" = direct ] && RESERVE_DEFAULT=20
        ;;
    eager|jit|grad)
        if [ "$SOLVER" = cpu-direct ]; then DOFS=(20000 60000 150000 300000); BATCH=8
        else DOFS=(50000 100000 200000 350000 500000 750000 1000000); BATCH=8; fi
        ;;
    *) echo "unknown mode '$MODE' (eager|jit|grad|vmap-lhs|vmap-rhs)"; exit 2 ;;
esac

RESERVE_GIB="${RESERVE_GIB:-$RESERVE_DEFAULT}"
RESERVE_KB=$(( RESERVE_GIB * 1024 * 1024 ))

echo "[guard] solver=$SOLVER mode=$MODE reserve=${RESERVE_GIB}GiB"
echo "[guard] defaults: dofs=${DOFS[*]} batch=$BATCH out=$OUT${*:+  (forwarded flags override: $*)}"

# -u => unbuffered python stdout so progress lines appear live.
stdbuf -oL -eL python -u "$SCRIPT" \
    --solver "$BENCH_SOLVER" \
    --modes "$MODE" \
    --batch "$BATCH" \
    --dofs "${DOFS[@]}" \
    --output "$OUT" \
    --tag "${SOLVER}_${MODE}" \
    "${EXTRA[@]}" "$@" &
PY_PID=$!

# ---- host-memory watchdog ----
while kill -0 "$PY_PID" 2>/dev/null; do
    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail_kb" -lt "$RESERVE_KB" ]; then
        echo ""
        echo "[guard] !!! MemAvailable $((avail_kb/1024/1024))GiB < ${RESERVE_GIB}GiB -> KILLING benchmark (pid $PY_PID) to protect host"
        kill -9 "$PY_PID" 2>/dev/null
        pkill -9 -P "$PY_PID" 2>/dev/null
        wait "$PY_PID" 2>/dev/null
        echo "[guard] benchmark killed. Host protected. Rows flushed so far are in $OUT"
        exit 137
    fi
    sleep "$POLL_S"
done

wait "$PY_PID"; rc=$?
echo "[guard] benchmark exited rc=$rc"
exit "$rc"
