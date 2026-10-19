#!/usr/bin/env bash
# Thin wrapper kept for the old command line; see run_guarded.sh.
#   bash run_cpu_direct.sh <mode> [extra bench flags]
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_guarded.sh" cpu-direct "$@"
