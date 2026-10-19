#!/usr/bin/env bash
# Thin wrapper kept for the old command line; see run_guarded.sh.
#   bash run_cudss_guarded.sh <mode> [extra bench flags]
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_guarded.sh" direct "$@"
