#!/usr/bin/env bash
# Build SuiteSparse from source for environments where the system package is
# too old (e.g. Google Colab / Ubuntu 22.04 ship 5.10.1, scikit-sparse >= 0.5.0
# requires SuiteSparse >= 7.4.0).
#
# Usage:
#   bash scripts/colab_install_suitesparse.sh
#   SUITESPARSE_VERSION=v7.8.0 bash scripts/colab_install_suitesparse.sh
#
# After this script completes, install feax with:
#   pip install feax[sksparse]

set -euo pipefail

SUITESPARSE_VERSION="${SUITESPARSE_VERSION:-v7.8.0}"
SUITESPARSE_SRC="${SUITESPARSE_SRC:-/tmp/suitesparse-src}"
SUITESPARSE_PREFIX="${SUITESPARSE_PREFIX:-/usr/local}"

apt-get install -y cmake libopenblas-dev liblapack-dev

rm -rf "$SUITESPARSE_SRC"
git clone --depth 1 --branch "$SUITESPARSE_VERSION" \
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git "$SUITESPARSE_SRC"

# SUITESPARSE_USE_CUDA=OFF is required: CMake otherwise auto-detects nvcc and
# tries to build CHOLMOD GPU kernels for compute_52, which CUDA 12+ rejects.
cmake -S "$SUITESPARSE_SRC" -B "$SUITESPARSE_SRC/build" \
    -DCMAKE_INSTALL_PREFIX="$SUITESPARSE_PREFIX" \
    -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;colamd;ccolamd;btf;cholmod;klu;umfpack;spqr" \
    -DSUITESPARSE_USE_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBLA_VENDOR=OpenBLAS

cmake --build "$SUITESPARSE_SRC/build" -j"$(nproc)"
cmake --install "$SUITESPARSE_SRC/build"
ldconfig
