#!/usr/bin/env bash
# Set up FEAX system dependencies on Google Colab and other Ubuntu environments.
# Mirrors the project Dockerfile so the same environment is reproducible outside
# the container.
#
# What this script does:
#   1. apt: gmsh + Python bindings, swig, NLopt headers, BLAS/LAPACK,
#      and the gmsh/OCCT runtime libraries needed on headless Colab.
#   2. Build SuiteSparse 7.x from source. Required because Ubuntu 22.04
#      ships SuiteSparse 5.10.1 via libsuitesparse-dev, but scikit-sparse
#      >= 0.5.0 needs >= 7.4.0. SUITESPARSE_USE_CUDA=OFF avoids an
#      `nvcc compute_52` failure on Colab GPU runtimes.
#   3. pip: NLopt Python bindings (--no-build-isolation, needs libnlopt-dev).
#
# After this script completes, install FEAX with one of:
#   pip install git+https://github.com/Naruki-Ichihara/feax.git                                # CPU baseline
#   SUITESPARSE_INCLUDE_DIR=/usr/local/include/suitesparse SUITESPARSE_LIBRARY_DIR=/usr/local/lib \
#     pip install "feax[sksparse] @ git+https://github.com/Naruki-Ichihara/feax.git"           # + cholmod/umfpack
#   SUITESPARSE_INCLUDE_DIR=/usr/local/include/suitesparse SUITESPARSE_LIBRARY_DIR=/usr/local/lib \
#     pip install "feax[cuda13,sksparse] @ git+https://github.com/Naruki-Ichihara/feax.git"    # GPU
# For the cuDSS direct solver also install spineax:
#   CMAKE_ARGS="-DBUILD_PBATCH_SOLVE=OFF" \
#     pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git
#
# The SUITESPARSE_* env vars are needed because scikit-sparse's setup.py only
# auto-searches /usr/include/suitesparse (the apt path); we install to /usr/local.
# CMAKE_ARGS=-DBUILD_PBATCH_SOLVE=OFF works around a __cudaGetKernel link error
# in spineax's optional pbatch_solve module on Colab GPU.
#
# Usage:
#   bash scripts/colab_setup.sh
#   SUITESPARSE_VERSION=v7.8.0 bash scripts/colab_setup.sh

set -euo pipefail

SUITESPARSE_VERSION="${SUITESPARSE_VERSION:-v7.8.0}"
SUITESPARSE_SRC="${SUITESPARSE_SRC:-/tmp/suitesparse-src}"
SUITESPARSE_PREFIX="${SUITESPARSE_PREFIX:-/usr/local}"

# 1. System packages (mirrors Dockerfile + Colab gmsh runtime libs)
apt-get update
apt-get install -y \
    cmake git swig \
    gmsh python3-gmsh \
    libnlopt-dev \
    libopenblas-dev liblapack-dev \
    libglu1 libxcursor-dev libxft2 libxinerama1 \
    libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev \
    libocct-foundation-dev libocct-data-exchange-dev

# 2. Build SuiteSparse from source (only the projects scikit-sparse links against)
rm -rf "$SUITESPARSE_SRC"
git clone --depth 1 --branch "$SUITESPARSE_VERSION" \
    https://github.com/DrTimothyAldenDavis/SuiteSparse.git "$SUITESPARSE_SRC"

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

# 3. NLopt Python bindings (Dockerfile installs this with --no-build-isolation)
pip install --no-build-isolation nlopt
