FROM nvcr.io/nvidia/jax:26.05-py3

RUN apt update
RUN apt upgrade -y

RUN apt -y install gmsh python3-gmsh libsuitesparse-dev swig libnlopt-dev
RUN pip install --upgrade pip
# nlopt's PyPI sdist pins cmake_minimum_required(VERSION <3.5); CMake 4.x dropped
# compat for policy versions <3.5 and hard-errors. CMAKE_POLICY_VERSION_MINIMUM=3.5
# tells CMake to configure anyway.
RUN CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation nlopt

# Copy feax source code
COPY . /workspace
WORKDIR /workspace

# Install feax with CUDA dependencies.
# JAX is intentionally not reinstalled here: the NVCR base image already ships
# JAX compiled for CUDA 13.1.1. Use pip install .[cuda13,jax] outside containers.
RUN pip install .[cuda13,sksparse]

# Install spineax (patched fork for cuDSS 0.8) WITHOUT touching the base JAX.
# Pre-install the build backend + cuDSS 0.8 (headers for build, libcudss.so.0 for
# runtime), then build spineax against the container's own jaxlib headers so the
# FFI ABI matches at runtime. --no-deps keeps the NVCR base JAX in place.
RUN pip install --no-build-isolation "scikit-build-core>=0.5" nanobind nvidia-cudss-cu13
RUN pip install --no-build-isolation --no-deps \
    "git+https://github.com/Naruki-Ichihara/spineax.git"

# Optional: Node.js 20 + pydoc-markdown + Docusaurus dependencies
# Build with: docker build --build-arg INSTALL_DOCS=true .
RUN if [ "$INSTALL_DOCS" = "true" ]; then \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs && \
        pip install pydoc-markdown && \
        cd /workspace/docs && npm install; \
    fi

CMD ["/bin/bash"]
