FROM nvcr.io/nvidia/jax:26.08-py3

ARG INSTALL_DOCS=false

RUN apt update
RUN apt upgrade -y

RUN apt -y install gmsh python3-gmsh libsuitesparse-dev swig libnlopt-dev
RUN pip install --upgrade pip

# Pin the base image's JAX / jaxlib. The NVCR image ships JAX built from source
# against its CUDA plugins; any PyPI jax/jaxlib pulled in by a dependency breaks
# the plugin ABI ("PJRT_FFI_UserData_Add_Args size" errors at the first GPU op).
# PIP_CONSTRAINT also applies to pip run later inside the container, so an
# accidental upgrade fails at resolve time instead of silently replacing JAX.
RUN python -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('jax', 'jaxlib')))" \
        > /etc/pip-constraints-jax.txt && \
    cat /etc/pip-constraints-jax.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints-jax.txt

# nlopt's PyPI sdist pins cmake_minimum_required(VERSION <3.5); CMake 4.x dropped
# compat for policy versions <3.5 and hard-errors. CMAKE_POLICY_VERSION_MINIMUM=3.5
# tells CMake to configure anyway.
RUN CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation nlopt

# Copy feax source code
COPY . /workspace
WORKDIR /workspace

# Install feax dependencies (core + cuda13 + sksparse + amg extras) from
# pyproject.toml, except asdex. asdex declares jax>=0.11.0, which an NVCR
# pre-release build (e.g. 0.11.0.devYYYYMMDD) does not satisfy under PEP 440, so
# it is installed with --no-deps and its remaining deps are listed explicitly.
RUN python -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; deps = p['dependencies'] + [d for e in ('cuda13', 'sksparse', 'amg') for d in p['optional-dependencies'][e]]; print('\n'.join(d for d in deps if not d.startswith('asdex')))" \
        > /tmp/feax-requirements.txt && \
    pip install -r /tmp/feax-requirements.txt && \
    pip install --no-deps asdex && \
    pip install "numba>=0.63.0"

# Editable install: docker-compose mounts the source tree over /workspace, so the
# package must point at it rather than at a stale copy in site-packages.
RUN pip install --no-deps -e .

# Install spineax WITHOUT touching the base JAX. --no-deps is essential: both the
# wheel and its lineax dep declare jax[cuda13]. The official PyPI release ships an
# x86_64-only abi3 wheel, so on other arches (e.g. aarch64 / GB10) fall back to
# building the fork from source against the container's own jaxlib (matching the
# cuDSS FFI ABI at runtime). cuDSS 0.8 (libcudss.so.0), cuSPARSE and lineax are the
# runtime deps installed here; jaxtyping / equinox already come from above.
# spineax's CMake only searches site-packages/nvidia/*/lib, so cuSPARSE must come from
# the pip wheel (not /usr/local/cuda). nvidia-cuda-cccl provides <nv/target>, which
# CUDA 13's cusparse.h -> cuda_fp16.h includes, for the source build.
RUN pip install nvidia-cudss-cu13 nvidia-cusparse nvidia-cuda-cccl && \
    pip install --no-deps lineax && \
    if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-deps spineax ; \
    else \
        pip install --no-build-isolation "scikit-build-core>=0.5" nanobind && \
        pip install --no-build-isolation --no-deps \
            "git+https://github.com/johnviljoen/spineax.git" ; \
    fi

# Fail the build if the base JAX was replaced or feax does not import.
RUN python -c "import importlib.metadata as m; pins = dict(l.strip().split('==') for l in open('/etc/pip-constraints-jax.txt')); bad = {p: m.version(p) for p, v in pins.items() if m.version(p) != v}; assert not bad, f'base JAX replaced: {bad} (expected {pins})'; import feax, asdex, amjax, pyamg"

# Optional: Node.js 20 + pydoc-markdown + Docusaurus dependencies
# Build with: docker build --build-arg INSTALL_DOCS=true .
RUN if [ "$INSTALL_DOCS" = "true" ]; then \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs && \
        pip install pydoc-markdown && \
        cd /workspace/docs && npm install; \
    fi

CMD ["/bin/bash"]
