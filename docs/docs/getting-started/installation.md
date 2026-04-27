# Installation

This guide provides detailed instructions for installing FEAX and its dependencies.

## JAX

FEAX requires JAX as its core dependency. JAX provides automatic differentiation and JIT compilation capabilities.

### CPU-only Installation

For CPU-only usage (Linux/macOS/Windows):

```bash
pip install -U jax
```

This is sufficient for learning and small-scale problems.

### NVIDIA GPU (CUDA 12)

For NVIDIA GPU acceleration with CUDA 12:

```bash
pip install -U "jax[cuda12]"
```

**Requirements:** NVIDIA driver ≥525

### NVIDIA GPU (CUDA 13)

For CUDA 13:

```bash
pip install -U "jax[cuda13]"
```

For more details, see the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## PyPI

### Install FEAX

Once JAX is installed, install FEAX using pip:

```bash
pip install feax
```

This will automatically install all required dependencies including:
- `numpy`, `scipy` - Numerical computing
- `meshio` - Mesh I/O operations
- `gmsh` - Mesh generation
- `fenics-basix` - Finite element basis functions
- `matplotlib` - Visualization
- `pandas` - Data handling
- `equinox` - Neural networks / pytree utilities
- `lineax` - Linear solvers

### Install from Source

To get the latest development version:

```bash
pip install git+https://github.com/Naruki-Ichihara/feax.git@main
```

For development with editable install:

```bash
git clone https://github.com/Naruki-Ichihara/feax.git
cd feax
pip install -e .
```

### Optional Extras

FEAX provides optional dependency groups via `pyproject.toml`:

| Extra | Contents | Usage |
|---|---|---|
| `.[cuda12]` | JAX (cuda12) + cuDSS for CUDA 12 | GPU acceleration (CUDA 12) |
| `.[cuda13]` | cuDSS + cuBLAS + cuDNN for CUDA 13 | GPU acceleration (CUDA 13, without JAX) |
| `.[jax]` | `jax[cuda13]` | JAX for CUDA 13 (use with `cuda13`) |
| `.[sksparse]` | `scikit-sparse` | CPU host-side direct solvers (`cholmod`, `umfpack`) |
| `.[dev]` | pytest, black, ruff, mypy | Development and testing |

### scikit-sparse (Optional)

The `cholmod` and `umfpack` direct solvers wrap [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), which compiles against the SuiteSparse C library. Without it, `DirectSolverOptions()` (auto) falls back to JAX's `spsolve` on CPU or `cudss` on GPU — no functionality is lost for typical use.

To enable `cholmod` / `umfpack`, install the system library first, then the extra:

```bash
# Debian / Ubuntu / Google Colab
apt-get install -y libsuitesparse-dev
pip install feax[sksparse]
```

For GPU use outside Docker, combine `cuda13` with `jax`:

```bash
pip install "feax[cuda13,jax]"
```

### cuDSS Direct Solver

When using the cuDSS direct solver, the `spineax` package is also required:

```bash
pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git
```

> **Note:** This is pre-installed in the Docker image.

## Docker

FEAX provides a Dockerfile based on NVIDIA's JAX image (`nvcr.io/nvidia/jax:25.10-py3`), which includes JAX pre-compiled for CUDA 13. JAX is **not** reinstalled during the build.

### Build Arguments

| Argument | Default | Description |
|---|---|---|
| `INSTALL_DOCS` | `false` | Install Node.js 20 + pydoc-markdown + Docusaurus dependencies |

### Build Docker Image

Standard build:

```bash
git clone https://github.com/Naruki-Ichihara/feax.git
cd feax
docker build -t feax:latest .
```

Build with Docusaurus support (for docs development):

```bash
docker build --build-arg INSTALL_DOCS=true -t feax:latest .
```

### Run Container

```bash
docker run --gpus all -it feax:latest
```

### Docker Compose (Recommended)

For development with persistent volumes and GPU support:

```bash
cd feax
docker-compose up -d
docker exec -it feax bash
```

The docker-compose configuration includes:
- GPU support (`deploy.resources.reservations.devices`)
- Volume mounting for development (`./:/workspace`)
- WSL2 display support (for GUI applications like gmsh)
- Shared memory configuration (`shm_size: 4gb`)

## Docs Development

To develop the documentation site locally, Node.js 20+ and pydoc-markdown are required.

### Install Dependencies

```bash
# Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# pydoc-markdown (API doc generator)
pip install pydoc-markdown

# Docusaurus + npm packages
cd docs
npm install
```

### Start Dev Server

Use the provided script to generate API docs and start the server in one step:

```bash
./docs/dev.sh
```

Or manually:

```bash
cd docs
npm run api:generate   # Generate API reference from Python source
npm run start          # Start dev server at http://localhost:3000
```

### Build for Production

```bash
cd docs
npm run build
```

## Colab

You can use FEAX in Google Colab notebooks with free GPU/TPU access.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naruki-Ichihara/feax/blob/main/examples/colab_quickstart.ipynb)

### Basic Setup

In a Colab notebook cell, install system dependencies via the bundled helper script. It mirrors the [Dockerfile](https://github.com/Naruki-Ichihara/feax/blob/main/Dockerfile): installs gmsh + Python bindings, swig, NLopt, BLAS/LAPACK, gmsh runtime libraries, builds SuiteSparse 7.x from source (with `SUITESPARSE_USE_CUDA=OFF` to avoid an `nvcc compute_52` failure on Colab GPU), and installs the `nlopt` Python bindings:

```python
!curl -fsSL https://raw.githubusercontent.com/Naruki-Ichihara/feax/main/scripts/colab_setup.sh | bash
!pip install git+https://github.com/Naruki-Ichihara/feax.git
```

The setup step takes ~5–8 minutes on Colab (one-time, dominated by the SuiteSparse build).

> **Note:** `scikit-sparse` is **not** required — the default `DirectSolverOptions()` auto-selects `cudss` on GPU or `spsolve` on CPU. Once `colab_setup.sh` has run, you can opt in to `cholmod` / `umfpack` with:
>
> ```python
> !pip install "feax[sksparse] @ git+https://github.com/Naruki-Ichihara/feax.git"
> ```

### With GPU Support

Colab provides CUDA-enabled GPUs by default. Run the setup script, then install the GPU extras:

```python
!curl -fsSL https://raw.githubusercontent.com/Naruki-Ichihara/feax/main/scripts/colab_setup.sh | bash
!pip install "feax[cuda13,sksparse] @ git+https://github.com/Naruki-Ichihara/feax.git"
!pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git

# Verify GPU is available
import jax
print(jax.devices())  # Should show GPU
```

## Verification

After installation, verify your setup:

```python
import jax
import feax
import jax.numpy as np

print(f"JAX version: {jax.__version__}")
print(f"FEAX version: {feax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX 64-bit enabled: {jax.config.jax_enable_x64}")
```

Expected output:
```
JAX version: 0.4.x
FEAX version: 0.1.0
JAX devices: [CpuDevice(id=0)] or [GpuDevice(id=0)]
JAX 64-bit enabled: True
```

## Next Steps

- Try the [Basic Tutorial](../basic/index.md) to get started
- Check out [example scripts](https://github.com/Naruki-Ichihara/feax/tree/main/examples)
- Read the [API documentation](../api/index.md)
