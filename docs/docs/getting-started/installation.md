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

### NVIDIA GPU (CUDA)

For NVIDIA GPU acceleration with CUDA 12:

```bash
pip install -U "jax[cuda12]"
```

For CUDA 12 with locally-installed CUDA/cuDNN:

```bash
pip install -U "jax[cuda12-local]"
```

**Requirements:**
- NVIDIA driver ≥525 (for CUDA 12)
- NVIDIA driver ≥550 (for CUDA 12.3+)

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

## Docker

FEAX provides a Dockerfile based on NVIDIA's JAX image with GPU support and optional dependencies.

### Build Options

The Dockerfile supports build arguments:
- `INSTALL_CLAUDE`: Install Claude Code (default: `false`)
- `INSTALL_FENICSX`: Install FEniCSX suite for debugging purposes (default: `false`, not required for normal use)

### Build Docker Image

```bash
git clone https://github.com/Naruki-Ichihara/feax.git
cd feax
docker build -t feax:latest .
```

Build with Claude Code (for development):

```bash
docker build --build-arg INSTALL_CLAUDE=true -t feax:latest .
```

Build with FEniCSX (for debugging):

```bash
docker build --build-arg INSTALL_FENICSX=true -t feax:latest .
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
- Shared memory configuration

## Colab

You can use FEAX in Google Colab notebooks with free GPU/TPU access.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naruki-Ichihara/feax/blob/main/examples/colab_quickstart.ipynb)

### Basic Setup

In a Colab notebook cell, first install system dependencies for gmsh:

```python
!apt update
!apt install -y libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
!pip install feax
```

### With GPU Support

Colab provides CUDA-enabled GPUs by default. Install system dependencies and FEAX:

```python
!apt update
!apt install -y libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
!pip install feax

# Verify GPU is available
import jax
print(jax.devices())  # Should show GPU
```

### With TPU Support

For TPU acceleration:

```python
!apt update
!apt install -y libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
!pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install feax

# Verify TPU
import jax
print(jax.devices())  # Should show TPU cores
```

### Example Colab Notebook

```python
# Install system dependencies
!apt update
!apt install -y libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev

# Install FEAX
!pip install feax

# Import and verify
import jax
import feax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# Run a simple example
from feax.mesh import box_mesh
mesh = box_mesh(size=1.0, mesh_size=0.2)
print(f"Mesh created with {mesh.points.shape[0]} nodes")
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
FEAX version: x.x.x
JAX devices: [CpuDevice(id=0)] or [GpuDevice(id=0)]
JAX 64-bit enabled: True
```

## Next Steps

- Try the [Basic Tutorial](../basic/index.md) to get started
- Check out [example scripts](https://github.com/Naruki-Ichihara/feax/tree/main/examples)
- Read the [API documentation](../../api/index.md)
