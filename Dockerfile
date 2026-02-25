FROM nvcr.io/nvidia/jax:25.10-py3

RUN apt update
RUN apt upgrade -y

# X11 libraries and libglu for gmsh (headless mode)
RUN apt -y install libglu1-mesa libglu1-mesa-dev \
    libxcursor1 libxinerama1 libxrandr2 libxi6 libxrender1 libxext6 libxft2
RUN pip install --upgrade pip

# Copy feax source code
COPY . /workspace
WORKDIR /workspace

# Install feax with CUDA dependencies.
# JAX is intentionally not reinstalled here: the NVCR base image already ships
# JAX compiled for CUDA 13.1.1. Use pip install .[cuda13,jax] outside containers.
RUN pip install .[cuda13]
RUN pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git

CMD ["/bin/bash"]
