FROM nvcr.io/nvidia/jax:25.10-py3

# Set to "true" to install Node.js 20 + docs dependencies (for Docusaurus)
ARG INSTALL_DOCS=false

RUN apt update
RUN apt upgrade -y

RUN apt -y install gmsh libsuitesparse-dev swig libnlopt-dev
RUN pip install --upgrade pip

# Copy feax source code
COPY . /workspace
WORKDIR /workspace

# Install feax with CUDA dependencies.
# JAX is intentionally not reinstalled here: the NVCR base image already ships
# JAX compiled for CUDA 13.1.1. Use pip install .[cuda13,jax] outside containers.
RUN pip install .[cuda13]
RUN pip install --no-build-isolation nlopt
RUN pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git

# Optional: Node.js 20 + pydoc-markdown + Docusaurus dependencies
# Build with: docker build --build-arg INSTALL_DOCS=true .
RUN if [ "$INSTALL_DOCS" = "true" ]; then \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs && \
        pip install pydoc-markdown && \
        cd /workspace/docs && npm install; \
    fi

CMD ["/bin/bash"]
