FROM nvcr.io/nvidia/jax:25.08-py3

# Build argument to control Claude Code installation
ARG INSTALL_CLAUDE=false
ARG INSTALL_FENICSX=false

RUN apt update
RUN apt upgrade -y

# Libglu for gmsh
RUN apt -y install libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
RUN pip install --upgrade pip

# Install Node.js and Claude Code (if enabled)
RUN if [ "$INSTALL_CLAUDE" = "true" ]; then \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs && \
    npm install -g @anthropic-ai/claude-code; \
    fi

# Install FEniCSX suite (if enabled)
RUN if [ "$INSTALL_FENICSX" = "true" ]; then \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:fenics-packages/fenics -y && \
    apt update -y && \
    apt install fenicsx -y; \
    fi

# Configure MPI to work in Docker containers
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# NOTE: FEniCSx requires proper IPC namespace for MPI shared memory.
# When running the container, you MUST use ONE of these options:
#   1. docker-compose up (recommended - already configured with ipc: host)
#   2. docker run --ipc=host ...
# Without proper IPC, dolfinx will fail to import with MPI initialization errors.

WORKDIR /home/
CMD ["/bin/bash"]
