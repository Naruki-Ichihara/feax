FROM nvcr.io/nvidia/jax:25.04-py3

# Build argument to control Claude Code installation
ARG INSTALL_CLAUDE=false

RUN apt update
RUN apt upgrade -y

# Libglu for gmsh
RUN apt -y install libglu1 libxcursor-dev libxft2 libxinerama1 libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libocct-foundation-dev libocct-data-exchange-dev
RUN pip install --upgrade pip

# Copy feax source code
COPY . /workspace
WORKDIR /workspace

# Install feax with CUDA support (includes all necessary NVIDIA packages)
RUN pip install .[cuda12]

# Install spineax from GitHub without build isolation
RUN pip install --no-build-isolation git+https://github.com/johnviljoen/spineax.git

# Install Node.js and Claude Code (if enabled)
RUN if [ "$INSTALL_CLAUDE" = "true" ]; then \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs && \
    npm install -g @anthropic-ai/claude-code; \
    fi

CMD ["/bin/bash"]
