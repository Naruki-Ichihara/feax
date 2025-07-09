FROM nvcr.io/nvidia/jax:25.04-py3

# Build argument to control Claude Code installation
ARG INSTALL_CLAUDE=true

RUN apt update
RUN apt upgrade -y
RUN pip install --upgrade pip

# Install Node.js and Claude Code (if enabled)
RUN if [ "$INSTALL_CLAUDE" = "true" ]; then \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs && \
    npm install -g @anthropic-ai/claude-code; \
    fi

WORKDIR /home/
CMD ["/bin/bash"]
