# Debug Text Engine Docker Image
#
# Minimal footprint text processing engine for testing and development.
# Uses simple punctuation-based segmentation instead of NLP models.
# No GPU, no ML frameworks - pure Python implementation.
#
# ==============================================================================
# Build (from repo root):
# ==============================================================================
#   docker build -t audiobook-maker/debug-text:latest -f text_processing/debug-text/Dockerfile .
#
# ==============================================================================
# Run:
# ==============================================================================
#   docker run -d -p 8772:8772 audiobook-maker/debug-text:latest
#
#   With custom port:
#   docker run -d -p 8772:8772 -e PORT=8772 audiobook-maker/debug-text:latest

# ==============================================================================
# Single stage build - minimal image
# ==============================================================================
FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY text_processing/debug-text/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy base server files (required by Debug Text server)
COPY base_server.py ./
COPY base_text_server.py ./

# Copy Debug Text engine files
COPY text_processing/debug-text/server.py ./
COPY text_processing/debug-text/engine.yaml ./

# Copy baked-in models
COPY text_processing/debug-text/models/ ./models/

# Copy generic entrypoint script
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# Create standard directories
# - models: primary model storage (baked-in or symlinked from external)
# - external_models: mount point for external models (symlinked to models/)
RUN mkdir -p /app/models /app/external_models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8772

# Expose default text processing port
EXPOSE 8772

# Health check (starts quickly - 10s start period is enough)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# Entrypoint handles external model symlinks, then runs CMD
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint -- Container runs isolated engine server with no shell access
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run server (bind to 0.0.0.0 for container accessibility)
# nosemgrep: dockerfile.security.missing-user.missing-user -- Container runs isolated engine server with no shell access
CMD ["sh", "-c", "python server.py --port ${PORT:-8772} --host 0.0.0.0"]
