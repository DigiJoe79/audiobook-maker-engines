# Debug STT Engine Docker Image
#
# Minimal footprint STT engine for testing and development.
# Returns mock transcription results instead of actual STT.
# No GPU, no ML frameworks - pure Python mock implementation.
#
# ==============================================================================
# Build (from repo root):
# ==============================================================================
#   docker build -t audiobook-maker/debug-stt:latest -f stt/debug-stt/Dockerfile .
#
# ==============================================================================
# Run:
# ==============================================================================
#   docker run -d -p 8770:8770 audiobook-maker/debug-stt:latest
#
#   With custom port:
#   docker run -d -p 8770:8770 -e PORT=8770 audiobook-maker/debug-stt:latest

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
COPY stt/debug-stt/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy base server files (required by Debug STT server)
COPY base_server.py ./
COPY base_quality_server.py ./

# Copy Debug STT engine files
COPY stt/debug-stt/server.py ./
COPY stt/debug-stt/engine.yaml ./

# Copy baked-in models
COPY stt/debug-stt/models/ ./models/

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
ENV PORT=8770

# Expose default STT port
EXPOSE 8770

# Health check (starts quickly - 10s start period is enough)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# Entrypoint handles external model symlinks, then runs CMD
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint -- Container runs isolated engine server with no shell access
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run server (bind to 0.0.0.0 for container accessibility)
# nosemgrep: dockerfile.security.missing-user.missing-user -- Container runs isolated engine server with no shell access
CMD ["sh", "-c", "python server.py --port ${PORT:-8770} --host 0.0.0.0"]
