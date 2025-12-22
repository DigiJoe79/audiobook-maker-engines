# =============================================================================
# Whisper STT Engine - GPU Docker Image
# =============================================================================
# OpenAI Whisper for speech-to-text transcription with CUDA acceleration
# Models are downloaded on-demand to /app/external_models
# =============================================================================

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
# ffmpeg is required by whisper for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# -----------------------------------------------------------------------------
# Python Virtual Environment
# -----------------------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# PyTorch with CUDA Support
# -----------------------------------------------------------------------------
# PyTorch 2.5.1 for stability (matches other GPU engines)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------------------------------------
# Application Dependencies
# -----------------------------------------------------------------------------
COPY stt/whisper/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Base Server Files
# -----------------------------------------------------------------------------
COPY base_server.py ./
COPY base_quality_server.py ./

# -----------------------------------------------------------------------------
# Engine Files
# -----------------------------------------------------------------------------
COPY stt/whisper/server.py ./
COPY stt/whisper/models.py ./
COPY stt/whisper/engine.yaml ./

# -----------------------------------------------------------------------------
# Entrypoint and Directories
# -----------------------------------------------------------------------------
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# Create required directories for Model Management Standard
# - models/: baked-in models + symlinks to external_models
# - external_models/: mount point for persistent model storage
# - samples/: speaker samples (not used by STT but required by entrypoint)
RUN mkdir -p /app/models /app/external_models /app/samples

# -----------------------------------------------------------------------------
# Runtime Configuration
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8767

EXPOSE 8767

# Health check using httpx (installed via requirements.txt)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "python server.py --port ${PORT:-8767} --host 0.0.0.0"]
