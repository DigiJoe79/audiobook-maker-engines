# XTTS Engine Docker Image
#
# Multi-stage build with v2.0.2 model downloaded from HuggingFace (~1.8GB)
# Supports GPU via NVIDIA Container Toolkit
#
# ==============================================================================
# Build (from repository root):
# ==============================================================================
#
#   docker build -t audiobook-maker/xtts:latest -f tts/xtts/Dockerfile .
#
# ==============================================================================
# Run:
# ==============================================================================
#   docker run -d --gpus all \
#     -p 8766:8766 \
#     -v /path/to/samples:/app/samples \
#     audiobook-maker/xtts:latest
#
# Run with custom models (mounted alongside default v2.0.2):
#   docker run -d --gpus all \
#     -p 8766:8766 \
#     -v /path/to/custom_models:/app/external_models \
#     -v /path/to/samples:/app/samples \
#     audiobook-maker/xtts:latest
#
# Run (CPU only):
#   docker run -d \
#     -p 8766:8766 \
#     -v /path/to/samples:/app/samples \
#     audiobook-maker/xtts:latest
#
# Model Structure:
#   /app/models/v2.0.2/      <- downloaded from HuggingFace during build
#   /app/external_models/    <- mount point for custom models (optional)
#   Custom models are symlinked to /app/models/ at startup

# ==============================================================================
# Stage 1: Builder - Install dependencies
# ==============================================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
WORKDIR /build
COPY tts/xtts/requirements.txt .

# Install PyTorch with CUDA 11.8 support first (separate layer for caching)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu118 \
    torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 runtime only (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy base server files (required by XTTS server)
COPY base_server.py ./
COPY base_tts_server.py ./

# Copy XTTS engine files
COPY tts/xtts/server.py ./
COPY tts/xtts/engine.yaml ./

# Copy shared entrypoint script
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Download XTTS v2.0.2 model from HuggingFace (coqui/XTTS-v2)
# Files: model.pth (~1.8GB), config.json, vocab.json
RUN mkdir -p /app/models/v2.0.2 && python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
files = ['model.pth', 'config.json', 'vocab.json']; \
[shutil.copy(hf_hub_download('coqui/XTTS-v2', f, revision='v2.0.2'), '/app/models/v2.0.2/') for f in files]; \
print('XTTS v2.0.2 model downloaded to /app/models/v2.0.2')" \
    && rm -rf /root/.cache/huggingface \
    && ls -la /app/models/v2.0.2/

# Create directories for volumes (external_models for custom models, samples for speaker samples)
RUN mkdir -p /app/external_models /app/samples

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8766

# Expose default TTS port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# Run entrypoint (symlinks external models, then starts server)
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["sh", "-c", "python server.py --port ${PORT:-8766} --host 0.0.0.0"]
