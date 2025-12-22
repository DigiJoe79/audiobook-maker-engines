# VibeVoice TTS Engine Docker Image
#
# Microsoft VibeVoice - Expressive multi-speaker TTS with voice cloning.
# Supports VibeVoice-1.5B (~3GB VRAM) and VibeVoice-7B (~18GB VRAM) models.
# Models are downloaded on first load (not baked in due to size: 3-18GB).
#
# ==============================================================================
# Build (from repository root):
# ==============================================================================
#
#   docker build -t audiobook-maker/vibevoice:latest -f tts/vibevoice/Dockerfile .
#
# ==============================================================================
# Run:
# ==============================================================================
#   docker run -d --gpus all \
#     -p 8766:8766 \
#     -v /path/to/samples:/app/samples \
#     -v /path/to/models:/app/external_models \
#     audiobook-maker/vibevoice:latest
#
# IMPORTANT: Mount external_models to persist downloaded models between restarts!

# ==============================================================================
# Stage 1: Builder - Install dependencies
# ==============================================================================
# Using base image (pre-built flash-attn wheel, no compilation needed)
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

# ==============================================================================
# Install dependencies in correct order
# ==============================================================================

# Step 1: Build dependencies
RUN pip install --no-cache-dir numpy cython

# Step 2: PyTorch with CUDA 12.4 support
# Using 2.5.1 for flash-attn compatibility (pre-built wheels available)
# --extra-index-url allows pip to find NVIDIA packages on PyPI
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Step 3: Audio processing
RUN pip install --no-cache-dir \
    librosa \
    scipy \
    numpy

# Step 4: HuggingFace Hub (for model downloads)
RUN pip install --no-cache-dir huggingface_hub

# Step 5: VibeVoice from community fork
RUN pip install --no-cache-dir git+https://github.com/vibevoice-community/VibeVoice.git

# Step 6: Flash Attention 2 (pre-built wheel for torch 2.5 + cu124 + python 3.12)
# Using pre-built wheel from community repo to avoid ABI compatibility issues
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl

# Step 7: FastAPI server stack
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0,<1.0.0" \
    "uvicorn>=0.32.0,<1.0.0" \
    "pydantic>=2.10.0,<3.0.0" \
    "python-multipart>=0.0.9,<1.0.0"

# Step 8: Utilities
RUN pip install --no-cache-dir \
    "loguru>=0.7.2,<0.8.0" \
    "httpx>=0.28.0,<1.0.0" \
    "PyYAML>=6.0.0,<7.0.0"

# ==============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 runtime only (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy base server files
COPY base_server.py ./
COPY base_tts_server.py ./

# Copy VibeVoice engine files
COPY tts/vibevoice/server.py ./
COPY tts/vibevoice/engine.yaml ./

# Copy generic entrypoint script
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# Create standard directories
# - models: primary model storage (symlinks to external after download)
# - external_models: mount point for external models (IMPORTANT: mount this to persist downloads!)
# - samples: speaker samples for voice cloning
RUN mkdir -p /app/models /app/external_models /app/samples

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8766

# Suppress library warnings
ENV TRANSFORMERS_VERBOSITY=error
ENV DIFFUSERS_VERBOSITY=error

# Expose default TTS port
EXPOSE 8766

# Health check (long start period - model download can take time)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# Entrypoint handles external model symlinks, then runs CMD
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run server (bind to 0.0.0.0 for container accessibility)
CMD ["sh", "-c", "python server.py --port ${PORT:-8766} --host 0.0.0.0"]
