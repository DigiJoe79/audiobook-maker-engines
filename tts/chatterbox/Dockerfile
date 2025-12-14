# Chatterbox Multilingual TTS Engine Docker Image
#
# Multi-stage build: dependencies in builder, model downloaded in runtime stage
# to avoid duplicate disk usage during build (~1.5GB model from HuggingFace)
# Supports GPU via NVIDIA Container Toolkit
#
# ==============================================================================
# Build (from repository root):
# ==============================================================================
#
#   docker build -t audiobook-maker/chatterbox:latest -f tts/chatterbox/Dockerfile .
#
# ==============================================================================
# Run:
# ==============================================================================
#   docker run -d --gpus all \
#     -p 8766:8766 \
#     -v /path/to/samples:/app/samples \
#     audiobook-maker/chatterbox:latest
#
# Run (CPU only):
#   docker run -d \
#     -p 8766:8766 \
#     -v /path/to/samples:/app/samples \
#     audiobook-maker/chatterbox:latest

# ==============================================================================
# Stage 1: Builder - Install dependencies
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements for reference
WORKDIR /build
COPY tts/chatterbox/requirements.txt .

# ==============================================================================
# Install dependencies in correct order (pkuseg build workaround)
# ==============================================================================

# Step 1: Build dependencies (numpy + cython first for pkuseg)
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<1.26.0" \
    cython

# Step 2: PyTorch with CUDA 12.4 support (BEFORE chatterbox-tts)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 3: Install pkuseg with --no-build-isolation (requires numpy pre-installed)
RUN pip install --no-cache-dir --no-build-isolation spacy-pkuseg

# Step 4: Install chatterbox-tts with --no-deps to preserve CUDA PyTorch
RUN pip install --no-cache-dir --no-build-isolation --no-deps chatterbox-tts

# Step 5: Install remaining Chatterbox dependencies
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    librosa==0.11.0 \
    safetensors==0.5.3 \
    conformer==0.3.2 \
    resemble-perth==1.0.1 \
    s3tokenizer \
    pykakasi==2.3.0

# Step 6: FastAPI server stack
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0,<1.0.0" \
    "uvicorn>=0.32.0,<1.0.0" \
    "python-multipart>=0.0.9,<1.0.0" \
    "pydantic>=2.10.0,<3.0.0"

# Step 7: Utilities
RUN pip install --no-cache-dir \
    "loguru>=0.7.2,<0.8.0" \
    "httpx>=0.28.0,<1.0.0" \
    "scipy>=1.11.0,<2.0.0" \
    "huggingface_hub>=0.20.0"

# ==============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 runtime only (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Download model from HuggingFace Hub (directly in runtime stage to save disk space during build)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox', local_dir_use_symlinks=False)" \
    && echo "Chatterbox model downloaded successfully"

# Set working directory
WORKDIR /app

# Copy base server files (required by Chatterbox server)
COPY base_server.py ./
COPY base_tts_server.py ./

# Copy Chatterbox engine files
COPY tts/chatterbox/server.py ./
COPY tts/chatterbox/engine.yaml ./

# Create directories for volumes (samples for speaker samples uploaded via API)
RUN mkdir -p /app/models /app/samples

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8766

# Suppress library warnings
ENV TRANSFORMERS_VERBOSITY=error
ENV DIFFUSERS_VERBOSITY=error

# Expose default TTS port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# Run server (bind to 0.0.0.0 for container accessibility)
# Uses PORT environment variable (set by docker_runner, default 8766)
CMD ["sh", "-c", "python server.py --port ${PORT:-8766} --host 0.0.0.0"]
