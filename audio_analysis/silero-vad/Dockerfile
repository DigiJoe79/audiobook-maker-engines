# =============================================================================
# Silero-VAD Audio Analysis Engine
# =============================================================================
#
# Lightweight audio analysis engine using Silero Voice Activity Detection.
# Detects speech/silence ratio, clipping, and volume issues.
#
# CPU-only (no GPU required) - model is bundled in silero-vad pip package.
#
# Build (from repository root):
#   docker build -t audiobook-maker/silero-vad:latest -f audio_analysis/silero-vad/Dockerfile .
#
# Run:
#   docker run -d -p 8769:8769 audiobook-maker/silero-vad:latest
#
# =============================================================================

FROM python:3.12-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python Environment
# -----------------------------------------------------------------------------
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Python Dependencies
# -----------------------------------------------------------------------------
COPY audio_analysis/silero-vad/requirements.txt ./requirements.txt

# Install PyTorch CPU-only (smaller image, silero-vad doesn't need GPU)
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Base Server Files
# -----------------------------------------------------------------------------
COPY base_server.py ./
COPY base_quality_server.py ./

# -----------------------------------------------------------------------------
# Engine Files
# -----------------------------------------------------------------------------
COPY audio_analysis/silero-vad/server.py ./
COPY audio_analysis/silero-vad/engine.yaml ./

# -----------------------------------------------------------------------------
# Entrypoint and Directories
# -----------------------------------------------------------------------------
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# Create standard directories (silero-vad model is in pip package, no external models)
RUN mkdir -p /app/models /app/external_models

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8769

EXPOSE 8769

# -----------------------------------------------------------------------------
# Health Check
# Note: PyTorch import is slow, so we need a longer start-period than lightweight engines
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "python server.py --port ${PORT:-8769} --host 0.0.0.0"]
