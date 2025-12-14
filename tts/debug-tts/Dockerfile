# Debug TTS Engine Docker Image
#
# Minimal footprint TTS engine for testing and development.
# Generates pleasant sine wave tones instead of actual TTS.
# No GPU, no ML frameworks - pure Python audio generation.
#
# ==============================================================================
# Build (from backend/engines directory):
# ==============================================================================
#   docker build -t audiobook-maker/debug-tts:latest -f tts/debug-tts/Dockerfile .
#
# ==============================================================================
# Run:
# ==============================================================================
#   docker run -d -p 8766:8766 audiobook-maker/debug-tts:latest
#
#   With custom port:
#   docker run -d -p 8770:8770 -e PORT=8770 audiobook-maker/debug-tts:latest

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
COPY tts/debug-tts/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy base server files (required by Debug TTS server)
COPY base_server.py ./
COPY base_tts_server.py ./

# Copy Debug TTS engine files
COPY tts/debug-tts/server.py ./
COPY tts/debug-tts/engine.yaml ./

# Create directories for volumes (samples for speaker samples - not used by debug-tts but for API consistency)
RUN mkdir -p /app/samples

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8766

# Expose default TTS port
EXPOSE 8766

# Health check (starts quickly - 10s start period is enough)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# Run server (bind to 0.0.0.0 for container accessibility)
CMD ["sh", "-c", "python server.py --port ${PORT:-8766} --host 0.0.0.0"]
