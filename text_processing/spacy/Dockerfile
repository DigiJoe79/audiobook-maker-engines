# =============================================================================
# spaCy Text Processing Engine - Docker Image
# =============================================================================
# Baked-in models: de_core_news_md, en_core_web_md
# On-demand download support for other languages
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# -----------------------------------------------------------------------------
# System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
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
COPY text_processing/spacy/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Baked-in spaCy Models (DE + EN)
# Downloaded to site-packages, then copied to /app/models for consistency
# -----------------------------------------------------------------------------
RUN python -m spacy download de_core_news_md && \
    python -m spacy download en_core_web_md

# Copy models from site-packages to /app/models for consistent structure
RUN mkdir -p /app/models && \
    cp -r /opt/venv/lib/python3.12/site-packages/de_core_news_md /app/models/ && \
    cp -r /opt/venv/lib/python3.12/site-packages/en_core_web_md /app/models/

# -----------------------------------------------------------------------------
# Base Server Files
# -----------------------------------------------------------------------------
COPY base_server.py ./
COPY base_text_server.py ./

# -----------------------------------------------------------------------------
# Engine Files
# -----------------------------------------------------------------------------
COPY text_processing/spacy/server.py ./
COPY text_processing/spacy/engine.yaml ./

# -----------------------------------------------------------------------------
# Entrypoint Script
# -----------------------------------------------------------------------------
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# -----------------------------------------------------------------------------
# Directory Structure (Model Management Standard)
# -----------------------------------------------------------------------------
# - models: baked-in models (de, en) + symlinks to external
# - external_models: mount point for on-demand downloaded models
# -----------------------------------------------------------------------------
RUN mkdir -p /app/external_models

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8770

EXPOSE 8770

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

# -----------------------------------------------------------------------------
# Entrypoint & Command
# -----------------------------------------------------------------------------
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "python server.py --port ${PORT:-8770} --host 0.0.0.0"]
