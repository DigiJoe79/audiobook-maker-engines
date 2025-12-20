# Model Management Standard

This document describes the unified model management pattern used across all engine containers.

## Overview

All engines follow the same pattern for model storage and loading, regardless of whether they run as Docker containers or local subprocesses.

## Directory Structure

```
/app/                          # Docker container root
├── models/                    # Primary model directory (server reads from here)
│   ├── default/               # Baked-in model (copied during build)
│   ├── v2.0.2/                # Another baked-in model
│   └── custom-model/          # Symlink -> /app/external_models/custom-model
│
├── external_models/           # Mount point for external models
│   └── custom-model/          # Mounted from host
│
└── samples/                   # Speaker samples for voice cloning
```

For subprocess execution, paths are relative to the engine directory:

```
backend/engines/tts/xtts/      # Engine directory
├── models/                    # Local models
├── external_models/           # Optional external models
└── samples/                   # Speaker samples
```

## Key Principles

1. **Server only reads from `models/`** - Engine code uses `self.models_dir` exclusively
2. **External models are symlinked** - The entrypoint script creates symlinks from `external_models/` to `models/`
3. **Baked-in models take precedence** - If a model exists in both locations, the baked-in version is used
4. **Same pattern for Docker and subprocess** - No conditional logic needed in engine code

## Docker Entrypoint Script

The generic entrypoint script (`scripts/docker-entrypoint.sh`) handles symlink creation:

```bash
#!/bin/bash
MODELS_DIR="/app/models"
EXTERNAL_DIR="/app/external_models"

mkdir -p "$MODELS_DIR"

if [ -d "$EXTERNAL_DIR" ]; then
    for item in "$EXTERNAL_DIR"/*; do
        [ -e "$item" ] || continue

        item_name=$(basename "$item")
        link_path="$MODELS_DIR/$item_name"

        if [ ! -e "$link_path" ]; then
            ln -s "$item" "$link_path"
            echo "[entrypoint] Linked: $item_name"
        fi
    done
fi

exec "$@"
```

## Dockerfile Pattern

```dockerfile
# Copy baked-in models (requires .dockerignore exception!)
COPY tts/engine-name/models/ ./models/

# Copy generic entrypoint script
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# Create standard directories
RUN mkdir -p /app/models /app/external_models /app/samples

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "python server.py --port ${PORT:-8766} --host 0.0.0.0"]
```

## .dockerignore Configuration

The root `.dockerignore` excludes all `**/models/` directories by default (models are typically mounted as volumes). For engines with baked-in models, add an exception:

```gitignore
# Root .dockerignore

# Models (mounted as volumes, not baked in)
**/models/

# Exception for engines with baked-in models
!tts/debug-tts/models/
!tts/chatterbox/models/
```

**Why this pattern?**
- Most engines have large models that should be mounted, not baked in
- Small engines (like debug-tts) can include models in the image
- Without the exception, `COPY` fails with "not found" error

**Troubleshooting:**
If you see this error during Docker build:
```
ERROR: failed to compute cache key: "/tts/engine-name/models": not found
```
Add `!tts/engine-name/models/` to `.dockerignore`.

## Base Server Integration

The `BaseEngineServer` automatically sets up model paths:

```python
class BaseEngineServer:
    def __init__(self, ...):
        # Automatic path detection
        if Path("/app").exists() and Path("/app/server.py").exists():
            self.base_path = Path("/app")
        else:
            self.base_path = Path(sys.argv[0]).parent.resolve()

        self.models_dir = self.base_path / "models"
        self.external_models_dir = self.base_path / "external_models"
```

Engines access models via `self.models_dir`:

```python
def load_model(self, model_name: str):
    model_path = self.models_dir / model_name
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    # Load model...
```

## Usage Scenarios

### Scenario 1: Small Models (Baked-in)

Models small enough to include in the Docker image (e.g., Chatterbox ~3GB).

```dockerfile
# Bake model into image during build
COPY models/ ./models/
# OR download during build
RUN python -c "download_model('/app/models/')"
```

```bash
# Run without mounting
docker run engine:latest
```

### Scenario 2: Baked-in Default + Optional External

Default model baked in, additional models can be mounted (e.g., XTTS).

```bash
# Run with default only
docker run xtts:latest

# Run with additional custom model
docker run -v /host/models:/app/external_models xtts:latest
```

### Scenario 3: Large Models (External Only)

Models too large for Docker image (e.g., VibeVoice ~20GB).

```dockerfile
# No models baked in - models/ stays empty
RUN mkdir -p /app/models
```

```bash
# Must mount models
docker run -v /host/vibevoice-models:/app/external_models vibevoice:latest
```

### Scenario 4: On-Demand Download (Whisper, etc.)

Models downloaded at runtime when first selected.

**Key insight:** Downloads must go to `external_models/` for persistence!

```python
# In server.py - BEFORE importing the library
import os
os.environ['WHISPER_CACHE'] = str(self.external_models_dir)
import whisper
```

**Why `external_models/` instead of `models/`?**
- `models/` is part of the container filesystem → deleted on container stop
- `external_models/` is the mount point → persists if volume is mounted

**Workflow:**
1. User mounts volume: `-v /data/whisper:/app/external_models`
2. User selects "base" model → Whisper downloads to `/app/external_models/base.pt`
3. Entrypoint creates symlink: `/app/models/base.pt -> /app/external_models/base.pt`
4. Server reads from `self.models_dir` as usual
5. Container restart → model still available (persisted in volume)

**Without mount:**
- Downloads work (stored in `external_models/`)
- But lost after container stop (expected behavior)

```bash
# For persistent on-demand downloads, always mount:
docker run -v /data/whisper-models:/app/external_models whisper:latest
```

## Mount Pattern

Always mount to `/app/external_models`:

```bash
docker run -v /host/path/to/models:/app/external_models engine:latest
```

The entrypoint script automatically symlinks the contents to `/app/models/`.

## Model Discovery

Engines can discover available models by scanning `self.models_dir`:

```python
def get_available_models(self) -> List[ModelInfo]:
    models = []
    for path in self.models_dir.iterdir():
        if path.is_dir():
            models.append(ModelInfo(name=path.name, ...))
    return models
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `PORT` | `8766` | Server port |

## Summary

- Mount external models to `/app/external_models`
- Server reads from `/app/models` (baked-in + symlinked external)
- No engine code changes needed for Docker vs subprocess
- Entrypoint script handles symlink creation automatically

## Conformance Checklist

Use this checklist when creating or reviewing engines:

### Server Implementation (server.py)

| Requirement | Implementation |
|-------------|----------------|
| Read models from `self.models_dir` | `model_path = self.models_dir / model_name` |
| Download to `self.external_models_dir` | `external_path = self.external_models_dir / model_name` |
| Symlink after download | `model_path.symlink_to(external_path)` |
| Baked-in models take precedence | `if not model_path.exists(): # then download` |
| Model discovery scans models_dir | `for path in self.models_dir.iterdir()` |

### Dockerfile

| Requirement | Implementation |
|-------------|----------------|
| Copy baked-in models | `COPY tts/engine/models/ ./models/` |
| Copy entrypoint script | `COPY scripts/docker-entrypoint.sh ./` |
| Make entrypoint executable | `RUN chmod +x /app/docker-entrypoint.sh` |
| Create standard directories | `RUN mkdir -p /app/models /app/external_models /app/samples` |
| Set entrypoint | `ENTRYPOINT ["/app/docker-entrypoint.sh"]` |

### .dockerignore

| Requirement | Implementation |
|-------------|----------------|
| Exception for baked-in models | `!tts/engine-name/models/` after `**/models/` |

### Example: Conforming Engine (debug-tts)

```python
# server.py - load_model()
def load_model(self, model_name: str) -> None:
    model_path = self.models_dir / model_name  # [OK] Read from models_dir

    if not model_path.exists():  # [OK] Baked-in takes precedence
        external_path = self.external_models_dir / model_name  # [OK] Download to external
        external_path.mkdir(parents=True, exist_ok=True)
        # ... download model files ...
        model_path.symlink_to(external_path)  # [OK] Symlink to models_dir

# server.py - get_available_models()
def get_available_models(self) -> List[ModelInfo]:
    models = []
    for model_path in self.models_dir.iterdir():  # [OK] Scan models_dir
        if model_path.is_dir() or model_path.is_symlink():
            models.append(ModelInfo(name=model_path.name, ...))
    return models
```
