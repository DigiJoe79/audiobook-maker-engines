# Engine Server API

Technical documentation for the standardized engine server API that all engines must implement.

## Base Server Hierarchy

All engine servers inherit from base classes that provide standardized endpoints with built-in validation and error handling:

```
BaseEngineServer (base_server.py)
├── BaseTTSServer (base_tts_server.py)      → /generate, /samples/*
├── BaseQualityServer (base_quality_server.py) → /analyze
└── BaseTextServer (base_text_server.py)    → /segment
```

## Centralized Features in base_server.py

The base server provides common functionality that all engines inherit:

| Feature | Method/Property | Description |
|---------|-----------------|-------------|
| Device Detection | `device` property | Auto-detects cuda/cpu, cached after first call |
| GPU Cleanup | `_cleanup_gpu_memory()` | Clears CUDA cache after OOM or unload |
| Error Handling | `_handle_processing_error()` | Centralized handling for GPU OOM and general errors |
| Auto Cleanup | After `unload_model()` | Automatically calls cleanup + gc.collect() + resets state |

### Automatic Cleanup after unload_model()

The `/load` and `/shutdown` endpoints automatically handle cleanup after calling `unload_model()`:

```python
# In base_server.py /load endpoint:
await loop.run_in_executor(self._executor, self.unload_model)
self.model_loaded = False      # Reset state
self.current_model = None      # Reset state
self._cleanup_gpu_memory()     # Free GPU memory
gc.collect()                   # Force garbage collection
```

This means engine implementations only need to delete their model objects - state management and cleanup are handled centrally:

```python
# Engine unload_model() - minimal implementation:
def unload_model(self) -> None:
    if self.model is not None:
        del self.model
        self.model = None
    # Note: GPU cleanup, gc.collect(), and state reset are handled by base_server.py
```

## Common Endpoints (All Engines)

| Endpoint | Method | Purpose | Implemented In |
|----------|--------|---------|----------------|
| `/health` | GET | Status, loaded model, device info | `base_server.py` |
| `/models` | GET | Available models for this engine | `base_server.py` |
| `/load` | POST | Load a specific model | `base_server.py` |
| `/shutdown` | POST | Graceful shutdown | `base_server.py` |
| `/info` | GET | Engine metadata from engine.yaml | `base_server.py` |

## Non-Blocking Operations

All long-running operations run in a ThreadPoolExecutor to keep the server responsive. This ensures `/health` always responds instantly, even during model loading or audio generation.

### Operations using ThreadPoolExecutor

| Operation | Method | Typical Duration |
|-----------|--------|------------------|
| Model loading | `load_model()` | Minutes (large models) |
| TTS generation | `generate_audio()` | 10-60 seconds |
| Quality analysis | `analyze_audio()` | 1-10 seconds |
| Text segmentation | `segment_text()` | < 1 second |

### Behavior during operations

| Server State | `/health` Response | Processing Endpoints |
|--------------|-------------------|---------------------|
| Loading model | `{"status": "loading", ...}` (instant) | 503 "Model loading in progress" |
| Processing request | `{"status": "processing", ...}` (instant) | Queued (wait for completion) |
| Ready | `{"status": "ready", ...}` (instant) | 200 (processed) |

**Implementation detail:** The executor uses `max_workers=1` to ensure sequential processing. This prevents GPU OOM errors from concurrent operations and provides natural request queuing.

### Client retry pattern

```python
import httpx
import time

def wait_for_ready(base_url: str, timeout: int = 300):
    """Wait for engine to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        resp = httpx.get(f"{base_url}/health")
        status = resp.json()["status"]
        if status == "ready":
            return True
        if status == "error":
            raise RuntimeError(resp.json().get("error", "Unknown error"))
        time.sleep(2)  # Poll every 2 seconds
    return False

# Start loading (returns immediately after POST)
httpx.post(f"{base_url}/load", json={"engineModelName": "1.5B"})

# Wait for loading to complete
if wait_for_ready(base_url):
    # Now safe to call /generate
    httpx.post(f"{base_url}/generate", json={...})
```

## Model Hotswap

Engines that support `capabilities.supports_model_hotswap: true` can switch models without restarting. The base server automatically handles GPU memory cleanup during hotswap.

### Hotswap sequence

```
POST /load {"engineModelName": "model-B"}
    │
    ├─ if model already loaded:
    │     ├─ unload_model()     ← Free GPU memory
    │     └─ model_loaded = False
    │
    ├─ load_model("model-B")    ← Load new model
    │
    └─ model_loaded = True
        current_model = "model-B"
```

**Why unload first?** GPU memory is limited. Loading a new model without freeing the old one causes OOM errors, especially with large models like VibeVoice 7B (~18GB VRAM).

**Client perspective:**

```python
# Switch from 1.5B to 7B model
httpx.post(f"{base_url}/load", json={"engineModelName": "7B"})

# Wait for loading to complete
wait_for_ready(base_url)

# Now using 7B model
```

**Status during hotswap:**
- `/health` returns `{"status": "loading", "currentEngineModel": null}`
- Processing endpoints return 503

## Type-Specific Endpoints

| Type | Endpoint | Method | Purpose |
|------|----------|--------|---------|
| TTS | `/generate` | POST | Generate audio from text |
| TTS | `/samples/check` | POST | Check which speaker samples exist |
| TTS | `/samples/upload` | POST | Upload speaker sample WAV |
| Quality (STT/Audio) | `/analyze` | POST | Analyze audio quality |
| Text | `/segment` | POST | Segment text into TTS-ready chunks |

## Input Validation

All base servers perform input validation before calling engine-specific implementations. Invalid requests return HTTP 400 (Bad Request) or 404 (Not Found).

### TTS `/generate` Validation

| Check | HTTP Status | Error Detail |
|-------|-------------|--------------|
| Model loading in progress | 503 | "Model loading in progress. Retry after loading completes." |
| Model not loaded | 400 | "Model not loaded" |
| Empty text | 400 | "Text cannot be empty" |
| Empty language | 400 | "Language cannot be empty" |
| Text exceeds max_text_length | 400 | "Text too long ({len} chars). {Engine} max is {max} chars. Use text segmentation to split into smaller chunks." |
| Invalid speaker filename format | 400 | "Invalid speaker filename format: {filename}" |
| Invalid speaker path | 400 | "Invalid speaker path" |
| Speaker sample not found | 404 | "Speaker sample not found: {sample}" |
| Speaker cloning required* | 400 | "{Engine} requires speaker samples for voice cloning..." |
| GPU out of memory | 503 | "[GPU_OOM]GPU out of memory. Try smaller input or restart engine." |

*\* Only for engines with `capabilities.supports_speaker_cloning: true` in engine.yaml*

### TTS `/samples/check` Validation

| Check | HTTP Status | Error Detail |
|-------|-------------|--------------|
| Invalid sample ID format | 400 | "Invalid sample ID format: {sample_id}" |
| Invalid sample path | 400 | "Invalid sample path" |

### TTS `/samples/upload` Validation

| Check | HTTP Status | Error Detail |
|-------|-------------|--------------|
| Invalid sample ID format | 400 | "Invalid sample ID format: {sample_id}" |
| Invalid sample path | 400 | "Invalid sample path" |
| Empty request body | 400 | "Empty request body" |

### Quality `/analyze` Validation

| Check | HTTP Status | Error Detail |
|-------|-------------|--------------|
| Model loading in progress | 503 | "Model loading in progress. Retry after loading completes." |
| Model not loaded | 400 | "Model not loaded" |
| No audio provided | 400 | "Either audio_base64 or audio_path must be provided" |
| Audio file not found | 404 | "Audio file not found: {path}" |
| Empty audio data | 400 | "Audio data is empty" |
| Audio too small | 400 | "Audio data too small to be valid WAV" |
| Invalid WAV format | 400 | "Invalid audio format: expected WAV file" |
| GPU out of memory | 503 | "[GPU_OOM]GPU out of memory. Try smaller input or restart engine." |

### Text `/segment` Validation

| Check | HTTP Status | Error Detail |
|-------|-------------|--------------|
| Model loading in progress | 503 | "Model loading in progress. Retry after loading completes." |
| Model not loaded | 400 | "Model not loaded" |
| Empty text | 400 | "Text cannot be empty" |
| Empty language | 400 | "Language cannot be empty" |
| Invalid max_length | 400 | "max_length must be positive" |
| Invalid min_length | 400 | "min_length cannot be negative" |
| min >= max | 400 | "min_length must be less than max_length" |

## Error Handling Pattern

All endpoints follow a consistent error handling pattern with centralized GPU OOM handling:

```python
@self.app.post("/endpoint")
async def endpoint(request: RequestModel):
    try:
        # 1. Check model loaded
        if not self.model_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded")

        # 2. Validate input (returns 400/404 for invalid input)
        # ... validation checks ...

        # 3. Call engine-specific implementation
        result = self.engine_method(...)

        return result

    except HTTPException:
        self.status = "ready"  # Client errors don't affect server readiness
        raise
    except Exception as e:
        # 4. Centralized error handling (in base_server.py)
        self._handle_processing_error(e, "operation_name")
```

### Centralized Error Handler (`_handle_processing_error`)

```python
def _handle_processing_error(self, e: Exception, operation: str) -> None:
    """Handles GPU OOM and general errors centrally."""
    if isinstance(e, RuntimeError):
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "cuda" in error_msg:
            self.status = "error"
            self._cleanup_gpu_memory()  # Try to recover
            raise HTTPException(
                status_code=503,
                detail="[GPU_OOM]GPU out of memory. Try smaller input or restart engine."
            )

    # General error handling
    self.status = "error"
    logger.error(f"[{self.engine_name}] {operation} failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

This centralization means:
- GPU OOM handling code exists only in `base_server.py`, not duplicated in engines
- Consistent error messages and status codes across all engine types
- Automatic GPU memory cleanup attempts on OOM errors

## Logging Levels

Engine servers use structured logging with consistent levels:

| Level | Events |
|-------|--------|
| INFO | Server start, model load/unload, shutdown, requests |
| DEBUG | Detailed request parameters, internal operations |
| ERROR | Failed operations with stack traces |

Example log output:
```
[engine-name] Starting server on 0.0.0.0:8766
[engine-name] Engine initialized
[engine-name] Model loaded: model-name
[engine-name] Generating audio | Model: model-name | Language: en | Speaker: sample.wav
[engine-name] Shutdown requested
```
