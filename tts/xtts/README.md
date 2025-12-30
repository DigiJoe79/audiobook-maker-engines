# XTTS v2

Multilingual TTS with voice cloning from Coqui AI. Supports 17 languages with high-quality speech synthesis from reference audio samples.

## Overview

XTTS v2 is a production-ready text-to-speech engine featuring multilingual support and voice cloning capabilities. It generates natural-sounding speech by learning from short audio samples of the target voice, making it ideal for audiobook narration with consistent speaker voices across multiple languages.

**Key Features:**
- Voice cloning from audio samples (5-30 seconds recommended)
- 17 supported languages with single model
- GPU-accelerated inference (CUDA 11.8/12.1/12.4)
- Model hotswap without restart
- Latent caching for faster multi-segment generation

## Supported Languages

| Language | Code | Language | Code |
|----------|------|----------|------|
| Arabic | ar | Polish | pl |
| Chinese | zh-cn | Portuguese | pt |
| Czech | cs | Russian | ru |
| Dutch | nl | Spanish | es |
| English | en | Turkish | tr |
| French | fr | Japanese | ja |
| German | de | Korean | ko |
| Hungarian | hu | Hindi | hi |
| Italian | it | | |

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/xtts:latest
docker run -d -p 8766:8766 --gpus all ghcr.io/digijoe79/audiobook-maker-engines/xtts:latest
```

**Note:** GPU passthrough (`--gpus all`) is required for optimal performance.

### Subprocess (Development)

```bash
# Windows
cd tts/xtts
setup.bat

# Linux/Mac
cd tts/xtts
./setup.sh
```

## API Endpoints

### Common Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/load` | POST | Load a specific model |
| `/models` | GET | List available models |
| `/health` | GET | Health check with status and device info |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### TTS Endpoints (from BaseTTSServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate audio from text |
| `/samples/check` | POST | Check which speaker samples exist |
| `/samples/upload/{sample_id}` | POST | Upload speaker sample WAV |

---

## API Reference

### POST /load

Load a specific XTTS model into memory. Auto-unloads previous model (hotswap).

**Request:**
```json
{
  "engineModelName": "v2.0.2"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "v2.0.2"
}
```

### GET /models

List available XTTS models discovered from the `models/` directory.

**Response:**
```json
{
  "models": [
    {
      "name": "v2.0.2",
      "displayName": "V2.0.2",
      "languages": ["ar", "pt", "zh-cn", "cs", "nl", "en", "fr", "de", "it", "pl", "ru", "es", "tr", "ja", "ko", "hu", "hi"],
      "fields": []
    }
  ]
}
```

### GET /health

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "v2.0.2",
  "device": "cuda",
  "packageVersion": "0.27.0",
  "gpuMemoryUsedMb": 2048,
  "gpuMemoryTotalMb": 8192
}
```

**Status Values:**
- `initializing` - Server starting up
- `ready` - Ready for requests (model not loaded)
- `loading` - Model loading in progress
- `processing` - Generating audio
- `error` - Error occurred (check `errorMessage` field)

### POST /generate

Generate audio from text using voice cloning.

**Request:**
```json
{
  "text": "Text to synthesize with the cloned voice.",
  "language": "en",
  "ttsSpeakerWav": "speaker-uuid.wav",
  "parameters": {
    "temperature": 0.65,
    "speed": 1.0,
    "lengthPenalty": 1.0,
    "repetitionPenalty": 2.0,
    "topK": 50,
    "topP": 0.8
  }
}
```

**Request Fields:**
- `text` (string, required): Text to synthesize (max 250 characters per engine.yaml constraint)
- `language` (string, required): Language code from supported languages
- `ttsSpeakerWav` (string or array, required): Speaker sample filename(s) in samples directory
  - Single sample: `"uuid.wav"`
  - Multiple samples: `["uuid1.wav", "uuid2.wav"]` (averaged for better quality)
- `parameters` (object, optional): Generation parameters (omit to use defaults)

**Response:** Binary WAV audio (Content-Type: audio/wav)

**Sample Rate:** 24000 Hz

**Error Responses:**
- `400` - Invalid request (empty text, missing speaker, text too long)
- `404` - Speaker sample not found
- `503` - Model loading in progress (retry after loading completes)

### POST /samples/check

Check which speaker samples exist in the engine's samples directory.

**Request:**
```json
{
  "sampleIds": ["uuid1", "uuid2", "uuid3"]
}
```

**Response:**
```json
{
  "missing": ["uuid2"]
}
```

### POST /samples/upload/{sample_id}

Upload a speaker sample WAV file to the engine.

**URL Parameters:**
- `sample_id` (string): Unique identifier for the sample (without .wav extension)

**Request Body:** Binary WAV audio (Content-Type: audio/wav)

**Requirements:**
- Format: WAV
- Sample Rate: 22050 Hz or higher (24000 Hz recommended)
- Duration: 5-30 seconds (optimal for voice cloning)
- Quality: Clear speech, minimal background noise

**Response:**
```json
{
  "status": "ok",
  "sampleId": "uuid1"
}
```

---

## Configuration

Parameters from `engine.yaml`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| temperature | float | 0.65 | 0.0-1.0 | Sampling temperature (lower = more consistent, higher = more varied) |
| speed | float | 1.0 | 0.5-2.0 | Playback speed multiplier |
| lengthPenalty | float | 1.0 | 0.5-2.0 | Length penalty for generation (higher = shorter output) |
| repetitionPenalty | float | 2.0 | 1.0-10.0 | Penalty for repeating tokens (higher = less repetition) |
| topK | int | 50 | 10-100 | Top-K sampling (number of tokens to consider) |
| topP | float | 0.8 | 0.0-1.0 | Top-P sampling (nucleus sampling threshold) |
| enableTextSplitting | bool | false | - | Internal text splitting (readonly, handled by app) |

**Constraint:** Maximum text length per generation is **250 characters** (prevents exceeding XTTS 400 token limit).

## Available Models

| Model | Description |
|-------|-------------|
| v2.0.2 | Default XTTS v2.0.2 model (baked into Docker image) |

**Auto-discovery:** Additional models placed in `/app/models/` (Docker) or `./models/` (subprocess) are automatically discovered.

**Model Structure:**
```
models/
  v2.0.2/
    config.json
    model.pth
    vocab.json
    speakers_xtts.pth
```

## Troubleshooting

### Model not loading

**Symptom:** `/load` returns error or timeout

**Solution:**
1. Check available disk space (models are ~1.8GB)
2. Verify model name matches output from `/models` endpoint
3. Check logs for download errors: `docker logs <container-id>`
4. Ensure model directory contains `config.json`

### GPU Out of Memory

**Symptom:** Generation fails with CUDA OOM error

**Solution:**
1. Ensure no other GPU processes are running: `nvidia-smi`
2. Reduce text length (keep under 250 chars)
3. Restart engine to clear GPU memory: `docker restart <container-id>`
4. Consider using smaller model variant (if available)

### Voice cloning quality poor

**Symptom:** Generated audio doesn't match speaker sample

**Solution:**
1. Use higher quality speaker samples:
   - 5-30 seconds duration (optimal)
   - Clear speech, minimal background noise
   - 24000 Hz sample rate
2. Use multiple speaker samples (3-5 recommended) for better averaging
3. Ensure language code matches speaker sample language
4. Lower `temperature` for more consistent output (try 0.5)

### Latency too high

**Symptom:** Generation takes too long per request

**Solution:**
1. Verify GPU is being used: Check `/health` endpoint for `device: "cuda"`
2. Use latent caching: Reuse same speaker samples across segments (cached automatically)
3. Reduce `topK` and increase `topP` for faster sampling
4. Ensure CUDA drivers are up to date

### Speaker sample upload fails

**Symptom:** `/samples/upload` returns 400 or 500 error

**Solution:**
1. Verify WAV format: `ffprobe speaker.wav`
2. Check sample_id format (alphanumeric, dash, underscore only)
3. Ensure file size is reasonable (<10MB)
4. Check disk space in samples directory

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running: `docker ps` or check process
2. Check port is correct (default: 8766)
3. Verify firewall allows connection
4. Check logs for startup errors: `docker logs <container-id>`

## Dependencies

### Package Migration Notice

**As of 2025-12, this engine uses `coqui-tts` instead of `TTS`.**

The original Coqui AI `TTS` package (PyPI: `TTS`) was abandoned in December 2023 after the company shutdown. The [Idiap Research Institute](https://github.com/idiap/coqui-ai-TTS) maintains an active fork published as `coqui-tts` on PyPI.

- **Old package:** `TTS==0.22.0` (abandoned, last update Dec 2023)
- **New package:** `coqui-tts>=0.27.0` (actively maintained)

The import remains unchanged (`from TTS import ...`), ensuring backward compatibility.

### Versioning Decisions (2025-12-05)

| Package | Version | Rationale |
|---------|---------|-----------|
| coqui-tts | >=0.27.0,<0.28.0 | Latest stable from maintained Idiap fork. Requires transformers >4.52.1 |
| torch | >=2.5.0,<2.6.0 | Stable release with CUDA 11.8/12.1/12.4 support. coqui-tts requires >2.1,<2.9 |
| torchaudio | >=2.5.0,<2.6.0 | Must match torch version |
| transformers | >=4.53.0,<4.56.0 | Required by coqui-tts 0.27.x (>4.52.1,<4.56). Avoid 4.50+ GenerationMixin deprecation warnings |
| fastapi | >=0.115.0,<1.0.0 | Latest stable, no breaking changes from 0.109.x |
| uvicorn | >=0.32.0,<1.0.0 | Latest stable, performance improvements |
| pydantic | >=2.10.0,<3.0.0 | Latest 2.x stable, v3 may have breaking changes |
| loguru | >=0.7.2,<0.8.0 | Stable, rarely updated |
| httpx | >=0.28.0,<1.0.0 | Latest stable async HTTP client |

### Compatibility Matrix

| coqui-tts | PyTorch | Transformers | Python | CUDA |
|-----------|---------|--------------|--------|------|
| 0.27.x | 2.2-2.8 | 4.53-4.55 | 3.10-3.13 | 11.8, 12.1, 12.4 |
| 0.26.x | 2.1-2.8 | 4.43-4.55 | 3.10-3.12 | 11.8, 12.1 |
| TTS 0.22.0 (legacy) | 2.1.x | <4.36 | 3.9-3.11 | 11.8 |

### Upstream Dependency Constraints

From `coqui-tts` 0.27.x pyproject.toml:
```
torch >2.1, <2.9
torchaudio >2.1.0, <2.9
transformers >4.52.1, <4.56
```

### Upgrade Notes

**PyTorch Version Selection:**
- **2.5.x chosen** over 2.8.x for stability (2.5.1 released Nov 2024, well-tested)
- CUDA 11.8 index URL retained for maximum GPU compatibility
- For CUDA 12.x, change to `--extra-index-url https://download.pytorch.org/whl/cu121`

**Transformers Deprecation Warning:**
- transformers 4.50+ deprecates `GenerationMixin` inheritance in `PreTrainedModel`
- XTTS uses `GPT2InferenceModel` which may be affected
- Monitor for warnings, may need pinning to <4.50 if issues arise

**Known Incompatibilities:**
- **Python 3.14**: Not yet supported by coqui-tts
- **transformers >=4.56**: Breaks coqui-tts 0.27.x
- **torch >=2.9**: Breaks coqui-tts 0.27.x

**CPU-Only Installation:**
Comment out the `--extra-index-url` line in requirements.txt:
```
# --extra-index-url https://download.pytorch.org/whl/cu118
```

## Testing

Validate your engine with the automated test suite:

```bash
# Start engine first
cd tts/xtts
python server.py --port 8766 --host 127.0.0.1

# Run tests from repo root
python scripts/test_engine.py --port 8766 --verbose
```

**Test Coverage:**
- Health endpoint validation
- Model loading and unloading
- Audio generation with speaker samples
- Parameter validation
- Error handling (missing samples, text too long, etc.)

See [docs/engine-development-guide.md](../../docs/engine-development-guide.md) for comprehensive testing documentation.

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md) - Complete guide for creating new engines
- [Engine Server API](../../docs/engine-server-api.md) - Detailed API documentation and error handling
- [Model Management Standard](../../docs/model-management.md) - Model loading and hotswap behavior
- [coqui-tts (Idiap Fork)](https://github.com/idiap/coqui-ai-TTS) - Upstream repository
- [PyPI: coqui-tts](https://pypi.org/project/coqui-tts/) - Package on PyPI
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) - CUDA compatibility
