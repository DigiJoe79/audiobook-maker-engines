# VibeVoice

Microsoft VibeVoice - Expressive multi-speaker TTS with voice cloning. Supports 1.5B and 7B models for high-quality conversational audio synthesis.

## Overview

VibeVoice is Microsoft's state-of-the-art text-to-speech engine designed for expressive, long-form, multi-speaker conversational audio synthesis. It excels at generating natural-sounding speech with voice cloning capabilities, making it ideal for audiobook narration, podcasts, and dialogue-heavy content.

**Key Features:**
- Voice cloning from audio samples (10-60 seconds recommended)
- Multi-speaker support (up to 4 speakers per generation)
- Long-form audio generation (up to 90 minutes for 1.5B model, 45 minutes for 7B model)
- Two model sizes: 1.5B (~3GB VRAM) and 7B (~18GB VRAM)
- Expressive prosody and natural conversational tone
- GPU-accelerated with CUDA support required

## Supported Languages

| Language | Code | Stability |
|----------|------|-----------|
| English  | en   | Stable |
| Chinese  | zh   | Stable |
| German   | de   | Experimental |
| French   | fr   | Experimental |
| Italian  | it   | Experimental |
| Japanese | ja   | Experimental |
| Korean   | ko   | Experimental |
| Dutch    | nl   | Experimental |
| Polish   | pl   | Experimental |
| Portuguese | pt | Experimental |
| Spanish  | es   | Experimental |

**Note:** English and Chinese have the most stable performance. Other languages are experimental and may have variable quality.

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/vibevoice:latest
docker run -d -p 8766:8766 --gpus all ghcr.io/digijoe79/audiobook-maker-engines/vibevoice:latest
```

**Important:** VibeVoice requires NVIDIA GPU with CUDA support. Use `--gpus all` flag to expose GPU to container.

### Subprocess (Development)

```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh
```

The setup script will:
1. Create Python 3.12 virtual environment
2. Install PyTorch 2.9.1 with CUDA 13.0
3. Install VibeVoice package from community fork
4. Install all dependencies

## API Endpoints

### Common Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/load` | POST | Load a specific model (1.5B or 7B) |
| `/models` | GET | List available models |
| `/health` | GET | Health check with status and device info |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### TTS Endpoints (from BaseTTSServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate audio from text with voice cloning |
| `/samples/check` | POST | Check which speaker samples exist |
| `/samples/upload/{sample_id}` | POST | Upload speaker sample WAV |

---

## API Reference

### POST /load

Load a specific VibeVoice model into memory. Auto-unloads previous model (hotswap).

**Request:**
```json
{
  "engineModelName": "1.5B"
}
```

**Valid model names:** `1.5B`, `7B`

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "1.5B"
}
```

### GET /models

List available VibeVoice models with metadata.

**Response:**
```json
{
  "models": [
    {
      "name": "1.5B",
      "displayName": "VibeVoice 1.5B (~3GB VRAM)",
      "languages": ["en", "zh", "de", "fr", "it", "ja", "ko", "nl", "pl", "pt", "es"],
      "fields": [
        {"key": "vramGb", "value": 3, "fieldType": "number"},
        {"key": "maxAudioMinutes", "value": 90, "fieldType": "number"},
        {"key": "parameters", "value": "1.5B", "fieldType": "string"},
        {"key": "voiceCloning", "value": true, "fieldType": "boolean"}
      ]
    },
    {
      "name": "7B",
      "displayName": "VibeVoice 7B (~18GB VRAM)",
      "languages": ["en", "zh", "de", "fr", "it", "ja", "ko", "nl", "pl", "pt", "es"],
      "fields": [
        {"key": "vramGb", "value": 18, "fieldType": "number"},
        {"key": "maxAudioMinutes", "value": 45, "fieldType": "number"},
        {"key": "parameters", "value": "9B", "fieldType": "string"},
        {"key": "voiceCloning", "value": true, "fieldType": "boolean"}
      ]
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
  "currentEngineModel": "1.5B",
  "device": "cuda",
  "packageVersion": "0.1.0",
  "gpuMemoryUsedMb": 3072,
  "gpuMemoryTotalMb": 8192
}
```

### POST /generate

Generate TTS audio with voice cloning from uploaded speaker samples.

**Request:**
```json
{
  "text": "Speaker 1: Hello, this is a demonstration of VibeVoice text-to-speech synthesis.",
  "language": "en",
  "ttsSpeakerWav": "speaker-uuid.wav",
  "parameters": {
    "cfgScale": 1.3,
    "doSample": false,
    "temperature": 0.95,
    "topP": 0.95,
    "maxNewTokens": 4096
  }
}
```

**Multi-speaker example:**
```json
{
  "text": "Speaker 1: Hello! Speaker 2: Hi there!",
  "language": "en",
  "ttsSpeakerWav": ["speaker1.wav", "speaker2.wav"],
  "parameters": {
    "cfgScale": 1.3
  }
}
```

**Response:** Binary WAV audio (Content-Type: audio/wav)

**Notes:**
- `ttsSpeakerWav` is required (voice cloning engine)
- Text must be formatted as "Speaker N: text" for multi-speaker
- Max text length: 5000 characters (from engine.yaml)
- Audio frame rate: 7.5 Hz (7.5 tokens = 1 second)
- Default max_new_tokens: 4096 (~9 minutes)

### POST /samples/check

Check which speaker samples exist in the engine.

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

Upload a speaker sample WAV file for voice cloning.

**Request:** Binary WAV data in request body

**Example:**
```bash
curl -X POST http://localhost:8766/samples/upload/speaker-uuid \
  -H "Content-Type: audio/wav" \
  --data-binary @speaker_sample.wav
```

**Response:**
```json
{
  "status": "ok",
  "sampleId": "speaker-uuid"
}
```

**Recommendations:**
- Sample duration: 10-60 seconds
- Sample rate: 24kHz (VibeVoice native rate)
- Format: Mono WAV
- Quality: Clean audio with minimal background noise

---

## Configuration

Parameters from `engine.yaml`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| cfg_scale | float | 1.3 | 1.0-3.0 | Classifier-free guidance scale. Controls adherence to text (1.3 recommended, values >2.0 can be unstable) |
| do_sample | bool | false | - | Enable sampling mode for varied output. False = deterministic, stable results |
| temperature | float | 0.95 | 0.1-2.0 | Sampling temperature (only used when do_sample=true). Higher = more random |
| top_p | float | 0.95 | 0.1-1.0 | Nucleus sampling threshold (only used when do_sample=true) |

**Advanced Parameters (not in UI):**
- `max_new_tokens`: Max tokens to generate (default: 4096 = ~9 minutes). VibeVoice uses 7.5 Hz frame rate

## Available Models

| Model | VRAM | Max Audio Length | Parameters | HuggingFace ID |
|-------|------|------------------|------------|----------------|
| 1.5B  | ~3GB | 90 minutes | 1.5 billion | microsoft/VibeVoice-1.5B |
| 7B    | ~18GB | 45 minutes | 9 billion | vibevoice/VibeVoice-7B |

**Model Selection:**
- **1.5B**: Recommended for most users. Faster generation, lower VRAM, longer audio limits
- **7B**: Highest quality output with more expressive prosody. Requires high-end GPU (RTX 3090/4090, A100)

**Note:** Models are downloaded from HuggingFace on first load and cached in `/app/external_models` (Docker) or `./external_models` (subprocess).

## Troubleshooting

### Model not loading

**Symptom:** `/load` returns error or timeout

**Solution:**
1. Verify GPU is available: Check `/health` endpoint shows `"device": "cuda"`
2. Check VRAM: Ensure GPU has sufficient memory (3GB for 1.5B, 18GB for 7B)
3. Verify HuggingFace connectivity: Model downloads require internet access
4. Check logs for specific error (flash attention, device mapping, etc.)

**Common load errors:**
- `flash_attention_2 not available`: Falls back to SDPA or eager attention (slower but works)
- `speech_bias_factor doesn't have any device`: Fixed by using `device_map="cuda"` strategy

### GPU Out of Memory

**Symptom:** Generation fails with `CUDA out of memory` error

**Solution:**
1. Use smaller model: Switch from 7B to 1.5B
2. Reduce max_new_tokens: Limit audio length per generation
3. Restart engine: Unload model via `/shutdown`, restart container/process
4. Close other GPU applications: Free up VRAM

### Audio quality issues

**Symptom:** Generated audio sounds robotic, choppy, or unnatural

**Solution:**
1. Use recommended cfg_scale: Keep at 1.3 (default). Values >2.0 can be "unhinged"
2. Disable sampling: Use `do_sample: false` for stable, production-quality output
3. Improve speaker samples: Use 10-60 second clean samples with minimal background noise
4. Match language: Use English or Chinese for best stability, experimental languages vary

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running: `docker ps` or check process
2. Check port is correct (default: 8766)
3. Verify firewall allows connection
4. For Docker: Ensure port mapping is correct (`-p 8766:8766`)

### Speaker sample upload fails

**Symptom:** `/samples/upload` returns 400 or 500 error

**Solution:**
1. Verify WAV format: Must be valid WAV file
2. Check filename: Only alphanumeric, dash, underscore allowed
3. Verify samples directory exists: Should be auto-created at `/app/samples`
4. Check file size: Ensure disk space available

## Dependencies

### Versioning Decisions (2025-12-30)

| Package | Version | Rationale |
|---------|---------|-----------|
| PyTorch | 2.9.1 (CUDA 13.0) | Latest stable with CUDA 13 support for modern GPUs (RTX 4000 series, H100) |
| Python | 3.12 | Required for VibeVoice package compatibility |
| vibevoice | git+https://github.com/vibevoice-community/VibeVoice.git | Community fork with fixes and improvements (official repo may have issues) |
| transformers | (via vibevoice) | Auto-installed as VibeVoice dependency |
| diffusers | (via vibevoice) | Auto-installed as VibeVoice dependency |
| flash-attn | (optional) | Installed if available, fallback to SDPA/eager attention |

### Upgrade Notes

**VibeVoice Package:**
- Install from community fork: `pip install git+https://github.com/vibevoice-community/VibeVoice.git`
- Official repo: `https://github.com/microsoft/VibeVoice` (may have installation issues)
- Version tracking: Uses git commit hash (no PyPI release)

**PyTorch:**
- CUDA 13.0 support enables latest GPU features (flash attention, improved memory management)
- Ensure NVIDIA drivers are up-to-date (525.60.13+ for CUDA 13)

**Attention Implementation:**
- Prefers flash_attention_2 (fastest, lowest memory)
- Falls back to SDPA (scaled dot-product attention)
- Falls back to eager (slowest, highest memory)

## Testing

```bash
python scripts/test_engine.py --port 8766 --verbose
```

**Manual Testing:**

```bash
# 1. Start engine
docker run -d -p 8766:8766 --gpus all ghcr.io/digijoe79/audiobook-maker-engines/vibevoice:latest

# 2. Check health
curl http://localhost:8766/health

# 3. Load model
curl -X POST http://localhost:8766/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName": "1.5B"}'

# 4. Upload speaker sample
curl -X POST http://localhost:8766/samples/upload/test-speaker \
  -H "Content-Type: audio/wav" \
  --data-binary @speaker.wav

# 5. Generate audio
curl -X POST http://localhost:8766/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Speaker 1: This is a test of VibeVoice.",
    "language": "en",
    "ttsSpeakerWav": "test-speaker.wav",
    "parameters": {"cfgScale": 1.3}
  }' \
  --output output.wav
```

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md)
- [Engine Server API](../../docs/engine-server-api.md)
- [Model Management Standard](../../docs/model-management.md)
- [VibeVoice Official Repo](https://github.com/microsoft/VibeVoice)
- [VibeVoice Community Fork](https://github.com/vibevoice-community/VibeVoice)
