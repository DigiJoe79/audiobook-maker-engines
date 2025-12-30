# Chatterbox

Resemble AI - Chatterbox TTS, SoTA open-source TTS

## Overview

Chatterbox is a state-of-the-art multilingual text-to-speech engine developed by Resemble AI, supporting 23 languages with high-quality voice cloning capabilities. It uses a diffusion-based architecture with speaker conditioning to generate natural-sounding speech at 24kHz sample rate.

**Key Features:**
- 23 language support with natural multilingual voice cloning
- High-quality 24kHz sample rate output
- GPU acceleration with CUDA support
- Controllable speech parameters (exaggeration, temperature, CFG weight)
- Cross-language voice transfer capability

## Supported Languages

| Code | Language   | Code | Language   | Code | Language   | Code | Language   |
|------|------------|------|------------|------|------------|------|------------|
| ar   | Arabic     | el   | Greek      | ja   | Japanese   | pt   | Portuguese |
| da   | Danish     | en   | English    | ko   | Korean     | ru   | Russian    |
| de   | German     | es   | Spanish    | ms   | Malay      | sv   | Swedish    |
|      |            | fi   | Finnish    | nl   | Dutch      | sw   | Swahili    |
|      |            | fr   | French     | no   | Norwegian  | tr   | Turkish    |
|      |            | he   | Hebrew     | pl   | Polish     | zh   | Chinese    |
|      |            | hi   | Hindi      |      |            |      |            |
|      |            | it   | Italian    |      |            |      |            |

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/chatterbox:latest
docker run -d -p 8766:8766 --gpus all ghcr.io/digijoe79/audiobook-maker-engines/chatterbox:latest
```

Note: Requires NVIDIA GPU with CUDA support. Use `--gpus all` to enable GPU acceleration.

### Subprocess (Development)

```bash
# Windows
cd tts/chatterbox
setup.bat

# Linux/Mac
cd tts/chatterbox
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Create isolated Python 3.11 virtual environment in `venv/`
2. Install CUDA-enabled PyTorch 2.6.0
3. Install chatterbox-tts with all dependencies
4. Handle pkuseg build dependencies correctly

Model files are automatically downloaded on first use (~2GB).

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

## API Reference

### POST /load

Load the Chatterbox multilingual model into memory.

**Request:**
```json
{
  "engineModelName": "multilingual"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "multilingual"
}
```

### GET /health

Check engine status and resource usage.

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "multilingual",
  "device": "cuda",
  "packageVersion": "0.1.4",
  "gpuMemoryUsedMb": 2048,
  "gpuMemoryTotalMb": 8192
}
```

### GET /models

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "multilingual",
      "displayName": "Multilingual (Pretrained)",
      "languages": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"]
    }
  ]
}
```

### POST /generate

Generate TTS audio with voice cloning.

**Request:**
```json
{
  "text": "Hello, this is a test of the Chatterbox engine.",
  "language": "en",
  "ttsSpeakerWav": "speaker-uuid.wav",
  "parameters": {
    "exaggeration": 0.5,
    "temperature": 0.8,
    "cfgWeight": 0.5,
    "seed": 0
  }
}
```

**Parameters:**
- `text`: Text to synthesize (max 300 characters recommended)
- `language`: Language code (e.g., "en", "de", "ja")
- `ttsSpeakerWav`: Speaker sample filename (required for voice cloning) or array of filenames
- `parameters`: Optional engine parameters (defaults used if omitted)

**Response:** Binary WAV audio (Content-Type: audio/wav)

**Error Responses:**
```json
{
  "detail": "Chatterbox requires speaker samples for voice cloning. Upload samples via /samples/upload and include sample IDs in ttsSpeakerWav."
}
```

### POST /samples/check

Check which speaker samples exist in the engine.

**Request:**
```json
{
  "sampleIds": ["uuid-1", "uuid-2", "uuid-3"]
}
```

**Response:**
```json
{
  "missing": ["uuid-2"]
}
```

### POST /samples/upload/{sample_id}

Upload a speaker sample WAV file.

**Request:** Binary WAV data in request body

**Response:**
```json
{
  "status": "ok",
  "sampleId": "uuid-1"
}
```

## Configuration

Parameters from `engine.yaml` (all optional, defaults applied if omitted):

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| exaggeration | float | 0.5 | 0.25-2.0 | Speech expressiveness control (0.5=neutral, >1.0 may sound unstable) |
| temperature | float | 0.8 | 0.05-5.0 | Randomness in generation (higher=more variation, may reduce quality) |
| cfgWeight | float | 0.5 | 0.2-1.0 | Classifier-free guidance weight (0.0=cross-language transfer mode) |
| seed | int | 0 | 0-2147483647 | Random seed for reproducible output (0=random each time) |

### Parameter Usage Examples

**Neutral speech (audiobook narration):**
```json
{
  "exaggeration": 0.5,
  "temperature": 0.8,
  "cfgWeight": 0.5
}
```

**Expressive speech (character dialogue):**
```json
{
  "exaggeration": 1.2,
  "temperature": 1.0,
  "cfgWeight": 0.5
}
```

**Cross-language voice transfer (English voice speaking German):**
```json
{
  "exaggeration": 0.5,
  "temperature": 0.8,
  "cfgWeight": 0.0
}
```

**Reproducible output:**
```json
{
  "exaggeration": 0.5,
  "temperature": 0.8,
  "cfgWeight": 0.5,
  "seed": 42
}
```

## Available Models

| Model | Description |
|-------|-------------|
| multilingual | Pretrained 23-language model with voice cloning (only model available) |

Chatterbox currently supports a single pretrained multilingual model that handles all 23 languages.

## Voice Cloning Best Practices

**Speaker Sample Requirements:**
- Audio length: 3-10 seconds recommended
- Quality: High-quality, noise-free recordings work best
- Format: Mono or stereo WAV files
- Language matching: Reference audio should match target language (or use `cfgWeight: 0.0` for cross-language transfer)

**Upload Process:**
1. Generate unique sample ID (UUID recommended)
2. Upload WAV via `POST /samples/upload/{sample_id}`
3. Include `{sample_id}.wav` in `ttsSpeakerWav` field when generating

**Cross-Language Voice Transfer:**
Set `cfgWeight: 0.0` to clone a voice across languages (e.g., use English reference audio to speak German text).

## Troubleshooting

### Model not loading

**Symptom:** `/load` returns error or timeout

**Solution:**
1. Verify internet connection for initial model download (~2GB)
2. Check available disk space (minimum 4GB free)
3. Check logs for download errors from HuggingFace
4. Verify CUDA drivers installed if using GPU

### GPU Out of Memory

**Symptom:** Generation fails with CUDA OOM error

**Solution:**
1. Reduce text length (use shorter segments, max 300 chars recommended)
2. Restart engine to clear GPU memory: `POST /shutdown` then restart
3. Close other GPU-intensive applications
4. Consider using CPU mode if GPU has insufficient memory (<4GB VRAM)

### Poor Quality Output

**Symptom:** Generated audio sounds robotic or unnatural

**Solution:**
1. Verify reference audio matches target language (or set `cfgWeight: 0.0` for cross-language)
2. Reduce `exaggeration` parameter (try 0.3-0.7 range)
3. Use higher-quality speaker samples (noise-free, clear speech)
4. Add punctuation to text for better prosody
5. Keep text segments under 300 characters

### pkuseg Build Failure

**Symptom:** `ModuleNotFoundError: No module named 'numpy'` during installation

**Solution:**
The setup scripts handle this automatically. For manual installation:
```bash
pip install "numpy>=1.24.0,<1.26.0" cython
pip install --no-build-isolation chatterbox-tts
```

The `--no-build-isolation` flag allows pkuseg to find numpy in the environment.

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running: `docker ps` or check process
2. Check port is correct (default 8766)
3. Verify firewall allows connection on port 8766
4. Check logs for startup errors: `docker logs <container_id>`

## Dependencies

### Versioning Decisions (2025-12-05)

| Package | Version | Rationale |
|---------|---------|-----------|
| chatterbox-tts | >=0.1.4,<0.2.0 | Latest stable release with strict internal dependency pinning |
| torch | ==2.6.0 | Pinned by chatterbox-tts, supports CUDA 11.8/12.1/12.4 |
| torchaudio | ==2.6.0 | Pinned by chatterbox-tts, must match torch version |
| transformers | ==4.46.3 | Pinned by chatterbox-tts for model compatibility |
| diffusers | ==0.29.0 | Pinned by chatterbox-tts for diffusion model support |
| numpy | >=1.24.0,<1.26.0 | Range required by chatterbox-tts, incompatible with Python 3.12 |
| fastapi | >=0.115.0,<1.0.0 | Latest stable server stack |
| uvicorn | >=0.32.0,<1.0.0 | Latest stable ASGI server |
| pydantic | >=2.10.0,<3.0.0 | Latest 2.x, v3 may introduce breaking changes |
| scipy | >=1.11.0,<2.0.0 | Audio resampling and processing utilities |

### Upgrade Notes

**chatterbox-tts:**
- Enforces strict version pinning for torch, transformers, diffusers, librosa, safetensors
- Cannot independently update PyTorch or transformers versions
- Includes gradio dependency (not used by engine server)
- Requires `--no-deps` flag during installation to preserve CUDA-enabled PyTorch

**Python Version:**
- Python 3.11 required (3.10-3.11 supported)
- Python 3.12+ incompatible due to numpy <1.26.0 constraint
- Recommended by Resemble AI for best compatibility

**CUDA Installation:**
Setup scripts automatically install PyTorch with CUDA 12.4 support:
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**Manual Installation (Advanced):**
```bash
# 1. Install build dependencies
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.24.0,<1.26.0" cython

# 2. Install CUDA PyTorch FIRST
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install chatterbox-tts without overwriting dependencies
pip install --no-build-isolation --no-deps chatterbox-tts

# 4. Install remaining chatterbox dependencies
pip install --no-build-isolation transformers==4.46.3 diffusers==0.29.0 librosa==0.11.0 \
  safetensors==0.5.3 conformer==0.3.2 resemble-perth==1.0.1 s3tokenizer \
  pykakasi==2.3.0 spacy-pkuseg

# 5. Install server stack
pip install fastapi uvicorn pydantic loguru httpx PyYAML scipy
```

The `--no-deps` flag prevents chatterbox-tts from installing CPU-only PyTorch.

## Testing

```bash
# Start engine
python server.py --port 8766 --host 127.0.0.1

# Run test suite
python scripts/test_engine.py --port 8766 --verbose
```

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md)
- [Engine Server API](../../docs/engine-server-api.md)
- [Model Management Standard](../../docs/model-management.md)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Chatterbox PyPI](https://pypi.org/project/chatterbox-tts/)
- [HuggingFace Model](https://huggingface.co/ResembleAI/chatterbox)
