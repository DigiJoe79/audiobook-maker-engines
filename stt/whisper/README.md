# OpenAI Whisper

State-of-the-art speech recognition with automatic language detection. Supports transcription and translation across 99+ languages.

## Overview

Whisper is an STT (Speech-to-Text) engine that provides transcription with confidence scoring and text comparison. This engine converts audio to text with word-level timestamps and validates transcription accuracy against expected text.

**Key Features:**
- **Whisper Transcription** - Convert audio to text with word-level timestamps
- **Confidence Scoring** - Per-word and overall confidence metrics
- **Text Comparison** - Compare transcription against expected text to detect pronunciation issues
- **Pronunciation Rules Support** - Filter false positives using custom pronunciation rules
- **Multi-language Support** - Automatic language detection across 99+ languages

**Note:** This engine provides transcription only. For audio quality analysis (speech ratio, silence detection, clipping), use the separate [Silero-VAD](../../audio_analysis/silero-vad/) engine.

## Supported Languages

Whisper supports 99+ languages with automatic language detection. Common languages include:

| Language | Code |
|----------|------|
| English  | en   |
| German   | de   |
| French   | fr   |
| Spanish  | es   |
| Italian  | it   |
| Portuguese | pt |
| Dutch    | nl   |
| Polish   | pl   |
| Russian  | ru   |
| Japanese | ja   |
| Chinese  | zh   |
| Korean   | ko   |

See `engine.yaml` for complete list of supported language codes.

## Installation

### Docker (Recommended)

```bash
# GPU version (recommended)
docker pull ghcr.io/digijoe79/audiobook-maker-engines/whisper:latest
docker run -d -p 8767:8767 --gpus all ghcr.io/digijoe79/audiobook-maker-engines/whisper:latest

# CPU version (slower)
docker pull ghcr.io/digijoe79/audiobook-maker-engines/whisper:cpu
docker run -d -p 8767:8767 ghcr.io/digijoe79/audiobook-maker-engines/whisper:cpu
```

### Subprocess (Development)

```bash
# Windows
setup.bat

# Linux/Mac
./setup.sh
```

This creates an isolated virtual environment in `venv/` with all Whisper dependencies.

## API Endpoints

### Common Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/load` | POST | Load a specific model |
| `/models` | GET | List available models |
| `/health` | GET | Health check with status and device info |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### Quality Endpoints (from BaseQualityServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze audio and return quality metrics |

---

## API Reference

### POST /load

Load a specific Whisper model into memory. Auto-unloads previous model (hotswap).

**Request:**
```json
{
  "engineModelName": "base"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "base"
}
```

### GET /models

List available Whisper models with metadata.

**Response:**
```json
{
  "models": [
    {
      "name": "tiny",
      "displayName": "Tiny (39 MB)",
      "languages": ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ko"],
      "fields": [
        {"key": "size_mb", "value": 39, "fieldType": "number"},
        {"key": "speed", "value": "~10x realtime", "fieldType": "string"},
        {"key": "accuracy", "value": "lowest", "fieldType": "string"}
      ]
    },
    {
      "name": "base",
      "displayName": "Base (74 MB)",
      "languages": ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ko"],
      "fields": [
        {"key": "size_mb", "value": 74, "fieldType": "number"},
        {"key": "speed", "value": "~7x realtime", "fieldType": "string"},
        {"key": "accuracy", "value": "basic", "fieldType": "string"}
      ]
    }
  ],
  "defaultModel": "base"
}
```

### GET /health

Health check with current status, loaded model, and device information.

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "base",
  "device": "cuda",
  "packageVersion": "20250625",
  "gpuMemoryUsedMb": 2048,
  "gpuMemoryTotalMb": 8192
}
```

**Status Values:**
- `ready` - Engine is ready to process requests
- `loading` - Model is currently loading (503 returned for analyze requests)
- `processing` - Audio is being analyzed
- `error` - Engine encountered an error

### POST /analyze

Analyze audio with Whisper transcription and optional text comparison.

**Request:**
```json
{
  "audioBase64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "language": "en",
  "expectedText": "Hello world, this is a test.",
  "pronunciationRules": [
    {
      "pattern": "test",
      "replacement": "tst",
      "isRegex": false,
      "isActive": true
    }
  ],
  "qualityThresholds": {
    "maxSilenceDurationWarning": 2500,
    "maxSilenceDurationCritical": 3750,
    "speechRatioIdealMin": 75,
    "speechRatioIdealMax": 90,
    "speechRatioWarningMin": 65,
    "speechRatioWarningMax": 93,
    "maxClippingPeak": 0.0,
    "minAverageVolume": -40.0
  }
}
```

**Request Fields:**
- `audioBase64` (string, required): Base64-encoded WAV audio file
- `language` (string, default: "en"): Language code for transcription
- `expectedText` (string, optional): Original text for comparison
- `pronunciationRules` (array, optional): Active pronunciation rules to filter false positives
- `qualityThresholds` (object, optional): Quality thresholds (not used by Whisper - confidence-based only)

**Response (Generic Quality Format) - Perfect Match:**
```json
{
  "engineType": "stt",
  "engineName": "whisper",
  "qualityScore": 95,
  "qualityStatus": "perfect",
  "details": {
    "topLabel": "whisperTranscription",
    "fields": [
      {
        "key": "confidence",
        "value": 95,
        "type": "percent"
      },
      {
        "key": "language",
        "value": "en",
        "type": "string"
      },
      {
        "key": "textMatch",
        "value": "perfect",
        "type": "string"
      }
    ],
    "infoBlocks": {}
  }
}
```

**Response - With Text Deviations:**
```json
{
  "engineType": "stt",
  "engineName": "whisper",
  "qualityScore": 80,
  "qualityStatus": "warning",
  "details": {
    "topLabel": "whisperTranscription",
    "fields": [
      {
        "key": "confidence",
        "value": 85,
        "type": "percent"
      },
      {
        "key": "language",
        "value": "en",
        "type": "string"
      }
    ],
    "infoBlocks": {
      "textDeviations": [
        {
          "text": "wordMismatch",
          "severity": "warning",
          "details": {
            "expected": "hello",
            "detected": "hallo",
            "confidence": 0.72
          }
        },
        {
          "text": "wordMismatch",
          "severity": "error",
          "details": {
            "expected": "world",
            "detected": "word",
            "confidence": 0.25
          }
        }
      ]
    }
  }
}
```

**Response Fields:**
- `engineType` (string): Always "stt" for STT engines
- `engineName` (string): Engine identifier ("whisper")
- `qualityScore` (integer): Overall quality score 0-100 (100 = best)
- `qualityStatus` (string): "perfect" (>=85), "warning" (70-84), or "defect" (<70)
- `details.topLabel` (string): i18n key for section header
- `details.fields` (array): Key-value pairs for display
  - `key` (string): i18n key suffix (frontend prepends "quality.fields.")
  - `value` (any): Field value
  - `type` (string): Rendering type (percent, number, seconds, string, text)
- `details.infoBlocks` (object): Grouped messages by category
  - Each block contains array of items with:
    - `text` (string): i18n key suffix (frontend prepends "quality.issues.")
    - `severity` (string): error, warning, or info
    - `details` (object, optional): Additional structured data

**Quality Score Calculation:**
1. Base score = Whisper confidence (0-100)
2. If `expectedText` provided, compare transcription:
   - Each word mismatch reduces score by 5 points (max 40 point penalty)
   - Pronunciation rules filter false positives
3. Incomplete transcription (>5 missing words) triggers error

**Error Responses:**

```json
{
  "detail": "Model not loaded"
}
```

```json
{
  "detail": "Invalid audio format: expected WAV file"
}
```

```json
{
  "detail": "Model loading in progress. Retry after loading completes."
}
```

---

## Configuration

Parameters from `engine.yaml`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| confidence_threshold | float | 0.80 | 0.5-1.0 | Minimum confidence threshold for quality scoring |

## Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39 MB | ~10x realtime | Lowest | Quick testing, development |
| base | 74 MB | ~7x realtime | Basic | Default, balanced performance |
| small | 244 MB | ~4x realtime | Good | Better accuracy for production |
| medium | 769 MB | ~2x realtime | Better | High quality transcription |
| large | 1550 MB | ~1x realtime | Best | Maximum accuracy (GPU required) |

**Model Selection Guidance:**
- **Development:** Use `tiny` or `base` for fast iteration
- **Production (CPU):** Use `small` for best quality/speed balance
- **Production (GPU):** Use `medium` or `large` for highest accuracy

All models are multilingual and support automatic language detection.

## Text Comparison

When `expectedText` is provided, the engine performs word-by-word comparison to detect transcription deviations:

### How It Works

1. **Normalization**
   - Both texts are lowercased
   - Punctuation is removed for comparison
   - Texts are split into words

2. **Word Alignment**
   - Words are compared sequentially
   - Shift detection handles missing/extra words
   - Pronunciation rules filter false positives

3. **Issue Detection**
   - **Word Mismatch:** Expected word differs from transcribed word
     - Severity: `warning` if confidence > 0.3, `error` otherwise
     - Details include expected, detected, and confidence
   - **Incomplete Transcription:** More than 5 words missing from end
     - Severity: `error`
     - Details include number of missing words

4. **Quality Score Adjustment**
   - Each issue reduces score by 5 points
   - Maximum penalty: 40 points
   - Perfect match adds `textMatch: "perfect"` field

### Pronunciation Rules

Pronunciation rules allow filtering of known variations (accents, regional pronunciations):

```json
{
  "pattern": "test",
  "replacement": "tst",
  "isRegex": false,
  "isActive": true
}
```

- Words affected by active rules are not counted as mismatches
- Supports both literal substring matching and regex patterns
- Rules are evaluated case-insensitively

### Example Comparison

**Input:**
```
expectedText: "Hello world, this is a test."
transcription: "Hallo world, this is a tst."
pronunciationRules: [{"pattern": "test", "replacement": "tst", "isActive": true}]
```

**Result:**
- Issue 1: "hello" → "hallo" (word mismatch, warning)
- Issue 2: "test" → "tst" (filtered by pronunciation rule, ignored)
- Quality Score: 95 (base) - 5 (1 issue) = 90 (perfect)

## Troubleshooting

### Model not loading

**Symptom:** `/load` returns error or timeout

**Solution:**
1. Check available disk space (models download to `/app/external_models/`)
2. Verify model name matches engine.yaml: `tiny`, `base`, `small`, `medium`, `large`
3. Check logs for download errors: `docker logs <container-id>`
4. For subprocess: Ensure internet connection for first-time download

### GPU Out of Memory

**Symptom:** Analysis fails with CUDA OOM error

**Solution:**
1. Use smaller model variant (`base` instead of `large`)
2. Reduce audio file duration (split long recordings)
3. Restart engine to clear GPU memory: `docker restart <container-id>`
4. Check GPU memory usage: `nvidia-smi`

### Low confidence scores

**Symptom:** All transcriptions have low confidence (<70%)

**Solution:**
1. Verify audio quality:
   - Sample rate should be 16000 Hz or higher
   - Audio should be clear with minimal background noise
   - Use Silero-VAD engine to check speech ratio
2. Use larger model (`medium` or `large`)
3. Verify correct language code is specified

### Text comparison showing false positives

**Symptom:** Many word mismatches that sound correct

**Solution:**
1. Create pronunciation rules for known variations
2. Check for punctuation differences (automatically ignored)
3. Verify expected text matches audio exactly
4. Regional accents may require pronunciation rules

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running: `docker ps` or check process
2. Check port is correct (default: 8767)
3. Verify firewall allows connection
4. For Docker: Ensure port mapping is correct `-p 8767:8767`

### Invalid audio format error

**Symptom:** `/analyze` returns "Invalid audio format: expected WAV file"

**Solution:**
1. Verify audio is WAV format (not MP3, FLAC, etc.)
2. Check base64 encoding is correct
3. Minimum file size: 44 bytes (WAV header)
4. File must have RIFF header signature

## Dependencies

- Python 3.12
- OpenAI Whisper v20250625+
- PyTorch 2.5+ / torchaudio 2.5+
- FastAPI 0.120+
- NumPy 2.1+
- ffmpeg (system requirement)

### Versioning Decisions (2025-12-04)

| Package | Version | Rationale |
|---------|---------|-----------|
| openai-whisper | >=20250625 | Latest stable (June 2025). Date-based versioning (YYYYMMDD). |
| torch | >=2.5.0,<3.0.0 | Range allows 2.5.x-2.9.x. CUDA 11.8/12.x support from 2.5. |
| torchaudio | >=2.5.0,<3.0.0 | **MUST** match torch version. 2.9: Maintenance mode, deprecated APIs removed. |
| numpy | >=2.1.0 | 2.1+ for Python 3.12 improvements. Latest: 2.3.5 (Nov 2025). |
| fastapi | >=0.120.0 | Pydantic 2.12 compatibility. Latest: 0.123.0 (Nov 2025). |
| uvicorn | >=0.34.0 | Python 3.13/3.14 Support. Latest: 0.38.0 (Oct 2025). |
| pydantic | >=2.10.0,<3.0.0 | Pydantic 2.x line. Latest: 2.12.5 (Nov 2025). |
| pydub | >=0.25.1 | No updates since March 2021 - project stable but inactive. |
| loguru | >=0.7.0,<1.0.0 | Latest: 0.7.3 (2025). No breaking changes expected. |
| pyyaml | >=6.0.1 | 6.0.1 fixes LibYAML Cython build issue. |

### Upgrade Notes

**OpenAI Whisper:**
- GitHub: https://github.com/openai/whisper
- Uses date-based versioning (YYYYMMDD) - check releases for new versions
- Models download automatically on first use via torch.hub
- Requires ffmpeg installed on system
- Previous release was v20240930 (September 2024)

**PyTorch / torchaudio:**
- **CRITICAL:** torch and torchaudio versions must be identical!
- PyTorch 2.9.1 (Nov 2025) is current, but torchaudio 2.9 has breaking changes
- torchaudio 2.9 is in maintenance mode - deprecated APIs were removed
- Recommendation: If you encounter issues, downgrade to 2.8.x: `pip install torch==2.8.0 torchaudio==2.8.0`
- CUDA: 2.5+ supports CUDA 11.8 and 12.x
- Set device to "cuda" in engine.yaml for GPU acceleration

**NumPy:**
- NumPy 2.x has breaking changes compared to 1.x (C-API, deprecations)
- 2.1+ recommended for Python 3.12

### Tested Configuration

This configuration has been tested and works:
```
openai-whisper==20250625
torch==2.5.1
torchaudio==2.5.1
numpy==2.1.3
fastapi==0.115.6
uvicorn==0.34.0
pydantic==2.10.4
```

### Breaking Changes Watch

| Package | Version | Change |
|---------|---------|--------|
| torchaudio | 2.9.0 | Maintenance mode, `torchaudio.load()`/`save()` now use TorchCodec |
| numpy | 2.0.0 | C-API changes, some deprecated functions removed |
| pydantic | 3.0.0 | Not yet released, but Pydantic v1 support will be removed in near future |

## Testing

Validate your engine with the automated test suite:

```bash
# Run full API test suite
python scripts/test_engine.py --port 8767 --verbose
```

Manual testing:

```bash
# Start service
# Windows
venv\Scripts\python.exe server.py --port 8767

# Linux/Mac
source venv/bin/activate
python server.py --port 8767

# Test analyze endpoint
curl -X POST http://localhost:8767/analyze \
  -H "Content-Type: application/json" \
  -d '{"audioBase64": "<base64_encoded_wav>", "language": "en"}'

# Test health endpoint
curl http://localhost:8767/health

# Test model loading
curl -X POST http://localhost:8767/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName": "small"}'
```

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md) - Complete development guide
- [Engine Server API](../../docs/engine-server-api.md) - API endpoint documentation
- [Model Management Standard](../../docs/model-management.md) - Model storage and discovery
