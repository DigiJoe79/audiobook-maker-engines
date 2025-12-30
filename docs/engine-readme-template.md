# Engine README Template

This template defines the standard structure for engine README files. Use the appropriate sections based on engine type.

---

## Template Structure

```markdown
# {Engine Display Name}

{One-line description from engine.yaml}

## Overview

{2-3 sentences explaining what the engine does and its primary use case}

**Key Features:**
- Feature 1
- Feature 2
- Feature 3

## Supported Languages

<!-- Include if engine has language-specific behavior -->

| Language | Code | Model (if applicable) |
|----------|------|----------------------|
| English  | en   | en_core_web_md |
| German   | de   | de_core_news_md |

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/{engine-name}:latest
docker run -d -p {port}:{port} ghcr.io/digijoe79/audiobook-maker-engines/{engine-name}:latest
```

### Subprocess (Development)

```bash
# Windows
setup.bat

# Linux/Mac
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

<!-- TTS Engines: Add this section -->
### TTS Endpoints (from BaseTTSServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate audio from text |
| `/samples/check` | POST | Check which speaker samples exist |
| `/samples/upload/{sample_id}` | POST | Upload speaker sample WAV |

<!-- STT/Audio Engines: Add this section -->
### Quality Endpoints (from BaseQualityServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze audio and return quality metrics |

<!-- Text Engines: Add this section -->
### Text Endpoints (from BaseTextServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/segment` | POST | Segment text into TTS-ready chunks |

---

## API Reference

### POST /load

Load a specific model into memory. Auto-unloads previous model (hotswap).

**Request:**
```json
{
  "engineModelName": "model-name"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "model-name"
}
```

### GET /health

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "model-name",
  "device": "cuda",
  "packageVersion": "1.0.0",
  "gpuMemoryUsedMb": 2048,
  "gpuMemoryTotalMb": 8192
}
```

<!-- TTS Engines: Include this -->
### POST /generate (TTS)

**Request:**
```json
{
  "text": "Text to synthesize",
  "language": "en",
  "ttsSpeakerWav": "speaker-uuid.wav",
  "parameters": {
    "temperature": 0.7,
    "speed": 1.0
  }
}
```

**Response:** Binary WAV audio (Content-Type: audio/wav)

### POST /samples/check (TTS)

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

<!-- STT/Audio Engines: Include this -->
### POST /analyze (Quality)

**Request:**
```json
{
  "audioBase64": "base64-encoded-wav",
  "language": "en",
  "expectedText": "Original text for comparison",
  "pronunciationRules": [],
  "qualityThresholds": {
    "maxSilenceDurationWarning": 2500,
    "maxSilenceDurationCritical": 3750,
    "speechRatioIdealMin": 75,
    "speechRatioIdealMax": 90
  }
}
```

**Response (Generic Quality Format):**
```json
{
  "engineType": "stt",
  "engineName": "whisper",
  "qualityScore": 85,
  "qualityStatus": "perfect",
  "details": {
    "topLabel": "qualityLabel",
    "fields": [
      {"key": "fieldName", "value": 85, "type": "percent"}
    ],
    "infoBlocks": {}
  }
}
```

<!-- Text Engines: Include this -->
### POST /segment (Text)

**Request:**
```json
{
  "text": "Long text to segment into chunks...",
  "language": "en",
  "maxLength": 250,
  "minLength": 10,
  "markOversized": true
}
```

**Response:**
```json
{
  "segments": [
    {
      "text": "First segment text.",
      "start": 0,
      "end": 19,
      "orderIndex": 0,
      "status": "ok"
    }
  ],
  "totalSegments": 5,
  "totalCharacters": 1250,
  "failedCount": 0
}
```

---

## Configuration

Parameters from `engine.yaml`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| param_name | float | 1.0 | 0.0-2.0 | Parameter description |

## Available Models

| Model | Size | Description |
|-------|------|-------------|
| model-name | 74 MB | Model description |

## Troubleshooting

### Model not loading

**Symptom:** `/load` returns error or timeout

**Solution:**
1. Check available disk space
2. Verify model name matches engine.yaml
3. Check logs for download errors

### GPU Out of Memory

**Symptom:** Generation fails with CUDA OOM error

**Solution:**
1. Use smaller model variant
2. Reduce batch size / text length
3. Restart engine to clear GPU memory

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running: `docker ps` or check process
2. Check port is correct
3. Verify firewall allows connection

## Dependencies

### Versioning Decisions ({YYYY-MM-DD})

| Package | Version | Rationale |
|---------|---------|-----------|
| package | >=X.Y.Z | Reason for version constraint |

### Upgrade Notes

**Package Name:**
- Important compatibility notes
- Breaking changes to watch

## Testing

```bash
python scripts/test_engine.py --port {port} --verbose
```

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md)
- [Engine Server API](../../docs/engine-server-api.md)
```

---

## Section Requirements by Engine Type

| Section | TTS | STT | Text | Audio |
|---------|-----|-----|------|-------|
| Overview | Required | Required | Required | Required |
| Supported Languages | Required | Required | Required | Optional |
| Installation | Required | Required | Required | Required |
| API Endpoints (Common) | Required | Required | Required | Required |
| API Endpoints (Type-specific) | /generate, /samples/* | /analyze | /segment | /analyze |
| Configuration | Required | Required | Required | Required |
| Available Models | If multiple | If multiple | If multiple | Optional |
| Troubleshooting | Required | Required | Required | Required |
| Dependencies | Required | Required | Required | Required |
| Testing | Required | Required | Required | Required |
| Documentation | Required | Required | Required | Required |

## Guidelines

### DO Include

- Accurate API examples matching actual request/response format (camelCase!)
- Parameter tables with type, default, and valid range
- Common error scenarios with solutions
- Version rationale for non-obvious dependency constraints

### DO NOT Include

- Manual venv setup commands (setup scripts handle this)
- Architecture diagrams (belong in docs/engine-development-guide.md)
- Implementation code examples (belong in _template/ directories)
- Future enhancements / roadmap (use GitHub Issues)
- Duplicate information from base server documentation

### Naming Conventions

- Use camelCase for all JSON field names in examples
- Use snake_case for Python code references
- Use kebab-case for endpoint paths
