# STT Engine Template

This is the template for creating new **Speech-to-Text (STT)** engine servers.

## Quick Start

1. **Copy template to new directory:**
   ```bash
   cp -r stt/_template stt/my_stt_engine
   cd stt/my_stt_engine
   ```

2. **Customize the template:**
   - Rename class in `server.py` (e.g., `MySTTServer`)
   - Update `engine_name` and `display_name`
   - Implement `/analyze` endpoint with your transcription logic
   - Update `engine.yaml` with your engine's configuration
   - Add dependencies to `requirements.txt`

3. **Create virtual environment:**
   ```bash
   # Windows
   setup.bat

   # Linux/Mac
   chmod +x setup.sh
   ./setup.sh
   ```

4. **Test standalone:**
   ```bash
   venv\Scripts\python.exe server.py --port 8767
   ```

5. **Restart backend** - Engine will be auto-discovered!

## Architecture

STT engines inherit from `BaseQualityServer` which extends `BaseEngineServer`:

```
BaseEngineServer (base_server.py)
├── /health - Health check
├── /load - Load model
├── /models - List available models
└── /shutdown - Graceful shutdown

BaseQualityServer (base_quality_server.py) extends BaseEngineServer
└── /analyze - Returns Generic Quality Format (shared with Audio engines)

STT Engine (your server.py) extends BaseQualityServer
└── Implements analyze_audio() method
```

## Required Endpoints

### Standard Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns status and loaded model |
| `/load` | POST | Load a model by name |
| `/models` | GET | List available models with metadata |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### STT-Specific Endpoint (you implement)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Transcribe audio and return **Generic Quality Format** |

## Generic Quality Format

**IMPORTANT:** The `/analyze` endpoint must return results in the **Generic Quality Format** for integration with the QualityWorker.

### POST /analyze

**Request:**
```json
{
  "audioBase64": "<base64-encoded-audio>",
  "language": "en"
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
    "topLabel": "whisperTranscription",
    "fields": [
      {"key": "confidence", "value": 85, "type": "percent"},
      {"key": "transcription", "value": "Hello world", "type": "text"},
      {"key": "language", "value": "en", "type": "string"},
      {"key": "duration", "value": 2.5, "type": "seconds"}
    ],
    "infoBlocks": {}
  }
}
```

### Generic Quality Format Schema

| Field | Type | Description |
|-------|------|-------------|
| `engineType` | string | Always `"stt"` for STT engines |
| `engineName` | string | Your engine name (e.g., `"whisper"`) |
| `qualityScore` | int | Score 0-100 (100 = best quality) |
| `qualityStatus` | string | `"perfect"` (>=85), `"warning"` (70-84), `"defect"` (<70) |
| `details` | object | Engine-specific details |

### Details Object Schema

| Field | Type | Description |
|-------|------|-------------|
| `topLabel` | string | i18n key for the details section header |
| `fields` | array | List of key-value fields to display |
| `infoBlocks` | object | Optional grouped info messages |

### Field Types

| Type | Description | Example |
|------|-------------|---------|
| `percent` | Percentage value (0-100) | `{"key": "confidence", "value": 85, "type": "percent"}` |
| `text` | Text content | `{"key": "transcription", "value": "Hello", "type": "text"}` |
| `string` | Simple string | `{"key": "language", "value": "en", "type": "string"}` |
| `seconds` | Duration in seconds | `{"key": "duration", "value": 2.5, "type": "seconds"}` |
| `number` | Numeric value | `{"key": "words", "value": 42, "type": "number"}` |

## Required Methods

Engines must implement these methods from BaseQualityServer:

```python
def analyze_audio(
    self,
    audio_bytes: bytes,
    language: str,
    thresholds: QualityThresholds
) -> AnalyzeResult:
    """
    Transcribe audio and return quality metrics.

    Args:
        audio_bytes: Raw audio file bytes (WAV format)
        language: Language code (e.g., "en", "de")
        thresholds: Quality thresholds (from request)

    Returns:
        AnalyzeResult with quality score, fields, and info blocks
    """
    pass

def load_model(self, model_name: str) -> None:
    """Load STT model into memory."""
    pass

def unload_model(self) -> None:
    """Unload model and free resources."""
    pass

def get_available_models(self) -> List[ModelInfo]:
    """Return list of available models with metadata."""
    pass
```

## Example Implementation

```python
from base_quality_server import (
    BaseQualityServer,
    QualityThresholds,
    QualityField,
    AnalyzeResult
)
from base_server import ModelInfo

class MySTTServer(BaseQualityServer):
    def __init__(self):
        super().__init__(
            engine_name="my_stt",
            display_name="My STT Engine",
            engine_type="stt"  # Always "stt" for STT engines
        )
        self.model = None

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds
    ) -> AnalyzeResult:
        # Your transcription logic here
        transcription = self.model.transcribe(audio_bytes, language)
        confidence = calculate_confidence(transcription)

        fields = [
            QualityField(key="confidence", value=confidence, type="percent"),
            QualityField(key="transcription", value=transcription, type="text"),
            QualityField(key="language", value=language, type="string"),
        ]

        return AnalyzeResult(
            quality_score=confidence,
            fields=fields,
            info_blocks={},
            top_label="mySTTTranscription"
        )

    def load_model(self, model_name: str) -> None:
        self.model = load_my_model(model_name)
        self.current_model = model_name
        self.model_loaded = True

    def unload_model(self) -> None:
        self.model = None
        # Note: model_loaded and current_model are reset by base_server.py

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(name="base", display_name="Base Model", languages=["en", "de"], fields=[]),
            ModelInfo(name="large", display_name="Large Model", languages=["en", "de"], fields=[]),
        ]
```

## Configuration (engine.yaml)

```yaml
schema_version: 2

name: "my-stt"
display_name: "My STT Engine"
engine_type: "stt"
description: "My custom STT engine"

upstream:
  name: "Original Project"
  url: "https://github.com/..."
  license: "MIT"

variants:
  - tag: "latest"
    platforms: ["linux/amd64"]
    requires_gpu: false

models:
  - name: "base"
    display_name: "Base Model"
  - name: "large"
    display_name: "Large Model"

default_model: "base"

supported_languages:
  - en
  - de
  - fr

capabilities:
  supports_model_hotswap: true
  supports_speaker_cloning: false
  supports_streaming: false

installation:
  python_version: "3.10"
  venv_path: "./venv"
  requires_gpu: false
```

## Directory Structure

```
my_stt_engine/
├── server.py          # Engine implementation
├── models.py          # Pydantic request/response models
├── engine.yaml        # Configuration
├── requirements.txt   # Dependencies
├── setup.bat          # Windows setup script
├── setup.sh           # Linux/Mac setup script
├── models/            # Downloaded models (optional)
│   └── .gitkeep
└── venv/              # Virtual environment (created by setup)
```

## Testing

```bash
# Start server
venv\Scripts\python.exe server.py --port 8767

# Test health
curl http://localhost:8767/health

# Test models
curl http://localhost:8767/models

# Test load
curl -X POST http://localhost:8767/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName": "base"}'

# Test analyze (with base64 audio)
curl -X POST http://localhost:8767/analyze \
  -H "Content-Type: application/json" \
  -d '{"audioBase64":"<base64-data>","language":"en","qualityThresholds":{}}'
```

### Automated API Testing

Use the comprehensive test suite:

```bash
# Run full test suite
python scripts/test_engine.py --port 8767 --verbose

# Skip shutdown test during development
python scripts/test_engine.py --port 8767 --skip-shutdown
```

## Examples

See `stt/whisper/` for a complete working example.

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md) - Complete development guide
- [Engine Server API](../../docs/engine-server-api.md) - API endpoint documentation
- [Model Management](../../docs/model-management.md) - Model handling patterns
