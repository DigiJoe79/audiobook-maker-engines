# Audio Analysis Engine Template

This is the template for creating new **Audio Analysis** engine servers.

Audio analysis engines analyze audio quality metrics like speech ratio, silence detection, clipping, volume levels, etc.

## Quick Start

1. **Copy template to new directory:**
   ```bash
   cp -r audio_analysis/_template audio_analysis/my_analyzer
   cd audio_analysis/my_analyzer
   ```

2. **Customize the template:**
   - Rename class in `server.py` (e.g., `MyAudioAnalyzer`)
   - Update `engine_name` and `display_name`
   - Implement `analyze_audio()` with your analysis logic
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
   venv\Scripts\python.exe server.py --port 8768
   ```

5. **Restart backend** - Engine will be auto-discovered!

## Architecture

Audio analysis engines inherit from `BaseQualityServer` which extends `BaseEngineServer`:

```
BaseEngineServer (base_server.py)
├── /health - Health check
├── /load - Load model
├── /models - List available models
└── /shutdown - Graceful shutdown

BaseQualityServer (base_quality_server.py) extends BaseEngineServer
└── /analyze - Returns Generic Quality Format (shared with STT engines)

Audio Analysis Engine (your server.py) extends BaseQualityServer
└── Implements analyze_audio() method
```

## Required Endpoints

### Standard Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns status |
| `/load` | POST | Load a model (optional - some analyzers are model-free) |
| `/models` | GET | List available models/configurations |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### Audio-Specific Endpoint (you implement)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze audio and return **Generic Quality Format** |

## Generic Quality Format

**IMPORTANT:** The `/analyze` endpoint must return results in the **Generic Quality Format** for integration with the QualityWorker.

### POST /analyze

**Request:**
```json
{
  "audioBase64": "<base64-encoded-audio>",
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

**Response (Generic Quality Format):**
```json
{
  "engineType": "audio",
  "engineName": "silero-vad",
  "qualityScore": 92,
  "qualityStatus": "perfect",
  "details": {
    "topLabel": "audioQuality",
    "fields": [
      {"key": "speechRatio", "value": 82, "type": "percent"},
      {"key": "maxSilence", "value": 1200, "type": "number"},
      {"key": "peakDb", "value": -3.2, "type": "number"},
      {"key": "avgVolumeDb", "value": -18.5, "type": "number"}
    ],
    "infoBlocks": {
      "warnings": [
        {"text": "audioIssue.highSpeechRatio", "severity": "warning"}
      ]
    }
  }
}
```

### Generic Quality Format Schema

| Field | Type | Description |
|-------|------|-------------|
| `engineType` | string | Always `"audio"` for audio analysis engines |
| `engineName` | string | Your engine name (e.g., `"silero-vad"`) |
| `qualityScore` | int | Score 0-100 (100 = best quality) |
| `qualityStatus` | string | `"perfect"` (>=85), `"warning"` (70-84), `"defect"` (<70) |
| `details` | object | Engine-specific details |

### Details Object Schema

| Field | Type | Description |
|-------|------|-------------|
| `topLabel` | string | i18n key for the details section header |
| `fields` | array | List of key-value fields to display |
| `infoBlocks` | object | Grouped info messages (warnings, errors, info) |

### Field Types

| Type | Description | Example |
|------|-------------|---------|
| `percent` | Percentage value (0-100) | `{"key": "speechRatio", "value": 82, "type": "percent"}` |
| `number` | Numeric value | `{"key": "maxSilence", "value": 1200, "type": "number"}` |
| `seconds` | Duration in seconds | `{"key": "duration", "value": 2.5, "type": "seconds"}` |
| `string` | Simple string | `{"key": "status", "value": "ok", "type": "string"}` |

### Info Block Severities

| Severity | Description | UI Treatment |
|----------|-------------|--------------|
| `error` | Critical issue | Red highlight |
| `warning` | Potential problem | Yellow highlight |
| `info` | Informational | Gray/neutral |

## Common Audio Metrics

Audio analysis engines typically measure:

| Metric | Description | Typical Thresholds |
|--------|-------------|-------------------|
| **Speech Ratio** | % of audio containing speech | Ideal: 75-90%, Warning: 65-93% |
| **Max Silence** | Longest silence gap (ms) | Warning: >2500ms, Critical: >3750ms |
| **Peak dB** | Maximum audio level | Critical if >0 dB (clipping) |
| **Avg Volume dB** | Average RMS volume | Warning if <-40 dB (too quiet) |
| **Noise Level** | Background noise estimate | Higher = worse quality |

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
    Analyze audio and return quality metrics.

    Args:
        audio_bytes: Raw audio file bytes (WAV format)
        language: Language code (not typically used for audio analysis)
        thresholds: Quality thresholds for determining warnings/errors

    Returns:
        AnalyzeResult with quality score, fields, and info blocks
    """
    pass

def load_model(self, model_name: str) -> None:
    """Load analysis model. For model-free engines, just set model_loaded = True."""
    pass

def unload_model(self) -> None:
    """Unload model and free resources."""
    pass

def get_available_models(self) -> List[ModelInfo]:
    """Return list of models. For model-free engines, return a single "default" entry."""
    pass
```

## Example Implementation

```python
from base_quality_server import (
    BaseQualityServer,
    QualityThresholds,
    QualityField,
    QualityInfoBlockItem,
    AnalyzeResult
)
from base_server import ModelInfo

class MyAudioAnalyzer(BaseQualityServer):
    def __init__(self):
        super().__init__(
            engine_name="my_analyzer",
            display_name="My Audio Analyzer",
            engine_type="audio"  # Always "audio" for audio analysis engines
        )

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds
    ) -> AnalyzeResult:
        # Your analysis logic here
        metrics = self._compute_metrics(audio_bytes)

        fields = [
            QualityField(key="speechRatio", value=metrics["speech_ratio"], type="percent"),
            QualityField(key="maxSilence", value=metrics["max_silence"], type="number"),
            QualityField(key="peakDb", value=metrics["peak_db"], type="number"),
        ]

        # Check for issues
        info_blocks = {}
        issues = []

        if metrics["speech_ratio"] < thresholds.speech_ratio_warning_min:
            issues.append(QualityInfoBlockItem(
                text="audioIssue.lowSpeechRatio",
                severity="error"
            ))

        if issues:
            info_blocks["issues"] = issues

        # Calculate quality score
        quality_score = self._calculate_score(metrics, thresholds)

        return AnalyzeResult(
            quality_score=quality_score,
            fields=fields,
            info_blocks=info_blocks,
            top_label="audioQuality"
        )

    def load_model(self, model_name: str) -> None:
        # Model-free - nothing to load
        self.model_loaded = True
        self.current_model = model_name

    def unload_model(self) -> None:
        # Note: model_loaded and current_model are reset by base_server.py
        pass

    def get_available_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="default", display_name="Default Analyzer", languages=[], fields=[])]
```

## Configuration (engine.yaml)

```yaml
schema_version: 2

name: "my-analyzer"
display_name: "My Audio Analyzer"
engine_type: "audio"
description: "My custom audio analyzer"

upstream:
  name: "Original Project"
  url: "https://github.com/..."
  license: "MIT"

variants:
  - tag: "latest"
    platforms: ["linux/amd64"]
    requires_gpu: false

models:
  - name: "default"
    display_name: "Default Configuration"

default_model: "default"

capabilities:
  supports_model_hotswap: false
  supports_speaker_cloning: false
  supports_streaming: false

installation:
  python_version: "3.10"
  venv_path: "./venv"
  requires_gpu: false
```

## Directory Structure

```
my_analyzer/
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
venv\Scripts\python.exe server.py --port 8768

# Test health
curl http://localhost:8768/health

# Test models
curl http://localhost:8768/models

# Test load
curl -X POST http://localhost:8768/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName": "default"}'

# Test analyze (with base64 audio)
curl -X POST http://localhost:8768/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "audioBase64":"<base64-data>",
    "language":"en",
    "qualityThresholds":{"speechRatioIdealMin":75}
  }'
```

### Automated API Testing

Use the comprehensive test suite:

```bash
# Run full test suite
python scripts/test_engine.py --port 8768 --verbose

# Skip shutdown test during development
python scripts/test_engine.py --port 8768 --skip-shutdown
```

## Examples

See this working implementation:
- `audio_analysis/silero-vad/` - VAD-based speech/silence detection

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md) - Complete development guide
- [Engine Server API](../../docs/engine-server-api.md) - API endpoint documentation
- [Model Management](../../docs/model-management.md) - Model handling patterns
