# TTS Engine Template

This is the template for creating new **Text-to-Speech (TTS)** engine servers.

## Quick Start

1. **Copy template to new directory:**
   ```bash
   cp -r tts/_template tts/my_engine
   cd tts/my_engine
   ```

2. **Customize the template:**
   - Rename class in `server.py` (e.g., `MyTTSServer`)
   - Update `engine_name` and `display_name`
   - Implement `load_model()`, `generate_audio()`, `unload_model()`, `get_available_models()`
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
   # Windows
   venv\Scripts\python.exe server.py --port 8766

   # Linux/Mac
   venv/bin/python server.py --port 8766
   ```

5. **Restart backend** - Engine will be auto-discovered!

## Architecture

TTS engines inherit from `BaseTTSServer` which extends `BaseEngineServer`:

```
BaseEngineServer (base_server.py)
├── /health - Health check
├── /load - Load model
├── /models - List available models
└── /shutdown - Graceful shutdown

BaseTTSServer (base_tts_server.py) extends BaseEngineServer
└── /generate - Generate TTS audio (WAV bytes)
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

### TTS-Specific Endpoints (from BaseTTSServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate TTS audio, returns WAV bytes |
| `/samples/check` | POST | Check which speaker samples exist |
| `/samples/upload/{sample_id}` | POST | Upload speaker sample WAV |

## Required Methods

Engines must implement these 4 methods:

```python
def load_model(self, model_name: str) -> None:
    """Load model into memory."""
    pass

def generate_audio(
    self,
    text: str,
    language: str,
    speaker_wav: Union[str, List[str]],
    parameters: Dict[str, Any]
) -> bytes:
    """Generate TTS audio, return WAV bytes."""
    pass

def unload_model(self) -> None:
    """Unload model and free resources."""
    pass

def get_available_models(self) -> List[ModelInfo]:
    """Return list of available models with metadata."""
    pass
```

## API Request/Response Formats

### POST /load

**Request:**
```json
{
  "engineModelName": "v2.0.3"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "v2.0.3"
}
```

### POST /generate

**Request:**
```json
{
  "text": "Hello world",
  "language": "en",
  "ttsSpeakerWav": "/path/to/speaker.wav",
  "parameters": {
    "speed": 1.0,
    "temperature": 0.7
  }
}
```

**Response:** Binary WAV audio data (Content-Type: audio/wav)

### GET /models

**Response:**
```json
{
  "models": [
    {
      "name": "v2.0.3",
      "displayName": "XTTS v2.0.3",
      "languages": ["en", "de", "fr"],
      "fields": [
        {"key": "size_mb", "value": 1800, "fieldType": "number"}
      ]
    }
  ],
  "defaultModel": "v2.0.3",
  "device": "cuda"
}
```

### GET /health

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "v2.0.3",
  "device": "cuda",
  "packageVersion": "1.0.0"
}
```

## Configuration (engine.yaml)

```yaml
schema_version: 2

name: "my-engine"
display_name: "My TTS Engine"
engine_type: "tts"
description: "My custom TTS engine"

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
    display_name: "Default Model"

default_model: "default"

supported_languages:
  - en
  - de
  - fr

constraints:
  max_text_length: 500

capabilities:
  supports_model_hotswap: true
  supports_speaker_cloning: true
  supports_streaming: false

installation:
  python_version: "3.10"
  venv_path: "./venv"
  requires_gpu: false
```

## Directory Structure

```
my_engine/
├── server.py          # Engine implementation (inherits from BaseTTSServer)
├── engine.yaml        # Configuration
├── requirements.txt   # Dependencies
├── setup.bat          # Windows setup script
├── setup.sh           # Linux/Mac setup script
├── models/            # Model files (optional)
│   └── .gitkeep
└── venv/              # Virtual environment (created by setup)
```

## Tips

- **Error Handling**: Raise exceptions - BaseTTSServer handles HTTP errors
- **Logging**: Use `logger.info()`, `logger.error()` - already imported from loguru
- **State**: Store model in `self.model`, use `self.current_model` for loaded model name
- **Audio Format**: Return WAV bytes (use helper `_convert_to_wav_bytes()` in template)
- **Speaker Samples**: `speaker_wav` can be a string (single file) or list (multiple files)

## Testing

Test your engine standalone before integrating:

```bash
# Start server
venv\Scripts\python.exe server.py --port 8766

# Test health check
curl http://localhost:8766/health

# Test models list
curl http://localhost:8766/models

# Test load
curl -X POST http://localhost:8766/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName":"default"}'

# Test generate
curl -X POST http://localhost:8766/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en","ttsSpeakerWav":"","parameters":{}}' \
  --output test.wav
```

### Automated API Testing

Use the comprehensive test suite:

```bash
# Run full test suite
python scripts/test_engine.py --port 8766 --verbose

# Skip shutdown test during development
python scripts/test_engine.py --port 8766 --skip-shutdown
```

## Examples

See `tts/xtts/` and `tts/chatterbox/` for complete working examples.

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md) - Complete development guide
- [Engine Server API](../../docs/engine-server-api.md) - API endpoint documentation
- [Model Management](../../docs/model-management.md) - Model handling patterns
