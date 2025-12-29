# Text Processing Engine Template

This is the template for creating new **Text Processing** engine servers.

Text processing engines segment text into chunks suitable for TTS generation, ensuring sentence boundaries are preserved and chunks don't exceed TTS engine limits.

## Quick Start

1. **Copy template to new directory:**
   ```bash
   cp -r text_processing/_template text_processing/my_processor
   cd text_processing/my_processor
   ```

2. **Customize the template:**
   - Rename class in `server.py` (e.g., `MyTextProcessor`)
   - Update `engine_name` and `display_name`
   - Implement `segment_text()` with your NLP logic
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
   venv\Scripts\python.exe server.py --port 8770
   ```

5. **Restart backend** - Engine will be auto-discovered!

## Architecture

Text processing engines inherit from `BaseTextServer` which extends `BaseEngineServer`:

```
BaseEngineServer (base_server.py)
├── /health - Health check
├── /load - Load model
├── /models - List available models
└── /shutdown - Graceful shutdown

BaseTextServer (base_text_server.py) extends BaseEngineServer
└── /segment - Segment text for TTS generation

Text Processing Engine (your server.py) extends BaseTextServer
└── Implements segment_text() method
```

## Required Endpoints

### Standard Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns status |
| `/load` | POST | Load a model (often language-specific) |
| `/models` | GET | List available models/languages |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### Text-Specific Endpoint (you implement)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/segment` | POST | Segment text into TTS-ready chunks |

## API Format

### POST /segment

**Request:**
```json
{
  "text": "This is a long text that needs to be segmented. It has multiple sentences. Some are short. Others might be quite long and need special handling.",
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
      "text": "This is a long text that needs to be segmented. It has multiple sentences.",
      "start": 0,
      "end": 74,
      "orderIndex": 0,
      "status": "ok"
    },
    {
      "text": "Some are short. Others might be quite long and need special handling.",
      "start": 75,
      "end": 144,
      "orderIndex": 1,
      "status": "ok"
    }
  ],
  "totalSegments": 2,
  "totalCharacters": 144,
  "failedCount": 0
}
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to segment |
| `language` | string | required | Language code (e.g., "en", "de") |
| `maxLength` | int | 250 | Maximum characters per segment |
| `minLength` | int | 10 | Minimum characters (merge short sentences) |
| `markOversized` | bool | true | Mark sentences > maxLength as "failed" |

### Response Schema

| Field | Type | Description |
|-------|------|-------------|
| `segments` | array | List of `SegmentItem` objects |
| `totalSegments` | int | Total number of segments |
| `totalCharacters` | int | Sum of all segment text lengths |
| `failedCount` | int | Number of segments with status="failed" |

### SegmentItem Schema

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Segment text content |
| `start` | int | Start position in original text |
| `end` | int | End position in original text |
| `orderIndex` | int | Segment order (0-based) |
| `status` | string | "ok" or "failed" |
| `length` | int? | Actual length (only for failed) |
| `maxLength` | int? | Max allowed (only for failed) |
| `issue` | string? | Issue type (e.g., "sentence_too_long") |

## Key Principles for TTS Segmentation

1. **NEVER split sentences in the middle** - This breaks TTS naturalness
2. **Combine short sentences** up to `maxLength` for better flow
3. **Mark oversized sentences** as "failed" for manual review
4. **Preserve sentence boundaries** - Use proper NLP sentence detection
5. **Sanitize text** - Normalize quotes, whitespace, and special characters

## Required Methods

Engines must implement these methods from BaseTextServer:

```python
def segment_text(
    self,
    text: str,
    language: str,
    max_length: int,
    min_length: int,
    mark_oversized: bool
) -> List[SegmentItem]:
    """
    Segment text into chunks for TTS generation.

    Args:
        text: Input text to segment
        language: Language code for NLP model selection
        max_length: Maximum characters per segment (TTS engine limit)
        min_length: Minimum characters (merge short sentences)
        mark_oversized: Mark sentences exceeding max_length as "failed"

    Returns:
        List of SegmentItem objects
    """
    pass

def load_model(self, model_name: str) -> None:
    """Load NLP model. Often language-specific (e.g., spaCy model)."""
    pass

def unload_model(self) -> None:
    """Unload model and free resources."""
    pass

def get_available_models(self) -> List[ModelInfo]:
    """Return list of available models/languages."""
    pass
```

## Example Implementation

```python
from base_text_server import BaseTextServer, SegmentItem
from base_server import ModelInfo

class MyTextProcessor(BaseTextServer):
    def __init__(self):
        super().__init__(
            engine_name="my_processor",
            display_name="My Text Processor"
        )
        self.nlp = None

    def segment_text(
        self,
        text: str,
        language: str,
        max_length: int,
        min_length: int,
        mark_oversized: bool
    ) -> List[SegmentItem]:
        # Your NLP-based segmentation logic here
        doc = self.nlp(text)

        segments = []
        for i, sent in enumerate(doc.sents):
            segments.append(SegmentItem(
                text=sent.text,
                start=sent.start_char,
                end=sent.end_char,
                order_index=i,
                status="ok" if len(sent.text) <= max_length else "failed"
            ))
        return segments

    def load_model(self, model_name: str) -> None:
        import spacy
        self.nlp = spacy.load(model_name)
        self.model_loaded = True
        self.current_model = model_name

    def unload_model(self) -> None:
        self.nlp = None
        # Note: model_loaded and current_model are reset by base_server.py

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(name="en_core_web_sm", display_name="English", languages=["en"], fields=[]),
            ModelInfo(name="de_core_news_sm", display_name="German", languages=["de"], fields=[]),
        ]
```

## Configuration (engine.yaml)

```yaml
schema_version: 2

name: "my-processor"
display_name: "My Text Processor"
engine_type: "text"
description: "My custom text processor"

upstream:
  name: "Original Project"
  url: "https://github.com/..."
  license: "MIT"

variants:
  - tag: "latest"
    platforms: ["linux/amd64"]
    requires_gpu: false

models:
  - name: "en_core_web_sm"
    display_name: "English"
  - name: "de_core_news_sm"
    display_name: "German"

default_model: "en_core_web_sm"

supported_languages:
  - en
  - de
  - es
  - fr

constraints:
  max_text_length: 1000000

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
my_processor/
├── server.py          # Engine implementation
├── models.py          # Pydantic request/response models
├── engine.yaml        # Configuration
├── requirements.txt   # Dependencies
├── setup.bat          # Windows setup script
├── setup.sh           # Linux/Mac setup script
├── models/            # Downloaded language models (if applicable)
│   └── .gitkeep
└── venv/              # Virtual environment (created by setup)
```

## Testing

```bash
# Start server
venv\Scripts\python.exe server.py --port 8770

# Test health
curl http://localhost:8770/health

# Test models
curl http://localhost:8770/models

# Load model
curl -X POST http://localhost:8770/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName": "en_core_web_sm"}'

# Test segment
curl -X POST http://localhost:8770/segment \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world. This is a test. It has multiple sentences.",
    "language": "en",
    "maxLength": 250,
    "minLength": 10,
    "markOversized": true
  }'
```

### Automated API Testing

Use the comprehensive test suite:

```bash
# Run full test suite
python scripts/test_engine.py --port 8770 --verbose

# Skip shutdown test during development
python scripts/test_engine.py --port 8770 --skip-shutdown
```

## Examples

See these working implementations:
- `text_processing/spacy/` - spaCy-based sentence segmentation with multi-language support

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md) - Complete development guide
- [Engine Server API](../../docs/engine-server-api.md) - API endpoint documentation
- [Model Management](../../docs/model-management.md) - Model handling patterns

## Common NLP Libraries

| Library | Pros | Cons |
|---------|------|------|
| **spaCy** | Fast, accurate, GPU support, many languages | Large model downloads |
| **NLTK** | Simple, well-documented | Slower, less accurate |
| **Stanza** | Very accurate, many languages | Slower than spaCy |
| **Custom regex** | Fast, no dependencies | Limited accuracy |
