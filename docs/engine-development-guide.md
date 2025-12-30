# Engine Development Guide

**How to add custom engines to Audiobook Maker**

## Table of Contents

1. [Overview](#overview)
2. [Architecture Concept](#architecture-concept)
3. [Quick Start](#quick-start)
4. [Implementation Guide](#implementation-guide)
5. [Configuration Reference](#configuration-reference)
6. [Testing Your Engine](#testing-your-engine)
7. [Subprocess Development](#subprocess-development)
8. [Best Practices](#best-practices)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Audiobook Maker v1.0.0+ features a **multi-engine architecture** with 4 engine types that allows developers to add custom engines without modifying the backend code. Each engine:

- Runs as a **separate FastAPI server** in its own process
- Has its own **isolated virtual environment** (no dependency conflicts)
- Is **auto-discovered** by the backend on startup
- Gets **automatic HTTP API** endpoints from base server classes
- Stays **"warm"** between requests (models loaded in memory)
- Can **crash independently** without affecting the backend
- Supports **enable/disable** with automatic **auto-stop** after inactivity

### Engine Types

| Type | Purpose | Base Class | Endpoint |
|------|---------|------------|----------|
| **TTS** | Text-to-Speech synthesis | `BaseTTSServer` | `/generate` |
| **STT** | Speech-to-Text transcription | `BaseQualityServer` | `/analyze` |
| **Text Processing** | Text segmentation (spaCy) | `BaseTextServer` | `/segment` |
| **Audio Analysis** | Audio quality analysis (VAD) | `BaseQualityServer` | `/analyze` |

### What You Need to Provide

Only **3-4 methods** and **1 config file**:

**For TTS Engines:**
```python
def load_model(self, model_name: str) -> None:
    """Load your TTS model into memory"""

def generate_audio(self, text, language, speaker_wav, parameters) -> bytes:
    """Generate audio and return WAV bytes"""

def unload_model(self) -> None:
    """Free resources"""

def get_available_models(self) -> List[ModelInfo]:
    """Return list of available models"""
```

**For STT/Audio Engines:**
```python
def load_model(self, model_name: str) -> None:
    """Load your analysis model into memory"""

def analyze_audio(
    self,
    audio_bytes: bytes,
    language: str,
    thresholds: QualityThresholds,
    expected_text: Optional[str] = None,
    pronunciation_rules: Optional[List[PronunciationRuleData]] = None
) -> AnalyzeResult:
    """Analyze audio and return quality metrics"""

def unload_model(self) -> None:
    """Free resources"""

def get_available_models(self) -> List[ModelInfo]:
    """Return list of available models"""
```

**For Text Processing Engines:**
```python
def load_model(self, model_name: str) -> None:
    """Load your NLP model into memory"""

def segment_text(
    self,
    text: str,
    language: str,
    max_length: int,
    min_length: int,
    mark_oversized: bool
) -> List[SegmentItem]:
    """Segment text into TTS-ready chunks"""

def unload_model(self) -> None:
    """Free resources"""

def get_available_models(self) -> List[ModelInfo]:
    """Return list of available models"""
```

Everything else (HTTP server, error handling, logging, health checks) is provided by the base server classes.

---

## Architecture Concept

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Backend (Main FastAPI, Port 8765)                              │
│  - Engine Discovery (scans engines by type)                     │
│  - 4 Engine Managers (TTS, STT, Text, Audio)                    │
│  - Workers (TTS Worker, Quality Worker)                         │
│  - Activity Tracking & Auto-Stop (5 min)                        │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ HTTP Requests (localhost or Docker network)
             │
    ┌────────┴────────┬─────────────┬─────────────┬─────────────┐
    │                 │             │             │             │
    ▼                 ▼             ▼             ▼             ▼
┌────────────┐  ┌─────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ TTS        │  │ TTS         │ │ STT      │ │ Text     │ │ Audio    │
│ XTTS       │  │ Chatterbox  │ │ Whisper  │ │ spaCy    │ │ Silero   │
│ Port 8766  │  │ Port 8767   │ │ Port 8770│ │ Port 8772│ │ Port 8774│
│ (Container)│  │ (Container) │ │(Container)│ │(Container)│ │(Container)│
└────────────┘  └─────────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Key Benefits:**

1. **Dependency Isolation** - XTTS needs PyTorch 2.5, Whisper needs 2.9? No problem!
2. **Version Conflicts** - Different Python versions, CUDA versions, etc.
3. **Crash Isolation** - Engine crash? Backend keeps running, just restart the engine
4. **Hot Swapping** - Switch between engines without backend restart
5. **Development** - Develop and test engines independently
6. **Auto-Stop** - Non-default engines stop after 5 minutes of inactivity

### Engine Lifecycle

```
1. Backend Startup
   ↓
2. Engine Discovery (from catalog or local)
   ↓
3. Parse engine.yaml
   ↓
4. Check enabled status (settings DB)
   ↓
5. Start engine (Docker container or subprocess)
   ↓
6. Health check (wait for /health to return 200)
   ↓
7. Register engine in EngineManager
   ↓
8. User selects engine in UI
   ↓
9. Backend sends /load request (load model)
   ↓
10. Backend sends /generate or /analyze requests
    ↓
11. Activity tracking (record last use timestamp)
    ↓
12. Auto-stop after 5 min inactivity (non-default engines)
    ↓
13. Backend sends /shutdown on exit (graceful shutdown)
```

### Engine Status Lifecycle

```
disabled → stopped → starting → running → stopping → stopped
    ↑                                                   │
    └───────────────────────────────────────────────────┘
```

- **disabled**: Engine is disabled in settings, won't start
- **stopped**: Engine is enabled but not running
- **starting**: Engine process is launching
- **running**: Engine is healthy and accepting requests
- **stopping**: Engine is shutting down gracefully

---

## Quick Start

### Step 1: Choose Engine Type

Decide which type of engine you're creating:
- **TTS**: Text-to-Speech synthesis → `tts/`
- **STT**: Speech-to-Text analysis → `stt/`
- **Text**: Text processing → `text_processing/`
- **Audio**: Audio analysis → `audio_analysis/`

### Step 2: Copy Template

```bash
cd {type}
cp -r _template my_engine
cd my_engine
```

### Step 3: Customize `engine.yaml`

```yaml
schema_version: 2

name: "my-engine"
display_name: "My Custom Engine"
engine_type: "tts"  # or "stt", "text", "audio"
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

constraints:
  max_text_length: 500

capabilities:
  supports_model_hotswap: true
  supports_speaker_cloning: true  # TTS only
  supports_streaming: false

parameters:
  speed:
    type: "float"
    label: "settings.tts.speed"
    description: "settings.tts.speedDesc"
    default: 1.0
    min: 0.5
    max: 2.0
    step: 0.1
```

### Step 4: Implement `server.py`

**For TTS Engine:**
```python
from pathlib import Path
from typing import Dict, Any, Union, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_tts_server import BaseTTSServer
from base_server import ModelInfo
from loguru import logger

class MyTTSServer(BaseTTSServer):
    """My Custom TTS Engine"""

    def __init__(self):
        super().__init__(
            engine_name="my-engine",
            display_name="My Custom TTS"
        )
        self.model = None
        self.default_model = "default"

    def load_model(self, model_name: str) -> None:
        """Load your TTS model"""
        model_path = self.models_dir / model_name
        # TODO: Load your model
        # self.model = YourTTSClass.from_pretrained(model_path)
        logger.info(f"[{self.engine_name}] Loaded model: {model_name}")

    def generate_audio(
        self,
        text: str,
        language: str,
        speaker_wav: Union[str, List[str]],
        parameters: Dict[str, Any]
    ) -> bytes:
        """Generate TTS audio"""
        # TODO: Generate audio with your model
        # audio_array = self.model.synthesize(text, language=language)

        # Convert to WAV bytes (example using wave module)
        import io
        import wave
        import numpy as np

        # Example: convert numpy array to WAV
        audio_int16 = (audio_array * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_int16.tobytes())
        return buffer.getvalue()

    def unload_model(self) -> None:
        """Free resources"""
        if self.model is not None:
            del self.model
            self.model = None
        # Note: GPU cleanup handled by base_server.py

    def get_available_models(self) -> List[ModelInfo]:
        """Return list of available models"""
        models = []
        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir() or model_dir.is_symlink():
                    models.append(ModelInfo(
                        name=model_dir.name,
                        display_name=model_dir.name.replace("_", " ").title(),
                        languages=["en", "de"],  # Set actual languages
                        fields=[]
                    ))
        return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = MyTTSServer()
    server.run(port=args.port, host=args.host)
```

**For STT/Audio Engine:**
```python
from pathlib import Path
from typing import List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_quality_server import (
    BaseQualityServer,
    QualityThresholds,
    QualityField,
    QualityInfoBlockItem,
    AnalyzeResult,
    PronunciationRuleData
)
from base_server import ModelInfo
from loguru import logger

class MySTTServer(BaseQualityServer):
    """My Custom STT Engine"""

    def __init__(self):
        super().__init__(
            engine_name="my-stt",
            display_name="My Custom STT",
            engine_type="stt"  # or "audio" for audio analysis
        )
        self.model = None
        self.default_model = "base"

    def load_model(self, model_name: str) -> None:
        """Load your STT model"""
        model_path = self.models_dir / model_name
        # TODO: Load your model
        # self.model = YourSTTClass.load(model_path)
        self.model_loaded = True
        self.current_model = model_name
        logger.info(f"[{self.engine_name}] Loaded model: {model_name}")

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds,
        expected_text: Optional[str] = None,
        pronunciation_rules: Optional[List[PronunciationRuleData]] = None
    ) -> AnalyzeResult:
        """Analyze audio and return quality metrics"""
        # TODO: Transcribe audio with your model
        # transcription = self.model.transcribe(audio_bytes, language)
        # confidence = self.model.get_confidence()

        transcription = "Transcribed text here"
        confidence = 85

        # Build quality fields
        fields = [
            QualityField(
                key="quality.stt.confidence",
                value=confidence,
                type="percent"
            ),
            QualityField(
                key="quality.stt.transcription",
                value=transcription,
                type="text"
            ),
        ]

        # Check for issues
        info_blocks = {}
        if confidence < 70:
            info_blocks["issues"] = [
                QualityInfoBlockItem(
                    text="quality.stt.lowConfidence",
                    severity="warning",
                    details={"confidence": confidence}
                )
            ]

        return AnalyzeResult(
            quality_score=confidence,
            fields=fields,
            info_blocks=info_blocks,
            top_label="quality.stt.myEngine"
        )

    def unload_model(self) -> None:
        """Free resources"""
        if self.model is not None:
            del self.model
            self.model = None
        # Note: GPU cleanup handled by base_server.py

    def get_available_models(self) -> List[ModelInfo]:
        """Return list of available models"""
        return [
            ModelInfo(name="base", display_name="Base Model", languages=["en", "de"], fields=[]),
            ModelInfo(name="large", display_name="Large Model", languages=["en", "de"], fields=[])
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = MySTTServer()
    server.run(port=args.port, host=args.host)
```

### Step 5: Create Dockerfile

```dockerfile
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY tts/my-engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy base server files
COPY base_server.py ./
COPY base_tts_server.py ./

# Copy engine files
COPY tts/my-engine/server.py ./
COPY tts/my-engine/engine.yaml ./

# Copy baked-in models (optional - requires .dockerignore exception!)
# COPY tts/my-engine/models/ ./models/

# Copy entrypoint and create directories
COPY scripts/docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh
RUN mkdir -p /app/models /app/external_models /app/samples

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8766

EXPOSE 8766

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["sh", "-c", "python server.py --port ${PORT:-8766} --host 0.0.0.0"]
```

### Step 6: Test Locally

```bash
# Build image
docker build -t audiobook-maker/my-engine:latest -f tts/my-engine/Dockerfile .

# Run container
docker run -d -p 8766:8766 audiobook-maker/my-engine:latest

# Test endpoints
curl http://localhost:8766/health
curl http://localhost:8766/models
curl http://localhost:8766/info
```

---

## Implementation Guide

### Base Server Classes

The engine system uses a hierarchy of base classes:

```
BaseEngineServer (Generic: /health, /load, /models, /info, /shutdown)
├── BaseTTSServer (TTS-specific, adds /generate, /samples/check, /samples/upload)
├── BaseQualityServer (STT + Audio, adds /analyze)
└── BaseTextServer (Text Processing, adds /segment)
```

### Required Methods by Engine Type

#### TTS Engines (inherit `BaseTTSServer`)

| Method | Purpose |
|--------|---------|
| `load_model(model_name)` | Load TTS model into memory |
| `generate_audio(text, language, speaker_wav, parameters)` | Synthesize text to WAV bytes |
| `unload_model()` | Free GPU/RAM resources |
| `get_available_models()` | Return list of `ModelInfo` objects |

#### STT/Audio Engines (inherit `BaseQualityServer`)

| Method | Purpose |
|--------|---------|
| `load_model(model_name)` | Load analysis model into memory |
| `analyze_audio(audio_bytes, language, thresholds, ...)` | Return `AnalyzeResult` |
| `unload_model()` | Free GPU/RAM resources |
| `get_available_models()` | Return list of `ModelInfo` objects |

#### Text Processing Engines (inherit `BaseTextServer`)

| Method | Purpose |
|--------|---------|
| `load_model(model_name)` | Load NLP model into memory |
| `segment_text(text, language, max_length, min_length, mark_oversized)` | Return `List[SegmentItem]` |
| `unload_model()` | Free resources |
| `get_available_models()` | Return list of `ModelInfo` objects |

### ModelInfo Structure

```python
from base_server import ModelInfo, ModelField

# ModelInfo is a Pydantic model (CamelCaseModel), not a dataclass
ModelInfo(
    name="model-id",           # Internal name (used in API calls)
    display_name="Model Name", # UI display name
    languages=["en", "de"],    # ISO language codes
    fields=[                   # Optional dynamic metadata
        ModelField(key="size_mb", value=500, field_type="number"),
        ModelField(key="speed", value="~10x realtime", field_type="string"),
    ]
)
```

### AnalyzeResult Structure (STT/Audio)

```python
from base_quality_server import AnalyzeResult, QualityField, QualityInfoBlockItem

# Note: quality_status is auto-calculated from score
# >= 85 = "perfect", >= 70 = "warning", < 70 = "defect"

AnalyzeResult(
    quality_score=85,      # 0-100 (int)
    fields=[               # List of QualityField
        QualityField(key="quality.stt.confidence", value=85, type="percent"),
        QualityField(key="quality.stt.transcription", value="Hello world", type="text"),
    ],
    info_blocks={          # Dict of issue lists
        "issues": [
            QualityInfoBlockItem(
                text="quality.stt.lowConfidence",
                severity="warning",  # "error", "warning", "info"
                details={"confidence": 65}
            )
        ]
    },
    top_label="quality.stt.myEngine"  # i18n key for section header
)
```

### SegmentItem Structure (Text Processing)

```python
from base_text_server import SegmentItem

SegmentItem(
    text="Segment text content.",
    start=0,           # Start position in original text
    end=25,            # End position in original text
    order_index=0,     # Segment order (0-based)
    status="ok",       # "ok" or "failed"
    # For failed segments only:
    length=350,        # Actual length
    max_length=250,    # Max allowed
    issue="sentence_too_long"
)
```

---

## Configuration Reference

### `engine.yaml` Structure

See [CLAUDE.md](../CLAUDE.md#engineyaml-schema) for the complete schema reference.

### Language Codes (ISO 639-1)

Common codes:
- `en` - English
- `de` - German (Deutsch)
- `fr` - French
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `nl` - Dutch
- `pl` - Polish
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `hi` - Hindi
- `tr` - Turkish

---

## Testing Your Engine

### 1. Local Docker Testing

```bash
# Build image
docker build -t audiobook-maker/my-engine:latest -f tts/my-engine/Dockerfile .

# Run container
docker run -d -p 8766:8766 --name my-engine audiobook-maker/my-engine:latest

# Test health check
curl http://localhost:8766/health
# Response: {"status":"ready","engineModelLoaded":false}

# Test available models
curl http://localhost:8766/models
# Response: [{"name":"default","displayName":"Default"}]

# Test engine info
curl http://localhost:8766/info
# Response: Full engine.yaml as JSON

# Test load model
curl -X POST http://localhost:8766/load \
  -H "Content-Type: application/json" \
  -d '{"engineModelName":"default"}'

# Test generate audio (TTS)
curl -X POST http://localhost:8766/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is a test.",
    "language": "en",
    "ttsSpeakerWav": "",
    "parameters": {}
  }' \
  --output test.wav

# Cleanup
docker stop my-engine && docker rm my-engine
```

### 2. Automated API Testing

Use the comprehensive test script to validate your engine against the API specification:

```bash
# Install test dependencies (if not already)
pip install httpx

# Run full test suite against your engine
python scripts/test_engine.py --port 8766

# Verbose output (shows all test details)
python scripts/test_engine.py --port 8766 --verbose

# Skip long-running tests during development
python scripts/test_engine.py --port 8766 --skip-robustness

# Skip shutdown test (keeps engine running)
python scripts/test_engine.py --port 8766 --skip-shutdown
```

**Test Phases:**

| Phase | Tests |
|-------|-------|
| 1. Discovery | `/health`, `/info`, `/models` |
| 2. Schema Validation | CamelCase keys, deep `/info` validation, GPU fields |
| 3. Model Loading | Non-existent model (expect error), default model |
| 4. Functional | Type-specific: `/generate`, `/analyze`, `/segment` |
| 5. Input Validation | Empty inputs, invalid formats (expect 4xx) |
| 6. Robustness | Hotswap, reload, large payloads, Unicode |
| 7. Shutdown | `POST /shutdown`, verify graceful stop |

**Exit Codes:**
- `0` - All tests passed (warnings ok)
- `1` - One or more tests failed
- `2` - Connection error (engine not reachable)

**Tip:** Run with `--skip-shutdown` during development to keep your engine running for manual testing.

### 3. Integration with Audiobook Maker

1. Add engine to online catalog or use custom Docker discovery
2. Install engine on a Docker host in the app
3. Enable engine in Settings
4. Test generation/analysis in the UI

---

## Examples

### Example 1: Cloud API Engine (OpenAI TTS)

For engines that call external APIs:

```python
import httpx
import os
from loguru import logger

class OpenAITTSServer(BaseTTSServer):
    def __init__(self):
        super().__init__(engine_name="openai-tts", display_name="OpenAI TTS")
        self.default_model = "tts-1"

    def load_model(self, model_name: str) -> None:
        self.current_model = model_name
        self.model_loaded = True
        logger.info(f"[{self.engine_name}] Using OpenAI model: {model_name}")

    def generate_audio(self, text, language, speaker_wav, parameters):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        response = httpx.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.current_model or "tts-1",
                "input": text,
                "voice": parameters.get("voice", "alloy")
            }
        )

        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.text}")

        # OpenAI returns MP3, convert to WAV
        return self._convert_mp3_to_wav(response.content)

    def unload_model(self) -> None:
        # Nothing to unload for API-based engines
        # Note: model_loaded and current_model are reset by base_server.py
        pass

    def get_available_models(self):
        return [
            ModelInfo(name="tts-1", display_name="TTS-1 (Fast)", languages=[], fields=[]),
            ModelInfo(name="tts-1-hd", display_name="TTS-1 HD (Quality)", languages=[], fields=[])
        ]
```

### Example 2: Multi-Speaker Engine (Piper)

For engines with built-in speaker voices:

```python
class PiperTTSServer(BaseTTSServer):
    def __init__(self):
        super().__init__(engine_name="piper", display_name="Piper TTS")
        self.default_model = "default"

    def load_model(self, model_name: str) -> None:
        model_path = self.models_dir / model_name
        self.model = PiperVoice.load(model_path)
        self.current_model = model_name
        self.model_loaded = True

    def generate_audio(self, text, language, speaker_wav, parameters):
        # Ignore speaker_wav (Piper uses built-in voices)
        speaker_id = parameters.get("speaker_id", 0)

        audio = self.model.synthesize(
            text=text,
            speaker_id=speaker_id,
            length_scale=parameters.get("speed", 1.0)
        )

        return self._to_wav_bytes(audio)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def get_available_models(self):
        return [
            ModelInfo(
                name=d.name,
                display_name=d.name.replace("-", " ").title(),
                languages=["en"],
                fields=[]
            )
            for d in self.models_dir.iterdir() if d.is_dir()
        ]
```

### Example 3: Audio Quality Analyzer

```python
from typing import List, Optional
from loguru import logger
from base_quality_server import (
    BaseQualityServer, QualityThresholds, QualityField,
    QualityInfoBlockItem, AnalyzeResult, PronunciationRuleData
)

class AudioQualityServer(BaseQualityServer):
    def __init__(self):
        super().__init__(
            engine_name="audio-quality",
            display_name="Audio Quality",
            engine_type="audio"
        )
        self.vad_model = None
        self.default_model = "default"

    def load_model(self, model_name: str) -> None:
        self.vad_model = load_silero_vad()
        self.model_loaded = True
        self.current_model = model_name
        logger.info(f"[{self.engine_name}] Loaded VAD model")

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds,
        expected_text: Optional[str] = None,
        pronunciation_rules: Optional[List[PronunciationRuleData]] = None
    ) -> AnalyzeResult:
        """Analyze audio quality using VAD"""
        import io
        import torchaudio

        # Load audio from bytes
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Get speech timestamps
        speech_timestamps = self.vad_model(audio, sr)

        # Calculate metrics
        total_duration = audio.shape[1] / sr
        speech_duration = sum(t['end'] - t['start'] for t in speech_timestamps)
        speech_ratio = (speech_duration / total_duration) * 100

        # Build quality fields
        fields = [
            QualityField(key="quality.audio.speechRatio", value=int(speech_ratio), type="percent"),
            QualityField(key="quality.audio.duration", value=f"{total_duration:.1f}s", type="string"),
        ]

        # Detect issues based on thresholds
        info_blocks = {}
        issues = []

        if speech_ratio < thresholds.speech_ratio_warning_min:
            issues.append(QualityInfoBlockItem(
                text="quality.audio.lowSpeechRatio",
                severity="warning",
                details={"speech_ratio": speech_ratio}
            ))

        if issues:
            info_blocks["issues"] = issues

        # Calculate quality score
        quality_score = min(100, int(speech_ratio + 20))

        return AnalyzeResult(
            quality_score=quality_score,
            fields=fields,
            info_blocks=info_blocks,
            top_label="quality.audio.analysis"
        )

    def unload_model(self) -> None:
        if self.vad_model is not None:
            del self.vad_model
            self.vad_model = None

    def get_available_models(self):
        return [
            ModelInfo(name="default", display_name="Silero VAD", languages=[], fields=[])
        ]
```

---

## Subprocess Development

> **Note:** Docker is the primary deployment method. Subprocess mode is only for engine development without rebuilding containers after every code change.

### Setup

1. Clone this repository into the audiobook-maker `engines/` directory:

   ```bash
   cd /path/to/audiobook-maker
   git clone https://github.com/user/audiobook-maker-engines engines
   ```

2. The backend auto-discovers engines in `engines/` subdirectories based on `engine.yaml` files.

3. Create a virtual environment and install dependencies:

   ```bash
   cd engines/tts/my-engine
   python -m venv venv

   # Windows
   venv\Scripts\pip install -r requirements.txt

   # Linux/Mac
   venv/bin/pip install -r requirements.txt
   ```

4. Start the engine server manually:

   ```bash
   # Windows
   venv\Scripts\python server.py --port 8766

   # Linux/Mac
   venv/bin/python server.py --port 8766
   ```

### Directory Structure

```
audiobook-maker/
├── backend/
├── frontend/
├── engines/                      # Clone audiobook-maker-engines here
│   ├── base_server.py
│   ├── base_tts_server.py
│   ├── tts/
│   │   ├── debug-tts/
│   │   │   ├── server.py
│   │   │   ├── engine.yaml
│   │   │   ├── requirements.txt
│   │   │   └── venv/            # Created by you
│   │   └── xtts/
│   └── stt/
│       └── whisper/
```

### When to Use Subprocess Mode

| Use Case | Recommended Mode |
|----------|------------------|
| Rapid iteration on `server.py` | Subprocess |
| Testing parameter changes | Subprocess |
| Debugging with breakpoints | Subprocess |
| Production deployment | Docker |
| GPU engines (XTTS, Whisper) | Docker |
| Sharing with others | Docker |

### Limitations

- **GPU engines**: Require matching CUDA/PyTorch versions on your system. Docker handles this automatically.
- **No isolation**: Dependencies may conflict with other Python projects.
- **Manual setup**: Each developer must create venv and install dependencies.

### Tips

- Use `--skip-shutdown` with `test_engine.py` to keep the engine running between test runs
- The backend detects subprocess engines by checking for `venv/` directory
- Logs appear directly in your terminal (no need to check Docker logs)

---

## Best Practices

### Memory Management

```python
def unload_model(self) -> None:
    if self.model is not None:
        del self.model
        self.model = None
    # Note: GPU cleanup, gc.collect(), and state reset
    # are handled automatically by base_server.py
```

### Error Handling

```python
from loguru import logger

def generate_audio(self, text, language, speaker_wav, parameters):
    # Note: Basic validation (model loaded, empty text, etc.)
    # is handled by base_tts_server.py

    try:
        audio = self.model.synthesize(text)
        return self._to_wav_bytes(audio)
    except Exception as e:
        logger.error(f"[{self.engine_name}] Generation failed: {e}")
        raise RuntimeError(f"Audio generation failed: {e}")
```

### Logging

```python
from loguru import logger

# Use loguru logger (imported at module level)
logger.info(f"[{self.engine_name}] Loading model: {model_name}")
logger.debug(f"[{self.engine_name}] Parameters: {parameters}")
logger.warning(f"[{self.engine_name}] Slow generation: {elapsed}s")
logger.error(f"[{self.engine_name}] Generation failed: {e}")
```

---

## Troubleshooting

### Container Fails to Start

**Check:**
1. Dockerfile syntax is correct
2. All COPY paths exist
3. Base images are available
4. Port is not already in use

**Debug:**
```bash
docker logs my-engine
docker run -it audiobook-maker/my-engine:latest /bin/bash
```

### Health Check Fails

**Check:**
1. Server starts without errors
2. `/health` endpoint returns 200
3. Port mapping is correct

**Debug:**
```bash
docker exec -it my-engine curl http://localhost:8766/health
```

### Generation Fails

**Check:**
1. Model is loaded (`/load` was called successfully)
2. Text length is within constraints
3. Language is supported
4. Speaker samples exist (if cloning)

**Debug:**
```python
# Add debug logging
logger.debug(f"[{self.engine_name}] Generating audio for: {text[:50]}...")
logger.debug(f"[{self.engine_name}] Language: {language}, Speaker: {speaker_wav}")
```

### Memory Issues

**Solutions:**
- Use smaller models
- Implement model offloading (CPU <-> GPU)
- Process text in smaller chunks
- Use appropriate Docker memory limits

---

## Related Documentation

- [engine-server-api.md](engine-server-api.md) - Detailed API endpoint documentation
- [model-management.md](model-management.md) - Model Management Standard
- [CLAUDE.md](../CLAUDE.md) - Quick reference for engine creation
- [scripts/test_engine.py](../scripts/test_engine.py) - Automated API test suite

---

**Happy Engine Building!**
