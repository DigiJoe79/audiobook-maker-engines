# spaCy Text Processor

NLP-based text segmentation for TTS pipelines using spaCy sentence boundaries

## Overview

The spaCy Text Processor provides intelligent sentence segmentation for audiobook production workflows. It uses spaCy's trained NLP models to detect natural sentence boundaries and segment text into TTS-ready chunks while respecting character length constraints.

**Key Features:**
- Sentence-boundary aware segmentation (never splits sentences mid-way)
- Multi-language support with 11 language models
- Text sanitization for consistent TTS processing (Unicode normalization, HTML entity decoding, quote normalization)
- Oversized sentence detection and marking for manual review
- CPU-optimized with disabled unnecessary pipeline components (NER, lemmatizer)
- Model hotswap support for switching languages without restart

## Supported Languages

| Language | Code | Model |
|----------|------|-------|
| German | de | de_core_news_md |
| English | en | en_core_web_md |
| Spanish | es | es_core_news_md |
| French | fr | fr_core_news_md |
| Italian | it | it_core_news_md |
| Dutch | nl | nl_core_news_md |
| Polish | pl | pl_core_news_md |
| Portuguese | pt | pt_core_news_md |
| Russian | ru | ru_core_news_md |
| Chinese | zh | zh_core_web_md |
| Japanese | ja | ja_core_news_md |

All models use the MD (medium) tier for balanced speed and accuracy (~40 MB each).

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/spacy:latest
docker run -d -p 8770:8770 ghcr.io/digijoe79/audiobook-maker-engines/spacy:latest
```

### Subprocess (Development)

```bash
# Windows
cd text_processing/spacy
setup.bat

# Linux/Mac
cd text_processing/spacy
./setup.sh
```

The setup scripts will:
1. Create a Python 3.12 virtual environment
2. Install dependencies from requirements.txt
3. Download default models (de_core_news_md, en_core_web_md)

## API Endpoints

### Common Endpoints (from BaseEngineServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/load` | POST | Load a specific model |
| `/models` | GET | List available models |
| `/health` | GET | Health check with status and device info |
| `/info` | GET | Engine metadata from engine.yaml |
| `/shutdown` | POST | Graceful shutdown |

### Text Endpoints (from BaseTextServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/segment` | POST | Segment text into TTS-ready chunks |

---

## API Reference

### POST /load

Load a spaCy language model. Accepts either a language code (e.g., "de", "en") or a full model name (e.g., "de_core_news_md").

**Request:**
```json
{
  "engineModelName": "de"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "de_core_news_md"
}
```

### GET /models

List all installed spaCy models in the virtual environment.

**Response:**
```json
{
  "models": [
    {
      "name": "de_core_news_md",
      "displayName": "de_core_news_md",
      "languages": ["de"]
    },
    {
      "name": "en_core_web_md",
      "displayName": "en_core_web_md",
      "languages": ["en"]
    }
  ],
  "defaultModel": "en_core_web_md",
  "device": "cpu"
}
```

### GET /health

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "de_core_news_md",
  "device": "cpu",
  "packageVersion": "3.8.11"
}
```

### POST /segment (Text)

Segment text into sentences while respecting TTS engine character limits.

**Important Principle:** The segmenter NEVER splits sentences in the middle. This is critical for maintaining TTS naturalness. Instead, it:
1. Combines short sentences up to `maxLength` for better flow
2. Marks individual sentences exceeding `maxLength` as "failed" for manual review

**Request:**
```json
{
  "text": "This is the first sentence. This is a very long second sentence that exceeds the maximum character limit and will be marked as failed for manual review. This is the third sentence.",
  "language": "en",
  "maxLength": 80,
  "minLength": 10,
  "markOversized": true
}
```

**Response:**
```json
{
  "segments": [
    {
      "text": "This is the first sentence.",
      "start": 0,
      "end": 27,
      "orderIndex": 0,
      "status": "ok"
    },
    {
      "text": "This is a very long second sentence that exceeds the maximum character limit and will be marked as failed for manual review.",
      "start": 28,
      "end": 152,
      "orderIndex": 1,
      "status": "failed",
      "length": 124,
      "maxLength": 80,
      "issue": "sentence_too_long"
    },
    {
      "text": "This is the third sentence.",
      "start": 153,
      "end": 180,
      "orderIndex": 2,
      "status": "ok"
    }
  ],
  "totalSegments": 3,
  "totalCharacters": 180,
  "failedCount": 1
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | string | (required) | Text to segment |
| language | string | (required) | Language code (e.g., "en", "de") |
| maxLength | int | 250 | Maximum characters per segment |
| minLength | int | 10 | Minimum characters (merge short sentences) |
| markOversized | bool | true | Mark sentences > maxLength as "failed" |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| segments | array | List of text segments |
| totalSegments | int | Total number of segments |
| totalCharacters | int | Sum of all segment text lengths |
| failedCount | int | Number of segments with status="failed" |

**Segment Object:**

| Field | Type | Description |
|-------|------|-------------|
| text | string | Segment text content |
| start | int | Start position in original text |
| end | int | End position in original text |
| orderIndex | int | Segment order (0-based) |
| status | string | "ok" or "failed" |
| length | int | (optional) Actual length for failed segments |
| maxLength | int | (optional) Max allowed for failed segments |
| issue | string | (optional) Issue type (e.g., "sentence_too_long") |

---

## Configuration

Parameters from `engine.yaml`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mark_oversized | bool | true | Mark sentences exceeding max_length as "failed" for manual review |

**Engine Config:**

| Setting | Value | Description |
|---------|-------|-------------|
| device | cpu | CPU-only processing (CUDA has no benefit for text segmentation) |
| model_tier | md | Uses MD models for balanced speed/accuracy |

## Available Models

The engine includes two pre-installed models in the Docker image:

| Model | Language | Size | Description |
|-------|----------|------|-------------|
| de_core_news_md | German | ~40 MB | Trained on news text |
| en_core_web_md | English | ~40 MB | Trained on web text |

Additional models can be downloaded on-demand using `/load` endpoint or manually:

```bash
# Activate venv first
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows

# Download additional language models
python -m spacy download fr_core_news_md  # French
python -m spacy download es_core_news_md  # Spanish
python -m spacy download it_core_news_md  # Italian
python -m spacy download nl_core_news_md  # Dutch
python -m spacy download pl_core_news_md  # Polish
python -m spacy download pt_core_news_md  # Portuguese
python -m spacy download ru_core_news_md  # Russian
python -m spacy download zh_core_web_md   # Chinese
python -m spacy download ja_core_news_md  # Japanese
```

## Troubleshooting

### Model not found

**Symptom:** `/load` returns 400 error with "Unknown spaCy model" message

**Solution:**
1. Verify the model name follows spaCy naming pattern: `{lang}_core_{type}_{size}`
2. Install the model manually:
   ```bash
   # Activate venv first
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate.bat  # Windows

   # Download model
   python -m spacy download de_core_news_md
   ```
3. Check installed models: `python -m spacy info`

### Model loading is slow

**Symptom:** First `/segment` request takes several seconds

**Solution:** This is normal on first load. The model is loaded once and cached in memory. Subsequent requests are fast (~5-50ms depending on text length).

To preload a model on startup:
```bash
# Call /load immediately after server starts
curl -X POST http://localhost:8770/load -H "Content-Type: application/json" -d '{"engineModelName": "de"}'
```

### Text segmentation produces unexpected results

**Symptom:** Sentences are split in unusual places

**Possible Causes:**
1. **Wrong language model:** Ensure the language code matches the text language
2. **Markdown artifacts:** The sanitizer removes most formatting, but complex nested structures may cause issues
3. **Very short segments:** Try reducing `minLength` parameter

**Solution:**
1. Verify correct language: `/load` with appropriate language code
2. Check input text for unusual characters or formatting
3. Adjust `minLength` and `maxLength` parameters

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running:
   - Docker: `docker ps | grep spacy`
   - Subprocess: Check if Python process exists
2. Check port is correct (default: 8770)
3. Verify firewall allows connection to port 8770

### Oversized sentences marked as failed

**Symptom:** Many segments have `status: "failed"` in response

**Explanation:** This is intentional behavior. Sentences exceeding `maxLength` are marked as failed to alert you that they need manual review or editing.

**Solutions:**
1. **Increase maxLength:** If your TTS engine supports longer text, increase the limit
2. **Edit source text:** Break long sentences into shorter ones in the original document
3. **Disable marking:** Set `markOversized: false` to accept oversized segments as-is (not recommended for TTS quality)

## Performance Optimizations

The engine includes several performance optimizations:

1. **Disabled Pipeline Components:** Only keeps essential components for sentence segmentation (tok2vec, tagger, parser/senter)
2. **Senter Preference:** Uses `senter` (sentence recognizer) instead of full `parser` when available (~10x faster)
3. **Model Caching:** Models stay loaded in memory for fast subsequent requests
4. **CPU-Only:** No CUDA overhead (text segmentation doesn't benefit from GPU)

**Typical Performance:**

| Text Length | Segments | Processing Time |
|-------------|----------|-----------------|
| 1 KB | ~10 | ~5 ms |
| 10 KB | ~100 | ~50 ms |
| 100 KB | ~1000 | ~500 ms |

*Benchmarked on Intel i7-12700K, CPU-only*

## Dependencies

### Versioning Decisions (2025-12-04)

| Package | Version | Rationale |
|---------|---------|-----------|
| spacy | >=3.8.7,<4.0.0 | v3.8.11 is latest stable (Nov 2024). v4.x excluded for stability. v3.8.7+ adds Python 3.13 support and Cython 3 compatibility |
| fastapi | >=0.120.0 | Aligned with Whisper engine for consistency across project |
| uvicorn[standard] | >=0.34.0 | Aligned with Whisper engine. Standard variant includes performance dependencies |
| pydantic | >=2.10.0,<3.0.0 | v2 API stable. v3 would bring breaking changes |
| httpx | >=0.28.0 | Required for Docker HEALTHCHECK endpoint |
| loguru | >=0.7.0,<1.0.0 | Logging library with stable API |
| pyyaml | >=6.0.1 | 6.0.1 fixes Cython build issue |

### Upgrade Notes

**spaCy:**
- v3.8.8+ dropped Python 3.9 support (project requires 3.10+, OK)
- v3.8.7+ added Python 3.13 support
- v4.x expected in future - currently pinned to 3.x for stability
- Model compatibility: MD models are forward-compatible within major version

**Pydantic:**
- v2 API is stable and widely adopted
- v3 would require migration (breaking changes expected)
- Pin to <3.0.0 to avoid unexpected breaking changes

## Testing

```bash
python scripts/test_engine.py --port 8770 --verbose
```

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md)
- [Engine Server API](../../docs/engine-server-api.md)
- [Model Management Standard](../../docs/model-management.md)
- [Text Sanitization for TTS](../../docs/subprocess-development.md)
