# Audiobook Maker Engines

Docker images for [Audiobook Maker](https://github.com/DigiJoe79/audiobook-maker) TTS/STT/Text/Audio engines.

> **Note:** For issues, feature requests, and discussions please use the [main repository](https://github.com/DigiJoe79/audiobook-maker/issues). This repo only contains the engine Docker images.

## Available Engines

| Engine | Type | GPU | Platforms | Description |
|--------|------|-----|-----------|-------------|
| [debug-tts](tts/debug-tts/) | TTS | No | amd64, arm64 | Test engine - generates sine waves |
| [chatterbox](tts/chatterbox/) | TTS | Yes | amd64 | Chatterbox multilingual TTS with voice cloning |
| [vibevoice](tts/vibevoice/) | TTS | Yes | amd64 | Microsoft VibeVoice - expressive multi-speaker TTS |
| [xtts](tts/xtts/) | TTS | Yes | amd64 | Coqui XTTS v2 - multilingual TTS with voice cloning |
| [whisper](stt/whisper/) | STT | Yes | amd64 | OpenAI Whisper - speech recognition (GPU, `:latest`) |
| [whisper](stt/whisper/) | STT | No | amd64, arm64 | OpenAI Whisper - speech recognition (CPU, `:cpu`) |
| [silero-vad](audio_analysis/silero-vad/) | Audio | No | amd64, arm64 | Silero VAD - voice activity detection |
| [spacy](text_processing/spacy/) | Text | No | amd64, arm64 | spaCy NLP - text processing and sentence splitting |

## Usage

### Pull an image

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/debug-tts:latest
docker pull ghcr.io/digijoe79/audiobook-maker-engines/vibevoice:latest
```

### Run an engine

```bash
# CPU engine
docker run -d -p 8766:8766 ghcr.io/digijoe79/audiobook-maker-engines/debug-tts:latest

# GPU engine (requires NVIDIA Container Toolkit)
docker run -d --gpus all -p 8766:8766 \
  -v /path/to/samples:/app/samples \
  -v /path/to/models:/app/external_models \
  ghcr.io/digijoe79/audiobook-maker-engines/vibevoice:latest
```

## Online Catalog

The `catalog.yaml` file contains metadata about all available engines. It is automatically generated and attached to each release.

**Fetch latest catalog:**
```
https://github.com/DigiJoe79/audiobook-maker-engines/releases/latest/download/catalog.yaml
```

## CI/CD Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `build-single.yml` | Manual | Build and push a single engine image to GHCR |
| `release-catalog.yml` | Release | Generate and upload catalog.yaml |

**Recommended workflow:**
1. Build & test image locally
2. Push to GHCR via `build-single.yml`
3. Pull & verify from GHCR
4. Create release to publish catalog.yaml

## Local Development

### Subprocess Mode

Run engines as local subprocesses instead of Docker containers:

```bash
# Clone this repo into the main audiobook-maker backend
cd /path/to/audiobook-maker
git clone https://github.com/DigiJoe79/audiobook-maker-engines.git backend/engines

# Set up a virtual environment for an engine
cd backend/engines/tts/debug-tts
python -m venv venv
venv/Scripts/pip install -r requirements.txt  # Windows
# or: venv/bin/pip install -r requirements.txt  # Linux/Mac

# Run the engine
venv/Scripts/python server.py --port 8766
```

### Building Images Locally

```bash
# From repo root
docker build -t audiobook-maker/debug-tts:latest -f tts/debug-tts/Dockerfile .
docker build -t audiobook-maker/vibevoice:latest -f tts/vibevoice/Dockerfile .
```

## Adding a New Engine

See [CLAUDE.md](CLAUDE.md) for detailed instructions on adding new engines.

## License

MIT License - see [LICENSE](LICENSE)
