# Audiobook Maker Engines

Docker images for [Audiobook Maker](https://github.com/DigiJoe79/audiobook-maker) TTS/STT/Text/Audio engines.

## Available Engines

| Engine | Type | GPU | Platforms | Description |
|--------|------|-----|-----------|-------------|
| debug-tts | TTS | No | amd64, arm64 | Test engine - generates sine waves |

## Usage

### Pull an image

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/debug-tts:latest
```

### Run an engine

```bash
docker run -d -p 8766:8766 ghcr.io/digijoe79/audiobook-maker-engines/debug-tts:latest
```

## Online Catalog

The `catalog.json` file contains metadata about all available engines. It is automatically attached to each release.

**Fetch latest catalog:**
```
https://github.com/DigiJoe79/audiobook-maker-engines/releases/latest/download/catalog.json
```

## Local Development (subprocess mode)

If you want to run engines as local subprocesses instead of Docker containers:

```bash
# Clone this repo into the main audiobook-maker backend
cd /path/to/audiobook-maker
git clone https://github.com/DigiJoe79/audiobook-maker-engines.git backend/engines

# Set up a virtual environment for an engine
cd backend/engines/tts/debug-tts
python -m venv venv
venv/Scripts/pip install -r requirements.txt  # Windows
# or: venv/bin/pip install -r requirements.txt  # Linux/Mac
```

## Building Images Locally

```bash
# From repo root
docker build -t audiobook-maker/debug-tts:latest -f tts/debug-tts/Dockerfile .
```

## License

MIT License - see [LICENSE](LICENSE)
