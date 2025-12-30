# Silero VAD Audio Analysis

Audio quality analysis using Silero Voice Activity Detection.

## Overview

Silero-VAD analyzes audio recordings to detect potential quality issues using Voice Activity Detection. The engine measures speech ratio, silence duration, clipping, and volume levels to provide a comprehensive quality score for audiobook segments.

**Key Features:**
- Voice Activity Detection for accurate speech/silence ratio calculation
- Silence segment detection to identify gaps in narration
- Clipping detection for distorted audio
- Volume analysis (RMS and peak measurements)
- Language-agnostic analysis (no language-specific models needed)

## Installation

### Docker (Recommended)

```bash
docker pull ghcr.io/digijoe79/audiobook-maker-engines/silero-vad:latest
docker run -d -p 8768:8768 ghcr.io/digijoe79/audiobook-maker-engines/silero-vad:latest
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

### Quality Endpoints (from BaseQualityServer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze audio and return quality metrics |

---

## API Reference

### POST /load

Load the Silero VAD model into memory.

**Request:**
```json
{
  "engineModelName": "silero-vad"
}
```

**Response:**
```json
{
  "status": "loaded",
  "engineModelName": "silero-vad"
}
```

### GET /health

**Response:**
```json
{
  "status": "ready",
  "engineModelLoaded": true,
  "currentEngineModel": "silero-vad",
  "device": "cpu",
  "packageVersion": "6.2.0"
}
```

### POST /analyze (Quality)

Analyze audio and return quality metrics using the Generic Quality Format.

**Request:**
```json
{
  "audioBase64": "UklGRiQAAABXQVZFZm10...",
  "language": "en",
  "qualityThresholds": {
    "speechRatioIdealMin": 75,
    "speechRatioIdealMax": 90,
    "speechRatioWarningMin": 65,
    "speechRatioWarningMax": 93,
    "maxSilenceDurationWarning": 2500,
    "maxSilenceDurationCritical": 3750,
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
  "qualityScore": 85,
  "qualityStatus": "perfect",
  "details": {
    "topLabel": "audioQuality",
    "fields": [
      {"key": "speechRatio", "value": 82, "type": "percent"},
      {"key": "maxSilence", "value": 1500, "type": "number"},
      {"key": "peakVolume", "value": "-2.3 dB", "type": "string"},
      {"key": "avgVolume", "value": "-35.1 dB", "type": "string"}
    ],
    "infoBlocks": {
      "audioIssues": [
        {
          "text": "speechRatioBelowIdeal",
          "severity": "warning",
          "details": {
            "speechRatio": 82.0,
            "threshold": 75
          }
        }
      ]
    }
  }
}
```

---

## Quality Metrics

### Speech Ratio (0-100%)

Percentage of audio containing speech vs silence.

- **Ideal Range**: 75-90% (configurable)
- **Warning Range**: 65-93% (configurable)
- **Critical**: Outside warning range

**Interpretation:**
- Too low (<65%): Excessive silence, may indicate recording issues
- Ideal (75-90%): Natural speech with appropriate pauses
- Too high (>93%): Insufficient pauses, may sound rushed or unnatural

### Max Silence Duration (milliseconds)

Longest continuous silence segment detected in the audio.

- **Warning**: >2500ms (configurable)
- **Critical**: >3750ms (configurable)

**Interpretation:**
- Long silence segments may indicate recording errors, forgotten text, or technical issues
- Short silences are normal for breath marks and sentence breaks

### Clipping Detection (dB)

Peak amplitude exceeding threshold indicates distorted audio.

- **Critical**: Peak volume >0.0 dB (configurable)

**Interpretation:**
- Clipping occurs when audio exceeds the maximum recordable level
- Results in harsh distortion and should always be avoided

### Average Volume (dB)

RMS (Root Mean Square) volume level across the entire audio.

- **Warning**: <-40.0 dB (configurable)

**Interpretation:**
- Too low: Audio may be difficult to hear or require excessive amplification
- Normal range: -35 to -15 dB for narration

---

## Quality Score Calculation

The quality score starts at 100 and subtracts points based on detected issues:

### Speech Ratio Penalties
- **Within ideal range (75-90%)**: No penalty
- **Outside ideal but within warning range**: Up to 15 points penalty (warning status)
- **Outside warning range (<65% or >93%)**: 31+ points penalty (forces defect status)

### Silence Duration Penalties
- **Below warning threshold (<2500ms)**: No penalty
- **Between warning and critical (2500-3750ms)**: Up to 15 points penalty (warning status)
- **Above critical (>3750ms)**: 31+ points penalty (forces defect status)

### Clipping Penalty
- **Detected**: 31 points penalty (instant defect status)

### Volume Penalty
- **Below minimum**: Up to 10 points penalty

### Status Thresholds
- **perfect**: score >= 85
- **warning**: 70 <= score < 85
- **defect**: score < 70

**Example Calculation:**
- Base score: 100
- Speech ratio 68% (below ideal 75%, within warning 65%): -8 points
- Max silence 2200ms (below warning): 0 points
- No clipping: 0 points
- Average volume -38 dB (above minimum -40): 0 points
- **Final score: 92 (perfect)**

---

## Configuration

Parameters from `engine.yaml`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| speech_ratio_ideal_min | int | 75 | 0-100 | Minimum ideal speech ratio (%) |
| speech_ratio_ideal_max | int | 90 | 0-100 | Maximum ideal speech ratio (%) |
| speech_ratio_warning_min | int | 65 | 0-100 | Minimum acceptable speech ratio (%) |
| speech_ratio_warning_max | int | 93 | 0-100 | Maximum acceptable speech ratio (%) |
| max_silence_duration_warning | int | 2500 | 0-10000 | Warning threshold for silence (ms) |
| max_silence_duration_critical | int | 3750 | 0-10000 | Critical threshold for silence (ms) |
| max_clipping_peak | float | 0.0 | -10.0 to 3.0 | Maximum peak volume (dB) |
| min_average_volume | float | -40.0 | -60.0 to 0.0 | Minimum average volume (dB) |

---

## Troubleshooting

### Model not loading

**Symptom:** `/load` returns error or timeout

**Solution:**
1. Check silero-vad package is installed: `pip list | grep silero-vad`
2. Verify PyTorch is installed correctly
3. Check logs for import errors
4. Try manual load: `python -c "from silero_vad import load_silero_vad; load_silero_vad()"`

### Analysis returns incorrect metrics

**Symptom:** Speech ratio is 0% or 100% on normal audio

**Solution:**
1. Verify audio is WAV format (RIFF header)
2. Check audio is not corrupted: play file manually
3. Verify sample rate (engine auto-resamples to 16kHz)
4. Check audio contains actual speech (not silence or noise)

### Slow processing

**Symptom:** Analysis takes longer than expected

**Solution:**
1. VAD model runs on CPU by default (sufficient for most use cases)
2. For faster processing, ensure PyTorch is compiled with optimizations
3. Long audio files (>5 minutes) naturally take longer
4. Check system is not under heavy load

### Connection refused

**Symptom:** Cannot connect to engine endpoint

**Solution:**
1. Verify engine is running: `docker ps` or check process
2. Check port 8768 is correct and not blocked
3. Verify firewall allows connection
4. Check Docker port mapping: `-p 8768:8768`

---

## Dependencies

### Versioning Decisions (2025-12-04)

| Package | Version | Rationale |
|---------|---------|-----------|
| silero-vad | >=6.2.0 | Pip package with model included. Exposes `__version__` for UI display. Version 6.x line is stable. |
| torch | >=2.9.0,<3.0.0 | Silero-VAD v6.2 confirmed PyTorch 2.9 compatibility. Upper bound prevents breaking changes. |
| torchaudio | >=2.9.0,<3.0.0 | Must match torch version for API compatibility. |
| scipy | >=1.14.0 | Python 3.12 allows latest. Used for WAV I/O and resampling. |
| numpy | >=2.0.0 | NumPy 2.x for Python 3.12. Breaking changes from 1.x are minimal for this use case. |
| fastapi | >=0.123.0 | Aligned with backend core. Latest stable line. |
| pydantic | >=2.10.0,<3.0.0 | Pydantic 2.x line. Upper bound prevents v3 breaking changes. |
| loguru | >=0.7.0,<1.0.0 | Stable logging library. No breaking changes expected in 0.x line. |

### Upgrade Notes

**Python Version:**
- Engine targets Python 3.12 (specified in engine.yaml)
- Each engine has isolated venv, no cross-engine Python constraints

**Silero-VAD Package:**
- Uses pip package instead of torch.hub (cleaner dependency management)
- Version displayed in UI is read from `silero_vad.__version__`
- Model included in package (no separate download on first use)
- Upgrading from 5.x to 6.x: API compatible, no code changes needed

**PyTorch:**
- PyTorch 2.9+ supports CUDA 12.x
- Silero-VAD runs on CPU (GPU not required for VAD inference)
- If GPU available, PyTorch will use it automatically (minimal benefit for VAD)

**SciPy WAV Warnings:**
- Engine suppresses scipy WAV chunk warnings (harmless metadata chunks)
- These warnings appear when reading WAV files with extra metadata chunks
- Suppressed via `warnings.filterwarnings("ignore", category=WavFileWarning)`

---

## Testing

```bash
python scripts/test_engine.py --port 8768 --verbose
```

## Documentation

- [Engine Development Guide](../../docs/engine-development-guide.md)
- [Engine Server API](../../docs/engine-server-api.md)
- [Model Management](../../docs/model-management.md)
