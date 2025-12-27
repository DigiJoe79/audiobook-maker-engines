#!/usr/bin/env python3
"""
Engine Functional Test Script

Comprehensive API testing for all engine types (TTS, STT, Text, Audio).
Tests are based on the documented API specifications in:
- docs/ENGINE_SYSTEM_ARCHITECTURE.md (API endpoints)
- base_server.py, base_tts_server.py, base_quality_server.py, base_text_server.py (Response models)

Test Phases:
    1. Discovery       - /health, /info, /models
    2. Schema Valid.   - CamelCase, deep /info, model structure, GPU fields
    3. Model Loading   - Non-existent model (expect error), then default model
    4. Functional      - Type-specific: /generate, /analyze, /segment, /samples
    5. Input Valid.    - Empty inputs, invalid formats (expect 4xx)
    6. Robustness      - Hotswap, reload, large payloads, Unicode
    7. Shutdown        - POST /shutdown, verify engine stops

Usage:
    python scripts/test_engine.py --port 8766
    python scripts/test_engine.py --port 8766 --verbose
    python scripts/test_engine.py --port 8766 --skip-functional
    python scripts/test_engine.py --port 8766 --skip-robustness
    python scripts/test_engine.py --port 8766 --skip-shutdown

Exit codes:
    0 - All tests passed (warnings ok)
    1 - One or more tests failed
    2 - Connection error (engine not reachable)

Note: Uses ASCII-only output for Windows console compatibility.
"""

import argparse
import base64
import json
import math
import os
import struct
import sys
import time
import wave
import io
from dataclasses import dataclass, field
from typing import Any, Optional


# =============================================================================
# Color Support
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    @classmethod
    def enabled(cls) -> bool:
        """Check if colors should be enabled."""
        # Disable colors if NO_COLOR env var is set or not a TTY
        if os.environ.get("NO_COLOR"):
            return False
        if not sys.stdout.isatty():
            return False
        # Enable ANSI colors on Windows 10+
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass
        return True


def colorize(text: str, color: str) -> str:
    """Apply color to text if colors are enabled."""
    if Colors.enabled():
        return f"{color}{text}{Colors.RESET}"
    return text

try:
    import httpx
except ImportError:
    print("[FAIL] httpx package not installed")
    print("       Install with: pip install httpx")
    sys.exit(1)


# =============================================================================
# Constants from API Specification
# =============================================================================

DEFAULT_TIMEOUT = 30.0
GENERATE_TIMEOUT = 120.0

# Expected response fields per endpoint (from base_server.py models)
# Using camelCase as that's how CamelCaseModel serializes

HEALTH_REQUIRED_FIELDS = ["status", "engineModelLoaded", "device"]
HEALTH_OPTIONAL_FIELDS = ["currentEngineModel", "error", "packageVersion",
                          "gpuMemoryUsedMb", "gpuMemoryTotalMb"]
HEALTH_STATUS_VALUES = ["ready", "loading", "processing", "error"]

MODELS_REQUIRED_FIELDS = ["models", "device"]
MODELS_OPTIONAL_FIELDS = ["defaultModel"]

INFO_REQUIRED_FIELDS = ["name", "displayName", "engineType"]
INFO_ENGINE_TYPES = ["tts", "stt", "text", "audio"]

LOAD_REQUIRED_FIELDS = ["status"]
LOAD_STATUS_VALUES = ["loaded", "error"]

# Quality (STT/Audio) response fields
ANALYZE_REQUIRED_FIELDS = ["engineType", "engineName", "qualityScore", "qualityStatus", "details"]
ANALYZE_STATUS_VALUES = ["perfect", "warning", "defect"]

# Text response fields
SEGMENT_REQUIRED_FIELDS = ["segments", "totalSegments", "totalCharacters"]


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str = ""
    details: Optional[dict] = None
    duration_ms: float = 0.0
    skipped: bool = False
    warning: Optional[str] = None  # Warning message (test passed but with caveat)


@dataclass
class TestSuite:
    """Collection of test results."""
    engine_name: str = ""
    engine_type: str = ""
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed and not r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)

    @property
    def warnings(self) -> list[str]:
        return [r.warning for r in self.results if r.warning]

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Create a simple WAV file with a sine wave tone for testing."""
    frequency = 440.0
    amplitude = 0.5
    num_samples = int(duration_sec * sample_rate)

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        sample = int(value * 32767)
        samples.append(struct.pack('<h', sample))

    audio_data = b''.join(samples)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    return buffer.getvalue()


def format_duration(ms: float) -> str:
    """Format duration in milliseconds."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.2f}s"


def check_required_fields(data: dict, required: list[str]) -> list[str]:
    """Check for missing required fields."""
    return [f for f in required if f not in data]


def check_enum_value(value: Any, allowed: list[str], field_name: str) -> Optional[str]:
    """Check if value is in allowed list."""
    if value not in allowed:
        return f"{field_name}='{value}' not in {allowed}"
    return None


def validate_camelcase_keys(data: dict, path: str = "") -> list[str]:
    """
    Check that all top-level keys are camelCase (no underscores).
    Returns list of invalid keys found.
    """
    errors = []
    for key in data.keys():
        if "_" in key:
            full_path = f"{path}.{key}" if path else key
            errors.append(full_path)
    return errors


def generate_text_of_length(length: int) -> str:
    """Generate a text string of exactly the specified length."""
    base = "This is test text for length validation. "
    repetitions = (length // len(base)) + 1
    return (base * repetitions)[:length]


# Unicode test text with various character sets
# Note: This text is sent to engines, not printed to console
UNICODE_TEST_TEXT = (
    u"German: \u00e4\u00f6\u00fc\u00df \u00c4\u00d6\u00dc. "  # aoeuess AOEU
    u"French: \u00e9\u00e8\u00ea\u00eb \u00e0\u00e2 \u00e7. "  # accented chars
    u"Quotes: \u201eHallo\u201c \u00abBonjour\u00bb. "  # German/French quotes
    u"Emoji: \U0001f3b5 \U0001f50a \U0001f3a7. "  # music, speaker, headphones
    u"CJK: \u4f60\u597d\u4e16\u754c. "  # Ni Hao Shi Jie
    u"Arabic: \u0645\u0631\u062d\u0628\u0627. "  # Marhaba
    u"Hebrew: \u05e9\u05dc\u05d5\u05dd."  # Shalom
)


# =============================================================================
# API Client
# =============================================================================

class EngineClient:
    """HTTP client for engine API."""

    def __init__(self, host: str, port: int, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def close(self):
        self.client.close()

    def get(self, path: str) -> httpx.Response:
        return self.client.get(f"{self.base_url}{path}")

    def post_json(self, path: str, data: dict, timeout: Optional[float] = None) -> httpx.Response:
        return self.client.post(
            f"{self.base_url}{path}",
            json=data,
            timeout=timeout or self.timeout
        )

    def post_bytes(self, path: str, data: bytes, timeout: Optional[float] = None) -> httpx.Response:
        return self.client.post(
            f"{self.base_url}{path}",
            content=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=timeout or self.timeout
        )


# =============================================================================
# Test Functions - Common Endpoints (from base_server.py)
# =============================================================================

def test_health(client: EngineClient) -> TestResult:
    """
    Test GET /health endpoint.

    Expected response (HealthResponse from base_server.py):
        status: str ("ready", "loading", "processing", "error")
        engineModelLoaded: bool
        currentEngineModel: Optional[str]
        device: str ("cpu" or "cuda")
        error: Optional[str]
        packageVersion: Optional[str]
        gpuMemoryUsedMb: Optional[int]
        gpuMemoryTotalMb: Optional[int]
    """
    start = time.time()
    try:
        resp = client.get("/health")
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="GET /health",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()

        # Check required fields
        missing = check_required_fields(data, HEALTH_REQUIRED_FIELDS)
        if missing:
            return TestResult(
                name="GET /health",
                passed=False,
                message=f"Missing required fields: {missing}",
                details=data,
                duration_ms=duration
            )

        # Validate status enum
        error = check_enum_value(data["status"], HEALTH_STATUS_VALUES, "status")
        if error:
            return TestResult(
                name="GET /health",
                passed=False,
                message=error,
                details=data,
                duration_ms=duration
            )

        # Validate device
        if data["device"] not in ["cpu", "cuda"]:
            return TestResult(
                name="GET /health",
                passed=False,
                message=f"device='{data['device']}' not in ['cpu', 'cuda']",
                details=data,
                duration_ms=duration
            )

        model = data.get("currentEngineModel", "none")
        loaded = data["engineModelLoaded"]
        return TestResult(
            name="GET /health",
            passed=True,
            message=f"status={data['status']}, modelLoaded={loaded}, model={model}",
            details=data,
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="GET /health",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_info(client: EngineClient) -> TestResult:
    """
    Test GET /info endpoint.

    Expected response (EngineInfoResponse from base_server.py):
        name: str
        displayName: str
        engineType: str ("tts", "stt", "text", "audio")
        description: Optional[str]
        upstream: Optional[UpstreamInfo]
        ... additional fields
    """
    start = time.time()
    try:
        resp = client.get("/info")
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="GET /info",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()

        # Check required fields
        missing = check_required_fields(data, INFO_REQUIRED_FIELDS)
        if missing:
            return TestResult(
                name="GET /info",
                passed=False,
                message=f"Missing required fields: {missing}",
                details=data,
                duration_ms=duration
            )

        # Validate engineType enum
        error = check_enum_value(data["engineType"], INFO_ENGINE_TYPES, "engineType")
        if error:
            return TestResult(
                name="GET /info",
                passed=False,
                message=error,
                details=data,
                duration_ms=duration
            )

        return TestResult(
            name="GET /info",
            passed=True,
            message=f"name={data['name']}, type={data['engineType']}",
            details=data,
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="GET /info",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_models(client: EngineClient) -> TestResult:
    """
    Test GET /models endpoint.

    Expected response (ModelsResponse from base_server.py):
        models: List[ModelInfo]
        defaultModel: Optional[str]
        device: str
    """
    start = time.time()
    try:
        resp = client.get("/models")
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="GET /models",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()

        # Check required fields
        missing = check_required_fields(data, MODELS_REQUIRED_FIELDS)
        if missing:
            return TestResult(
                name="GET /models",
                passed=False,
                message=f"Missing required fields: {missing}",
                details=data,
                duration_ms=duration
            )

        # Validate models is a list
        if not isinstance(data["models"], list):
            return TestResult(
                name="GET /models",
                passed=False,
                message=f"'models' should be list, got {type(data['models']).__name__}",
                details=data,
                duration_ms=duration
            )

        models = data["models"]
        model_names = [m.get("name", "?") for m in models]
        default = data.get("defaultModel", "none")

        return TestResult(
            name="GET /models",
            passed=True,
            message=f"Found {len(models)} model(s), default={default}",
            details=data,
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="GET /models",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_load_model(client: EngineClient, model_name: str) -> TestResult:
    """
    Test POST /load endpoint.

    Request (LoadRequest from base_server.py):
        engineModelName: str

    Expected response (LoadResponse from base_server.py):
        status: str ("loaded", "error")
        engineModelName: Optional[str]
        error: Optional[str]
    """
    start = time.time()
    try:
        # Request field is engine_model_name (snake_case) -> engineModelName (camelCase)
        resp = client.post_json("/load", {"engine_model_name": model_name})
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name=f"POST /load ({model_name})",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()

        # Check required fields
        missing = check_required_fields(data, LOAD_REQUIRED_FIELDS)
        if missing:
            return TestResult(
                name=f"POST /load ({model_name})",
                passed=False,
                message=f"Missing required fields: {missing}",
                details=data,
                duration_ms=duration
            )

        # Validate status
        error = check_enum_value(data["status"], LOAD_STATUS_VALUES, "status")
        if error:
            return TestResult(
                name=f"POST /load ({model_name})",
                passed=False,
                message=error,
                details=data,
                duration_ms=duration
            )

        if data["status"] == "error":
            return TestResult(
                name=f"POST /load ({model_name})",
                passed=False,
                message=f"Load failed: {data.get('error', 'unknown')}",
                details=data,
                duration_ms=duration
            )

        return TestResult(
            name=f"POST /load ({model_name})",
            passed=True,
            message=f"Model loaded in {format_duration(duration)}",
            details=data,
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name=f"POST /load ({model_name})",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - TTS Endpoints (from base_tts_server.py)
# =============================================================================

def test_tts_generate(client: EngineClient, text: str, language: str,
                      speaker_sample: Optional[str] = None) -> TestResult:
    """
    Test POST /generate endpoint.

    Request (GenerateRequest from base_tts_server.py):
        text: str
        language: str
        ttsSpeakerWav: Union[str, List[str]]
        parameters: Optional[Dict]

    Expected response: Audio bytes (WAV) with Content-Type audio/*

    Args:
        client: Engine client
        text: Text to synthesize
        language: Language code
        speaker_sample: Optional speaker sample ID (for voice cloning engines)
    """
    start = time.time()
    try:
        # Use speaker sample if provided (for voice cloning engines)
        speaker_wav = [speaker_sample] if speaker_sample else []

        payload = {
            "text": text,
            "language": language,
            "tts_speaker_wav": speaker_wav,
            "parameters": {}
        }

        resp = client.post_json("/generate", payload, timeout=GENERATE_TIMEOUT)
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="POST /generate",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        # Check content type
        content_type = resp.headers.get("content-type", "")
        if "audio" not in content_type and "octet-stream" not in content_type:
            return TestResult(
                name="POST /generate",
                passed=False,
                message=f"Expected audio content-type, got: {content_type}",
                duration_ms=duration
            )

        # Check audio size
        audio_size = len(resp.content)
        if audio_size < 100:
            return TestResult(
                name="POST /generate",
                passed=False,
                message=f"Audio too small ({audio_size} bytes)",
                duration_ms=duration
            )

        return TestResult(
            name="POST /generate",
            passed=True,
            message=f"Generated {audio_size:,} bytes in {format_duration(duration)}",
            details={"size_bytes": audio_size},
            duration_ms=duration
        )

    except httpx.ReadTimeout:
        return TestResult(
            name="POST /generate",
            passed=False,
            message=f"Timeout after {GENERATE_TIMEOUT}s",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="POST /generate",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_tts_samples_check(client: EngineClient) -> TestResult:
    """
    Test POST /samples/check endpoint.

    Request (SampleCheckRequest from base_tts_server.py):
        sampleIds: List[str]

    Expected response (SampleCheckResponse):
        missing: List[str]
    """
    start = time.time()
    try:
        payload = {"sample_ids": ["nonexistent-sample-id"]}
        resp = client.post_json("/samples/check", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="POST /samples/check",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()

        if "missing" not in data:
            return TestResult(
                name="POST /samples/check",
                passed=False,
                message="Missing 'missing' field in response",
                details=data,
                duration_ms=duration
            )

        return TestResult(
            name="POST /samples/check",
            passed=True,
            message=f"Returned {len(data['missing'])} missing sample(s)",
            details=data,
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /samples/check",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_tts_generate_empty_text(client: EngineClient) -> TestResult:
    """Test that POST /generate rejects empty text with HTTP 400."""
    start = time.time()
    try:
        payload = {"text": "", "language": "en", "tts_speaker_wav": [], "parameters": {}}
        resp = client.post_json("/generate", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="POST /generate (empty text)",
                passed=True,
                message="Correctly rejected with HTTP 400",
                duration_ms=duration
            )

        return TestResult(
            name="POST /generate (empty text)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /generate (empty text)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_tts_generate_no_speaker(client: EngineClient, language: str) -> TestResult:
    """
    Test that speaker-cloning TTS engines reject requests without speaker samples.
    Expected: HTTP 400 (not 500)
    """
    start = time.time()
    try:
        payload = {
            "text": "Test without speaker sample",
            "language": language,
            "tts_speaker_wav": [],  # Empty - should be rejected
            "parameters": {}
        }
        resp = client.post_json("/generate", payload, timeout=GENERATE_TIMEOUT)
        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="POST /generate (no speaker)",
                passed=True,
                message="Correctly rejected with HTTP 400",
                duration_ms=duration
            )

        if resp.status_code == 500:
            return TestResult(
                name="POST /generate (no speaker)",
                passed=False,
                message=f"Got HTTP 500 (crash) instead of 400 validation error: {resp.text[:100]}",
                duration_ms=duration
            )

        return TestResult(
            name="POST /generate (no speaker)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.ReadTimeout:
        return TestResult(
            name="POST /generate (no speaker)",
            passed=False,
            message=f"Timeout - engine may be stuck",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="POST /generate (no speaker)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_tts_generate_invalid_language(client: EngineClient) -> TestResult:
    """
    Test that TTS engines handle invalid language codes gracefully.
    Expected: HTTP 400 or graceful fallback (not 500)
    """
    start = time.time()
    try:
        payload = {
            "text": "Test with invalid language",
            "language": "xx-invalid",  # Invalid language code
            "tts_speaker_wav": [],
            "parameters": {}
        }
        resp = client.post_json("/generate", payload, timeout=GENERATE_TIMEOUT)
        duration = (time.time() - start) * 1000

        # Accept 400 (validation) or 200 (graceful fallback)
        if resp.status_code in [200, 400]:
            return TestResult(
                name="POST /generate (invalid lang)",
                passed=True,
                message=f"Handled gracefully with HTTP {resp.status_code}",
                duration_ms=duration
            )

        if resp.status_code == 500:
            return TestResult(
                name="POST /generate (invalid lang)",
                passed=False,
                message=f"Got HTTP 500 (crash) instead of graceful handling: {resp.text[:100]}",
                duration_ms=duration
            )

        return TestResult(
            name="POST /generate (invalid lang)",
            passed=False,
            message=f"Unexpected HTTP {resp.status_code}",
            duration_ms=duration
        )

    except httpx.ReadTimeout:
        return TestResult(
            name="POST /generate (invalid lang)",
            passed=False,
            message=f"Timeout",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="POST /generate (invalid lang)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - Quality Endpoints (from base_quality_server.py)
# =============================================================================

def test_quality_analyze(client: EngineClient, audio_bytes: bytes, language: str = "en") -> TestResult:
    """
    Test POST /analyze endpoint.

    Request (AnalyzeRequest from base_quality_server.py):
        audioBase64: Optional[str]
        audioPath: Optional[str]
        language: str
        qualityThresholds: QualityThresholds
        expectedText: Optional[str]
        pronunciationRules: List[PronunciationRuleData]

    Expected response (AnalyzeResponse):
        engineType: str
        engineName: str
        qualityScore: int (0-100)
        qualityStatus: str ("perfect", "warning", "defect")
        details: QualityEngineDetails
    """
    start = time.time()
    try:
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        payload = {
            "audio_base64": audio_base64,
            "language": language,
            "quality_thresholds": {
                "speech_ratio_ideal_min": 60,
                "speech_ratio_ideal_max": 95,
                "speech_ratio_warning_min": 40,
                "speech_ratio_warning_max": 98,
                "max_silence_duration_warning": 2000,
                "max_silence_duration_critical": 5000,
                "max_clipping_peak": -1.0,
                "min_average_volume": -40.0
            }
        }

        resp = client.post_json("/analyze", payload, timeout=GENERATE_TIMEOUT)
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="POST /analyze",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:300]}",
                duration_ms=duration
            )

        data = resp.json()

        # Check required fields
        missing = check_required_fields(data, ANALYZE_REQUIRED_FIELDS)
        if missing:
            return TestResult(
                name="POST /analyze",
                passed=False,
                message=f"Missing required fields: {missing}",
                details=data,
                duration_ms=duration
            )

        # Validate qualityStatus
        error = check_enum_value(data["qualityStatus"], ANALYZE_STATUS_VALUES, "qualityStatus")
        if error:
            return TestResult(
                name="POST /analyze",
                passed=False,
                message=error,
                details=data,
                duration_ms=duration
            )

        # Validate qualityScore range
        score = data["qualityScore"]
        if not (0 <= score <= 100):
            return TestResult(
                name="POST /analyze",
                passed=False,
                message=f"qualityScore={score} not in range 0-100",
                details=data,
                duration_ms=duration
            )

        return TestResult(
            name="POST /analyze",
            passed=True,
            message=f"score={score}, status={data['qualityStatus']}",
            details=data,
            duration_ms=duration
        )

    except httpx.ReadTimeout:
        return TestResult(
            name="POST /analyze",
            passed=False,
            message=f"Timeout after {GENERATE_TIMEOUT}s",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="POST /analyze",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_quality_analyze_empty_audio(client: EngineClient) -> TestResult:
    """Test that POST /analyze rejects empty audio with HTTP 400."""
    start = time.time()
    try:
        payload = {"audio_base64": "", "language": "en", "quality_thresholds": {}}
        resp = client.post_json("/analyze", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="POST /analyze (empty audio)",
                passed=True,
                message="Correctly rejected with HTTP 400",
                duration_ms=duration
            )

        return TestResult(
            name="POST /analyze (empty audio)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /analyze (empty audio)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_quality_analyze_invalid_audio(client: EngineClient) -> TestResult:
    """
    Test that POST /analyze rejects invalid audio format with HTTP 400.
    Expected: HTTP 400 (not 500)
    """
    start = time.time()
    try:
        # Send invalid data (not a WAV file)
        invalid_audio = base64.b64encode(b"This is not a WAV file").decode("utf-8")
        payload = {
            "audio_base64": invalid_audio,
            "language": "en",
            "quality_thresholds": {}
        }
        resp = client.post_json("/analyze", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="POST /analyze (invalid audio)",
                passed=True,
                message="Correctly rejected with HTTP 400",
                duration_ms=duration
            )

        if resp.status_code == 500:
            return TestResult(
                name="POST /analyze (invalid audio)",
                passed=False,
                message=f"Got HTTP 500 (crash) instead of 400 validation: {resp.text[:100]}",
                duration_ms=duration
            )

        return TestResult(
            name="POST /analyze (invalid audio)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /analyze (invalid audio)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - Text Endpoints (from base_text_server.py)
# =============================================================================

def test_text_segment(client: EngineClient, text: str, language: str = "en") -> TestResult:
    """
    Test POST /segment endpoint.

    Request (SegmentRequest from base_text_server.py):
        text: str
        language: str
        maxLength: int
        minLength: int
        markOversized: bool

    Expected response (SegmentResponse):
        segments: List[SegmentItem]
        totalSegments: int
        totalCharacters: int
        failedCount: int
    """
    start = time.time()
    try:
        payload = {
            "text": text,
            "language": language,
            "max_length": 250,
            "min_length": 10,
            "mark_oversized": True
        }

        resp = client.post_json("/segment", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="POST /segment",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()

        # Check required fields
        missing = check_required_fields(data, SEGMENT_REQUIRED_FIELDS)
        if missing:
            return TestResult(
                name="POST /segment",
                passed=False,
                message=f"Missing required fields: {missing}",
                details=data,
                duration_ms=duration
            )

        # Validate segments is a list
        if not isinstance(data["segments"], list):
            return TestResult(
                name="POST /segment",
                passed=False,
                message=f"'segments' should be list, got {type(data['segments']).__name__}",
                details=data,
                duration_ms=duration
            )

        return TestResult(
            name="POST /segment",
            passed=True,
            message=f"Created {data['totalSegments']} segment(s), {data['totalCharacters']} chars",
            details=data,
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /segment",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_text_segment_empty(client: EngineClient) -> TestResult:
    """Test that POST /segment rejects empty text with HTTP 400."""
    start = time.time()
    try:
        payload = {"text": "", "language": "en", "max_length": 250, "min_length": 10, "mark_oversized": True}
        resp = client.post_json("/segment", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="POST /segment (empty text)",
                passed=True,
                message="Correctly rejected with HTTP 400",
                duration_ms=duration
            )

        return TestResult(
            name="POST /segment (empty text)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /segment (empty text)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_text_segment_invalid_params(client: EngineClient) -> TestResult:
    """
    Test that POST /segment rejects invalid length parameters with HTTP 400.
    According to docs: min_length >= max_length should return 400.
    """
    start = time.time()
    try:
        payload = {
            "text": "Test text",
            "language": "en",
            "max_length": 10,
            "min_length": 100,  # min > max - invalid!
            "mark_oversized": True
        }
        resp = client.post_json("/segment", payload)
        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="POST /segment (min > max)",
                passed=True,
                message="Correctly rejected with HTTP 400",
                duration_ms=duration
            )

        if resp.status_code == 500:
            return TestResult(
                name="POST /segment (min > max)",
                passed=False,
                message=f"Got HTTP 500 (crash) instead of 400 validation: {resp.text[:100]}",
                duration_ms=duration
            )

        return TestResult(
            name="POST /segment (min > max)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /segment (min > max)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - Common Edge Cases
# =============================================================================

def test_load_nonexistent_model(client: EngineClient) -> TestResult:
    """
    Test that POST /load with non-existent model returns proper error.
    Expected: HTTP 404 or 400 with error status (not 500)
    """
    start = time.time()
    try:
        resp = client.post_json("/load", {"engine_model_name": "nonexistent-model-xyz-12345"})
        duration = (time.time() - start) * 1000

        # Accept 400, 404 or 200 with error status
        if resp.status_code in [400, 404]:
            return TestResult(
                name="POST /load (nonexistent)",
                passed=True,
                message=f"Correctly rejected with HTTP {resp.status_code}",
                duration_ms=duration
            )

        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "error":
                return TestResult(
                    name="POST /load (nonexistent)",
                    passed=True,
                    message="Correctly returned error status",
                    duration_ms=duration
                )

        if resp.status_code == 500:
            return TestResult(
                name="POST /load (nonexistent)",
                passed=False,
                message=f"Got HTTP 500 (crash) instead of proper error: {resp.text[:100]}",
                duration_ms=duration
            )

        return TestResult(
            name="POST /load (nonexistent)",
            passed=False,
            message=f"Unexpected response: HTTP {resp.status_code}",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /load (nonexistent)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - Phase 2: Schema Validation
# =============================================================================

def test_info_camelcase(client: EngineClient, info_data: dict) -> TestResult:
    """
    Test that all top-level keys in /info response are camelCase.
    Strict validation: fail if any snake_case keys found.
    """
    start = time.time()
    duration = 0.0  # Already have data, no request needed

    invalid_keys = validate_camelcase_keys(info_data)

    if invalid_keys:
        return TestResult(
            name="/info CamelCase check",
            passed=False,
            message=f"Found snake_case keys: {invalid_keys}",
            details={"invalid_keys": invalid_keys},
            duration_ms=duration
        )

    return TestResult(
        name="/info CamelCase check",
        passed=True,
        message=f"All {len(info_data)} top-level keys are camelCase",
        duration_ms=duration
    )


def test_info_deep_validation(client: EngineClient, info_data: dict) -> TestResult:
    """
    Deep validation of /info response fields.
    Checks optional fields have correct types when present.
    """
    start = time.time()
    duration = 0.0  # Already have data
    errors = []

    # supportedLanguages: should be list of strings
    if "supportedLanguages" in info_data:
        langs = info_data["supportedLanguages"]
        if not isinstance(langs, list):
            errors.append(f"supportedLanguages should be list, got {type(langs).__name__}")
        elif langs and not all(isinstance(x, str) for x in langs):
            errors.append("supportedLanguages should contain only strings")

    # constraints: should be dict
    if "constraints" in info_data:
        constraints = info_data["constraints"]
        if not isinstance(constraints, dict):
            errors.append(f"constraints should be dict, got {type(constraints).__name__}")

    # capabilities: should be dict
    if "capabilities" in info_data:
        caps = info_data["capabilities"]
        if not isinstance(caps, dict):
            errors.append(f"capabilities should be dict, got {type(caps).__name__}")

    # parameters: should be dict
    if "parameters" in info_data:
        params = info_data["parameters"]
        if not isinstance(params, dict):
            errors.append(f"parameters should be dict, got {type(params).__name__}")

    # models: should be list of dicts with 'name'
    if "models" in info_data:
        models = info_data["models"]
        if not isinstance(models, list):
            errors.append(f"models should be list, got {type(models).__name__}")
        else:
            for i, model in enumerate(models):
                if not isinstance(model, dict):
                    errors.append(f"models[{i}] should be dict")
                elif "name" not in model:
                    errors.append(f"models[{i}] missing 'name' field")

    # defaultModel: if present, should exist in models
    if "defaultModel" in info_data and "models" in info_data:
        default = info_data["defaultModel"]
        model_names = [m.get("name") for m in info_data["models"] if isinstance(m, dict)]
        if default and default not in model_names:
            errors.append(f"defaultModel '{default}' not found in models list")

    # upstream: if present, should have name, url, license
    if "upstream" in info_data:
        upstream = info_data["upstream"]
        if isinstance(upstream, dict):
            for field in ["name", "url", "license"]:
                if field not in upstream:
                    errors.append(f"upstream missing '{field}' field")

    if errors:
        return TestResult(
            name="/info deep validation",
            passed=False,
            message=f"Found {len(errors)} issue(s): {errors[0]}",
            details={"errors": errors},
            duration_ms=duration
        )

    checked = ["supportedLanguages", "constraints", "capabilities", "parameters", "models", "defaultModel", "upstream"]
    present = [f for f in checked if f in info_data]
    return TestResult(
        name="/info deep validation",
        passed=True,
        message=f"Validated {len(present)} optional field(s): {', '.join(present) or 'none'}",
        duration_ms=duration
    )


def test_models_item_structure(client: EngineClient, models_data: dict) -> TestResult:
    """
    Validate structure of each model in /models response.
    Each model should have 'name' and 'displayName'.
    """
    start = time.time()
    duration = 0.0  # Already have data
    errors = []

    models = models_data.get("models", [])
    if not models:
        return TestResult(
            name="/models item structure",
            passed=True,
            message="No models to validate",
            duration_ms=duration
        )

    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"models[{i}] is not a dict")
            continue
        if "name" not in model:
            errors.append(f"models[{i}] missing 'name'")
        if "displayName" not in model:
            errors.append(f"models[{i}] missing 'displayName'")

    if errors:
        return TestResult(
            name="/models item structure",
            passed=False,
            message=f"Found {len(errors)} issue(s): {errors[0]}",
            details={"errors": errors},
            duration_ms=duration
        )

    return TestResult(
        name="/models item structure",
        passed=True,
        message=f"All {len(models)} model(s) have valid structure",
        duration_ms=duration
    )


def test_health_gpu_fields(client: EngineClient, health_data: dict) -> TestResult:
    """
    When device=cuda, GPU memory fields should be present.
    """
    start = time.time()
    duration = 0.0  # Already have data

    device = health_data.get("device", "cpu")
    if device != "cuda":
        return TestResult(
            name="/health GPU fields",
            passed=True,
            message=f"Device is '{device}', GPU fields not required",
            skipped=True,
            duration_ms=duration
        )

    errors = []

    # Check gpuMemoryUsedMb
    if "gpuMemoryUsedMb" not in health_data:
        errors.append("Missing gpuMemoryUsedMb")
    elif not isinstance(health_data["gpuMemoryUsedMb"], (int, float)):
        errors.append(f"gpuMemoryUsedMb should be number, got {type(health_data['gpuMemoryUsedMb']).__name__}")

    # Check gpuMemoryTotalMb
    if "gpuMemoryTotalMb" not in health_data:
        errors.append("Missing gpuMemoryTotalMb")
    elif not isinstance(health_data["gpuMemoryTotalMb"], (int, float)):
        errors.append(f"gpuMemoryTotalMb should be number, got {type(health_data['gpuMemoryTotalMb']).__name__}")

    if errors:
        return TestResult(
            name="/health GPU fields",
            passed=False,
            message=f"GPU fields issue: {errors[0]}",
            details={"errors": errors},
            duration_ms=duration
        )

    used = health_data.get("gpuMemoryUsedMb", 0)
    total = health_data.get("gpuMemoryTotalMb", 0)
    return TestResult(
        name="/health GPU fields",
        passed=True,
        message=f"GPU memory: {used}MB / {total}MB",
        duration_ms=duration
    )


# =============================================================================
# Test Functions - Phase 4: TTS Samples Upload
# =============================================================================

def test_tts_samples_upload(client: EngineClient, sample_id: Optional[str] = None) -> TestResult:
    """
    Test POST /samples/upload/{sample_id} endpoint with roundtrip verification.
    Uploads a test sample as raw WAV bytes, then verifies with /samples/check.

    Args:
        client: Engine client
        sample_id: Optional sample ID (generated if not provided)
    """
    start = time.time()

    # Generate unique sample ID if not provided
    if sample_id is None:
        sample_id = f"test-sample-{int(time.time())}"
    test_audio = create_test_wav(duration_sec=1.0)

    try:
        # Upload the sample (raw WAV bytes to /samples/upload/{sample_id})
        resp = client.post_bytes(f"/samples/upload/{sample_id}", test_audio)
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="POST /samples/upload",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        # Verify with /samples/check
        check_resp = client.post_json("/samples/check", {"sample_ids": [sample_id]})
        if check_resp.status_code != 200:
            return TestResult(
                name="POST /samples/upload",
                passed=False,
                message=f"Upload succeeded but /samples/check failed: HTTP {check_resp.status_code}",
                duration_ms=duration
            )

        check_data = check_resp.json()
        missing = check_data.get("missing", [])

        if sample_id in missing:
            return TestResult(
                name="POST /samples/upload",
                passed=False,
                message=f"Upload succeeded but sample not found in /samples/check",
                duration_ms=duration
            )

        return TestResult(
            name="POST /samples/upload",
            passed=True,
            message=f"Uploaded and verified sample '{sample_id}'",
            warning=f"Test sample '{sample_id}' remains on engine",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="POST /samples/upload",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - Phase 6: Robustness
# =============================================================================

def test_model_hotswap(client: EngineClient, models: list, capabilities: dict, current_model: str) -> TestResult:
    """
    Test model hotswap if supported and multiple models available.
    """
    start = time.time()

    # Check if hotswap is supported
    supports_hotswap = capabilities.get("supports_model_hotswap", False)
    if not supports_hotswap:
        return TestResult(
            name="Model hotswap",
            passed=True,
            message="Hotswap not supported by engine",
            skipped=True,
            duration_ms=0.0
        )

    # Check if we have multiple models
    model_names = [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]
    if len(model_names) < 2:
        return TestResult(
            name="Model hotswap",
            passed=True,
            message=f"Only {len(model_names)} model(s) available, need 2+ for hotswap test",
            skipped=True,
            duration_ms=0.0
        )

    # Find a different model to switch to
    other_model = None
    for name in model_names:
        if name != current_model:
            other_model = name
            break

    if not other_model:
        return TestResult(
            name="Model hotswap",
            passed=True,
            message="Could not find different model to test",
            skipped=True,
            duration_ms=0.0
        )

    try:
        # Load different model
        resp = client.post_json("/load", {"engine_model_name": other_model})
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="Model hotswap",
                passed=False,
                message=f"Failed to load '{other_model}': HTTP {resp.status_code}",
                duration_ms=duration
            )

        data = resp.json()
        if data.get("status") == "error":
            return TestResult(
                name="Model hotswap",
                passed=False,
                message=f"Load returned error: {data.get('error', 'unknown')}",
                duration_ms=duration
            )

        # Verify model changed
        health_resp = client.get("/health")
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            loaded_model = health_data.get("currentEngineModel")
            if loaded_model != other_model:
                return TestResult(
                    name="Model hotswap",
                    passed=False,
                    message=f"Model did not change: expected '{other_model}', got '{loaded_model}'",
                    duration_ms=duration
                )

        # Restore original model
        client.post_json("/load", {"engine_model_name": current_model})

        return TestResult(
            name="Model hotswap",
            passed=True,
            message=f"Switched from '{current_model}' to '{other_model}' and back",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="Model hotswap",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_reload_same_model(client: EngineClient, current_model: str) -> TestResult:
    """
    Test reloading the same model that is already loaded.
    Should be handled gracefully.
    """
    start = time.time()

    if not current_model:
        return TestResult(
            name="Reload same model",
            passed=True,
            message="No model loaded, cannot test reload",
            skipped=True,
            duration_ms=0.0
        )

    try:
        resp = client.post_json("/load", {"engine_model_name": current_model})
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="Reload same model",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        data = resp.json()
        if data.get("status") == "error":
            return TestResult(
                name="Reload same model",
                passed=False,
                message=f"Reload failed: {data.get('error', 'unknown')}",
                duration_ms=duration
            )

        # Verify model still loaded
        health_resp = client.get("/health")
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            if not health_data.get("engineModelLoaded"):
                return TestResult(
                    name="Reload same model",
                    passed=False,
                    message="Model no longer loaded after reload",
                    duration_ms=duration
                )

        return TestResult(
            name="Reload same model",
            passed=True,
            message=f"Successfully reloaded '{current_model}'",
            duration_ms=duration
        )

    except httpx.RequestError as e:
        return TestResult(
            name="Reload same model",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_large_payload_at_limit(
    client: EngineClient,
    info_data: dict,
    engine_type: str,
    language: str,
    speaker_sample: Optional[str] = None
) -> TestResult:
    """
    Test with text at exactly maxTextLength limit.
    """
    start = time.time()

    # Get max_text_length from constraints (snake_case in engine.yaml)
    constraints = info_data.get("constraints", {})
    max_length = constraints.get("max_text_length", 10000)  # Fallback to 10000

    # Generate text at limit
    test_text = generate_text_of_length(max_length)

    try:
        if engine_type == "tts":
            speaker_wav = [speaker_sample] if speaker_sample else []
            payload = {
                "text": test_text,
                "language": language,
                "tts_speaker_wav": speaker_wav,
                "parameters": {}
            }
            resp = client.post_json("/generate", payload, timeout=GENERATE_TIMEOUT)
        elif engine_type == "text":
            payload = {
                "text": test_text,
                "language": language,
                "max_length": 250,
                "min_length": 10,
                "mark_oversized": True
            }
            resp = client.post_json("/segment", payload)
        else:
            return TestResult(
                name="Large payload (at limit)",
                passed=True,
                message=f"Not applicable for engine type '{engine_type}'",
                skipped=True,
                duration_ms=0.0
            )

        duration = (time.time() - start) * 1000

        if resp.status_code == 200:
            return TestResult(
                name="Large payload (at limit)",
                passed=True,
                message=f"Handled {max_length} chars successfully",
                duration_ms=duration
            )

        return TestResult(
            name="Large payload (at limit)",
            passed=False,
            message=f"Expected HTTP 200, got {resp.status_code}: {resp.text[:100]}",
            duration_ms=duration
        )

    except httpx.ReadTimeout:
        return TestResult(
            name="Large payload (at limit)",
            passed=False,
            message=f"Timeout processing {max_length} chars",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="Large payload (at limit)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_large_payload_over_limit(
    client: EngineClient,
    info_data: dict,
    engine_type: str,
    language: str,
    speaker_sample: Optional[str] = None
) -> TestResult:
    """
    Test with text exceeding maxTextLength limit.
    Should return HTTP 400.
    """
    start = time.time()

    # Get max_text_length from constraints (snake_case in engine.yaml)
    constraints = info_data.get("constraints", {})
    max_length = constraints.get("max_text_length")

    if max_length is None:
        return TestResult(
            name="Large payload (over limit)",
            passed=True,
            message="No max_text_length constraint defined",
            skipped=True,
            duration_ms=0.0
        )

    # Generate text over limit
    over_length = max_length + 100
    test_text = generate_text_of_length(over_length)

    try:
        if engine_type == "tts":
            speaker_wav = [speaker_sample] if speaker_sample else []
            payload = {
                "text": test_text,
                "language": language,
                "tts_speaker_wav": speaker_wav,
                "parameters": {}
            }
            resp = client.post_json("/generate", payload, timeout=GENERATE_TIMEOUT)
        elif engine_type == "text":
            payload = {
                "text": test_text,
                "language": language,
                "max_length": 250,
                "min_length": 10,
                "mark_oversized": True
            }
            resp = client.post_json("/segment", payload)
        else:
            return TestResult(
                name="Large payload (over limit)",
                passed=True,
                message=f"Not applicable for engine type '{engine_type}'",
                skipped=True,
                duration_ms=0.0
            )

        duration = (time.time() - start) * 1000

        if resp.status_code == 400:
            return TestResult(
                name="Large payload (over limit)",
                passed=True,
                message=f"Correctly rejected {over_length} chars with HTTP 400",
                duration_ms=duration
            )

        if resp.status_code == 200:
            return TestResult(
                name="Large payload (over limit)",
                passed=False,
                message=f"Expected HTTP 400 for {over_length} chars, got 200 (limit not enforced?)",
                duration_ms=duration
            )

        return TestResult(
            name="Large payload (over limit)",
            passed=False,
            message=f"Expected HTTP 400, got {resp.status_code}",
            duration_ms=duration
        )

    except httpx.ReadTimeout:
        return TestResult(
            name="Large payload (over limit)",
            passed=False,
            message=f"Timeout (should reject quickly, not process)",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="Large payload (over limit)",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


def test_unicode_handling(
    client: EngineClient,
    engine_type: str,
    language: str,
    speaker_sample: Optional[str] = None
) -> TestResult:
    """
    Test handling of Unicode characters (umlauts, emojis, CJK, RTL).
    """
    start = time.time()

    try:
        if engine_type == "tts":
            speaker_wav = [speaker_sample] if speaker_sample else []
            payload = {
                "text": UNICODE_TEST_TEXT,
                "language": language,
                "tts_speaker_wav": speaker_wav,
                "parameters": {}
            }
            resp = client.post_json("/generate", payload, timeout=GENERATE_TIMEOUT)
            duration = (time.time() - start) * 1000

            if resp.status_code != 200:
                return TestResult(
                    name="Unicode handling",
                    passed=False,
                    message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    duration_ms=duration
                )

            audio_size = len(resp.content)
            if audio_size < 100:
                return TestResult(
                    name="Unicode handling",
                    passed=False,
                    message=f"Audio too small ({audio_size} bytes)",
                    duration_ms=duration
                )

            return TestResult(
                name="Unicode handling",
                passed=True,
                message=f"Generated {audio_size:,} bytes from Unicode text",
                duration_ms=duration
            )

        elif engine_type == "text":
            payload = {
                "text": UNICODE_TEST_TEXT,
                "language": language,
                "max_length": 250,
                "min_length": 10,
                "mark_oversized": True
            }
            resp = client.post_json("/segment", payload)
            duration = (time.time() - start) * 1000

            if resp.status_code != 200:
                return TestResult(
                    name="Unicode handling",
                    passed=False,
                    message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    duration_ms=duration
                )

            data = resp.json()
            segments = data.get("segments", [])
            return TestResult(
                name="Unicode handling",
                passed=True,
                message=f"Segmented Unicode text into {len(segments)} segment(s)",
                duration_ms=duration
            )

        else:
            return TestResult(
                name="Unicode handling",
                passed=True,
                message=f"Not applicable for engine type '{engine_type}'",
                skipped=True,
                duration_ms=0.0
            )

    except httpx.ReadTimeout:
        return TestResult(
            name="Unicode handling",
            passed=False,
            message="Timeout processing Unicode text",
            duration_ms=(time.time() - start) * 1000
        )
    except httpx.RequestError as e:
        return TestResult(
            name="Unicode handling",
            passed=False,
            message=f"Connection error: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Test Functions - Phase 7: Shutdown
# =============================================================================

def test_shutdown(client: EngineClient) -> TestResult:
    """
    Test POST /shutdown endpoint.
    After this test, the engine should no longer be reachable.
    """
    start = time.time()

    try:
        resp = client.post_json("/shutdown", {})
        duration = (time.time() - start) * 1000

        if resp.status_code != 200:
            return TestResult(
                name="POST /shutdown",
                passed=False,
                message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                duration_ms=duration
            )

        # Wait a moment for shutdown
        time.sleep(1.0)

        # Verify engine is down
        try:
            verify_resp = client.get("/health")
            # If we get here, engine is still running
            return TestResult(
                name="POST /shutdown",
                passed=False,
                message=f"Engine still responding after shutdown (HTTP {verify_resp.status_code})",
                duration_ms=duration
            )
        except httpx.RequestError:
            # Expected: connection should fail
            return TestResult(
                name="POST /shutdown",
                passed=True,
                message="Engine shut down successfully",
                duration_ms=duration
            )

    except httpx.RequestError as e:
        # If shutdown request itself fails, that's an error
        return TestResult(
            name="POST /shutdown",
            passed=False,
            message=f"Shutdown request failed: {e}",
            duration_ms=(time.time() - start) * 1000
        )


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests(
    host: str,
    port: int,
    verbose: bool = False,
    skip_functional: bool = False,
    skip_robustness: bool = False,
    skip_shutdown: bool = False
) -> TestSuite:
    """Run all tests against an engine."""
    suite = TestSuite()
    client = EngineClient(host, port)

    # Store data for later phases
    info_data = {}
    models_data = {}
    health_data = {}
    current_model = None
    test_sample_id = None  # Set in Phase 4 for TTS, reused in Phase 6

    try:
        # =================================================================
        # Phase 1: Discovery (Common Endpoints)
        # =================================================================
        print("\n--- Phase 1: Discovery ---\n")

        # GET /health
        result = test_health(client)
        suite.add(result)
        print_result(result, verbose)
        health_data = result.details or {}

        if not result.passed:
            print("\n[FAIL] Engine health check failed, aborting")
            return suite

        # GET /info
        result = test_info(client)
        suite.add(result)
        print_result(result, verbose)

        if not result.passed:
            print("\n[FAIL] Cannot determine engine type, aborting")
            return suite

        # Extract engine info
        info_data = result.details or {}
        suite.engine_name = info_data.get("name", "unknown")
        suite.engine_type = info_data.get("engineType", "unknown")
        languages = info_data.get("supportedLanguages", ["en"])
        test_language = languages[0] if languages else "en"

        print(f"\n    Engine: {suite.engine_name}")
        print(f"    Type:   {suite.engine_type}")
        print(f"    Language for tests: {test_language}")

        # GET /models
        result = test_models(client)
        suite.add(result)
        print_result(result, verbose)

        if not result.passed:
            print("\n[FAIL] Cannot list models, aborting")
            return suite

        # Extract model info
        models_data = result.details or {}
        models = models_data.get("models", [])
        default_model = models_data.get("defaultModel") or (models[0].get("name") if models else None)

        # =================================================================
        # Phase 2: Schema Validation
        # =================================================================
        print("\n--- Phase 2: Schema Validation ---\n")

        # CamelCase check
        result = test_info_camelcase(client, info_data)
        suite.add(result)
        print_result(result, verbose)

        # Deep /info validation
        result = test_info_deep_validation(client, info_data)
        suite.add(result)
        print_result(result, verbose)

        # /models item structure
        result = test_models_item_structure(client, models_data)
        suite.add(result)
        print_result(result, verbose)

        # GPU fields (if applicable)
        result = test_health_gpu_fields(client, health_data)
        suite.add(result)
        print_result(result, verbose)

        # =================================================================
        # Phase 3: Model Loading
        # =================================================================
        print("\n--- Phase 3: Model Loading ---\n")

        # First: Load non-existent model (expect error)
        result = test_load_nonexistent_model(client)
        suite.add(result)
        print_result(result, verbose)

        # Then: Load default model (keep for Phase 4)
        if default_model:
            result = test_load_model(client, default_model)
            suite.add(result)
            print_result(result, verbose)

            if result.passed:
                current_model = default_model
            else:
                print("\n[WARN] Model loading failed, functional tests may fail")
        else:
            print("    [SKIP] No models available to load")

        # =================================================================
        # Phase 4: Functional Tests (type-specific)
        # =================================================================
        if skip_functional:
            print("\n--- Phase 4: Functional Tests (SKIPPED) ---\n")
        else:
            print(f"\n--- Phase 4: Functional Tests ({suite.engine_type}) ---\n")

            if suite.engine_type == "tts":
                # TTS: /samples/check, /samples/upload, /generate
                result = test_tts_samples_check(client)
                suite.add(result)
                print_result(result, verbose)

                # Generate sample ID for upload test (reused for generate if voice cloning)
                test_sample_id = f"test-sample-{int(time.time())}"
                result = test_tts_samples_upload(client, test_sample_id)
                suite.add(result)
                print_result(result, verbose)

                # Check if engine requires speaker samples for voice cloning
                capabilities = info_data.get("capabilities", {})
                needs_speaker = capabilities.get("supports_speaker_cloning", False)

                test_text = "Hello, this is a test of the text to speech engine."
                # Note: samples are stored as {sample_id}.wav, so add extension for /generate
                speaker_for_test = f"{test_sample_id}.wav" if needs_speaker else None
                result = test_tts_generate(client, test_text, test_language, speaker_for_test)
                suite.add(result)
                print_result(result, verbose)

            elif suite.engine_type in ["stt", "audio"]:
                # Quality: /analyze
                test_audio = create_test_wav(duration_sec=2.0)
                result = test_quality_analyze(client, test_audio, test_language)
                suite.add(result)
                print_result(result, verbose)

            elif suite.engine_type == "text":
                # Text: /segment
                test_text = (
                    "This is the first sentence. This is the second sentence. "
                    "And here is a third one for good measure."
                )
                result = test_text_segment(client, test_text, test_language)
                suite.add(result)
                print_result(result, verbose)

        # =================================================================
        # Phase 5: Input Validation (4xx expected)
        # =================================================================
        print("\n--- Phase 5: Input Validation (4xx expected) ---\n")

        if suite.engine_type == "tts":
            result = test_tts_generate_empty_text(client)
            suite.add(result)
            print_result(result, verbose)

            # Speaker cloning without samples
            capabilities = info_data.get("capabilities", {})
            if capabilities.get("supports_speaker_cloning", False):
                result = test_tts_generate_no_speaker(client, test_language)
                suite.add(result)
                print_result(result, verbose)

            # Invalid language code
            result = test_tts_generate_invalid_language(client)
            suite.add(result)
            print_result(result, verbose)

        elif suite.engine_type in ["stt", "audio"]:
            result = test_quality_analyze_empty_audio(client)
            suite.add(result)
            print_result(result, verbose)

            result = test_quality_analyze_invalid_audio(client)
            suite.add(result)
            print_result(result, verbose)

        elif suite.engine_type == "text":
            result = test_text_segment_empty(client)
            suite.add(result)
            print_result(result, verbose)

            result = test_text_segment_invalid_params(client)
            suite.add(result)
            print_result(result, verbose)

        # =================================================================
        # Phase 6: Robustness
        # =================================================================
        if skip_robustness:
            print("\n--- Phase 6: Robustness (SKIPPED) ---\n")
        else:
            print("\n--- Phase 6: Robustness ---\n")

            # Model hotswap (if supported and 2+ models)
            capabilities = info_data.get("capabilities", {})
            result = test_model_hotswap(client, models, capabilities, current_model)
            suite.add(result)
            print_result(result, verbose)

            # Reload same model
            result = test_reload_same_model(client, current_model)
            suite.add(result)
            print_result(result, verbose)

            # Determine speaker sample for TTS robustness tests
            # Reuse sample from Phase 4 if engine requires speaker cloning
            speaker_for_robustness = None
            if suite.engine_type == "tts":
                needs_speaker = capabilities.get("supports_speaker_cloning", False)
                if needs_speaker and test_sample_id:
                    speaker_for_robustness = f"{test_sample_id}.wav"

            # Large payload at limit
            result = test_large_payload_at_limit(
                client, info_data, suite.engine_type, test_language, speaker_for_robustness
            )
            suite.add(result)
            print_result(result, verbose)

            # Large payload over limit
            result = test_large_payload_over_limit(
                client, info_data, suite.engine_type, test_language, speaker_for_robustness
            )
            suite.add(result)
            print_result(result, verbose)

            # Unicode handling
            result = test_unicode_handling(
                client, suite.engine_type, test_language, speaker_for_robustness
            )
            suite.add(result)
            print_result(result, verbose)

        # =================================================================
        # Phase 7: Shutdown
        # =================================================================
        if skip_shutdown:
            print("\n--- Phase 7: Shutdown (SKIPPED) ---\n")
        else:
            print("\n--- Phase 7: Shutdown ---\n")

            result = test_shutdown(client)
            suite.add(result)
            print_result(result, verbose)

    finally:
        client.close()

    return suite


def print_result(result: TestResult, verbose: bool):
    """Print a single test result with colors."""
    if result.skipped:
        status = colorize("[SKIP]", Colors.YELLOW)
    elif result.warning:
        status = colorize("[WARN]", Colors.YELLOW)
    elif result.passed:
        status = colorize("[OK]", Colors.GREEN)
    else:
        status = colorize("[FAIL]", Colors.RED)

    print(f"    {status} {result.name} ({format_duration(result.duration_ms)})")
    print(f"         {result.message}")

    if result.warning:
        print(f"         {colorize('Warning:', Colors.YELLOW)} {result.warning}")

    if verbose and result.details and not result.passed:
        details_str = json.dumps(result.details, indent=2)
        if len(details_str) > 500:
            details_str = details_str[:500] + "..."
        print(f"         Details: {details_str}")


def print_summary(suite: TestSuite):
    """Print test summary with colors."""
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {suite.engine_name} ({suite.engine_type})")
    print("=" * 60)

    # Color the counts based on values
    passed_str = colorize(str(suite.passed), Colors.GREEN) if suite.passed > 0 else str(suite.passed)
    skipped_str = colorize(str(suite.skipped), Colors.YELLOW) if suite.skipped > 0 else str(suite.skipped)
    failed_str = colorize(str(suite.failed), Colors.RED) if suite.failed > 0 else str(suite.failed)

    print(f"\n    Total:   {len(suite.results)}")
    print(f"    Passed:  {passed_str}")
    print(f"    Skipped: {skipped_str}")
    print(f"    Failed:  {failed_str}")

    if suite.failed > 0:
        print(f"\n    {colorize('Failed tests:', Colors.RED)}")
        for result in suite.results:
            if not result.passed and not result.skipped:
                print(f"      - {result.name}")
                print(f"        {result.message}")

    if suite.warnings:
        print(f"\n    {colorize('Warnings:', Colors.YELLOW)}")
        for warning in suite.warnings:
            print(f"      - {warning}")

    print()
    if suite.all_passed:
        print(f"    {colorize('RESULT: ALL TESTS PASSED', Colors.GREEN + Colors.BOLD)}")
    else:
        print(f"    {colorize('RESULT: SOME TESTS FAILED', Colors.RED + Colors.BOLD)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test engine API endpoints against documented specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_engine.py --port 8766
    python scripts/test_engine.py --port 8766 --verbose
    python scripts/test_engine.py --host 192.168.1.100 --port 8766
    python scripts/test_engine.py --port 8766 --skip-functional
    python scripts/test_engine.py --port 8766 --skip-robustness
    python scripts/test_engine.py --port 8766 --skip-shutdown
        """
    )
    parser.add_argument("--host", default="127.0.0.1", help="Engine host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, required=True, help="Engine port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--skip-functional", action="store_true", help="Skip functional tests (generate/analyze/segment)")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness tests (hotswap/unicode/large payload)")
    parser.add_argument("--skip-shutdown", action="store_true", help="Skip shutdown test (engine stays running)")

    args = parser.parse_args()

    print(f"\nTesting engine at {args.host}:{args.port}")
    print("=" * 60)

    # Check connectivity
    try:
        client = httpx.Client(timeout=5.0)
        client.get(f"http://{args.host}:{args.port}/health")
        client.close()
    except httpx.RequestError as e:
        print(f"\n[FAIL] Cannot connect to engine: {e}")
        print(f"\n       Make sure the engine is running on port {args.port}")
        sys.exit(2)

    # Run tests
    suite = run_tests(
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        skip_functional=args.skip_functional,
        skip_robustness=args.skip_robustness,
        skip_shutdown=args.skip_shutdown
    )

    print_summary(suite)
    sys.exit(0 if suite.all_passed else 1)


if __name__ == "__main__":
    main()
