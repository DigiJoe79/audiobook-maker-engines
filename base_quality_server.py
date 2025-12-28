"""
Base Quality Server - Abstract FastAPI Server for Quality Analysis Engines (STT + Audio)

Extends BaseEngineServer with quality analysis functionality:
- /analyze endpoint for audio quality analysis
- Generic Quality Format response structure
- Configurable quality thresholds

Both STT engines (Whisper) and Audio Analysis engines (Silero-VAD) inherit from this class.
They share the same response format for unified quality reporting.

Quality engines implement:
- load_model(model_name: str)
- get_available_models() -> List[ModelInfo]
- analyze_audio(audio_bytes, language, thresholds) -> AnalyzeResult
- unload_model()
"""
from fastapi import HTTPException
from typing import Dict, Any, List, Optional
from abc import abstractmethod
from loguru import logger
import base64
import asyncio

from base_server import BaseEngineServer, CamelCaseModel, ModelInfo as ModelInfo


# ============= Quality Thresholds =============

class QualityThresholds(CamelCaseModel):
    """
    Configurable thresholds for quality checks.

    These are passed from the backend and can be customized per-request.
    Engines should use these to determine warning/error severity.
    """
    # Silence thresholds (milliseconds)
    max_silence_duration_warning: int = 2500
    max_silence_duration_critical: int = 3750

    # Speech ratio thresholds (0-100 percent)
    speech_ratio_ideal_min: float = 75
    speech_ratio_ideal_max: float = 90
    speech_ratio_warning_min: float = 65
    speech_ratio_warning_max: float = 93

    # Volume thresholds (dB)
    max_clipping_peak: float = 0.0
    min_average_volume: float = -40.0


# ============= Generic Quality Format Models =============

class QualityField(CamelCaseModel):
    """
    Single field in quality analysis details.

    Field types determine how the frontend renders the value:
    - percent: Show as percentage (e.g., "82%")
    - number: Show as raw number (e.g., "1200")
    - seconds: Show as duration (e.g., "2.5s")
    - string: Show as-is (e.g., "-3.2 dB")
    - text: Show as text with potential i18n lookup
    """
    key: str   # i18n key for field label
    value: Any  # Field value
    type: str   # Rendering type: percent, seconds, text, string, number


class QualityInfoBlockItem(CamelCaseModel):
    """
    Single item in an info block (warning, error, or info message).

    Severities:
    - error: Critical issue (red highlight)
    - warning: Potential problem (yellow highlight)
    - info: Informational message (gray/neutral)
    """
    text: str  # i18n key or display text
    severity: str  # error, warning, info
    details: Optional[Dict[str, Any]] = None  # Additional data for debugging


class QualityEngineDetails(CamelCaseModel):
    """
    Engine-specific details for UI rendering.

    This structure allows the frontend to render quality details
    without knowing the specific engine implementation.
    """
    top_label: str  # i18n key for section header
    fields: List[QualityField] = []  # Key-value pairs for display
    info_blocks: Dict[str, List[QualityInfoBlockItem]] = {}  # Grouped messages


class AnalyzeResponse(CamelCaseModel):
    """
    Generic quality analysis response format.

    This is the standard format that all quality engines (STT + Audio) must return.
    The QualityWorker expects this format for unified quality reporting.

    Quality score thresholds:
    - perfect: score >= 85
    - warning: 70 <= score < 85
    - defect: score < 70
    """
    engine_type: str  # "stt" or "audio"
    engine_name: str  # Engine identifier (e.g., "whisper", "silero-vad")
    quality_score: int  # 0-100 (100 = best)
    quality_status: str  # "perfect", "warning", "defect"
    details: QualityEngineDetails  # Engine-specific details


# ============= Request Model =============

class PronunciationRuleData(CamelCaseModel):
    """Pronunciation rule data for text comparison (STT only)."""
    pattern: str
    replacement: str
    is_regex: bool = False
    is_active: bool = True


class AnalyzeRequest(CamelCaseModel):
    """Request model for quality analysis."""
    audio_base64: Optional[str] = None  # Base64-encoded audio
    audio_path: Optional[str] = None  # Alternative: file path
    language: str = "en"  # Language code (mainly for STT)
    quality_thresholds: QualityThresholds = QualityThresholds()

    # STT-specific: for text comparison
    expected_text: Optional[str] = None  # Original segment text
    pronunciation_rules: List[PronunciationRuleData] = []  # Active rules

    def get_audio_bytes(self) -> Optional[bytes]:
        """Decode base64 audio data to bytes if provided."""
        if self.audio_base64:
            return base64.b64decode(self.audio_base64)
        return None


# ============= Analysis Result (Internal) =============

class AnalyzeResult:
    """
    Internal result structure returned by analyze_audio().

    Engines return this, and BaseQualityServer converts it to AnalyzeResponse.
    """
    def __init__(
        self,
        quality_score: int,
        fields: List[QualityField],
        info_blocks: Optional[Dict[str, List[QualityInfoBlockItem]]] = None,
        top_label: str = "quality.analysis"
    ):
        self.quality_score = quality_score
        self.fields = fields
        self.info_blocks = info_blocks or {}
        self.top_label = top_label

        # Auto-determine status from score
        if quality_score >= 85:
            self.quality_status = "perfect"
        elif quality_score >= 70:
            self.quality_status = "warning"
        else:
            self.quality_status = "defect"


# ============= Base Quality Server =============

class BaseQualityServer(BaseEngineServer):
    """
    Abstract base class for quality analysis engines (STT + Audio)

    Extends BaseEngineServer with:
    - /analyze endpoint for quality analysis
    - Generic Quality Format response
    - Quality thresholds handling

    Quality engines implement:
    - load_model(model_name) - Load analysis model
    - get_available_models() - Return available models
    - analyze_audio(audio_bytes, language, thresholds) -> AnalyzeResult
    - unload_model() - Unload model and free resources
    """

    # Subclasses must set this to "stt" or "audio"
    engine_type: str = "quality"

    def __init__(self, engine_name: str, display_name: str, engine_type: str, config_path: Optional[str] = None):
        """
        Initialize quality analysis engine server

        Args:
            engine_name: Engine identifier (e.g., "whisper", "silero-vad")
            display_name: Human-readable name (e.g., "Whisper STT", "Silero VAD")
            engine_type: Either "stt" or "audio"
            config_path: Optional path to engine.yaml (for /info endpoint)
        """
        super().__init__(engine_name, display_name, config_path)
        self.engine_type = engine_type

        # Setup quality-specific routes
        self._setup_analyze_route()

        logger.info(f"[{self.engine_name}] Quality server initialized (type: {engine_type})")

    def _setup_analyze_route(self):
        """Setup quality-specific /analyze endpoint"""

        @self.app.post("/analyze", response_model=AnalyzeResponse)
        async def analyze_endpoint(request: AnalyzeRequest):
            """Analyze audio and return quality metrics"""
            try:
                # Return 503 if model is still loading (non-blocking response)
                if self.status == "loading":
                    raise HTTPException(
                        status_code=503,
                        detail="Model loading in progress. Retry after loading completes."
                    )

                if not self.model_loaded:
                    raise HTTPException(status_code=400, detail="Model not loaded")

                # Get audio data
                audio_bytes = None
                if request.audio_base64:
                    audio_bytes = request.get_audio_bytes()
                elif request.audio_path:
                    from pathlib import Path
                    audio_path = Path(request.audio_path)
                    if not audio_path.exists():
                        raise HTTPException(
                            status_code=404,
                            detail=f"Audio file not found: {request.audio_path}"
                        )
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Either audio_base64 or audio_path must be provided"
                    )

                # Validate audio data
                if not audio_bytes or len(audio_bytes) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Audio data is empty"
                    )

                # Validate WAV format (check RIFF header)
                if len(audio_bytes) < 44:  # Minimum WAV header size
                    raise HTTPException(
                        status_code=400,
                        detail="Audio data too small to be valid WAV"
                    )
                if audio_bytes[:4] != b'RIFF' or audio_bytes[8:12] != b'WAVE':
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid audio format: expected WAV file"
                    )

                logger.info(
                    f"[{self.engine_name}] Analyzing audio | "
                    f"Model: {self.current_model} | "
                    f"Language: {request.language} | "
                    f"Size: {len(audio_bytes)} bytes"
                )

                self.status = "processing"

                # Call engine-specific implementation in thread pool
                # This prevents blocking the event loop during analysis
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: self.analyze_audio(
                        audio_bytes=audio_bytes,
                        language=request.language,
                        thresholds=request.quality_thresholds,
                        expected_text=request.expected_text,
                        pronunciation_rules=request.pronunciation_rules
                    )
                )

                self.status = "ready"

                # Build response
                return AnalyzeResponse(
                    engine_type=self.engine_type,
                    engine_name=self.engine_name,
                    quality_score=result.quality_score,
                    quality_status=result.quality_status,
                    details=QualityEngineDetails(
                        top_label=result.top_label,
                        fields=result.fields,
                        info_blocks=result.info_blocks
                    )
                )

            except HTTPException:
                self.status = "ready"  # HTTP errors are client errors, server is still ready
                raise
            except Exception as e:
                self._handle_processing_error(e, "analysis")

    # ============= Quality-Specific Abstract Method =============

    @abstractmethod
    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds,
        expected_text: Optional[str] = None,
        pronunciation_rules: Optional[List[PronunciationRuleData]] = None
    ) -> AnalyzeResult:
        """
        Analyze audio and return quality metrics (engine-specific)

        Args:
            audio_bytes: Raw audio file bytes (WAV format)
            language: Language code (e.g., "en", "de") - mainly for STT
            thresholds: Quality thresholds for determining warnings/errors
            expected_text: Original text for comparison (STT only)
            pronunciation_rules: Active pronunciation rules (STT only)

        Returns:
            AnalyzeResult with quality score, fields, and info blocks

        Raises:
            Exception: If analysis fails
        """
        pass
