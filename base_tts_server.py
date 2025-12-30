"""
Base TTS Engine Server - Abstract FastAPI Server for TTS Engines

Extends BaseEngineServer with TTS-specific functionality:
- /generate endpoint for audio synthesis
- /samples/check and /samples/upload endpoints for speaker sample management
- generate_audio() abstract method

TTS engine servers inherit from this class and implement:
- load_model(model_name: str)
- get_available_models() -> List[ModelInfo]
- generate_audio(text, language, speaker_wav, parameters) -> bytes
- unload_model()
"""
from fastapi import Response, HTTPException, Request
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
from abc import abstractmethod
from loguru import logger
import asyncio

from base_server import BaseEngineServer, CamelCaseModel, ModelInfo as ModelInfo


# ============= TTS-Specific Request Models =============

class GenerateRequest(CamelCaseModel):
    """Request to generate TTS audio"""
    text: str
    language: str  # Required (engine can ignore if not needed)
    tts_speaker_wav: Union[str, List[str]]  # Filename(s) in samples_dir
    parameters: Optional[Dict[str, Any]] = None  # Engine-specific params (null/missing = use defaults)


class SampleCheckRequest(CamelCaseModel):
    """Check which samples exist in engine"""
    sample_ids: List[str]  # List of sample UUIDs (without .wav extension)


class SampleCheckResponse(CamelCaseModel):
    """Response with missing sample IDs"""
    missing: List[str]  # Sample UUIDs not found in engine


# ============= Base TTS Server =============

class BaseTTSServer(BaseEngineServer):
    """
    Abstract base class for TTS engine servers

    Extends BaseEngineServer with:
    - /generate endpoint for audio synthesis
    - generate_audio() abstract method

    TTS engines implement:
    - load_model(model_name) - Load a TTS model
    - get_available_models() - Return available TTS models
    - generate_audio(text, language, speaker_wav, parameters) -> bytes
    - unload_model() - Unload model and free resources
    """

    def __init__(self, engine_name: str, display_name: str, config_path: Optional[str] = None):
        """
        Initialize TTS engine server

        Args:
            engine_name: Engine identifier (e.g., "xtts", "chatterbox")
            display_name: Human-readable name (e.g., "XTTS v2", "Chatterbox TTS")
            config_path: Optional path to engine.yaml (for /info endpoint)
        """
        super().__init__(engine_name, display_name, config_path)

        # Determine samples directory based on environment
        # Docker: /app/samples (can be mounted for persistence)
        # Subprocess: ./samples/ (relative to engine dir)
        if Path("/app").exists():
            # Docker container - use /app/samples
            self.samples_dir = Path("/app/samples")
        else:
            # Subprocess on Windows/Linux - use engine-local dir
            # Derived from server.py location (e.g., backend/engines/tts/xtts/samples/)
            import sys
            if sys.argv[0]:
                self.samples_dir = Path(sys.argv[0]).parent / "samples"
            else:
                self.samples_dir = Path(__file__).parent / "samples"

        self.samples_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[{self.engine_name}] Samples directory: {self.samples_dir}")

        # Setup TTS-specific routes
        self._setup_generate_route()
        self._setup_sample_routes()

        logger.info(f"[{self.engine_name}] BaseTTSServer initialized")

    def _validate_sample_id(self, sample_id: str) -> Path:
        """
        Validate sample ID format and return safe path.

        Args:
            sample_id: Sample identifier (UUID or filename without extension)

        Returns:
            Safe Path object within samples_dir

        Raises:
            HTTPException: If sample_id is invalid or path escapes samples_dir
        """
        # Format validation: only alphanumeric, dash, underscore, dot allowed
        if not sample_id or not all(c.isalnum() or c in '-_.' for c in sample_id):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sample ID format: {sample_id}"
            )

        # Build path and verify it stays within samples_dir
        sample_path = self.samples_dir / f"{sample_id}.wav"
        try:
            resolved = sample_path.resolve()
            samples_resolved = self.samples_dir.resolve()
            if not str(resolved).startswith(str(samples_resolved) + "/") and resolved != samples_resolved:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid sample path"
                )
        except (OSError, RuntimeError):
            raise HTTPException(
                status_code=400,
                detail="Invalid sample path"
            )

        return sample_path

    def _validate_speaker_wav(self, filename: str) -> Path:
        """
        Validate speaker wav filename and return safe path.

        Args:
            filename: Speaker sample filename (with .wav extension)

        Returns:
            Safe Path object within samples_dir

        Raises:
            HTTPException: If filename is invalid or path escapes samples_dir
        """
        if not filename or not filename.strip():
            raise HTTPException(
                status_code=400,
                detail="Empty speaker filename"
            )

        # Format validation: only alphanumeric, dash, underscore, dot allowed
        if not all(c.isalnum() or c in '-_.' for c in filename):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid speaker filename format: {filename}"
            )

        # Build path and verify it stays within samples_dir
        sample_path = self.samples_dir / filename
        try:
            resolved = sample_path.resolve()
            samples_resolved = self.samples_dir.resolve()
            if not str(resolved).startswith(str(samples_resolved) + "/") and resolved != samples_resolved:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid speaker path"
                )
        except (OSError, RuntimeError):
            raise HTTPException(
                status_code=400,
                detail="Invalid speaker path"
            )

        return sample_path

    def _setup_generate_route(self):
        """Setup TTS-specific /generate endpoint"""

        @self.app.post("/generate")
        async def generate_endpoint(request: GenerateRequest):
            """Generate TTS audio"""
            try:
                # Validate model is loaded and ready
                self._require_model_ready()

                # Validate text input
                if not request.text or not request.text.strip():
                    raise HTTPException(status_code=400, detail="Text cannot be empty")

                # Validate language
                if not request.language or not request.language.strip():
                    raise HTTPException(status_code=400, detail="Language cannot be empty")

                # Validate text length (from engine.yaml constraints)
                max_text_length = self._engine_config.get("constraints", {}).get("max_text_length")
                if max_text_length and len(request.text) > max_text_length:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Text too long ({len(request.text)} chars). "
                               f"{self.display_name} max is {max_text_length} chars. "
                               "Use text segmentation to split into smaller chunks."
                    )

                # Validate speaker samples exist (if provided)
                if request.tts_speaker_wav:
                    samples_to_check = (
                        [request.tts_speaker_wav] if isinstance(request.tts_speaker_wav, str)
                        else request.tts_speaker_wav
                    )
                    for sample in samples_to_check:
                        if sample and sample.strip():
                            # Validate filename format and path
                            sample_path = self._validate_speaker_wav(sample)
                            if not sample_path.exists():
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Speaker sample not found: {sample}"
                                )

                # Validate speaker samples required for cloning engines
                supports_cloning = self._engine_config.get("capabilities", {}).get(
                    "supports_speaker_cloning", False
                )
                if supports_cloning:
                    has_samples = bool(
                        request.tts_speaker_wav and
                        (isinstance(request.tts_speaker_wav, str) and request.tts_speaker_wav.strip()) or
                        (isinstance(request.tts_speaker_wav, list) and len(request.tts_speaker_wav) > 0)
                    )
                    if not has_samples:
                        raise HTTPException(
                            status_code=400,
                            detail=f"{self.display_name} requires speaker samples for voice cloning. "
                                   "Upload samples via /samples/upload and include sample IDs in tts_speaker_wav."
                        )

                # Clear previous error state on new request
                self.error_message = None
                self.status = "processing"

                # Normalize parameters (null/None -> empty dict, engine applies its defaults)
                parameters = request.parameters or {}

                # Format speaker for logging (basename only, not full path)
                if isinstance(request.tts_speaker_wav, str):
                    speaker_info = Path(request.tts_speaker_wav).name
                else:
                    speaker_info = f'{len(request.tts_speaker_wav)} samples'

                # Log TTS generation request
                logger.info(
                    f"[{self.engine_name}] Generating audio | "
                    f"Model: {self.current_model} | "
                    f"Language: {request.language} | "
                    f"Speaker: {speaker_info}"
                )

                # Call engine-specific implementation in thread pool
                # This prevents blocking the event loop during long generations
                loop = asyncio.get_event_loop()
                audio_bytes = await loop.run_in_executor(
                    self._executor,
                    lambda: self.generate_audio(
                        text=request.text,
                        language=request.language,
                        speaker_wav=request.tts_speaker_wav,
                        parameters=parameters
                    )
                )

                self.status = "ready"

                # Return binary audio
                return Response(content=audio_bytes, media_type="audio/wav")

            except HTTPException:
                self.status = "ready"  # HTTP errors are client errors, server is still ready
                raise
            except Exception as e:
                self._handle_processing_error(e, "generation")

    def _setup_sample_routes(self):
        """Setup speaker sample management endpoints"""

        @self.app.post("/samples/check", response_model=SampleCheckResponse)
        async def check_samples(request: SampleCheckRequest):
            """Check which samples exist in the engine's samples directory"""
            try:
                missing = []
                for sample_id in request.sample_ids:
                    # Validate sample_id format and path
                    sample_path = self._validate_sample_id(sample_id)
                    if not sample_path.exists():
                        missing.append(sample_id)

                logger.debug(
                    f"[{self.engine_name}] Sample check: "
                    f"{len(request.sample_ids)} requested, {len(missing)} missing"
                )
                return SampleCheckResponse(missing=missing)
            except Exception as e:
                logger.error(f"[{self.engine_name}] Sample check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/samples/upload/{sample_id}")
        async def upload_sample(sample_id: str, request: Request):
            """Upload a speaker sample to the engine's samples directory"""
            try:
                # Validate sample_id format and path
                dest = self._validate_sample_id(sample_id)

                # Read raw WAV bytes from request body
                content = await request.body()

                if not content:
                    raise HTTPException(status_code=400, detail="Empty request body")

                with open(dest, "wb") as f:
                    f.write(content)

                logger.info(
                    f"[{self.engine_name}] Sample uploaded: {sample_id}.wav "
                    f"({len(content)} bytes)"
                )
                return {"status": "ok", "sampleId": sample_id}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"[{self.engine_name}] Sample upload failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    # ============= TTS-Specific Abstract Method =============

    @abstractmethod
    def generate_audio(
        self,
        text: str,
        language: str,
        speaker_wav: Union[str, List[str]],
        parameters: Dict[str, Any]
    ) -> bytes:
        """
        Generate TTS audio (engine-specific)

        Args:
            text: Text to synthesize
            language: Language code (e.g., "en", "de")
            speaker_wav: Filename(s) in samples_dir (e.g., "uuid.wav" or ["uuid1.wav", "uuid2.wav"])
                         Engine implementations should resolve to full path via self.samples_dir
            parameters: Engine-specific parameters (temperature, speed, etc.)

        Returns:
            WAV audio as bytes

        Raises:
            Exception: If generation fails
        """
        pass
