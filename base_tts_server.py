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
import traceback

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

    def _setup_generate_route(self):
        """Setup TTS-specific /generate endpoint"""

        @self.app.post("/generate")
        async def generate_endpoint(request: GenerateRequest):
            """Generate TTS audio"""
            try:
                if not self.model_loaded:
                    raise HTTPException(status_code=400, detail="Model not loaded")

                # Validate text input
                if not request.text or not request.text.strip():
                    raise HTTPException(status_code=400, detail="Text cannot be empty")

                # Validate language
                if not request.language or not request.language.strip():
                    raise HTTPException(status_code=400, detail="Language cannot be empty")

                # Validate speaker samples exist (if provided)
                if request.tts_speaker_wav:
                    samples_to_check = (
                        [request.tts_speaker_wav] if isinstance(request.tts_speaker_wav, str)
                        else request.tts_speaker_wav
                    )
                    for sample in samples_to_check:
                        if sample and sample.strip():
                            sample_path = self.samples_dir / sample
                            if not sample_path.exists():
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Speaker sample not found: {sample}"
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

                # Call engine-specific implementation
                audio_bytes = self.generate_audio(
                    text=request.text,
                    language=request.language,
                    speaker_wav=request.tts_speaker_wav,
                    parameters=parameters
                )

                self.status = "ready"

                # Return binary audio
                return Response(content=audio_bytes, media_type="audio/wav")

            except HTTPException:
                self.status = "ready"  # HTTP errors are client errors, server is still ready
                raise
            except Exception as e:
                self.status = "error"
                self.error_message = str(e)
                logger.error(f"[{self.engine_name}] Generation failed: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_sample_routes(self):
        """Setup speaker sample management endpoints"""

        @self.app.post("/samples/check", response_model=SampleCheckResponse)
        async def check_samples(request: SampleCheckRequest):
            """Check which samples exist in the engine's samples directory"""
            missing = []
            for sample_id in request.sample_ids:
                sample_path = self.samples_dir / f"{sample_id}.wav"
                if not sample_path.exists():
                    missing.append(sample_id)

            logger.debug(
                f"[{self.engine_name}] Sample check: "
                f"{len(request.sample_ids)} requested, {len(missing)} missing"
            )
            return SampleCheckResponse(missing=missing)

        @self.app.post("/samples/upload/{sample_id}")
        async def upload_sample(sample_id: str, request: Request):
            """Upload a speaker sample to the engine's samples directory"""
            dest = self.samples_dir / f"{sample_id}.wav"

            # Read raw WAV bytes from request body
            content = await request.body()
            with open(dest, "wb") as f:
                f.write(content)

            logger.info(
                f"[{self.engine_name}] Sample uploaded: {sample_id}.wav "
                f"({len(content)} bytes)"
            )
            return {"status": "ok", "sampleId": sample_id}

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
