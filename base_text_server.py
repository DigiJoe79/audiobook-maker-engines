"""
Base Text Server - Abstract FastAPI Server for Text Processing Engines

Extends BaseEngineServer with text processing functionality:
- /segment endpoint for text segmentation
- Segment response structure for TTS pipelines

Text processing engines inherit from this class and implement:
- load_model(model_name: str)
- get_available_models() -> List[ModelInfo]
- segment_text(text, language, max_length, min_length, mark_oversized) -> List[SegmentItem]
- unload_model()
"""
from fastapi import HTTPException
from typing import List, Optional
from abc import abstractmethod
from loguru import logger
import traceback
import asyncio

from base_server import BaseEngineServer, CamelCaseModel, ModelInfo as ModelInfo


# ============= Segment Models =============

class SegmentItem(CamelCaseModel):
    """
    Single text segment from segmentation.

    Status values:
    - "ok": Segment is within limits and ready for TTS
    - "failed": Segment exceeds max_length and needs manual review
    """
    text: str  # Segment text content
    start: int  # Start position in original text
    end: int  # End position in original text
    order_index: int  # Segment order (0-based)
    status: str = "ok"  # "ok" or "failed"

    # Optional metadata for failed segments
    length: Optional[int] = None  # Actual length (only for failed)
    max_length: Optional[int] = None  # Max allowed (only for failed)
    issue: Optional[str] = None  # Issue type (e.g., "sentence_too_long")


class SegmentResponse(CamelCaseModel):
    """Response model for text segmentation."""
    segments: List[SegmentItem]  # List of text segments
    total_segments: int  # Total number of segments
    total_characters: int  # Sum of all segment text lengths
    failed_count: int = 0  # Number of segments with status="failed"


class SegmentRequest(CamelCaseModel):
    """Request model for text segmentation."""
    text: str  # Text to segment
    language: str  # Language code (e.g., "en", "de")
    max_length: int = 250  # Maximum characters per segment
    min_length: int = 10  # Minimum characters (merge short sentences)
    mark_oversized: bool = True  # Mark sentences > max_length as "failed"


# ============= Base Text Server =============

class BaseTextServer(BaseEngineServer):
    """
    Abstract base class for text processing engines

    Extends BaseEngineServer with:
    - /segment endpoint for text segmentation
    - Segment response structure

    Text engines implement:
    - load_model(model_name) - Load NLP model (often language-specific)
    - get_available_models() - Return available models/languages
    - segment_text(text, language, max_length, min_length, mark_oversized) -> List[SegmentItem]
    - unload_model() - Unload model and free resources
    """

    def __init__(self, engine_name: str, display_name: str, config_path: Optional[str] = None):
        """
        Initialize text processing engine server

        Args:
            engine_name: Engine identifier (e.g., "spacy", "nltk")
            display_name: Human-readable name (e.g., "spaCy Text Processor")
            config_path: Optional path to engine.yaml (for /info endpoint)
        """
        super().__init__(engine_name, display_name, config_path)

        # Setup text-specific routes
        self._setup_segment_route()

        logger.info(f"[{self.engine_name}] Text server initialized")

    def _setup_segment_route(self):
        """Setup text-specific /segment endpoint"""

        @self.app.post("/segment", response_model=SegmentResponse)
        async def segment_endpoint(request: SegmentRequest):
            """Segment text into TTS-ready chunks"""
            try:
                # Return 503 if model is still loading (non-blocking response)
                if self.status == "loading":
                    raise HTTPException(
                        status_code=503,
                        detail="Model loading in progress. Retry after loading completes."
                    )

                if not self.model_loaded:
                    raise HTTPException(status_code=400, detail="Model not loaded")

                # Validate text input
                if not request.text or not request.text.strip():
                    raise HTTPException(status_code=400, detail="Text cannot be empty")

                # Validate language
                if not request.language or not request.language.strip():
                    raise HTTPException(status_code=400, detail="Language cannot be empty")

                # Validate length parameters
                if request.max_length <= 0:
                    raise HTTPException(status_code=400, detail="max_length must be positive")
                if request.min_length < 0:
                    raise HTTPException(status_code=400, detail="min_length cannot be negative")
                if request.min_length >= request.max_length:
                    raise HTTPException(
                        status_code=400,
                        detail="min_length must be less than max_length"
                    )

                logger.info(
                    f"📝 [{self.engine_name}] Segmenting text | "
                    f"Model: {self.current_model} | "
                    f"Language: {request.language} | "
                    f"Length: {len(request.text)} chars | "
                    f"Max segment: {request.max_length} chars"
                )

                self.status = "processing"

                # Call engine-specific implementation in thread pool
                # This prevents blocking the event loop during segmentation
                loop = asyncio.get_event_loop()
                segments = await loop.run_in_executor(
                    self._executor,
                    lambda: self.segment_text(
                        text=request.text,
                        language=request.language,
                        max_length=request.max_length,
                        min_length=request.min_length,
                        mark_oversized=request.mark_oversized
                    )
                )

                self.status = "ready"

                # Calculate totals
                total_chars = sum(len(seg.text) for seg in segments)
                failed_count = sum(1 for seg in segments if seg.status == "failed")

                logger.info(
                    f"[{self.engine_name}] Segmentation complete | "
                    f"Segments: {len(segments)} | "
                    f"Failed: {failed_count} | "
                    f"Total chars: {total_chars}"
                )

                return SegmentResponse(
                    segments=segments,
                    total_segments=len(segments),
                    total_characters=total_chars,
                    failed_count=failed_count
                )

            except HTTPException:
                raise
            except Exception as e:
                self.status = "error"
                self.error_message = str(e)
                logger.error(f"[{self.engine_name}] Segmentation failed: {e}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

    # ============= Text-Specific Abstract Method =============

    @abstractmethod
    def segment_text(
        self,
        text: str,
        language: str,
        max_length: int,
        min_length: int,
        mark_oversized: bool
    ) -> List[SegmentItem]:
        """
        Segment text into chunks for TTS generation (engine-specific)

        Key principles for TTS segmentation:
        1. NEVER split sentences in the middle (breaks TTS naturalness)
        2. Combine short sentences up to max_length for better flow
        3. Mark sentences > max_length as "failed" for manual review

        Args:
            text: Input text to segment
            language: Language code for NLP model selection
            max_length: Maximum characters per segment (TTS engine limit)
            min_length: Minimum characters (merge short sentences)
            mark_oversized: Mark sentences exceeding max_length as "failed"

        Returns:
            List of SegmentItem objects

        Raises:
            Exception: If segmentation fails
        """
        pass
