"""
Debug STT Engine Server

A lightweight STT engine for testing and development that behaves
exactly like a real STT engine but returns mock transcription results.

Features:
- Returns perfect transcription (expected_text if provided)
- Simulates speech detection with configurable results
- No GPU required, minimal dependencies
- Instant model loading (no actual model)
"""
from pathlib import Path
from typing import List, Optional
import sys
import wave
import io

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_quality_server import (
    BaseQualityServer,
    QualityThresholds,
    QualityField,
    QualityInfoBlockItem,
    AnalyzeResult,
    PronunciationRuleData
)
from base_server import ModelInfo, ModelField
from loguru import logger


class DebugSTTServer(BaseQualityServer):
    """Debug STT Engine - Returns mock transcription for development"""

    def __init__(self):
        super().__init__(
            engine_name="debug-stt",
            display_name="Debug STT",
            engine_type="stt"
        )

        # No actual model needed
        self.model = None
        self.default_model = "default"
        # Note: self.device property is provided by BaseEngineServer (auto-detects cuda/cpu)

        logger.info("[debug-stt] Debug STT Engine initialized (mock transcription)")

    def load_model(self, model_name: str) -> None:
        """
        Simulate model loading with on-demand download behavior.

        Args:
            model_name: Model identifier
        """
        logger.info(f"[debug-stt] Loading model: {model_name}")

        model_path = self.models_dir / model_name

        # Check if model exists (baked-in or symlinked from external)
        if not model_path.exists():
            logger.info(f"[debug-stt] Model '{model_name}' not found, downloading...")

            # Simulate download to external_models (for persistence)
            external_path = self.external_models_dir / model_name
            external_path.mkdir(parents=True, exist_ok=True)

            # Create dummy model file
            model_file = external_path / "model.json"
            model_file.write_text(
                f'{{\n  "name": "{model_name}",\n  "type": "mock_stt",\n'
                f'  "version": "1.0.0",\n  "description": "Auto-downloaded debug model"\n}}\n'
            )

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)

            logger.info(f"[debug-stt] Model '{model_name}' downloaded to {external_path}")

        # Simulate successful model load
        self.current_model = model_name
        self.model_loaded = True

        logger.info(f"[debug-stt] Model '{model_name}' loaded successfully")

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds,
        expected_text: Optional[str] = None,
        pronunciation_rules: Optional[List[PronunciationRuleData]] = None
    ) -> AnalyzeResult:
        """
        Analyze audio and return mock transcription results.

        Always returns perfect quality score with expected_text as transcription.

        Args:
            audio_bytes: Raw audio file bytes (WAV format)
            language: Language code
            thresholds: Quality thresholds
            expected_text: Original text for comparison (returned as transcription)
            pronunciation_rules: Pronunciation rules (ignored)

        Returns:
            AnalyzeResult with perfect score
        """
        # Parse WAV to get duration
        duration_sec = self._get_wav_duration(audio_bytes)

        # Use expected_text as "transcription" or generate placeholder
        transcription = expected_text if expected_text else "Debug transcription text"

        logger.debug(
            f"[debug-stt] Analyzing audio: duration={duration_sec:.2f}s, "
            f"language={language}, text_len={len(transcription)}"
        )

        # Build quality fields
        fields = [
            QualityField(key="quality.stt.transcription", value=transcription, type="text"),
            QualityField(key="quality.stt.confidence", value=100, type="percent"),
            QualityField(key="quality.stt.duration", value=round(duration_sec, 2), type="seconds"),
            QualityField(key="quality.stt.language", value=language, type="string"),
        ]

        # Perfect match info
        info_blocks = {
            "match": [
                QualityInfoBlockItem(
                    text="quality.stt.perfectMatch",
                    severity="info",
                    details={"similarity": 100}
                )
            ]
        }

        return AnalyzeResult(
            quality_score=100,  # Perfect score
            fields=fields,
            info_blocks=info_blocks,
            top_label="quality.stt.analysis"
        )

    def _get_wav_duration(self, audio_bytes: bytes) -> float:
        """Parse WAV bytes to get duration in seconds."""
        try:
            buffer = io.BytesIO(audio_bytes)
            with wave.open(buffer, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate) if rate > 0 else 0.0
        except Exception:
            return 0.0

    def unload_model(self) -> None:
        """Free resources (nothing to free for debug engine)."""
        logger.info("[debug-stt] Unloading model")
        self.model = None
        # Note: gc.collect() and state reset are handled by base_server.py

    def get_available_models(self) -> List[ModelInfo]:
        """Return available debug models by scanning models directory."""
        models = []

        if self.models_dir.exists():
            for model_path in sorted(self.models_dir.iterdir()):
                if model_path.is_dir() or model_path.is_symlink():
                    model_name = model_path.name
                    models.append(
                        ModelInfo(
                            name=model_name,
                            display_name=f"{model_name.title()} (Mock STT)",
                            languages=["de", "en", "fr", "es", "it"],
                            fields=[
                                ModelField(key="type", value="mock_stt", field_type="string"),
                                ModelField(key="size_mb", value=0, field_type="number"),
                            ]
                        )
                    )

        return models

    def get_package_version(self) -> str:
        """Return engine version."""
        return "1.0.0"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug STT Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = DebugSTTServer()
    server.run(port=args.port, host=args.host)
