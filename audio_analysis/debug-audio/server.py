"""
Debug Audio Analysis Engine Server

A lightweight audio analysis engine for testing and development that behaves
exactly like a real audio analysis engine but returns mock quality metrics.

Features:
- Returns perfect audio quality scores
- Simulates VAD (Voice Activity Detection) with high speech ratio
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


class DebugAudioServer(BaseQualityServer):
    """Debug Audio Engine - Returns mock audio quality metrics for development"""

    def __init__(self):
        super().__init__(
            engine_name="debug-audio",
            display_name="Debug Audio",
            engine_type="audio"
        )

        # No actual model needed
        self.model = None
        self.default_model = "default"
        # Note: self.device property is provided by BaseEngineServer (auto-detects cuda/cpu)

        logger.info("[debug-audio] Debug Audio Engine initialized (mock analysis)")

    def load_model(self, model_name: str) -> None:
        """
        Simulate model loading with on-demand download behavior.

        Args:
            model_name: Model identifier
        """
        logger.info(f"[debug-audio] Loading model: {model_name}")

        model_path = self.models_dir / model_name

        # Check if model exists (baked-in or symlinked from external)
        if not model_path.exists():
            logger.info(f"[debug-audio] Model '{model_name}' not found, downloading...")

            # Simulate download to external_models (for persistence)
            external_path = self.external_models_dir / model_name
            external_path.mkdir(parents=True, exist_ok=True)

            # Create dummy model file
            model_file = external_path / "model.json"
            model_file.write_text(
                f'{{\n  "name": "{model_name}",\n  "type": "mock_vad",\n'
                f'  "version": "1.0.0",\n  "description": "Auto-downloaded debug model"\n}}\n'
            )

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)

            logger.info(f"[debug-audio] Model '{model_name}' downloaded to {external_path}")

        # Simulate successful model load
        self.current_model = model_name
        self.model_loaded = True

        logger.info(f"[debug-audio] Model '{model_name}' loaded successfully")

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds,
        expected_text: Optional[str] = None,
        pronunciation_rules: Optional[List[PronunciationRuleData]] = None
    ) -> AnalyzeResult:
        """
        Analyze audio and return mock quality metrics.

        Always returns perfect quality score with ideal audio metrics.

        Args:
            audio_bytes: Raw audio file bytes (WAV format)
            language: Language code (ignored for audio analysis)
            thresholds: Quality thresholds (used to generate ideal values)
            expected_text: Not used for audio analysis
            pronunciation_rules: Not used for audio analysis

        Returns:
            AnalyzeResult with perfect score
        """
        # Parse WAV to get duration and basic info
        duration_sec, sample_rate, channels = self._get_wav_info(audio_bytes)

        logger.debug(
            f"[debug-audio] Analyzing audio: duration={duration_sec:.2f}s, "
            f"sample_rate={sample_rate}, channels={channels}"
        )

        # Generate ideal metrics within threshold ranges
        speech_ratio = (thresholds.speech_ratio_ideal_min + thresholds.speech_ratio_ideal_max) / 2
        max_silence_ms = int(thresholds.max_silence_duration_warning * 0.5)  # Well below warning
        avg_volume_db = -20.0  # Good average volume
        peak_volume_db = -3.0  # Below clipping

        # Build quality fields
        fields = [
            QualityField(key="quality.audio.speechRatio", value=round(speech_ratio, 1), type="percent"),
            QualityField(key="quality.audio.maxSilence", value=max_silence_ms, type="number"),
            QualityField(key="quality.audio.avgVolume", value=f"{avg_volume_db:.1f} dB", type="string"),
            QualityField(key="quality.audio.peakVolume", value=f"{peak_volume_db:.1f} dB", type="string"),
            QualityField(key="quality.audio.duration", value=round(duration_sec, 2), type="seconds"),
            QualityField(key="quality.audio.sampleRate", value=sample_rate, type="number"),
        ]

        # Perfect quality info
        info_blocks = {
            "quality": [
                QualityInfoBlockItem(
                    text="quality.audio.excellent",
                    severity="info",
                    details={"speech_ratio": speech_ratio, "max_silence_ms": max_silence_ms}
                )
            ]
        }

        return AnalyzeResult(
            quality_score=95,  # Excellent score (not 100 to be realistic)
            fields=fields,
            info_blocks=info_blocks,
            top_label="quality.audio.analysis"
        )

    def _get_wav_info(self, audio_bytes: bytes) -> tuple:
        """Parse WAV bytes to get duration, sample rate, and channels."""
        try:
            buffer = io.BytesIO(audio_bytes)
            with wave.open(buffer, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                duration = frames / float(rate) if rate > 0 else 0.0
                return duration, rate, channels
        except Exception:
            return 0.0, 0, 0

    def unload_model(self) -> None:
        """Free resources (nothing to free for debug engine)."""
        logger.info("[debug-audio] Unloading model")
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
                            display_name=f"{model_name.title()} (Mock VAD)",
                            languages=[],  # Audio analysis is language-independent
                            fields=[
                                ModelField(key="type", value="mock_vad", field_type="string"),
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

    parser = argparse.ArgumentParser(description="Debug Audio Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = DebugAudioServer()
    server.run(port=args.port, host=args.host)
