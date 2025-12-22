"""
Template STT Engine Server

Inherits from BaseQualityServer for consistent lifecycle management.
Provides STT-specific analysis endpoint:
- /analyze - Transcribe audio and return Generic Quality Format

Standard endpoints from BaseEngineServer:
- /health - Health check
- /load - Load model
- /models - List available models
- /info - Engine metadata from engine.yaml
- /shutdown - Graceful shutdown

Model Management Pattern:
- Always read models from self.models_dir
- Download on-demand models to self.external_models_dir, then symlink
- Baked-in models take precedence over external models

See docs/model-management.md for the complete standard.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for base_server imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base_quality_server import (
    BaseQualityServer,
    QualityThresholds,
    QualityField,
    QualityInfoBlockItem,
    AnalyzeResult,
    PronunciationRuleData
)
from base_server import ModelInfo
from loguru import logger


class TemplateSTTServer(BaseQualityServer):
    """
    Template STT engine server.

    TODO: Rename this class to match your engine (e.g., MySTTServer).

    STT engines transcribe audio and return quality metrics based on
    transcription confidence and accuracy.
    """

    def __init__(self):
        super().__init__(
            engine_name="template-stt",  # TODO: Change to your engine name (must match engine.yaml)
            display_name="Template STT",  # TODO: Change display name
            engine_type="stt"
        )

        # TODO: Initialize your STT model here
        self.model = None
        self.default_model = "base"
        self.device = "cpu"  # or "cuda" for GPU engines

        logger.info(f"[{self.engine_name}] STT engine initialized")

    def load_model(self, model_name: str) -> None:
        """
        Load STT model.

        Model Management Pattern:
        1. Check if model exists in self.models_dir (baked-in or symlinked)
        2. If not, download to self.external_models_dir and create symlink
        3. Load model from self.models_dir
        """
        logger.info(f"[{self.engine_name}] Loading model: {model_name}")

        model_path = self.models_dir / model_name

        # Check if model exists (baked-in or already symlinked)
        if not model_path.exists():
            logger.info(f"[{self.engine_name}] Model not found, downloading...")

            # Download to external_models for persistence
            external_path = self.external_models_dir / model_name
            external_path.mkdir(parents=True, exist_ok=True)

            # TODO: Download model files to external_path
            # Example for Whisper:
            # import whisper
            # whisper.load_model(model_name, download_root=str(external_path))

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)
            logger.info(f"[{self.engine_name}] Model downloaded to {external_path}")

        # TODO: Load your model here
        # Example:
        # self.model = whisper.load_model(model_name)

        self.model = True  # Placeholder - remove after implementing
        self.model_loaded = True
        self.current_model = model_name

        logger.info(f"[{self.engine_name}] Model '{model_name}' loaded successfully")

    def analyze_audio(
        self,
        audio_bytes: bytes,
        language: str,
        thresholds: QualityThresholds,
        expected_text: Optional[str] = None,
        pronunciation_rules: Optional[List[PronunciationRuleData]] = None
    ) -> AnalyzeResult:
        """
        Transcribe audio and return quality metrics.

        TODO: Implement your transcription logic here.

        Args:
            audio_bytes: Raw audio file bytes (WAV format)
            language: Language code (e.g., "en", "de")
            thresholds: Quality thresholds (mainly used by audio engines)
            expected_text: Original segment text for comparison
            pronunciation_rules: Active pronunciation rules to filter false positives

        Returns:
            AnalyzeResult with transcription quality metrics
        """
        # TODO: Replace this placeholder implementation

        # Example: Transcribe audio
        # transcription = self.model.transcribe(audio_bytes, language)
        # confidence = self.model.get_confidence()

        # Placeholder values
        transcription = "This is a placeholder transcription."
        confidence = 85
        duration = 2.5
        word_count = len(transcription.split())

        # Build quality fields
        fields = [
            QualityField(
                key="quality.stt.confidence",
                value=confidence,
                type="percent"
            ),
            QualityField(
                key="quality.stt.transcription",
                value=transcription,
                type="text"
            ),
            QualityField(
                key="quality.stt.language",
                value=language,
                type="string"
            ),
            QualityField(
                key="quality.stt.duration",
                value=duration,
                type="seconds"
            ),
            QualityField(
                key="quality.stt.wordCount",
                value=word_count,
                type="number"
            ),
        ]

        # Check for issues
        info_blocks = {}
        issues = []

        # TODO: Add your quality checks here
        if confidence < 70:
            issues.append(QualityInfoBlockItem(
                text="quality.stt.lowConfidence",
                severity="warning",
                details={"confidence": confidence}
            ))

        if issues:
            info_blocks["issues"] = issues

        logger.info(
            f"[{self.engine_name}] Transcription complete | "
            f"Confidence: {confidence}% | "
            f"Words: {word_count} | "
            f"Duration: {duration}s"
        )

        return AnalyzeResult(
            quality_score=confidence,
            fields=fields,
            info_blocks=info_blocks,
            top_label="quality.stt.templateEngine"  # TODO: Change i18n key
        )

    def unload_model(self) -> None:
        """Unload model and free resources."""
        logger.info(f"[{self.engine_name}] Unloading model")

        if self.model is not None:
            self.model = None

        self.model_loaded = False
        self.current_model = None

        # Force garbage collection
        import gc
        gc.collect()

    def get_available_models(self) -> List[ModelInfo]:
        """
        Return available STT models by scanning models directory.
        """
        models = []

        # Scan models directory
        if self.models_dir.exists():
            for model_path in sorted(self.models_dir.iterdir()):
                if model_path.is_dir() or model_path.is_symlink():
                    model_name = model_path.name
                    models.append(
                        ModelInfo(
                            name=model_name,
                            display_name=model_name.replace("-", " ").title(),
                            languages=["en", "de"],  # TODO: Set actual languages
                            fields=[]
                        )
                    )

        # Fallback if no models found
        if not models:
            models.append(
                ModelInfo(name="base", display_name="Base Model", languages=["en", "de"], fields=[])
            )
            models.append(
                ModelInfo(name="large", display_name="Large Model", languages=["en", "de"], fields=[])
            )

        return models

    def get_package_version(self) -> str:
        """Return engine/package version."""
        return "1.0.0"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Template STT Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    server = TemplateSTTServer()
    server.run(port=args.port, host=args.host)
