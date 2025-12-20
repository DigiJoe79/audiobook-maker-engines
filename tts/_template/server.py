"""
Template TTS Engine Server

Copy this template to create a new TTS engine.
Replace TODO comments with your engine-specific code.

Inherits from BaseTTSServer which provides:
- /health - Health check
- /load - Load model
- /models - List available models
- /info - Engine metadata from engine.yaml
- /generate - Generate TTS audio
- /samples/check - Check speaker samples
- /samples/upload - Upload speaker samples
- /shutdown - Graceful shutdown

Model Management Pattern:
- Always read models from self.models_dir
- Download on-demand models to self.external_models_dir, then symlink
- Baked-in models take precedence over external models

See docs/model-management.md for the complete standard.
"""
from pathlib import Path
from typing import Dict, Any, Union, List
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_tts_server import BaseTTSServer
from base_server import ModelInfo, ModelField
from loguru import logger


class TemplateServer(BaseTTSServer):
    """TODO: Your Engine Name - Description"""

    def __init__(self):
        super().__init__(
            engine_name="template-tts",  # TODO: Change to your engine name (must match engine.yaml)
            display_name="Template TTS"  # TODO: Change to display name
        )

        # TODO: Initialize your engine-specific state
        self.model = None
        self.default_model = "default"
        self.device = "cpu"  # or "cuda" for GPU engines

        logger.info(f"[{self.engine_name}] Engine initialized")

    def load_model(self, model_name: str) -> None:
        """
        Load a TTS model into memory.

        Model Management Pattern:
        1. Check if model exists in self.models_dir (baked-in or symlinked)
        2. If not, download to self.external_models_dir and create symlink
        3. Load model from self.models_dir

        Args:
            model_name: Model identifier (e.g., 'default', 'v2.0.3')

        Raises:
            ValueError: If model_name is invalid
            Exception: If loading fails
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
            # Example:
            # download_model_files(model_name, external_path)

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)
            logger.info(f"[{self.engine_name}] Model downloaded to {external_path}")

        # TODO: Load the model
        # Example:
        # self.model = YourModelClass.load(model_path)
        # self.model.to(self.device)

        # Update state after loading
        self.current_model = model_name
        self.model_loaded = True

        logger.info(f"[{self.engine_name}] Model '{model_name}' loaded successfully")

        # Remove this line after implementing:
        raise NotImplementedError("TODO: Implement load_model()")

    def generate_audio(
        self,
        text: str,
        language: str,
        speaker_wav: Union[str, List[str]],
        parameters: Dict[str, Any]
    ) -> bytes:
        """
        Generate TTS audio from text.

        Args:
            text: Text to synthesize
            language: Language code (e.g., 'en', 'de')
            speaker_wav: Path(s) to speaker sample file(s)
            parameters: Engine-specific parameters (speed, temperature, etc.)

        Returns:
            WAV audio as bytes

        Raises:
            Exception: If generation fails
        """
        # TODO: Implement audio generation
        # Example:
        # audio_array = self.model.synthesize(
        #     text=text,
        #     language=language,
        #     speaker_wav=speaker_wav,
        #     **parameters
        # )
        # return self._convert_to_wav_bytes(audio_array)

        raise NotImplementedError("TODO: Implement generate_audio()")

    def unload_model(self) -> None:
        """Free model resources."""
        logger.info(f"[{self.engine_name}] Unloading model")

        if self.model is not None:
            # TODO: Add engine-specific cleanup if needed
            del self.model
            self.model = None

        self.current_model = None
        self.model_loaded = False

        # Force garbage collection
        import gc
        gc.collect()

    def get_available_models(self) -> List[ModelInfo]:
        """
        Return list of available models by scanning models directory.

        Returns:
            List of ModelInfo objects with name, display_name, and fields
        """
        models = []

        # Scan models directory for available models
        if self.models_dir.exists():
            for model_path in sorted(self.models_dir.iterdir()):
                if model_path.is_dir() or model_path.is_symlink():
                    model_name = model_path.name
                    models.append(
                        ModelInfo(
                            name=model_name,
                            display_name=model_name.replace("-", " ").title(),
                            languages=["en", "de"],  # TODO: Set actual languages
                            fields=[
                                # TODO: Add model-specific metadata
                                # ModelField(key="size_mb", value=500, field_type="number"),
                            ]
                        )
                    )

        # Fallback if no models found
        if not models:
            models.append(
                ModelInfo(
                    name="default",
                    display_name="Default Model",
                    languages=["en", "de"],
                    fields=[]
                )
            )

        return models

    def get_package_version(self) -> str:
        """Return engine/package version."""
        # TODO: Return actual version from your TTS library
        # Example:
        # import your_tts_library
        # return your_tts_library.__version__
        return "1.0.0"

    # ============= Helper Methods =============

    def _convert_to_wav_bytes(self, audio_array, sample_rate: int = 24000) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        import io
        import wave
        import numpy as np

        # Normalize to int16
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Write to WAV
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        return buffer.getvalue()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Template TTS Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = TemplateServer()
    server.run(port=args.port, host=args.host)
