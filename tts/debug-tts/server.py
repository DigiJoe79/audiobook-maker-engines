"""
Debug TTS Engine Server

A lightweight TTS engine for testing and development that behaves
exactly like a real TTS engine but outputs pleasant test audio.

Features:
- Generates sine wave tones with text-proportional duration
- Different tones based on parameters (pitch, speed)
- No GPU required, minimal dependencies
- Instant model loading (no actual model)
"""
from pathlib import Path
from typing import Dict, Any, Union, List
import sys
import io
import wave
import math
import struct

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_tts_server import BaseTTSServer
from base_server import ModelInfo, ModelField
from loguru import logger


class DebugTTSServer(BaseTTSServer):
    """Debug TTS Engine - Generates test audio for development"""

    # Audio constants
    SAMPLE_RATE = 24000  # Standard TTS sample rate
    BASE_FREQUENCY = 440.0  # A4 note (pleasant base tone)

    def __init__(self):
        super().__init__(
            engine_name="debug-tts",
            display_name="Debug TTS"
        )

        # No actual model needed
        self.model = None
        self.default_model = "default"
        self.device = "cpu"

        logger.info("[debug-tts] Debug TTS Engine initialized (test audio generator)")

    def load_model(self, model_name: str) -> None:
        """
        Simulate model loading with on-demand download behavior.

        If model doesn't exist, creates it in external_models (simulating download).
        This demonstrates the on-demand download pattern (like Whisper).

        Args:
            model_name: Model identifier
        """
        logger.info(f"[debug-tts] Loading model: {model_name}")

        model_path = self.models_dir / model_name

        # Check if model exists (baked-in or symlinked from external)
        if not model_path.exists():
            logger.info(f"[debug-tts] Model '{model_name}' not found, downloading...")

            # Simulate download to external_models (for persistence)
            external_path = self.external_models_dir / model_name
            external_path.mkdir(parents=True, exist_ok=True)

            # Create dummy model file
            model_file = external_path / "model.json"
            model_file.write_text(f'{{\n  "name": "{model_name}",\n  "type": "sine_wave",\n  "version": "1.0.0",\n  "description": "Auto-downloaded debug model"\n}}\n')

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)

            logger.info(f"[debug-tts] Model '{model_name}' downloaded to {external_path}")

        # Simulate successful model load
        self.current_model = model_name
        self.model_loaded = True

        logger.info(f"[debug-tts] Model '{model_name}' loaded successfully")

    def generate_audio(
        self,
        text: str,
        language: str,
        speaker_wav: Union[str, List[str]],
        parameters: Dict[str, Any]
    ) -> bytes:
        """
        Generate test audio based on text length.

        The audio duration is proportional to text length, simulating
        real TTS behavior. Parameters affect the generated tone.

        Args:
            text: Text to "synthesize" (length determines duration)
            language: Language code (affects tone frequency slightly)
            speaker_wav: Speaker sample path(s) (ignored)
            parameters: Engine parameters (pitch, speed affect output)

        Returns:
            WAV audio as bytes
        """
        from fastapi import HTTPException

        if not self.model_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Call POST /load first."
            )

        # Calculate duration based on text length
        # Assume ~10 characters per second of speech (typical speaking rate)
        chars_per_second = parameters.get("speed", 1.0) * 10.0
        duration = max(0.5, len(text) / chars_per_second)

        # Limit duration to reasonable bounds
        duration = min(duration, 60.0)  # Max 60 seconds

        # Get tone parameters
        pitch_factor = parameters.get("pitch", 1.0)
        frequency = self.BASE_FREQUENCY * pitch_factor

        # Slightly vary frequency based on language for variety
        language_offsets = {
            "de": 0.95,   # Slightly lower for German
            "en": 1.0,    # Default for English
            "fr": 1.05,   # Slightly higher for French
            "es": 1.02,   # Spanish
            "it": 1.03,   # Italian
        }
        frequency *= language_offsets.get(language, 1.0)

        logger.debug(
            f"[debug-tts] Generating audio: "
            f"duration={duration:.2f}s, freq={frequency:.1f}Hz, "
            f"text_len={len(text)}"
        )

        # Generate the audio
        audio_bytes = self._generate_sine_wave(frequency, duration)

        return audio_bytes

    def _generate_sine_wave(self, frequency: float, duration: float) -> bytes:
        """
        Generate a pleasant sine wave tone as WAV bytes.

        Uses a soft attack/release envelope to avoid clicks.

        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds

        Returns:
            WAV audio as bytes
        """
        num_samples = int(self.SAMPLE_RATE * duration)

        # Generate samples
        samples = []
        for i in range(num_samples):
            t = i / self.SAMPLE_RATE

            # Sine wave at specified frequency
            sample = math.sin(2.0 * math.pi * frequency * t)

            # Add soft harmonics for a richer tone (like a soft synth)
            sample += 0.3 * math.sin(2.0 * math.pi * frequency * 2 * t)  # 2nd harmonic
            sample += 0.1 * math.sin(2.0 * math.pi * frequency * 3 * t)  # 3rd harmonic

            # Apply envelope (fade in/out to avoid clicks)
            fade_samples = int(self.SAMPLE_RATE * 0.05)  # 50ms fade
            if i < fade_samples:
                # Fade in
                sample *= i / fade_samples
            elif i > num_samples - fade_samples:
                # Fade out
                sample *= (num_samples - i) / fade_samples

            # Normalize amplitude (0.5 to avoid clipping with harmonics)
            sample *= 0.5

            samples.append(sample)

        # Convert to 16-bit PCM
        audio_data = b''
        for sample in samples:
            # Clamp and convert to int16
            clamped = max(-1.0, min(1.0, sample))
            int_sample = int(clamped * 32767)
            audio_data += struct.pack('<h', int_sample)

        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_data)

        return buffer.getvalue()

    def unload_model(self) -> None:
        """Free resources (nothing to free for debug engine)."""
        logger.info("[debug-tts] Unloading model")
        self.model = None
        self.current_model = None
        self.model_loaded = False

    def get_available_models(self) -> List[ModelInfo]:
        """
        Return available debug models by scanning models directory.

        Returns:
            List of ModelInfo objects
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
                            display_name=f"{model_name.title()} (Sine Wave)",
                            languages=["de", "en", "fr", "es", "it"],
                            fields=[
                                ModelField(key="type", value="sine_wave", field_type="string"),
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

    parser = argparse.ArgumentParser(description="Debug TTS Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = DebugTTSServer()
    server.run(port=args.port, host=args.host)
