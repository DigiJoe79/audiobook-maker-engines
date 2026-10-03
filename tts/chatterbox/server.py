"""
Chatterbox Multilingual TTS Engine Server

Supports 23 languages with high-quality voice cloning and multilingual TTS.
Based on Chatterbox by Resemble AI.
"""
from pathlib import Path
from typing import Dict, Any, Union, List
import sys
import io
import warnings
import logging  # Only for suppressing third-party library logs
import os
import threading
from time import perf_counter
import numpy as np
import torch

# Suppress ALL warnings (library deprecations, FutureWarnings, etc.)
warnings.filterwarnings('ignore')

# Suppress transformers/diffusers specific warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DIFFUSERS_VERBOSITY'] = 'error'

# Suppress chatterbox internal warnings (token repetition, alignment, etc.)
# NOTE: Using standard logging module ONLY for suppressing third-party library logs
logging.getLogger('chatterbox').setLevel(logging.ERROR)
logging.getLogger('chatterbox.models.t3.inference.alignment_stream_analyzer').setLevel(logging.ERROR)

# Suppress perth watermarker output
logging.getLogger('perth').setLevel(logging.ERROR)

# Add parent directory to path to import base_server
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_tts_server import BaseTTSServer, ModelInfo  # noqa: E402
from loguru import logger  # noqa: E402
from conditioning_cache import ConditionalsCache, prepare_or_reuse_conditionals  # noqa: E402
from text_chunking import MAX_TOTAL_TEXT_CHARS, split_text  # noqa: E402

# Import Chatterbox (after warnings/logging suppression)
import contextlib  # noqa: E402
import chatterbox  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

# Chatterbox package version (for health endpoint)
CHATTERBOX_VERSION = getattr(chatterbox, '__version__', 'unknown')


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes"""
    import scipy.io.wavfile

    # Normalize audio to int16 range
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        # Ensure audio is in [-1, 1] range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        # Convert to int16
        audio_array = (audio_array * 32767).astype(np.int16)

    # Write to bytes buffer
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, sample_rate, audio_array)
    wav_buffer.seek(0)
    return wav_buffer.read()


class ChatterboxServer(BaseTTSServer):
    """Chatterbox Multilingual TTS Engine"""

    def __init__(self):
        # Engine state (before super().__init__)
        self.model = None
        self._model_lock = threading.RLock()
        self._conditionals_cache = ConditionalsCache(max_entries=2)

        super().__init__(
            engine_name="chatterbox",
            display_name="Chatterbox",
            config_path=str(Path(__file__).parent / "engine.yaml")
        )
        # Note: self.device property is provided by BaseEngineServer (auto-detects cuda/cpu)
        logger.info(f"[chatterbox] Running on device: {self.device}")

    # Chatterbox supported languages (23 languages)
    SUPPORTED_LANGUAGES = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"
    ]

    def get_available_models(self) -> List[ModelInfo]:
        """Return available Chatterbox models (only 'multilingual' supported)"""
        logger.debug("[chatterbox] Returning available models (multilingual only)")
        return [
            ModelInfo(
                name="multilingual",
                display_name="Multilingual (Pretrained)",
                languages=self.SUPPORTED_LANGUAGES
            )
        ]

    def load_model(self, model_name: str) -> None:
        """Load Chatterbox TTS model"""
        from fastapi import HTTPException

        # Validate model name (Chatterbox only supports 'multilingual' pretrained model)
        if model_name != "multilingual":
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}'. Chatterbox only supports 'multilingual'."
            )

        logger.info(f"[chatterbox] Loading Chatterbox Multilingual model on {self.device}...")

        # Load from models directory (set by BaseEngineServer)
        model_marker = self.models_dir / 't3_mtl23ls_v2.safetensors'

        if not model_marker.exists():
            raise HTTPException(
                status_code=400,
                detail="Model files not found. Ensure model is baked into Docker image or mounted."
            )

        with self._model_lock:
            self._conditionals_cache.clear()
            logger.info("[chatterbox] Loading model files from the configured model directory")
            model = ChatterboxMultilingualTTS.from_local(self.models_dir, self.device)

            # Ensure model is on correct device
            if hasattr(model, 'to') and str(getattr(model, 'device', None)) != self.device:
                model.to(self.device)

            self.model = model
            logger.info(f"[chatterbox] Model loaded successfully. Sample rate: {self.model.sr} Hz")

    def generate_audio(
        self,
        text: str,
        language: str,
        speaker_wav: Union[str, List[str]],
        parameters: Dict[str, Any]
    ) -> bytes:
        """Generate TTS audio using Chatterbox"""
        # Note: Model loaded check, speaker cloning validation, and max_text_length
        # validation are handled in base_tts_server.py

        request_started_at = perf_counter()
        exaggeration = float(parameters.get("exaggeration", 0.5))
        temperature = float(parameters.get("temperature", 0.8))
        cfg_weight = float(parameters.get("cfg_weight", 0.5))
        seed = int(parameters.get("seed", 0))

        # Resolve speaker filename to full path in samples_dir
        # speaker_wav is now a filename (e.g., "uuid.wav") not a full path
        audio_prompt_path = None
        if speaker_wav:
            if isinstance(speaker_wav, list):
                # Use first sample for audio prompt
                if speaker_wav:
                    audio_prompt_path = str(self.samples_dir / speaker_wav[0])
            else:
                if speaker_wav.strip():
                    audio_prompt_path = str(self.samples_dir / speaker_wav)

        max_text_length = self._engine_config.get("constraints", {}).get(
            "max_text_length", MAX_TOTAL_TEXT_CHARS
        )
        chunks = split_text(text, max_total_chars=max_text_length)
        sample_hash_ms = 0.0
        conditioning_ms = 0.0
        cache_hit: bool | None = None
        effective_exaggeration = exaggeration

        with self._model_lock:
            if self.model is None:
                raise RuntimeError("Chatterbox model is not loaded")

            # Seed and all mutable model state stay inside this lock.
            if seed != 0:
                torch.manual_seed(seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                import random
                random.seed(seed)
                np.random.seed(seed)

            if audio_prompt_path:
                cache_hit, effective_exaggeration, sample_hash_ms, conditioning_ms = (
                    prepare_or_reuse_conditionals(
                        self.model,
                        self._conditionals_cache,
                        Path(audio_prompt_path),
                        exaggeration,
                    )
                )
                if not cache_hit:
                    synchronization_started_at = perf_counter()
                    self._synchronize_device()
                    conditioning_ms += (perf_counter() - synchronization_started_at) * 1000

            generated_chunks = []
            synthesis_started_at = perf_counter()
            for chunk in chunks:
                self._synchronize_device()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    wav_tensor = self.model.generate(
                        chunk,
                        language_id=language,
                        exaggeration=effective_exaggeration,
                        temperature=temperature,
                        cfg_weight=cfg_weight,
                    )
                generated_chunks.append(wav_tensor.squeeze(0).detach().cpu().numpy())
                self._synchronize_device()
            synthesis_ms = (perf_counter() - synthesis_started_at) * 1000

            encoding_started_at = perf_counter()
            audio_parts = []
            boundary_pause = np.zeros(round(self.model.sr * 0.15), dtype=generated_chunks[0].dtype)
            for index, generated_audio in enumerate(generated_chunks):
                audio_parts.append(generated_audio)
                if index < len(generated_chunks) - 1:
                    audio_parts.append(boundary_pause)
            wav_array = np.concatenate(audio_parts)
            wav_bytes = audio_to_wav_bytes(wav_array, self.model.sr)
            wav_encoding_ms = (perf_counter() - encoding_started_at) * 1000

        total_ms = (perf_counter() - request_started_at) * 1000
        cache_status = "not_applicable" if cache_hit is None else ("hit" if cache_hit else "miss")
        logger.info(
            "[chatterbox] generation timing cache_hit={} chunks={} sample_hash_ms={:.1f} "
            "conditioning_ms={:.1f} synthesis_ms={:.1f} wav_encoding_ms={:.1f} total_ms={:.1f}",
            cache_status,
            len(chunks),
            sample_hash_ms,
            conditioning_ms,
            synthesis_ms,
            wav_encoding_ms,
            total_ms,
        )
        return wav_bytes

    def _synchronize_device(self) -> None:
        """Synchronize CUDA so per-stage durations include asynchronous work."""

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def unload_model(self) -> None:
        """Free resources"""
        with self._model_lock:
            self._conditionals_cache.clear()
            if self.model is not None:
                logger.info("[chatterbox] Unloading model...")
                del self.model
                self.model = None
                # Note: GPU cleanup, gc.collect(), and state reset are handled by base_server.py

    def get_package_version(self) -> str:
        """Return Chatterbox package version for health endpoint"""
        return CHATTERBOX_VERSION


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chatterbox Multilingual Engine Server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    server = ChatterboxServer()
    server.run(port=args.port, host=args.host)
