"""Optional integration coverage using the installed Chatterbox runtime stack."""

from __future__ import annotations

import io
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from tts.chatterbox.conditioning_cache import ConditionalsCache
    from tts.chatterbox.server import ChatterboxServer
except (ImportError, OSError):
    torch = None
    ConditionalsCache = None
    ChatterboxServer = None


@unittest.skipIf(ChatterboxServer is None, "Chatterbox runtime dependencies are not installed")
class ChatterboxGenerationIntegrationTests(unittest.TestCase):
    def test_long_text_is_complete_and_cached_generation_is_repeatable(self) -> None:
        from loguru import logger

        class FakeModel:
            sr = 24000

            def __init__(self) -> None:
                self.conds = None
                self.preparations = 0
                self.calls: list[tuple[str, float]] = []

            def prepare_conditionals(self, sample_path: str, *, exaggeration: float) -> None:
                self.preparations += 1
                self.conds = object()

            def generate(
                self,
                text: str,
                *,
                language_id: str,
                exaggeration: float,
                temperature: float,
                cfg_weight: float,
            ):
                self.calls.append((text, exaggeration))
                value = (len(text) % 7) / 10
                return torch.full((1, 32), value, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "speaker.wav"
            sample.write_bytes(b"synthetic sample")
            server = object.__new__(ChatterboxServer)
            server._device = "cpu"
            server.samples_dir = root
            server._engine_config = {"constraints": {"max_text_length": 3000}}
            server.model = FakeModel()
            server._model_lock = threading.RLock()
            server._conditionals_cache = ConditionalsCache(max_entries=2)
            text = "x" * 301
            timing_output = io.StringIO()
            sink_id = logger.add(timing_output, format="{message}")
            try:
                first_audio = server.generate_audio(
                    text, "en", sample.name, {"seed": 123, "exaggeration": 0.5}
                )
                second_audio = server.generate_audio(
                    text, "en", sample.name, {"seed": 123, "exaggeration": 0.5}
                )
            finally:
                logger.remove(sink_id)

            self.assertEqual(first_audio, second_audio)
            self.assertEqual(server.model.preparations, 1)
            self.assertEqual([text for text, _ in server.model.calls], ["x" * 300, "x"] * 2)
            self.assertTrue(all(value == 0.5 for _, value in server.model.calls))
            self.assertIn("cache_hit=miss", timing_output.getvalue())
            self.assertIn("cache_hit=hit", timing_output.getvalue())
            self.assertIn("chunks=2", timing_output.getvalue())
            self.assertNotIn(sample.name, timing_output.getvalue())
            self.assertNotIn(text, timing_output.getvalue())
            with wave.open(io.BytesIO(first_audio), "rb") as generated_wav:
                self.assertEqual(generated_wav.getframerate(), 24000)
                self.assertEqual(generated_wav.getnframes(), 32 + 3600 + 32)

    def test_model_load_and_unload_clear_cached_conditionals(self) -> None:
        class FakeModel:
            sr = 24000

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "t3_mtl23ls_v2.safetensors").write_bytes(b"model marker")
            server = object.__new__(ChatterboxServer)
            server._device = "cpu"
            server.models_dir = root
            server.model = FakeModel()
            server._model_lock = threading.RLock()
            server._conditionals_cache = ConditionalsCache(max_entries=2)
            server._conditionals_cache.put(("old-sample", 0.5), object())
            replacement_model = FakeModel()

            with patch(
                "tts.chatterbox.server.ChatterboxMultilingualTTS.from_local",
                return_value=replacement_model,
            ):
                server.load_model("multilingual")

            self.assertIs(server.model, replacement_model)
            self.assertEqual(len(server._conditionals_cache), 0)
            server._conditionals_cache.put(("new-sample", 0.5), object())
            server.unload_model()
            self.assertIsNone(server.model)
            self.assertEqual(len(server._conditionals_cache), 0)


if __name__ == "__main__":
    unittest.main()
