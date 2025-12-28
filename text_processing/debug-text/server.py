"""
Debug Text Engine Server

A lightweight text processing engine for testing and development that behaves
exactly like a real text engine but uses simple punctuation-based segmentation.

Features:
- Segments text at sentence boundaries (., !, ?)
- Respects max_length and min_length parameters
- No NLP models required, minimal dependencies
- Instant model loading (no actual model)
"""
from pathlib import Path
from typing import List
import sys
import re

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base_text_server import BaseTextServer, SegmentItem
from base_server import ModelInfo, ModelField
from loguru import logger


class DebugTextServer(BaseTextServer):
    """Debug Text Engine - Simple punctuation-based segmentation for development"""

    # Sentence-ending punctuation pattern
    SENTENCE_END_PATTERN = re.compile(r'([.!?]+)\s*')

    def __init__(self):
        super().__init__(
            engine_name="debug-text",
            display_name="Debug Text"
        )

        # No actual model needed
        self.model = None
        self.default_model = "default"
        # Note: self.device property is provided by BaseEngineServer (auto-detects cuda/cpu)

        logger.info("[debug-text] Debug Text Engine initialized (punctuation-based segmentation)")

    def load_model(self, model_name: str) -> None:
        """
        Simulate model loading with on-demand download behavior.

        Args:
            model_name: Model identifier
        """
        logger.info(f"[debug-text] Loading model: {model_name}")

        model_path = self.models_dir / model_name

        # Check if model exists (baked-in or symlinked from external)
        if not model_path.exists():
            logger.info(f"[debug-text] Model '{model_name}' not found, downloading...")

            # Simulate download to external_models (for persistence)
            external_path = self.external_models_dir / model_name
            external_path.mkdir(parents=True, exist_ok=True)

            # Create dummy model file
            model_file = external_path / "model.json"
            model_file.write_text(
                f'{{\n  "name": "{model_name}",\n  "type": "punctuation",\n'
                f'  "version": "1.0.0",\n  "description": "Auto-downloaded debug model"\n}}\n'
            )

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)

            logger.info(f"[debug-text] Model '{model_name}' downloaded to {external_path}")

        # Simulate successful model load
        self.current_model = model_name
        self.model_loaded = True

        logger.info(f"[debug-text] Model '{model_name}' loaded successfully")

    def segment_text(
        self,
        text: str,
        language: str,
        max_length: int,
        min_length: int,
        mark_oversized: bool
    ) -> List[SegmentItem]:
        """
        Segment text using simple punctuation-based splitting.

        Splits at sentence boundaries (., !, ?) and combines short sentences.

        Args:
            text: Input text to segment
            language: Language code (ignored - punctuation-based)
            max_length: Maximum characters per segment
            min_length: Minimum characters (merge short sentences)
            mark_oversized: Mark sentences exceeding max_length as "failed"

        Returns:
            List of SegmentItem objects
        """
        logger.debug(
            f"[debug-text] Segmenting text: len={len(text)}, "
            f"max_length={max_length}, min_length={min_length}"
        )

        # Split text into sentences at punctuation
        raw_sentences = self._split_into_sentences(text)

        # Build segments by combining short sentences
        segments = []
        current_text = ""
        current_start = 0
        text_pos = 0

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find position of this sentence in original text
            sent_start = text.find(sentence, text_pos)
            if sent_start == -1:
                sent_start = text_pos
            sent_end = sent_start + len(sentence)

            # Check if adding this sentence exceeds max_length
            potential_text = (current_text + " " + sentence).strip() if current_text else sentence

            if len(potential_text) <= max_length:
                # Can add to current segment
                if not current_text:
                    current_start = sent_start
                current_text = potential_text
                text_pos = sent_end
            else:
                # Current segment is full, save it
                if current_text:
                    segment = self._create_segment(
                        current_text, current_start, text_pos,
                        len(segments), max_length, mark_oversized
                    )
                    segments.append(segment)

                # Start new segment with current sentence
                current_text = sentence
                current_start = sent_start
                text_pos = sent_end

        # Don't forget the last segment
        if current_text:
            segment = self._create_segment(
                current_text, current_start, text_pos,
                len(segments), max_length, mark_oversized
            )
            segments.append(segment)

        # Handle minimum length - merge very short segments
        if min_length > 0:
            segments = self._merge_short_segments(segments, min_length, max_length)

        logger.debug(f"[debug-text] Created {len(segments)} segments")
        return segments

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text at sentence-ending punctuation."""
        # Split at sentence boundaries while keeping punctuation
        parts = self.SENTENCE_END_PATTERN.split(text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i].strip()
            punct = parts[i + 1] if i + 1 < len(parts) else ""
            if sentence:
                sentences.append(sentence + punct)

        # Handle any remaining text without ending punctuation
        if parts and parts[-1].strip():
            sentences.append(parts[-1].strip())

        return sentences

    def _create_segment(
        self,
        text: str,
        start: int,
        end: int,
        order_index: int,
        max_length: int,
        mark_oversized: bool
    ) -> SegmentItem:
        """Create a SegmentItem, marking as failed if oversized."""
        text_len = len(text)

        if mark_oversized and text_len > max_length:
            return SegmentItem(
                text=text,
                start=start,
                end=end,
                order_index=order_index,
                status="failed",
                length=text_len,
                max_length=max_length,
                issue="sentence_too_long"
            )

        return SegmentItem(
            text=text,
            start=start,
            end=end,
            order_index=order_index,
            status="ok"
        )

    def _merge_short_segments(
        self,
        segments: List[SegmentItem],
        min_length: int,
        max_length: int
    ) -> List[SegmentItem]:
        """Merge segments shorter than min_length with neighbors."""
        if not segments:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]

            # Skip failed segments - don't merge them
            if current.status == "failed":
                merged.append(current)
                i += 1
                continue

            # Check if current is too short and can be merged
            if len(current.text) < min_length and i + 1 < len(segments):
                next_seg = segments[i + 1]
                combined_text = current.text + " " + next_seg.text

                # Only merge if result is within max_length
                if len(combined_text) <= max_length and next_seg.status != "failed":
                    # Create merged segment
                    merged_segment = SegmentItem(
                        text=combined_text.strip(),
                        start=current.start,
                        end=next_seg.end,
                        order_index=len(merged),
                        status="ok"
                    )
                    merged.append(merged_segment)
                    i += 2  # Skip both segments
                    continue

            # Re-index and add
            current.order_index = len(merged)
            merged.append(current)
            i += 1

        return merged

    def unload_model(self) -> None:
        """Free resources (nothing to free for debug engine)."""
        logger.info("[debug-text] Unloading model")
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
                            display_name=f"{model_name.title()} (Punctuation)",
                            languages=["de", "en", "fr", "es", "it"],
                            fields=[
                                ModelField(key="type", value="punctuation", field_type="string"),
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

    parser = argparse.ArgumentParser(description="Debug Text Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    server = DebugTextServer()
    server.run(port=args.port, host=args.host)
