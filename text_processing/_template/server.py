"""
Template Text Processing Engine Server

Inherits from BaseTextServer for consistent lifecycle management.
Provides text-specific endpoint:
- /segment - Text segmentation for TTS processing

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

from base_text_server import BaseTextServer, SegmentItem
from base_server import ModelInfo
from loguru import logger


class TemplateTextProcessor(BaseTextServer):
    """
    Template text processing engine.

    TODO: Rename this class to match your engine (e.g., MyTextProcessor).

    Text processing engines segment text into chunks suitable for TTS
    generation, ensuring sentence boundaries are preserved.
    """

    def __init__(self):
        super().__init__(
            engine_name="template-text",  # TODO: Change to your engine name (must match engine.yaml)
            display_name="Template Text Processor"  # TODO: Change display name
        )

        # TODO: Initialize your NLP model here
        self.nlp = None
        self.default_model = "default"
        self.current_language: Optional[str] = None

        logger.info(f"[{self.engine_name}] Text processing engine initialized")

    def load_model(self, model_name: str) -> None:
        """
        Load text processing model.

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
            # Example for spaCy:
            # import spacy.cli
            # spacy.cli.download(model_name)

            # Create symlink from models/ to external_models/
            model_path.symlink_to(external_path)
            logger.info(f"[{self.engine_name}] Model downloaded to {external_path}")

        # TODO: Load your NLP model here
        # Example:
        # import spacy
        # self.nlp = spacy.load(model_name)

        self.nlp = True  # Placeholder - remove after implementing
        self.current_language = model_name
        self.model_loaded = True
        self.current_model = model_name

        logger.info(f"[{self.engine_name}] Model '{model_name}' loaded successfully")

    def segment_text(
        self,
        text: str,
        language: str,
        max_length: int,
        min_length: int,
        mark_oversized: bool
    ) -> List[SegmentItem]:
        """
        Segment text into chunks for TTS generation.

        TODO: Implement your segmentation logic here.

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
        """
        if not text or not text.strip():
            return []

        # Sanitize text
        text = self._sanitize_text(text)

        # TODO: Replace this placeholder with your NLP-based segmentation
        # Example using spaCy:
        # doc = self.nlp(text)
        # sentences = [sent.text for sent in doc.sents]

        # Placeholder: simple period-based splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        segments: List[SegmentItem] = []
        order_index = 0
        current_pos = 0
        current_segment_text = ""
        current_start = 0

        for sentence in sentences:
            sent_text = sentence + "."
            sent_length = len(sent_text)

            # Check if single sentence exceeds max_length
            if sent_length > max_length:
                # Save accumulated segment first
                if current_segment_text:
                    segments.append(SegmentItem(
                        text=current_segment_text.strip(),
                        start=current_start,
                        end=current_start + len(current_segment_text.strip()),
                        order_index=order_index,
                        status="ok"
                    ))
                    order_index += 1
                    current_segment_text = ""

                # Mark oversized sentence
                segments.append(SegmentItem(
                    text=sent_text,
                    start=current_pos,
                    end=current_pos + sent_length,
                    order_index=order_index,
                    status="failed" if mark_oversized else "ok",
                    length=sent_length if mark_oversized else None,
                    max_length=max_length if mark_oversized else None,
                    issue="sentence_too_long" if mark_oversized else None
                ))
                order_index += 1
                current_start = current_pos + sent_length
                current_pos += sent_length
                continue

            # Check if adding sentence would exceed max_length
            would_exceed = current_segment_text and \
                (len(current_segment_text) + 1 + sent_length) > max_length

            if would_exceed:
                # Save current segment
                segments.append(SegmentItem(
                    text=current_segment_text.strip(),
                    start=current_start,
                    end=current_start + len(current_segment_text.strip()),
                    order_index=order_index,
                    status="ok"
                ))
                order_index += 1
                current_segment_text = sent_text
                current_start = current_pos
            else:
                # Add to current segment
                if current_segment_text:
                    current_segment_text += " " + sent_text
                else:
                    current_segment_text = sent_text
                    current_start = current_pos

            current_pos += sent_length + 1

        # Add remaining segment
        if current_segment_text.strip():
            segments.append(SegmentItem(
                text=current_segment_text.strip(),
                start=current_start,
                end=current_start + len(current_segment_text.strip()),
                order_index=order_index,
                status="ok"
            ))

        return segments

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for TTS processing."""
        import unicodedata
        import html

        if not text:
            return ""

        # Unicode normalization (NFC)
        text = unicodedata.normalize('NFC', text)

        # Remove BOM and zero-width characters
        text = text.replace('\ufeff', '')
        text = text.replace('\u200b', '')
        text = text.replace('\u200c', '')
        text = text.replace('\u200d', '')

        # Normalize whitespace
        text = ' '.join(text.split())

        # Decode HTML entities
        text = html.unescape(text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text.strip()

    def unload_model(self) -> None:
        """Unload model and free resources."""
        logger.info(f"[{self.engine_name}] Unloading model")

        if self.nlp is not None:
            self.nlp = None

        self.current_language = None
        # Note: GPU cleanup, gc.collect(), and state reset are handled by base_server.py

    def get_available_models(self) -> List[ModelInfo]:
        """Return available models by scanning models directory."""
        models = []

        # Scan models directory
        if self.models_dir.exists():
            for model_path in sorted(self.models_dir.iterdir()):
                if model_path.is_dir() or model_path.is_symlink():
                    model_name = model_path.name
                    models.append(
                        ModelInfo(
                            name=model_name,
                            display_name=model_name.replace("_", " ").title(),
                            languages=[],
                            fields=[]
                        )
                    )

        # Fallback if no models found
        if not models:
            models.append(
                ModelInfo(
                    name="default",
                    display_name="Default Configuration",
                    languages=[],
                    fields=[]
                )
            )

        return models

    def get_package_version(self) -> str:
        """Return engine/package version."""
        return "1.0.0"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Template Text Processing Engine Server")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    server = TemplateTextProcessor()
    server.run(port=args.port, host=args.host)
