"""
PaddleOCR extractor.

PaddleOCR is an OCR toolkit developed by Baidu/PaddlePaddle.
- Excellent for CJK languages (Chinese, Japanese, Korean)
- GPU accelerated
- Supports layout analysis + OCR

Python Package:
    pip install paddleocr paddlepaddle  # CPU version
    pip install paddleocr paddlepaddle-gpu  # GPU version

Model Download Location:
    By default, PaddleOCR downloads models to ~/.paddleocr/
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base import BaseOCRExtractor
from .models import BoundingBox, OCRGranularity, OCROutput, TextBlock

# Language code mapping for common aliases
LANG_CODES: Dict[str, str] = {
    "en": "en",
    "english": "en",
    "ch": "ch",
    "chinese": "ch",
    "chinese_cht": "chinese_cht",  # Traditional Chinese
    "ja": "japan",
    "japanese": "japan",
    "japan": "japan",
    "ko": "korean",
    "korean": "korean",
    "de": "german",
    "german": "german",
    "fr": "french",
    "french": "french",
    "ar": "arabic",
    "arabic": "arabic",
    "hi": "devanagari",  # Hindi
    "ta": "tamil",
    "te": "telugu",
    "ru": "cyrillic",
    "russian": "cyrillic",
}


class PaddleOCRConfig(BaseModel):
    """
    Configuration for PaddleOCR extractor.

    This is a single-backend model (PaddlePaddle - CPU/GPU).

    Example:
        >>> config = PaddleOCRConfig(lang="ch", device="gpu")
        >>> ocr = PaddleOCR(config=config)
    """

    lang: str = Field(
        default="en",
        description="Language code (en, ch, japan, korean, german, french, etc.). "
        "See https://github.com/PaddlePaddle/PaddleOCR for full list.",
    )
    device: str = Field(
        default="cpu",
        description="Device: 'cpu' or 'gpu'",
    )

    model_config = ConfigDict(extra="forbid")


class PaddleOCR(BaseOCRExtractor):
    """
    PaddleOCR text extractor.

    Single-backend model (PaddlePaddle - CPU/GPU).

    Example:
        >>> from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig
        >>>
        >>> ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        >>> result = ocr.extract(image)
        >>>
        >>> for block in result.text_blocks:
        ...     print(f"'{block.text}' @ {block.bbox.to_list()}")
    """

    MODEL_NAME = "paddleocr"

    def __init__(self, config: PaddleOCRConfig):
        """
        Initialize PaddleOCR extractor.

        Args:
            config: Configuration object

        Raises:
            ImportError: If paddleocr or paddlepaddle is not installed
        """
        self.config = config
        self._ocr = None

        # Normalize language code
        self._lang = LANG_CODES.get(config.lang.lower(), config.lang)

        self._load_model()

    def _load_model(self) -> None:
        """Initialize PaddleOCR engine."""
        # Check paddlepaddle
        try:
            import paddle  # noqa: F401
        except ImportError:
            raise ImportError(
                "paddlepaddle is required for PaddleOCR.\n"
                "Install with:\n"
                "  CPU: pip install paddlepaddle\n"
                "  GPU: pip install paddlepaddle-gpu"
            )

        # Check paddleocr
        try:
            from paddleocr import PaddleOCR as PaddleOCREngine
        except ImportError:
            raise ImportError("paddleocr is required for PaddleOCR. Install with: pip install paddleocr")

        # Initialize OCR engine (PaddleOCR v3.x API)
        self._ocr = PaddleOCREngine(
            lang=self._lang,
            device=self.config.device,
        )

    def extract(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCROutput:
        """
        Run OCR on an image.

        Args:
            image: Input image (PIL Image, numpy array, or path)

        Returns:
            OCROutput with detected text blocks
        """
        if self._ocr is None:
            raise RuntimeError("PaddleOCR not initialized. Call _load_model() first.")

        # Prepare image
        pil_image = self._prepare_image(image)
        image_width, image_height = pil_image.size

        # Convert to numpy array
        image_array = np.array(pil_image)

        # Run PaddleOCR v3.x - use predict() method
        results = self._ocr.predict(image_array)

        # Parse results
        text_blocks = []

        # PaddleOCR may return None or empty results
        if results is None or len(results) == 0:
            return OCROutput(
                text_blocks=[],
                full_text="",
                image_width=image_width,
                image_height=image_height,
                model_name=self.MODEL_NAME,
                languages_detected=[self._lang],
            )

        # PaddleOCR v3.x returns list of dicts with 'rec_texts', 'rec_scores', 'dt_polys'
        for result in results:
            if result is None:
                continue

            rec_texts = result.get("rec_texts", [])
            rec_scores = result.get("rec_scores", [])
            dt_polys = result.get("dt_polys", [])

            for i, text in enumerate(rec_texts):
                if not text.strip():
                    continue

                confidence = rec_scores[i] if i < len(rec_scores) else 1.0

                # Get polygon and convert to list
                polygon: Optional[List[List[float]]] = None
                if i < len(dt_polys) and dt_polys[i] is not None:
                    poly_array = dt_polys[i]
                    # Handle numpy array
                    if hasattr(poly_array, "tolist"):
                        polygon = poly_array.tolist()
                    else:
                        polygon = list(poly_array)

                # Convert polygon to bbox
                if polygon:
                    bbox = BoundingBox.from_polygon(polygon)
                else:
                    bbox = BoundingBox(x1=0, y1=0, x2=0, y2=0)

                text_blocks.append(
                    TextBlock(
                        text=text,
                        bbox=bbox,
                        confidence=float(confidence),
                        granularity=OCRGranularity.LINE,
                        polygon=polygon,
                        language=self._lang,
                    )
                )

        # Sort by position (top to bottom, left to right)
        text_blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))

        # Build full_text from sorted blocks to ensure reading order
        full_text = " ".join(block.text for block in text_blocks)

        return OCROutput(
            text_blocks=text_blocks,
            full_text=full_text,
            image_width=image_width,
            image_height=image_height,
            model_name=self.MODEL_NAME,
            languages_detected=[self._lang],
        )
