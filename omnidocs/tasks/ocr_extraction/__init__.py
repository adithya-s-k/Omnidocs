"""
OCR Extraction Module.

Provides extractors for detecting text with bounding boxes from document images.
Returns text content along with spatial coordinates (unlike Text Extraction which
returns formatted Markdown/HTML without coordinates).

Available Extractors:
    - TesseractOCR: Open-source OCR (CPU, requires system Tesseract)
    - EasyOCR: PyTorch-based OCR (CPU/GPU, 80+ languages)
    - PaddleOCR: PaddlePaddle-based OCR (CPU/GPU, excellent CJK support)

Key Difference from Text Extraction:
    - OCR Extraction: Text + Bounding Boxes (spatial location)
    - Text Extraction: Markdown/HTML (formatted document export)

Example:
    >>> from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig
    >>>
    >>> ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
    >>> result = ocr.extract(image)
    >>>
    >>> for block in result.text_blocks:
    ...     print(f"'{block.text}' @ {block.bbox.to_list()} (conf: {block.confidence:.2f})")

    >>> # With EasyOCR
    >>> from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
    >>>
    >>> ocr = EasyOCR(config=EasyOCRConfig(languages=["en", "ch_sim"], gpu=True))
    >>> result = ocr.extract(image)

    >>> # With PaddleOCR
    >>> from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig
    >>>
    >>> ocr = PaddleOCR(config=PaddleOCRConfig(lang="ch", device="cpu"))
    >>> result = ocr.extract(image)
"""

from .base import BaseOCRExtractor
from .easyocr import EasyOCR, EasyOCRConfig
from .models import (
    NORMALIZED_SIZE,
    BoundingBox,
    OCRGranularity,
    OCROutput,
    TextBlock,
)
from .paddleocr import PaddleOCR, PaddleOCRConfig
from .tesseract import TesseractOCR, TesseractOCRConfig

__all__ = [
    # Base
    "BaseOCRExtractor",
    # Models
    "OCRGranularity",
    "BoundingBox",
    "TextBlock",
    "OCROutput",
    "NORMALIZED_SIZE",
    # Tesseract
    "TesseractOCR",
    "TesseractOCRConfig",
    # EasyOCR
    "EasyOCR",
    "EasyOCRConfig",
    # PaddleOCR
    "PaddleOCR",
    "PaddleOCRConfig",
]
