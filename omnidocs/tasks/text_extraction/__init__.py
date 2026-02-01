"""
Text Extraction Module.

Provides extractors for converting document images to structured text formats
(HTML, Markdown, JSON). Uses Vision-Language Models for accurate text extraction
with formatting preservation and optional layout detection.

Available Extractors:
    - QwenTextExtractor: Qwen3-VL based extractor (multi-backend)
    - DotsOCRTextExtractor: Dots OCR with layout-aware extraction (PyTorch/VLLM/API)

Example:
    >>> from omnidocs.tasks.text_extraction import QwenTextExtractor
    >>> from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    >>>
    >>> extractor = QwenTextExtractor(
    ...     backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
    ... )
    >>> result = extractor.extract(image, output_format="markdown")
    >>> print(result.content)
"""

from .base import BaseTextExtractor
from .dotsocr import DotsOCRTextExtractor
from .models import DotsOCRTextOutput, LayoutElement, OutputFormat, TextOutput
from .qwen import QwenTextExtractor

__all__ = [
    # Base
    "BaseTextExtractor",
    # Models
    "TextOutput",
    "OutputFormat",
    "LayoutElement",
    "DotsOCRTextOutput",
    # Extractors
    "QwenTextExtractor",
    "DotsOCRTextExtractor",
]
