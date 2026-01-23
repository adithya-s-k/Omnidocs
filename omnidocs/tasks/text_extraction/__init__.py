"""
Text Extraction Module.

Provides extractors for converting document images to structured text formats
(HTML, Markdown). Uses Vision-Language Models for accurate text extraction
with formatting preservation.

Available Extractors:
    - QwenTextExtractor: Qwen3-VL based extractor (multi-backend)

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
from .models import OutputFormat, TextOutput
from .qwen import QwenTextExtractor

__all__ = [
    # Base
    "BaseTextExtractor",
    # Models
    "TextOutput",
    "OutputFormat",
    # Extractors
    "QwenTextExtractor",
]
