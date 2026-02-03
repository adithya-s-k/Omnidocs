"""
Text Extraction Module.

Provides extractors for converting document images to structured text formats
(HTML, Markdown, JSON). Uses Vision-Language Models for accurate text extraction
with formatting preservation and optional layout detection.

Available Extractors:
    - QwenTextExtractor: Qwen3-VL based extractor (multi-backend)
    - DotsOCRTextExtractor: Dots OCR with layout-aware extraction (PyTorch/VLLM/API)
    - NanonetsTextExtractor: Nanonets OCR2-3B for text extraction (PyTorch/VLLM)
    - GraniteDoclingTextExtractor: IBM Granite Docling for document conversion (multi-backend)

Example:
    ```python
    from omnidocs.tasks.text_extraction import QwenTextExtractor
    from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

    extractor = QwenTextExtractor(
            backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
        )
    result = extractor.extract(image, output_format="markdown")
    print(result.content)
    ```
"""

from .base import BaseTextExtractor
from .dotsocr import DotsOCRTextExtractor
from .granitedocling import GraniteDoclingTextExtractor
from .models import DotsOCRTextOutput, LayoutElement, OutputFormat, TextOutput
from .nanonets import NanonetsTextExtractor
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
    "NanonetsTextExtractor",
    "GraniteDoclingTextExtractor",
]
