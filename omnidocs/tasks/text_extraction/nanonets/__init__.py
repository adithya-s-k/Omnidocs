"""
Nanonets OCR2-3B backend configurations and extractor for text extraction.

Available backends:
    - NanonetsTextPyTorchConfig: PyTorch/HuggingFace backend
    - NanonetsTextVLLMConfig: VLLM high-throughput backend
    - NanonetsTextMLXConfig: MLX backend for Apple Silicon

Example:
    ```python
    from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig
    config = NanonetsTextPyTorchConfig()
    ```
"""

from .extractor import NanonetsTextExtractor
from .mlx import NanonetsTextMLXConfig
from .pytorch import NanonetsTextPyTorchConfig
from .vllm import NanonetsTextVLLMConfig

__all__ = [
    "NanonetsTextExtractor",
    "NanonetsTextPyTorchConfig",
    "NanonetsTextVLLMConfig",
    "NanonetsTextMLXConfig",
]
