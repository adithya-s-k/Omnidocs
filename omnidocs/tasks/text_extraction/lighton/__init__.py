"""LightOn text extraction module.

LightOn OCR is optimized for document text extraction with multi-lingual support.
Supports multiple backends: PyTorch, VLLM, MLX, and API.

Example:
    ```python
    from omnidocs.tasks.text_extraction import LightOnTextExtractor
    from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

    # Initialize with PyTorch backend
    extractor = LightOnTextExtractor(
        backend=LightOnTextPyTorchConfig(device="cuda", torch_dtype="bfloat16")
    )

    # Extract text
    result = extractor.extract(image)
    print(result.content)
    print(f"Confidence: {result.format}")
    ```
"""

from .extractor import LightOnTextExtractor
from .mlx import LightOnTextMLXConfig
from .pytorch import LightOnTextPyTorchConfig
from .vllm import LightOnTextVLLMConfig

__all__ = [
    # Main extractor
    "LightOnTextExtractor",
    # Config classes
    "LightOnTextPyTorchConfig",
    "LightOnTextVLLMConfig",
    "LightOnTextMLXConfig",
]
