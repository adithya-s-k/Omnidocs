"""
Qwen3-VL backend configurations and extractor for text extraction.

Available backends:
    - QwenTextPyTorchConfig: PyTorch/HuggingFace backend
    - QwenTextVLLMConfig: VLLM high-throughput backend
    - QwenTextMLXConfig: MLX backend for Apple Silicon
    - QwenTextAPIConfig: API backend (OpenRouter, etc.)

Example:
    >>> from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
    >>> config = QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
"""

from .api import QwenTextAPIConfig
from .extractor import QwenTextExtractor
from .mlx import QwenTextMLXConfig
from .pytorch import QwenTextPyTorchConfig
from .vllm import QwenTextVLLMConfig

__all__ = [
    "QwenTextExtractor",
    "QwenTextPyTorchConfig",
    "QwenTextVLLMConfig",
    "QwenTextMLXConfig",
    "QwenTextAPIConfig",
]
