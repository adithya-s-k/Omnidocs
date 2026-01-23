"""
Qwen3-VL backend configurations and detector for layout detection.

Available backends:
    - QwenLayoutPyTorchConfig: PyTorch/HuggingFace backend
    - QwenLayoutVLLMConfig: VLLM high-throughput backend
    - QwenLayoutMLXConfig: MLX backend for Apple Silicon
    - QwenLayoutAPIConfig: API backend (OpenRouter, etc.)

Example:
    >>> from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
    >>> config = QwenLayoutPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
"""

from .api import QwenLayoutAPIConfig
from .detector import QwenLayoutDetector
from .mlx import QwenLayoutMLXConfig
from .pytorch import QwenLayoutPyTorchConfig
from .vllm import QwenLayoutVLLMConfig

__all__ = [
    "QwenLayoutDetector",
    "QwenLayoutPyTorchConfig",
    "QwenLayoutVLLMConfig",
    "QwenLayoutMLXConfig",
    "QwenLayoutAPIConfig",
]
