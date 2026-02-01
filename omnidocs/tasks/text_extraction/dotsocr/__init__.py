"""
Dots OCR text extractor and backend configurations.

Available backends:
- PyTorch: DotsOCRPyTorchConfig (local GPU inference)
- VLLM: DotsOCRVLLMConfig (offline batch inference)
- API: DotsOCRAPIConfig (online VLLM server via OpenAI-compatible API)
"""

from .api import DotsOCRAPIConfig
from .extractor import DotsOCRTextExtractor
from .pytorch import DotsOCRPyTorchConfig
from .vllm import DotsOCRVLLMConfig

__all__ = [
    "DotsOCRPyTorchConfig",
    "DotsOCRVLLMConfig",
    "DotsOCRAPIConfig",
    "DotsOCRTextExtractor",
]
