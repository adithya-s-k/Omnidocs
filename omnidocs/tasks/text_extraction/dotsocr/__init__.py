"""
Dots OCR text extractor and backend configurations.

Available backends:
- PyTorch: DotsOCRPyTorchConfig (local GPU inference)
- VLLM: DotsOCRVLLMConfig (offline batch inference)
- API: DotsOCRAPIConfig (online VLLM server via OpenAI-compatible API)
"""

from .pytorch import DotsOCRPyTorchConfig
from .vllm import DotsOCRVLLMConfig
from .api import DotsOCRAPIConfig
from .extractor import DotsOCRTextExtractor

__all__ = [
    "DotsOCRPyTorchConfig",
    "DotsOCRVLLMConfig",
    "DotsOCRAPIConfig",
    "DotsOCRTextExtractor",
]
