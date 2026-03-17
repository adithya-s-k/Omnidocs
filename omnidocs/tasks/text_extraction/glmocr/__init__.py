"""
GLM-OCR backend configurations and extractor for text extraction.

GLM-OCR from zai-org (Feb 2026) — 0.9B OCR-specialist model.
Architecture: CogViT visual encoder (0.4B) + GLM decoder (0.5B).
Scores #1 on OmniDocBench V1.5 (94.62), beating models 10x its size.

Unlike GLM-V (which is a general VLM), GLM-OCR is purpose-built for document OCR.
Uses AutoModelForImageTextToText + AutoProcessor (NOT Glm4vForConditionalGeneration).
Requires transformers>=5.3.0.

Available backends:
    - GLMOCRPyTorchConfig: PyTorch/HuggingFace backend
    - GLMOCRVLLMConfig: VLLM high-throughput backend (with MTP speculative decoding)
    - GLMOCRAPIConfig: API backend

HuggingFace: zai-org/GLM-OCR
License: Apache 2.0
"""

from .api import GLMOCRAPIConfig
from .extractor import GLMOCRTextExtractor
from .mlx import GLMOCRMLXConfig
from .pytorch import GLMOCRPyTorchConfig
from .vllm import GLMOCRVLLMConfig

__all__ = [
    "GLMOCRTextExtractor",
    "GLMOCRPyTorchConfig",
    "GLMOCRVLLMConfig",
    "GLMOCRMLXConfig",
    "GLMOCRAPIConfig",
]
