"""
DeepSeek-OCR backend configurations and extractor for text extraction.

Two generations of DeepSeek OCR models from deepseek-ai:

  DeepSeek-OCR   (Oct 2024, arXiv:2510.18234) — v1, MIT license, 3B params, ~6.7 GB BF16
  DeepSeek-OCR-2 (Jan 2026, arXiv:2601.20552) — v2, Apache 2.0, 3B params, improved "Visual Causal Flow"

Both share the same inference interface (AutoModel + AutoTokenizer with model.infer()).
The default model is DeepSeek-OCR-2 (latest).

Supported prompts:
  "<image>\n<|grounding|>Convert the document to markdown."   ← structured document
  "<image>\n<|grounding|>OCR this image."                     ← general image
  "<image>\nFree OCR."                                        ← plain text, no layout
  "<image>\nParse the figure."                                ← figures in document

Available backends:
    - DeepSeekOCRTextPyTorchConfig: PyTorch/HuggingFace backend
    - DeepSeekOCRTextVLLMConfig: VLLM high-throughput backend (recommended, ~2500 tok/s on A100)
    - DeepSeekOCRTextMLXConfig: MLX backend for Apple Silicon
    - DeepSeekOCRTextAPIConfig: API backend (Novita AI)

HuggingFaces:
    deepseek-ai/DeepSeek-OCR-2  (latest, Apache 2.0)
    deepseek-ai/DeepSeek-OCR    (v1, MIT)
GitHub: https://github.com/deepseek-ai/DeepSeek-OCR-2

Example:
    ```python
    from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig
    config = DeepSeekOCRTextVLLMConfig()  # uses DeepSeek-OCR-2 by default
    ```
"""

from .api import DeepSeekOCRTextAPIConfig
from .extractor import DeepSeekOCRTextExtractor
from .mlx import DeepSeekOCRTextMLXConfig
from .pytorch import DeepSeekOCRTextPyTorchConfig
from .vllm import DeepSeekOCRTextVLLMConfig

__all__ = [
    "DeepSeekOCRTextExtractor",
    "DeepSeekOCRTextPyTorchConfig",
    "DeepSeekOCRTextVLLMConfig",
    "DeepSeekOCRTextMLXConfig",
    "DeepSeekOCRTextAPIConfig",
]
