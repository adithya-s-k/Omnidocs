"""MinerU VL text extraction module.

MinerU VL is a vision-language model for document layout detection and
text/table/equation recognition. It performs two-step extraction:
1. Layout Detection: Detect regions with types (text, table, equation, etc.)
2. Content Recognition: Extract content from each detected region

Example:
    ```python
    from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
    from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

    # Initialize with PyTorch backend
    extractor = MinerUVLTextExtractor(
        backend=MinerUVLTextPyTorchConfig(device="cuda")
    )

    # Extract text
    result = extractor.extract(image)
    print(result.content)

    # Extract with detailed blocks
    result, blocks = extractor.extract_with_blocks(image)
    for block in blocks:
        print(f"{block.type}: {block.content[:50]}...")
    ```
"""

from .extractor import MinerUVLTextExtractor
from .pytorch import MinerUVLTextPyTorchConfig
from .vllm import MinerUVLTextVLLMConfig
from .mlx import MinerUVLTextMLXConfig
from .api import MinerUVLTextAPIConfig
from .utils import (
    BlockType,
    ContentBlock,
    SamplingParams,
    MinerUSamplingParams,
    parse_layout_output,
    convert_otsl_to_html,
)

__all__ = [
    # Main extractor
    "MinerUVLTextExtractor",
    # Config classes
    "MinerUVLTextPyTorchConfig",
    "MinerUVLTextVLLMConfig",
    "MinerUVLTextMLXConfig",
    "MinerUVLTextAPIConfig",
    # Data structures
    "BlockType",
    "ContentBlock",
    "SamplingParams",
    "MinerUSamplingParams",
    # Utilities
    "parse_layout_output",
    "convert_otsl_to_html",
]
