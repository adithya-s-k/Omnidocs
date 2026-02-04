"""MinerU VL layout detection module.

MinerU VL can be used for standalone layout detection, returning
detected regions with types and bounding boxes.

For full document extraction (layout + content), use MinerUVLTextExtractor
from the text_extraction module instead.

Example:
    ```python
    from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
    from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

    detector = MinerUVLLayoutDetector(
        backend=MinerUVLLayoutPyTorchConfig(device="cuda")
    )
    result = detector.extract(image)

    for box in result.bboxes:
        print(f"{box.label}: {box.confidence:.2f}")
    ```
"""

from .api import MinerUVLLayoutAPIConfig
from .detector import MinerUVLLayoutDetector
from .mlx import MinerUVLLayoutMLXConfig
from .pytorch import MinerUVLLayoutPyTorchConfig
from .vllm import MinerUVLLayoutVLLMConfig

__all__ = [
    "MinerUVLLayoutDetector",
    "MinerUVLLayoutPyTorchConfig",
    "MinerUVLLayoutVLLMConfig",
    "MinerUVLLayoutMLXConfig",
    "MinerUVLLayoutAPIConfig",
]
