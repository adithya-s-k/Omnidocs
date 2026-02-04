"""MLX backend configuration for MinerU VL layout detection (Apple Silicon)."""

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field


class MinerUVLLayoutMLXConfig(BaseModel):
    """
    MLX backend config for MinerU VL layout detection on Apple Silicon.

    Example:
        ```python
        from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutMLXConfig

        detector = MinerUVLLayoutDetector(
            backend=MinerUVLLayoutMLXConfig()
        )
        result = detector.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="opendatalab/MinerU2.5-2509-1.2B",
        description="Model ID (will be converted to MLX format)",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    layout_image_size: Tuple[int, int] = Field(
        default=(1036, 1036),
        description="Resize image to this size for layout detection",
    )
