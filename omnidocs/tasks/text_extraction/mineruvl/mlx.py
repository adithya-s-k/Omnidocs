"""MLX backend configuration for MinerU VL text extraction (Apple Silicon)."""

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MinerUVLTextMLXConfig(BaseModel):
    """
    MLX backend config for MinerU VL text extraction on Apple Silicon.

    Uses MLX-VLM for efficient inference on M1/M2/M3/M4 chips.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

        extractor = MinerUVLTextExtractor(
            backend=MinerUVLTextMLXConfig()
        )
        result = extractor.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="opendatalab/MinerU2.5-2509-1.2B",
        description="Model ID (will be converted to MLX format)",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
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
