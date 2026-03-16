"""MLX backend configuration for LightOn text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LightOnTextMLXConfig(BaseModel):
    """
    MLX backend config for LightOn text extraction.

    Uses MLX for efficient inference on Apple Silicon.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import LightOnTextExtractor
        from omnidocs.tasks.text_extraction.lighton import LightOnTextMLXConfig

        extractor = LightOnTextExtractor(
            backend=LightOnTextMLXConfig()
        )
        result = extractor.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="lightonai/LightOnOCR-2-1B",
        description="Model identifier on HuggingFace or MLX community",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Cache directory for downloaded models",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate",
    )
