"""MLX backend configuration for GLM-OCR text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GLMOCRMLXConfig(BaseModel):
    """
    MLX backend configuration for GLM-OCR.

    Uses mlx-vlm for Apple Silicon native inference.
    GLM-OCR at 0.9B runs comfortably on any M-series Mac with 8GB+ unified memory.
    Requires: mlx, mlx-vlm>=0.3.11

    Note: Only works on Apple Silicon Macs. Do NOT use for Modal/cloud deployments.

    Available models:
        mlx-community/GLM-OCR-bf16   (default — full precision, 2.21 GB)
        mlx-community/GLM-OCR-6bit   (quantized, smaller)

    Example:
```python
        config = GLMOCRMLXConfig()  # bf16, default
        config = GLMOCRMLXConfig(model="mlx-community/GLM-OCR-6bit")  # quantized
```
    """

    model: str = Field(
        default="mlx-community/GLM-OCR-bf16",
        description="MLX model path or HuggingFace ID. "
        "Default: mlx-community/GLM-OCR-bf16 (bfloat16, 2.21 GB).",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for deterministic output.",
    )

    model_config = ConfigDict(extra="forbid")
