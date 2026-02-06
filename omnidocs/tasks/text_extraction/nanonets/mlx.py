"""
MLX backend configuration for Nanonets OCR2-3B text extraction.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NanonetsTextMLXConfig(BaseModel):
    """
    MLX backend configuration for Nanonets OCR2-3B text extraction.

    This backend uses MLX for Apple Silicon native inference.
    Best for local development and testing on macOS M1/M2/M3/M4+.
    Requires: mlx, mlx-vlm

    Note: This backend only works on Apple Silicon Macs.
    Do NOT use for Modal/cloud deployments.

    Example:
        ```python
        config = NanonetsTextMLXConfig(
                model="mlx-community/Nanonets-OCR2-3B-bf16",
            )
        ```
    """

    model: str = Field(
        default="mlx-community/Nanonets-OCR2-3B-bf16",
        description="MLX model path or HuggingFace ID. "
        "Default: mlx-community/Nanonets-OCR2-3B-bf16 (bfloat16 precision)",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODEL_CACHE env var or default cache.",
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
