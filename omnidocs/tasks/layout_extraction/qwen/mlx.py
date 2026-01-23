"""
MLX backend configuration for Qwen3-VL layout detection.
"""

from pydantic import BaseModel, ConfigDict, Field


class QwenLayoutMLXConfig(BaseModel):
    """
    MLX backend configuration for Qwen layout detection.

    This backend uses MLX for Apple Silicon native inference.
    Best for local development and testing on macOS M1/M2/M3+.
    Requires: mlx, mlx-vlm

    Note: This backend only works on Apple Silicon Macs.
    Do NOT use for Modal/cloud deployments.

    Example:
        >>> config = QwenLayoutMLXConfig(
        ...     model="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        ... )
    """

    model: str = Field(
        default="mlx-community/Qwen3-VL-8B-Instruct-4bit",
        description="MLX model path or HuggingFace ID. "
        "Recommended: mlx-community/Qwen3-VL-8B-Instruct-4bit (4-bit quantized)",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )

    model_config = ConfigDict(extra="forbid")
