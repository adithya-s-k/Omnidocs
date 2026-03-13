"""
MLX backend configuration for DeepSeek-OCR text extraction.

Available MLX quantized variants (mlx-community):
  mlx-community/DeepSeek-OCR-4bit   (4-bit, recommended)
  mlx-community/DeepSeek-OCR-8bit   (8-bit, higher fidelity)

Note: DeepSeek-OCR-2 MLX variants may not yet be available — check
https://huggingface.co/mlx-community for latest uploads.
Fall back to DeepSeek-OCR v1 4bit/8bit for Apple Silicon.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DeepSeekOCRTextMLXConfig(BaseModel):
    """
    MLX backend configuration for DeepSeek-OCR text extraction.

    Apple Silicon only (M1/M2/M3+). Do NOT deploy to Modal/cloud.
    Uses standard mlx-vlm generate interface.

    Note: MLX variants currently available for DeepSeek-OCR v1.
    Check mlx-community for DeepSeek-OCR-2 variants as they are published.

    Example:
        ```python
        config = DeepSeekOCRTextMLXConfig(
            model="mlx-community/DeepSeek-OCR-4bit",
        )
        ```
    """

    model: str = Field(
        default="mlx-community/DeepSeek-OCR-4bit",
        description="MLX HuggingFace model ID. "
        "Options: 'mlx-community/DeepSeek-OCR-4bit' (recommended), "
        "'mlx-community/DeepSeek-OCR-8bit' (higher fidelity). "
        "Check mlx-community for DeepSeek-OCR-2 variants.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses HF_HOME or default.",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for greedy decoding.",
    )

    model_config = ConfigDict(extra="forbid")
