"""MLX backend configuration for Granite Docling text extraction (Apple Silicon)."""

from pydantic import BaseModel, ConfigDict, Field


class GraniteDoclingTextMLXConfig(BaseModel):
    """Configuration for Granite Docling text extraction with MLX backend.

    This backend is optimized for Apple Silicon Macs (M1/M2/M3/M4).
    Uses the MLX-optimized model variant.
    """

    model: str = Field(
        default="ibm-granite/granite-docling-258M-mlx",
        description="MLX model path or HuggingFace ID",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=8192,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 for deterministic)",
    )

    model_config = ConfigDict(extra="forbid")
