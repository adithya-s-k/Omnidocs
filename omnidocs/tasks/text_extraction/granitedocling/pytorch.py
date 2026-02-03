"""PyTorch backend configuration for Granite Docling text extraction."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GraniteDoclingTextPyTorchConfig(BaseModel):
    """Configuration for Granite Docling text extraction with PyTorch backend."""

    model: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="HuggingFace model ID for Granite Docling",
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on ('cuda', 'mps', 'cpu')",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="bfloat16",
        description="Torch dtype for model weights",
    )
    device_map: Optional[str] = Field(
        default="auto",
        description="Device map for model parallelism",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code for model loading",
    )
    use_flash_attention: bool = Field(
        default=True,
        description="Use flash attention 2 if available",
    )
    max_new_tokens: int = Field(
        default=8192,
        ge=256,
        le=16384,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 for deterministic)",
    )
    do_sample: bool = Field(
        default=True,
        description="Whether to use sampling during generation",
    )

    model_config = ConfigDict(extra="forbid")
