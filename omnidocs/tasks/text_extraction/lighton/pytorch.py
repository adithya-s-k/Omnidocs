"""PyTorch/HuggingFace backend configuration for LightOn text extraction."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LightOnTextPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend config for LightOn text extraction.

    Uses HuggingFace Transformers with LightOnOcrForConditionalGeneration.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import LightOnTextExtractor
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        extractor = LightOnTextExtractor(
            backend=LightOnTextPyTorchConfig(device="cuda", torch_dtype="bfloat16")
        )
        result = extractor.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="lightonai/LightOnOCR-2-1B",
        description="HuggingFace model ID",
    )
    device: Literal["cuda", "cpu", "mps", "auto"] = Field(
        default="auto",
        description="Device for inference",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="bfloat16",
        description="Model dtype (bfloat16 recommended for LightOn)",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Use Flash Attention 2 if available (requires flash-attn package). Uses SDPA by default.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code from HuggingFace",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate",
    )
    device_map: Optional[str] = Field(
        default="auto",
        description="Device map for model parallelism",
    )
