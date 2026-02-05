"""PyTorch/HuggingFace backend configuration for MinerU VL text extraction."""

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MinerUVLTextPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend config for MinerU VL text extraction.

    Uses HuggingFace Transformers with Qwen2VLForConditionalGeneration.

    Example:
        ```python
        from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        extractor = MinerUVLTextExtractor(
            backend=MinerUVLTextPyTorchConfig(device="cuda")
        )
        result = extractor.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="opendatalab/MinerU2.5-2509-1.2B",
        description="HuggingFace model ID",
    )
    device: Literal["cuda", "cpu", "mps", "auto"] = Field(
        default="auto",
        description="Device for inference",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="float16",
        description="Model dtype",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Use Flash Attention 2 if available (requires flash-attn package). Uses SDPA by default.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code from HuggingFace",
    )
    layout_image_size: Tuple[int, int] = Field(
        default=(1036, 1036),
        description="Resize image to this size for layout detection",
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
