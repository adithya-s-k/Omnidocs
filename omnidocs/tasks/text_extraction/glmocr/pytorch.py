"""PyTorch backend configuration for GLM-OCR text extraction."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GLMOCRPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend configuration for GLM-OCR.

    GLM-OCR uses AutoModelForImageTextToText + AutoProcessor.
    Requires transformers>=5.3.0.

    Example:
```python
        config = GLMOCRPyTorchConfig()  # zai-org/GLM-OCR, default
        config = GLMOCRPyTorchConfig(device="mps")  # Apple Silicon
```
    """

    model: str = Field(
        default="zai-org/GLM-OCR",
        description="HuggingFace model ID. Default: 'zai-org/GLM-OCR' (0.9B, Feb 2026).",
    )
    device: str = Field(
        default="cuda",
        description="Device: 'cuda', 'mps', 'cpu'.",
    )
    torch_dtype: Literal["bfloat16", "float16", "float32", "auto"] = Field(
        default="bfloat16",
        description="Torch dtype. bfloat16 recommended.",
    )
    device_map: Optional[str] = Field(
        default="auto",
        description="Device map for model parallelism.",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Use Flash Attention 2. Requires flash-attn installed.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory.",
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for greedy decoding.",
    )

    model_config = ConfigDict(extra="forbid")