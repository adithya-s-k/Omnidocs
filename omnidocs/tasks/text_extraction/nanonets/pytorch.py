"""
PyTorch/HuggingFace backend configuration for Nanonets OCR2-3B text extraction.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class NanonetsTextPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend configuration for Nanonets OCR2-3B text extraction.

    This backend uses the transformers library with PyTorch for local GPU inference.
    Requires: torch, transformers, accelerate

    Example:
        ```python
        config = NanonetsTextPyTorchConfig(
                device="cuda",
                torch_dtype="float16",
            )
        ```
    """

    model: str = Field(
        default="nanonets/Nanonets-OCR2-3B",
        description="HuggingFace model ID for Nanonets OCR2.",
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on. Options: 'cuda', 'mps', 'cpu'. "
        "Auto-detects best available if specified device is unavailable.",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="float16",
        description="Torch dtype for model weights.",
    )
    device_map: Optional[str] = Field(
        default="auto",
        description="Device map strategy for model parallelism. Options: 'auto', 'balanced', 'sequential', or None.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from HuggingFace Hub.",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Use SDPA attention for faster inference.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODEL_CACHE env var or default cache.",
    )
    max_new_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for greedy decoding.",
    )

    model_config = ConfigDict(extra="forbid")
