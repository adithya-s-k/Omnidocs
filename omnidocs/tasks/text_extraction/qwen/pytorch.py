"""
PyTorch/HuggingFace backend configuration for Qwen3-VL text extraction.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class QwenTextPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend configuration for Qwen text extraction.

    This backend uses the transformers library with PyTorch for local GPU inference.
    Requires: torch, transformers, accelerate, qwen-vl-utils

    Example:
        >>> config = QwenTextPyTorchConfig(
        ...     model="Qwen/Qwen3-VL-8B-Instruct",
        ...     device="cuda",
        ...     torch_dtype="bfloat16",
        ... )
    """

    model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="HuggingFace model ID (e.g., Qwen/Qwen3-VL-2B-Instruct, "
        "Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct)",
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on. Options: 'cuda', 'mps', 'cpu'. "
        "Auto-detects best available if specified device is unavailable.",
    )
    torch_dtype: Literal["float16", "bfloat16", "float32", "auto"] = Field(
        default="auto",
        description="Torch dtype for model weights. 'auto' lets the model decide.",
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
        description="Use Flash Attention 2 for faster inference. Requires flash-attn package to be installed.",
    )
    max_new_tokens: int = Field(
        default=8192,
        ge=256,
        le=32768,
        description="Maximum number of tokens to generate. "
        "Text extraction typically needs more tokens than layout detection.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower values are more deterministic.",
    )

    model_config = ConfigDict(extra="forbid")
