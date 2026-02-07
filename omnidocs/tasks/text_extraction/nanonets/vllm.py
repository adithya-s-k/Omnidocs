"""
VLLM backend configuration for Nanonets OCR2-3B text extraction.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NanonetsTextVLLMConfig(BaseModel):
    """
    VLLM backend configuration for Nanonets OCR2-3B text extraction.

    This backend uses VLLM for high-throughput inference.
    Best for batch processing and production deployments.
    Requires: vllm, torch, transformers, qwen-vl-utils

    Example:
        ```python
        config = NanonetsTextVLLMConfig(
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
            )
        ```
    """

    model: str = Field(
        default="nanonets/Nanonets-OCR2-3B",
        description="HuggingFace model ID for Nanonets OCR2.",
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism.",
    )
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use (0.0-1.0).",
    )
    max_model_len: int = Field(
        default=32768,
        ge=1024,
        description="Maximum sequence length for the model context.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from HuggingFace Hub.",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graph optimization for faster cold start.",
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
        description="Sampling temperature. 0.0 for greedy decoding.",
    )
    download_dir: Optional[str] = Field(
        default=None,
        description="Directory to download model weights. If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom all-reduce for tensor parallelism.",
    )

    model_config = ConfigDict(extra="forbid")
