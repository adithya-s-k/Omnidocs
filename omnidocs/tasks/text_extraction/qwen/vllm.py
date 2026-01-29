"""
VLLM backend configuration for Qwen3-VL text extraction.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QwenTextVLLMConfig(BaseModel):
    """
    VLLM backend configuration for Qwen text extraction.

    This backend uses VLLM for high-throughput inference.
    Best for batch processing and production deployments.
    Requires: vllm, torch, transformers, qwen-vl-utils

    Example:
        >>> config = QwenTextVLLMConfig(
        ...     model="Qwen/Qwen3-VL-8B-Instruct",
        ...     tensor_parallel_size=2,
        ...     gpu_memory_utilization=0.9,
        ... )
    """

    model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        description="HuggingFace model ID (e.g., Qwen/Qwen3-VL-2B-Instruct, "
        "Qwen/Qwen3-VL-4B-Instruct, Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct)",
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism. Use higher values for larger models.",
    )
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use (0.0-1.0). Lower values leave headroom for other processes.",
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
        description="Disable CUDA graph optimization for faster cold start. "
        "Useful for Modal deployments where startup time matters.",
    )
    max_tokens: int = Field(
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
    download_dir: Optional[str] = Field(
        default=None,
        description="Directory to download model weights. If None, uses OMNIDOCS_MODEL_CACHE env var or default cache.",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom all-reduce for tensor parallelism. Set to False for best performance.",
    )

    model_config = ConfigDict(extra="forbid")
