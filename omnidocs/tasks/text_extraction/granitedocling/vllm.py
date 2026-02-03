"""VLLM backend configuration for Granite Docling text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GraniteDoclingTextVLLMConfig(BaseModel):
    """Configuration for Granite Docling text extraction with VLLM backend.

    IMPORTANT: This config uses revision="untied" by default, which is required
    for VLLM compatibility with Granite Docling's tied weights.
    """

    model: str = Field(
        default="ibm-granite/granite-docling-258M",
        description="HuggingFace model ID for Granite Docling",
    )
    revision: str = Field(
        default="untied",
        description="Model revision. IMPORTANT: Must use 'untied' for VLLM compatibility",
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism",
    )
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization fraction",
    )
    max_model_len: int = Field(
        default=8192,
        ge=1024,
        description="Maximum model context length",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code for model loading",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graphs for debugging",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=16384,
        description="Maximum tokens to generate per request",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 for deterministic)",
    )
    download_dir: Optional[str] = Field(
        default=None,
        description="Directory for model downloads",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom all-reduce for multi-GPU",
    )
    fast_boot: bool = Field(
        default=True,
        description="Disable torch.compile and CUDA graphs for faster startup",
    )
    limit_mm_per_prompt: int = Field(
        default=1,
        ge=1,
        description="Maximum images per prompt (Granite Docling processes 1 at a time)",
    )

    model_config = ConfigDict(extra="forbid")
