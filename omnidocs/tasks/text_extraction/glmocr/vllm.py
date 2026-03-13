"""VLLM backend configuration for GLM-OCR text extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GLMOCRVLLMConfig(BaseModel):
    """
    VLLM backend configuration for GLM-OCR.

    GLM-OCR supports VLLM with MTP (Multi-Token Prediction) speculative decoding
    for significantly higher throughput. Requires vllm>=0.17.0 and transformers>=5.3.0.

    Example:
```python
        config = GLMOCRVLLMConfig(gpu_memory_utilization=0.85)
```
    """

    model: str = Field(
        default="zai-org/GLM-OCR",
        description="HuggingFace model ID.",
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
        description="Fraction of GPU memory to use.",
    )
    max_model_len: int = Field(
        default=8192,
        ge=512,
        description="Maximum sequence length.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code.",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graph optimization.",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    repetition_penalty: float = Field(
        default=1.05,
        ge=1.0,
        le=2.0,
        description="Repetition penalty to prevent looping at temperature=0.0.",
    )
    download_dir: Optional[str] = Field(
        default=None,
        description="Directory to download model weights.",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom all-reduce for tensor parallelism.",
    )

    model_config = ConfigDict(extra="forbid")