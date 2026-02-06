"""VLLM backend configuration for Dots OCR."""

from typing import Literal

from pydantic import BaseModel, Field


class DotsOCRVLLMConfig(BaseModel):
    """
    VLLM backend configuration for Dots OCR.

    VLLM provides high-throughput inference with optimizations like:
    - PagedAttention for efficient KV cache management
    - Continuous batching for higher throughput
    - Optimized CUDA kernels

    Example:
        ```python
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        config = DotsOCRVLLMConfig(
                model="rednote-hilab/dots.ocr",
                tensor_parallel_size=2,
                gpu_memory_utilization=0.9,
            )
        extractor = DotsOCRTextExtractor(backend=config)
        ```
    """

    model: str = Field(
        default="rednote-hilab/dots.ocr",
        description="HuggingFace model ID for Dots OCR",
    )
    tensor_parallel_size: int = Field(default=1, ge=1, description="Number of GPUs for tensor parallelism")
    gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=1.0, description="GPU memory utilization (0.1-1.0)")
    max_model_len: int = Field(
        default=32768,
        ge=1024,
        description="Maximum context length (Dots OCR supports up to 32K)",
    )
    trust_remote_code: bool = Field(default=True, description="Trust remote code for model loading")
    dtype: Literal["float16", "bfloat16"] = Field(
        default="bfloat16", description="Data type for inference (bfloat16 recommended)"
    )
    enforce_eager: bool = Field(
        default=False,
        description="Use eager execution mode (disable CUDA graphs for faster cold start)",
    )
    disable_custom_all_reduce: bool = Field(
        default=False, description="Disable custom all-reduce for tensor parallelism"
    )
    download_dir: str | None = Field(
        default=None,
        description="Model download directory. If None, uses OMNIDOCS_MODEL_CACHE env var or default cache.",
    )

    class Config:
        extra = "forbid"  # Raise error on unknown params
