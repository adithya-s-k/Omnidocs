"""VLLM backend configuration for MinerU VL layout detection."""

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MinerUVLLayoutVLLMConfig(BaseModel):
    """
    VLLM backend config for MinerU VL layout detection.

    Example:
        ```python
        from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutVLLMConfig

        detector = MinerUVLLayoutDetector(
            backend=MinerUVLLayoutVLLMConfig(tensor_parallel_size=1)
        )
        result = detector.extract(image)
        ```
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default="opendatalab/MinerU2.5-2509-1.2B",
        description="Model ID",
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
        description="Fraction of GPU memory to use",
    )
    max_model_len: int = Field(
        default=16384,
        ge=1024,
        description="Maximum model sequence length",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code from HuggingFace",
    )
    enforce_eager: bool = Field(
        default=True,
        description="Disable CUDA graphs for faster startup",
    )
    download_dir: Optional[str] = Field(
        default=None,
        description="Directory for model downloads",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom all-reduce kernel",
    )
    layout_image_size: Tuple[int, int] = Field(
        default=(1036, 1036),
        description="Resize image to this size for layout detection",
    )
    max_tokens: int = Field(
        default=4096,
        ge=256,
        le=16384,
        description="Maximum tokens to generate",
    )
