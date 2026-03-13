"""
VLLM backend configuration for DeepSeek-OCR text extraction.

DeepSeek-OCR has official upstream VLLM support (announced Oct 23 2025).
Achieves ~2500 tokens/s on A100-40G — the recommended backend for production.

DeepSeek-OCR-2 VLLM support: refer to https://github.com/deepseek-ai/DeepSeek-OCR-2
for the latest vLLM setup instructions (may require nightly build).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DeepSeekOCRTextVLLMConfig(BaseModel):
    """
    VLLM backend configuration for DeepSeek-OCR / DeepSeek-OCR-2 text extraction.

    DeepSeek-OCR has official upstream VLLM support (~2500 tokens/s on A100).
    Recommended for high-throughput batch document processing in production.
    Requires: vllm>=0.11.1 (or nightly for OCR-2), torch, transformers==4.46.3

    Example:
        ```python
        config = DeepSeekOCRTextVLLMConfig(
            model="deepseek-ai/DeepSeek-OCR-2",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        ```
    """

    model: str = Field(
        default="deepseek-ai/DeepSeek-OCR",
        description="HuggingFace model ID. "
        "'deepseek-ai/DeepSeek-OCR-2' (latest) or 'deepseek-ai/DeepSeek-OCR' (v1).",
    )
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism.",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use.",
    )
    max_model_len: int = Field(
        default=8192,
        ge=1024,
        description="Maximum sequence length for the model context.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Required — DeepSeek-OCR uses custom model code.",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graph optimization. Useful for Modal cold-start.",
    )
    max_tokens: int = Field(
        default=8192,
        ge=256,
        le=8192,
        description="Maximum number of tokens to generate per page.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 for greedy decoding.",
    )
    download_dir: Optional[str] = Field(
        default=None,
        description="Directory to download model weights. If None, uses OMNIDOCS_MODELS_DIR.",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom all-reduce for tensor parallelism.",
    )
    enable_prefix_caching: bool = Field(
        default=False,
        description="Must be False for DeepSeek-OCR v1 (incompatible with NGram processor).",
    )
    mm_processor_cache_gb: float = Field(
        default=0,
        description="Must be 0 for DeepSeek-OCR v1.",
    )
    use_ngram_logits_processor: bool = Field(
        default=True,
        description="Enable NGramPerReqLogitsProcessor — required for correct v1 output.",
    )
    ngram_size: int = Field(
        default=30,
        description="NGram window size for the logits processor.",
    )
    ngram_window_size: int = Field(
        default=90,
        description="NGram context window size for the logits processor.",
    )
    skip_special_tokens: bool = Field(
        default=False,
        description="Whether to skip special tokens in output. Must be False for v1.",
    )
    model_config = ConfigDict(extra="forbid")
