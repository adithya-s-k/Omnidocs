"""
PyTorch/HuggingFace backend configuration for DeepSeek-OCR text extraction.

Both DeepSeek-OCR and DeepSeek-OCR-2 use:
  - AutoModel (not AutoModelForCausalLM)
  - AutoTokenizer (not AutoProcessor)
  - model.infer(tokenizer, prompt=..., image_file=...) for inference

Requirements (from official README):
  python==3.12.9, CUDA==11.8
  torch==2.6.0, transformers==4.46.3, tokenizers==0.20.3
  einops, addict, easydict
  flash-attn==2.7.3 (optional, --no-build-isolation)
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DeepSeekOCRTextPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend configuration for DeepSeek-OCR / DeepSeek-OCR-2.

    Uses AutoModel + AutoTokenizer. Inference via model.infer() — the model
    handles tiling and multi-page PDF stitching internally.

    Models:
        deepseek-ai/DeepSeek-OCR-2  (default, latest — Jan 2026, Apache 2.0)
        deepseek-ai/DeepSeek-OCR    (v1 — Oct 2024, MIT)

    GPU requirements: L4 / A100 (≥16 GB VRAM recommended).

    Example:
        ```python
        config = DeepSeekOCRTextPyTorchConfig(
            model="deepseek-ai/DeepSeek-OCR-2",
            use_flash_attention=True,  # requires flash-attn==2.7.3
        )
        ```
    """

    model: str = Field(
        default="deepseek-ai/DeepSeek-OCR-2",
        description="HuggingFace model ID. "
        "'deepseek-ai/DeepSeek-OCR-2' (latest, Jan 2026, Apache 2.0) or "
        "'deepseek-ai/DeepSeek-OCR' (v1, Oct 2024, MIT).",
    )
    device: str = Field(
        default="cuda",
        description="Device to run inference on. Options: 'cuda', 'cpu'. "
        "MPS not tested for DeepSeek-OCR.",
    )
    torch_dtype: Literal["bfloat16", "float16", "float32", "auto"] = Field(
        default="bfloat16",
        description="Torch dtype. BF16 is required per the official README.",
    )
    use_flash_attention: bool = Field(
        default=False,
        description="Use Flash Attention 2 (_attn_implementation='flash_attention_2'). "
        "Requires flash-attn==2.7.3 installed with --no-build-isolation. "
        "Falls back to 'eager' if False.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Required — DeepSeek-OCR uses custom model code.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory. If None, uses OMNIDOCS_MODELS_DIR env var or default cache.",
    )
    # Inference parameters passed to model.infer()
    base_size: int = Field(
        default=1024,
        ge=512,
        le=2048,
        description="Base canvas size for the visual encoder patch grid.",
    )
    image_size: int = Field(
        default=768,
        ge=256,
        le=1024,
        description="Size each image tile is resized to before encoding.",
    )
    crop_mode: bool = Field(
        default=True,
        description="Enable adaptive tiling ('Gundam mode'): splits dense pages into "
        "overlapping tiles for better recognition of small fonts.",
    )

    model_config = ConfigDict(extra="forbid")
