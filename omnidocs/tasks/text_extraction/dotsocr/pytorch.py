"""PyTorch backend configuration for Dots OCR."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DotsOCRPyTorchConfig(BaseModel):
    """
    PyTorch/HuggingFace backend configuration for Dots OCR.

    Dots OCR provides layout-aware text extraction with 11 predefined layout
    categories (Caption, Footnote, Formula, List-item, Page-footer, Page-header,
    Picture, Section-header, Table, Text, Title).

    Example:
        ```python
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        config = DotsOCRPyTorchConfig(
                model="rednote-hilab/dots.ocr",
                device="cuda",
                torch_dtype="bfloat16",
            )
        extractor = DotsOCRTextExtractor(backend=config)
        ```
    """

    model: str = Field(
        default="rednote-hilab/dots.ocr",
        description="HuggingFace model ID for Dots OCR",
    )
    device: str = Field(default="cuda", description="Device to run on (cuda/cpu/mps)")
    torch_dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="bfloat16", description="Torch dtype for model weights"
    )
    trust_remote_code: bool = Field(default=True, description="Trust remote code for model loading")
    device_map: Optional[str] = Field(default="auto", description="Device mapping strategy (auto/balanced/sequential)")
    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = Field(
        default="flash_attention_2",
        description="Attention implementation (flash_attention_2 recommended for speed)",
    )

    class Config:
        extra = "forbid"  # Raise error on unknown params
