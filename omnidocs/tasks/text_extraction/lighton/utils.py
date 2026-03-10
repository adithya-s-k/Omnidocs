"""Utility functions for LightOn text extraction."""

from typing import Dict, List

from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """Sampling parameters for text generation."""

    max_tokens: int = Field(default=4096, ge=1, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


DEFAULT_PROMPTS = {
    "transcribe": "Transcribe this document.",
    "extract": "Extract all text from this document.",
    "ocr": "Perform OCR on this document.",
}

DEFAULT_SAMPLING_PARAMS = SamplingParams(
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
)

SYSTEM_PROMPT = "You are a document parser and OCR system. Extract all text content from documents accurately."


def get_rgb_image(image):
    """Ensure image is in RGB format."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def simple_post_process(text: str) -> str:
    """
    Simple post-processing of extracted text.

    Removes common artifacts and cleans up formatting.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Remove special tokens
    for token in ["<|im_start|>", "<|im_end|>", "<|end_header_id|>"]:
        text = text.replace(token, "")

    # Remove extra whitespace
    text = text.strip()
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join([line for line in lines if line])

    return text


def parse_layout_output(raw_output: str) -> Dict[str, List[str]]:
    """
    Parse layout detection output.

    LightOn outputs plain text. This extracts sections based on heuristics.

    Args:
        raw_output: Raw model output

    Returns:
        Dict with layout sections
    """
    text = simple_post_process(raw_output)

    # Split by double newlines as potential section breaks
    sections = text.split("\n\n")

    return {
        "text": text,
        "sections": sections,
        "line_count": len(text.split("\n")),
    }
