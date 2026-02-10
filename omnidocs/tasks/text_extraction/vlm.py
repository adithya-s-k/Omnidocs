"""
VLM text extractor.

A provider-agnostic Vision-Language Model text extractor using litellm.
Works with any cloud API: Gemini, OpenRouter, Azure, OpenAI, Anthropic, etc.

Example:
    ```python
    from omnidocs.vlm import VLMAPIConfig
    from omnidocs.tasks.text_extraction import VLMTextExtractor

    config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
    extractor = VLMTextExtractor(config=config)
    result = extractor.extract("document.png", output_format="markdown")
    print(result.content)

    # With custom prompt
    result = extractor.extract("document.png", prompt="Extract only table data as markdown")
    ```
"""

import re
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from PIL import Image

from omnidocs.vlm import VLMAPIConfig, vlm_completion

from .base import BaseTextExtractor
from .models import OutputFormat, TextOutput

DEFAULT_PROMPTS = {
    "markdown": (
        "Extract all text from this document image and return it in clean Markdown format. "
        "Preserve the document structure including headings, lists, tables, and paragraphs. "
        "For tables, use Markdown table syntax. For formulas, use LaTeX notation."
    ),
    "html": (
        "Extract all text from this document image and return it as clean HTML. "
        "Preserve the document structure using appropriate HTML tags: "
        "<h1>-<h6> for headings, <p> for paragraphs, <ul>/<ol> for lists, "
        "<table> for tables, and <code> for code blocks."
    ),
}


def _extract_plain_text(output: str, output_format: str) -> str:
    """Extract plain text from HTML or Markdown output."""
    if output_format == "html":
        text = re.sub(r"<[^>]+>", " ", output)
    else:
        text = re.sub(r"```[^`]*```", "", output)
        text = re.sub(r"<!--[^>]+-->", "", text)
        text = re.sub(r"\|[-:]+\|", "", text)
        text = re.sub(r"[#*_`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class VLMTextExtractor(BaseTextExtractor):
    """
    Provider-agnostic VLM text extractor using litellm.

    Works with any cloud VLM API: Gemini, OpenRouter, Azure, OpenAI,
    Anthropic, etc. Supports custom prompts for specialized extraction.

    Example:
        ```python
        from omnidocs.vlm import VLMAPIConfig
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        # Gemini (reads GOOGLE_API_KEY from env)
        config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
        extractor = VLMTextExtractor(config=config)

        # Default extraction
        result = extractor.extract("document.png", output_format="markdown")

        # Custom prompt
        result = extractor.extract(
            "document.png",
            prompt="Extract only the table data as markdown",
        )
        ```
    """

    def __init__(self, config: VLMAPIConfig):
        """
        Initialize VLM text extractor.

        Args:
            config: VLM API configuration with model and provider details.
        """
        self.config = config
        self._loaded = True

    def _load_model(self) -> None:
        """No-op for API-only extractor."""
        pass

    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
        prompt: Optional[str] = None,
    ) -> TextOutput:
        """
        Extract text from an image.

        Args:
            image: Input image (PIL Image, numpy array, or file path).
            output_format: Desired output format ("html" or "markdown").
            prompt: Custom prompt. If None, uses a task-specific default prompt.

        Returns:
            TextOutput containing extracted text content.
        """
        if output_format not in ("html", "markdown"):
            raise ValueError(f"Invalid output_format: {output_format}. Expected 'html' or 'markdown'.")

        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        final_prompt = prompt or DEFAULT_PROMPTS[output_format]
        raw_output = vlm_completion(self.config, final_prompt, pil_image)
        plain_text = _extract_plain_text(raw_output, output_format)

        return TextOutput(
            content=raw_output,
            format=OutputFormat(output_format),
            raw_output=raw_output,
            plain_text=plain_text,
            image_width=width,
            image_height=height,
            model_name=f"VLM ({self.config.model})",
        )
