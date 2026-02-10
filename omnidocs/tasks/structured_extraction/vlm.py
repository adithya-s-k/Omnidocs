"""
VLM structured extractor.

A provider-agnostic Vision-Language Model structured extractor using litellm.
Extracts structured data matching a Pydantic schema from document images.

Example:
    ```python
    from pydantic import BaseModel
    from omnidocs.vlm import VLMAPIConfig
    from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

    class Invoice(BaseModel):
        vendor: str
        total: float
        items: list[str]
        date: str

    config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
    extractor = VLMStructuredExtractor(config=config)

    result = extractor.extract(
        image="invoice.png",
        schema=Invoice,
        prompt="Extract invoice details from this document.",
    )
    print(result.data.vendor, result.data.total)
    ```
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from pydantic import BaseModel

from omnidocs.vlm import VLMAPIConfig, vlm_structured_completion

from .base import BaseStructuredExtractor
from .models import StructuredOutput


class VLMStructuredExtractor(BaseStructuredExtractor):
    """
    Provider-agnostic VLM structured extractor using litellm.

    Extracts structured data from document images using any cloud VLM API.
    Uses litellm's native response_format support to send Pydantic schemas
    to providers that support structured output (OpenAI, Gemini, etc.).

    Example:
        ```python
        from pydantic import BaseModel
        from omnidocs.vlm import VLMAPIConfig
        from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

        class Invoice(BaseModel):
            vendor: str
            total: float
            items: list[str]

        config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
        extractor = VLMStructuredExtractor(config=config)
        result = extractor.extract("invoice.png", schema=Invoice, prompt="Extract invoice fields")
        print(result.data.vendor)
        ```
    """

    def __init__(self, config: VLMAPIConfig):
        """
        Initialize VLM structured extractor.

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
        schema: type[BaseModel],
        prompt: str,
    ) -> StructuredOutput:
        """
        Extract structured data from an image.

        Args:
            image: Input image (PIL Image, numpy array, or file path).
            schema: Pydantic model class defining the expected output structure.
            prompt: Extraction prompt describing what to extract.

        Returns:
            StructuredOutput containing the validated data.
        """
        pil_image = self._prepare_image(image)
        width, height = pil_image.size

        data = vlm_structured_completion(self.config, prompt, pil_image, schema)

        return StructuredOutput(
            data=data,
            image_width=width,
            image_height=height,
            model_name=f"VLM ({self.config.model})",
        )
