"""
Base class for structured extractors.

Defines the abstract interface for extracting structured data from document images.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from pydantic import BaseModel

from .models import StructuredOutput


class BaseStructuredExtractor(ABC):
    """
    Abstract base class for structured extractors.

    Structured extractors return data matching a user-provided Pydantic schema.

    Example:
        ```python
        class MyExtractor(BaseStructuredExtractor):
            def __init__(self, config):
                self.config = config

            def _load_model(self):
                pass

            def extract(self, image, schema, prompt):
                return StructuredOutput(data=schema(...), ...)
        ```
    """

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights into memory."""
        pass

    @abstractmethod
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
        pass

    def _prepare_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """Convert various input formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB")

        raise ValueError(f"Unsupported image type: {type(image)}. Expected PIL.Image, numpy array, or file path.")
