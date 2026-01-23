"""
Base class for text extractors.

Defines the abstract interface that all text extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union

import numpy as np
from PIL import Image

from .models import TextOutput


class BaseTextExtractor(ABC):
    """
    Abstract base class for text extractors.

    All text extraction models must inherit from this class and implement
    the required methods.

    Example:
        >>> class MyTextExtractor(BaseTextExtractor):
        ...     def __init__(self, config: MyConfig):
        ...         self.config = config
        ...         self._load_model()
        ...
        ...     def _load_model(self):
        ...         # Load model weights
        ...         pass
        ...
        ...     def extract(self, image, output_format="markdown"):
        ...         # Run extraction
        ...         return TextOutput(...)
    """

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load model weights into memory.

        This method should handle:
        - Downloading model if not present locally
        - Loading model onto the configured device
        - Setting model to evaluation mode

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If model loading fails
        """
        pass

    @abstractmethod
    def extract(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        output_format: Literal["html", "markdown"] = "markdown",
    ) -> TextOutput:
        """
        Extract text from an image.

        Args:
            image: Input image as:
                - PIL.Image.Image: PIL image object
                - np.ndarray: Numpy array (HWC format, RGB)
                - str or Path: Path to image file
            output_format: Desired output format:
                - "html": Structured HTML
                - "markdown": Markdown format

        Returns:
            TextOutput containing extracted text content

        Raises:
            ValueError: If image format or output_format is not supported
            RuntimeError: If model is not loaded or inference fails
        """
        pass

    def _prepare_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
        """
        Convert various input formats to PIL Image.

        Args:
            image: Input in various formats

        Returns:
            PIL Image in RGB mode

        Raises:
            ValueError: If input format is not supported
            FileNotFoundError: If image path does not exist
        """
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
