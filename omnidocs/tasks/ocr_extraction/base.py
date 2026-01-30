"""
Base class for OCR extractors.

Defines the abstract interface that all OCR extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .models import OCROutput


class BaseOCRExtractor(ABC):
    """
    Abstract base class for OCR extractors.

    All OCR extraction models must inherit from this class and implement
    the required methods.

    Example:
        >>> class MyOCRExtractor(BaseOCRExtractor):
        ...     def __init__(self, config: MyConfig):
        ...         self.config = config
        ...         self._load_model()
        ...
        ...     def _load_model(self):
        ...         # Initialize OCR engine
        ...         pass
        ...
        ...     def extract(self, image):
        ...         # Run OCR extraction
        ...         return OCROutput(...)
    """

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load model/initialize OCR engine.

        This method should handle:
        - Installing/verifying required dependencies
        - Loading model weights if applicable
        - Setting up the OCR engine

        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    def extract(self, image: Union[Image.Image, np.ndarray, str, Path]) -> OCROutput:
        """
        Run OCR extraction on an image.

        Args:
            image: Input image as:
                - PIL.Image.Image: PIL image object
                - np.ndarray: Numpy array (HWC format, RGB)
                - str or Path: Path to image file

        Returns:
            OCROutput containing detected text blocks with bounding boxes

        Raises:
            ValueError: If image format is not supported
            RuntimeError: If OCR engine is not initialized or extraction fails
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
            # Use context manager to ensure file handle is properly closed
            with Image.open(path) as img:
                # Load image data into memory before closing the file
                return img.convert("RGB").copy()

        raise ValueError(f"Unsupported image type: {type(image)}. Expected PIL.Image, numpy array, or file path.")
