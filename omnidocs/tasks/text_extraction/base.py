"""
Base class for text extractors.

Defines the abstract interface that all text extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Union

import numpy as np
from PIL import Image

from .models import TextOutput

if TYPE_CHECKING:
    from omnidocs.document import Document


class BaseTextExtractor(ABC):
    """
    Abstract base class for text extractors.

    All text extraction models must inherit from this class and implement
    the required methods.

    Example:
        ```python
        class MyTextExtractor(BaseTextExtractor):
                def __init__(self, config: MyConfig):
                    self.config = config
                    self._load_model()

                def _load_model(self):
                    # Load model weights
                    pass

                def extract(self, image, output_format="markdown"):
                    # Run extraction
                    return TextOutput(...)
        ```
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

    def batch_extract(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        output_format: Literal["html", "markdown"] = "markdown",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TextOutput]:
        """
        Extract text from multiple images.

        Default implementation loops over extract(). Subclasses can override
        for optimized batching (e.g., VLLM).

        Args:
            images: List of images in any supported format
            output_format: Desired output format
            progress_callback: Optional function(current, total) for progress

        Returns:
            List of TextOutput in same order as input

        Examples:
            ```python
            images = [doc.get_page(i) for i in range(doc.page_count)]
            results = extractor.batch_extract(images, output_format="markdown")
            ```
        """
        results = []
        total = len(images)

        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self.extract(image, output_format=output_format)
            results.append(result)

        return results

    def extract_document(
        self,
        document: "Document",
        output_format: Literal["html", "markdown"] = "markdown",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TextOutput]:
        """
        Extract text from all pages of a document.

        Args:
            document: Document instance
            output_format: Desired output format
            progress_callback: Optional function(current, total) for progress

        Returns:
            List of TextOutput, one per page

        Examples:
            ```python
            doc = Document.from_pdf("paper.pdf")
            results = extractor.extract_document(doc, output_format="markdown")
            ```
        """
        results = []
        total = document.page_count

        for i, page in enumerate(document.iter_pages()):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self.extract(page, output_format=output_format)
            results.append(result)

        return results
