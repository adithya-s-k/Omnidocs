"""
Base class for table extractors.

Defines the abstract interface that all table extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import numpy as np
from PIL import Image

from .models import TableOutput

if TYPE_CHECKING:
    from omnidocs.document import Document
    from omnidocs.tasks.ocr_extraction.models import OCROutput


class BaseTableExtractor(ABC):
    """
    Abstract base class for table structure extractors.

    Table extractors analyze table images to detect cell structure,
    identify headers, and extract text content.

    Example:
        ```python
        class MyTableExtractor(BaseTableExtractor):
            def __init__(self, config: MyConfig):
                self.config = config
                self._load_model()

            def _load_model(self):
                # Load model weights
                pass

            def extract(self, image):
                # Run extraction
                return TableOutput(...)
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
        ocr_output: Optional["OCROutput"] = None,
    ) -> TableOutput:
        """
        Extract table structure from an image.

        Args:
            image: Table image (should be cropped to table region)
            ocr_output: Optional OCR results for cell text matching.
                       If not provided, model will attempt to extract text.

        Returns:
            TableOutput with cells, structure, and export methods

        Example:
            ```python
            # Without OCR (model extracts text)
            result = extractor.extract(table_image)

            # With OCR (better text quality)
            ocr = some_ocr.extract(table_image)
            result = extractor.extract(table_image, ocr_output=ocr)
            ```
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
        ocr_outputs: Optional[List["OCROutput"]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TableOutput]:
        """
        Extract tables from multiple images.

        Default implementation loops over extract(). Subclasses can override
        for optimized batching.

        Args:
            images: List of table images
            ocr_outputs: Optional list of OCR results (same length as images)
            progress_callback: Optional function(current, total) for progress

        Returns:
            List of TableOutput in same order as input

        Examples:
            ```python
            results = extractor.batch_extract(table_images)
            ```
        """
        results = []
        total = len(images)

        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)

            ocr = ocr_outputs[i] if ocr_outputs else None
            result = self.extract(image, ocr_output=ocr)
            results.append(result)

        return results

    def extract_document(
        self,
        document: "Document",
        table_bboxes: Optional[List[List[float]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TableOutput]:
        """
        Extract tables from all pages of a document.

        Args:
            document: Document instance
            table_bboxes: Optional list of table bounding boxes per page.
                         Each element should be a list of [x1, y1, x2, y2] coords.
            progress_callback: Optional function(current, total) for progress

        Returns:
            List of TableOutput, one per detected table

        Examples:
            ```python
            doc = Document.from_pdf("paper.pdf")
            results = extractor.extract_document(doc)
            ```
        """
        results = []
        total = document.page_count

        for i, page in enumerate(document.iter_pages()):
            if progress_callback:
                progress_callback(i + 1, total)

            # If no bboxes provided, process entire page
            if table_bboxes is None:
                result = self.extract(page)
                results.append(result)
            else:
                # Crop and process each table region
                for bbox in table_bboxes:
                    x1, y1, x2, y2 = bbox
                    table_region = page.crop((x1, y1, x2, y2))
                    result = self.extract(table_region)
                    results.append(result)

        return results
