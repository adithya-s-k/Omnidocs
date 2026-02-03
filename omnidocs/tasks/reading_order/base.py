"""
Base class for reading order predictors.

Defines the abstract interface that all reading order predictors must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from .models import ReadingOrderOutput

if TYPE_CHECKING:
    from omnidocs.tasks.layout_extraction.models import LayoutOutput
    from omnidocs.tasks.ocr_extraction.models import OCROutput


class BaseReadingOrderPredictor(ABC):
    """
    Abstract base class for reading order predictors.

    Reading order predictors take layout detection and OCR results
    and produce a properly ordered sequence of document elements.

    Example:
        ```python
        predictor = RuleBasedReadingOrderPredictor()

        # Get layout and OCR
        layout = layout_extractor.extract(image)
        ocr = ocr_extractor.extract(image)

        # Predict reading order
        result = predictor.predict(layout, ocr)

        # Or with multiple pages
        results = predictor.predict_multi_page(layouts, ocrs)
        ```
    """

    @abstractmethod
    def predict(
        self,
        layout: "LayoutOutput",
        ocr: Optional["OCROutput"] = None,
        page_no: int = 0,
    ) -> ReadingOrderOutput:
        """
        Predict reading order for a single page.

        Args:
            layout: Layout detection results with bounding boxes
            ocr: Optional OCR results. If provided, text will be
                 matched to layout elements by bbox overlap.
            page_no: Page number (for multi-page documents)

        Returns:
            ReadingOrderOutput with ordered elements and associations

        Example:
            ```python
            layout = layout_extractor.extract(page_image)
            ocr = ocr_extractor.extract(page_image)
            order = predictor.predict(layout, ocr, page_no=0)
            ```
        """
        pass

    def predict_multi_page(
        self,
        layouts: List["LayoutOutput"],
        ocrs: Optional[List["OCROutput"]] = None,
    ) -> List[ReadingOrderOutput]:
        """
        Predict reading order for multiple pages.

        Args:
            layouts: List of layout results, one per page
            ocrs: Optional list of OCR results, one per page

        Returns:
            List of ReadingOrderOutput, one per page
        """
        results = []

        for i, layout in enumerate(layouts):
            ocr = ocrs[i] if ocrs else None
            result = self.predict(layout, ocr, page_no=i)
            results.append(result)

        return results
