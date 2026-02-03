"""
Reading Order Module.

Provides predictors for determining the logical reading sequence of
document elements based on layout detection and spatial analysis.

Available Predictors:
    - RuleBasedReadingOrderPredictor: Rule-based predictor using R-tree indexing

Example:
    ```python
    from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor
    from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
    from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

    # Initialize components
    layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())
    ocr = EasyOCR(config=EasyOCRConfig())
    predictor = RuleBasedReadingOrderPredictor()

    # Process document
    layout = layout_extractor.extract(image)
    ocr_result = ocr.extract(image)
    reading_order = predictor.predict(layout, ocr_result)

    # Get text in reading order
    text = reading_order.get_full_text()

    # Get elements by type
    tables = reading_order.get_elements_by_type(ElementType.TABLE)

    # Get caption associations
    for elem in reading_order.ordered_elements:
        if elem.element_type == ElementType.FIGURE:
            captions = reading_order.get_captions_for(elem.original_id)
            print(f"Figure {elem.original_id} captions: {[c.text for c in captions]}")
    ```
"""

from .base import BaseReadingOrderPredictor
from .models import (
    NORMALIZED_SIZE,
    BoundingBox,
    ElementType,
    OrderedElement,
    ReadingOrderOutput,
)
from .rule_based import RuleBasedReadingOrderPredictor

__all__ = [
    # Base
    "BaseReadingOrderPredictor",
    # Models
    "BoundingBox",
    "ElementType",
    "OrderedElement",
    "ReadingOrderOutput",
    "NORMALIZED_SIZE",
    # Predictors
    "RuleBasedReadingOrderPredictor",
]
