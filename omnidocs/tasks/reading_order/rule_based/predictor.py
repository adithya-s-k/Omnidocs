"""
Rule-based reading order predictor.

Uses spatial analysis and R-tree indexing to determine the logical
reading sequence of document elements.
"""

from typing import TYPE_CHECKING, Dict, Optional

from omnidocs.tasks.reading_order.base import BaseReadingOrderPredictor
from omnidocs.tasks.reading_order.models import (
    BoundingBox,
    ElementType,
    OrderedElement,
    ReadingOrderOutput,
)

if TYPE_CHECKING:
    from omnidocs.tasks.layout_extraction.models import LayoutOutput
    from omnidocs.tasks.ocr_extraction.models import OCROutput


# Mapping from layout labels to reading order element types
LABEL_TO_ELEMENT_TYPE: Dict[str, ElementType] = {
    "title": ElementType.TITLE,
    "text": ElementType.TEXT,
    "list": ElementType.LIST,
    "figure": ElementType.FIGURE,
    "table": ElementType.TABLE,
    "caption": ElementType.CAPTION,
    "formula": ElementType.FORMULA,
    "footnote": ElementType.FOOTNOTE,
    "page_header": ElementType.PAGE_HEADER,
    "page_footer": ElementType.PAGE_FOOTER,
    "code": ElementType.CODE,
    "abandon": ElementType.OTHER,
    "unknown": ElementType.OTHER,
}


class RuleBasedReadingOrderPredictor(BaseReadingOrderPredictor):
    """
    Rule-based reading order predictor using spatial analysis.

    Uses R-tree spatial indexing and rule-based algorithms to determine
    the logical reading sequence of document elements. This is a CPU-only
    implementation that doesn't require GPU resources.

    Features:
    - Multi-column layout detection
    - Header/footer separation
    - Caption-to-figure/table association
    - Footnote linking
    - Element merge suggestions

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
        ```
    """

    def __init__(self):
        """Initialize the reading order predictor."""
        self._predictor = None
        self._load_predictor()

    def _load_predictor(self) -> None:
        """Load the underlying reading order predictor."""
        try:
            from docling_ibm_models.reading_order.reading_order_rb import (
                ReadingOrderPredictor,
            )

            self._predictor = ReadingOrderPredictor()
        except ImportError:
            raise ImportError(
                "docling-ibm-models is required for RuleBasedReadingOrderPredictor. "
                "Install with: pip install docling-ibm-models"
            )

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
            ocr: Optional OCR results for text content
            page_no: Page number (for multi-page documents)

        Returns:
            ReadingOrderOutput with ordered elements and associations

        Example:
            ```python
            layout = layout_extractor.extract(page_image)
            ocr = ocr_extractor.extract(page_image)
            order = predictor.predict(layout, ocr, page_no=0)
            print(order.get_full_text())
            ```
        """
        from docling_core.types.doc.base import CoordOrigin, Size
        from docling_ibm_models.reading_order.reading_order_rb import PageElement

        # Convert layout boxes to PageElements
        page_elements = []
        text_map: Dict[int, str] = {}

        # Build text map from OCR if available
        if ocr:
            text_map = self._build_text_map(layout, ocr)

        # Page size from layout
        page_width = layout.image_width
        page_height = layout.image_height

        for i, box in enumerate(layout.bboxes):
            # Convert label
            label_str = box.label.value.lower()
            docling_label = self._get_docling_label(label_str)

            # Create PageElement with bottom-left origin (docling convention)
            # Convert from top-left origin (our convention) to bottom-left
            page_elem = PageElement(
                cid=i,
                text=text_map.get(i, ""),
                page_no=page_no,
                page_size=Size(width=page_width, height=page_height),
                label=docling_label,
                l=box.bbox.x1,
                b=page_height - box.bbox.y2,  # Convert y2 to bottom
                r=box.bbox.x2,
                t=page_height - box.bbox.y1,  # Convert y1 to top
                coord_origin=CoordOrigin.BOTTOMLEFT,
            )
            page_elements.append(page_elem)

        # Run reading order prediction
        sorted_elements = self._predictor.predict_reading_order(page_elements=page_elements)

        # Get caption associations
        caption_map = self._predictor.predict_to_captions(sorted_elements=sorted_elements)

        # Get footnote associations
        footnote_map = self._predictor.predict_to_footnotes(sorted_elements=sorted_elements)

        # Get merge suggestions
        merge_map = self._predictor.predict_merges(sorted_elements=sorted_elements)

        # Convert to OrderedElements
        ordered_elements = []
        for idx, elem in enumerate(sorted_elements):
            element_type = LABEL_TO_ELEMENT_TYPE.get(
                str(elem.label.value).lower().replace("-", "_"),
                ElementType.OTHER,
            )

            # Convert back from bottom-left to top-left origin
            bbox = BoundingBox(
                x1=elem.l,
                y1=page_height - elem.t,  # Convert top back
                x2=elem.r,
                y2=page_height - elem.b,  # Convert bottom back
            )

            ordered_elem = OrderedElement(
                index=idx,
                element_type=element_type,
                bbox=bbox,
                text=elem.text,
                confidence=layout.bboxes[elem.cid].confidence if elem.cid < len(layout.bboxes) else 1.0,
                page_no=page_no,
                original_id=elem.cid,
            )
            ordered_elements.append(ordered_elem)

        return ReadingOrderOutput(
            ordered_elements=ordered_elements,
            caption_map=dict(caption_map),
            footnote_map=dict(footnote_map),
            merge_map=dict(merge_map),
            image_width=page_width,
            image_height=page_height,
            model_name="RuleBasedReadingOrderPredictor",
        )

    def _build_text_map(self, layout: "LayoutOutput", ocr: "OCROutput") -> Dict[int, str]:
        """
        Build a map from layout element IDs to text content.

        Matches OCR text blocks to layout elements by bounding box overlap.
        """
        text_map: Dict[int, str] = {}

        for i, box in enumerate(layout.bboxes):
            matched_texts = []

            for text_block in ocr.text_blocks:
                # Check for significant overlap
                if self._boxes_overlap(box.bbox, text_block.bbox):
                    matched_texts.append(text_block.text)

            if matched_texts:
                text_map[i] = " ".join(matched_texts)

        return text_map

    def _boxes_overlap(self, box1, box2, min_overlap: float = 0.5) -> bool:
        """Check if two boxes have significant overlap."""
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        if box2_area <= 0:
            return False

        return intersection / box2_area >= min_overlap

    def _get_docling_label(self, label_str: str):
        """Convert string label to DocItemLabel."""
        from docling_core.types.doc.labels import DocItemLabel

        # Mapping from our labels to DocItemLabel
        label_mapping = {
            "title": DocItemLabel.TITLE,
            "text": DocItemLabel.TEXT,
            "list": DocItemLabel.LIST_ITEM,
            "figure": DocItemLabel.PICTURE,
            "table": DocItemLabel.TABLE,
            "caption": DocItemLabel.CAPTION,
            "formula": DocItemLabel.FORMULA,
            "footnote": DocItemLabel.FOOTNOTE,
            "page_header": DocItemLabel.PAGE_HEADER,
            "page_footer": DocItemLabel.PAGE_FOOTER,
            "code": DocItemLabel.CODE,
            "abandon": DocItemLabel.TEXT,
            "unknown": DocItemLabel.TEXT,
        }

        return label_mapping.get(label_str, DocItemLabel.TEXT)
