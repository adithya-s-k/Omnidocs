"""
Pydantic models for reading order prediction.

Takes layout detection and OCR results, produces ordered element sequence
with caption and footnote associations.

Example:
    ```python
    # Get layout and OCR
    layout = layout_extractor.extract(image)
    ocr = ocr_extractor.extract(image)

    # Predict reading order
    reading_order = predictor.predict(layout, ocr)

    # Iterate in reading order
    for element in reading_order.ordered_elements:
        print(f"{element.index}: [{element.element_type}] {element.text[:50]}...")

    # Get caption associations
    for fig_id, caption_ids in reading_order.caption_map.items():
        print(f"Figure {fig_id} has captions: {caption_ids}")
    ```
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

# Normalization constant - all coordinates normalized to this range
NORMALIZED_SIZE = 1024


class ElementType(str, Enum):
    """Type of document element for reading order."""

    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    FORMULA = "formula"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    CODE = "code"
    OTHER = "other"


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates."""

    x1: float = Field(..., description="Left x coordinate")
    y1: float = Field(..., description="Top y coordinate")
    x2: float = Field(..., description="Right x coordinate")
    y2: float = Field(..., description="Bottom y coordinate")

    model_config = ConfigDict(extra="forbid")

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] list."""
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        """Create from [x1, y1, x2, y2] list."""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_normalized(self, image_width: int, image_height: int) -> "BoundingBox":
        """
        Convert to normalized coordinates (0-1024 range).

        Args:
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            New BoundingBox with coordinates in 0-1024 range
        """
        return BoundingBox(
            x1=self.x1 / image_width * NORMALIZED_SIZE,
            y1=self.y1 / image_height * NORMALIZED_SIZE,
            x2=self.x2 / image_width * NORMALIZED_SIZE,
            y2=self.y2 / image_height * NORMALIZED_SIZE,
        )


class OrderedElement(BaseModel):
    """
    A document element with its reading order position.

    Combines layout detection results with OCR text and
    assigns a reading order index.
    """

    index: int = Field(..., ge=0, description="Reading order index (0-based)")
    element_type: ElementType = Field(..., description="Type of element")
    bbox: BoundingBox = Field(..., description="Bounding box in pixels")
    text: str = Field(default="", description="Text content (from OCR)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    page_no: int = Field(default=0, ge=0, description="Page number (0-indexed)")
    original_id: Optional[int] = Field(
        default=None,
        description="Original element ID from layout detection",
    )

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "index": self.index,
            "element_type": self.element_type.value,
            "bbox": self.bbox.to_list(),
            "text": self.text,
            "confidence": self.confidence,
            "page_no": self.page_no,
            "original_id": self.original_id,
        }


class ReadingOrderOutput(BaseModel):
    """
    Complete reading order prediction result.

    Provides:
    - Ordered list of document elements
    - Caption-to-element associations
    - Footnote-to-element associations
    - Merge suggestions for split elements

    Example:
        ```python
        result = predictor.predict(layout, ocr)

        # Get full text in reading order
        full_text = result.get_full_text()

        # Get elements by type
        tables = result.get_elements_by_type(ElementType.TABLE)

        # Find caption for a figure
        captions = result.get_captions_for(figure_element.original_id)
        ```
    """

    ordered_elements: List[OrderedElement] = Field(
        default_factory=list,
        description="Elements sorted by reading order",
    )
    caption_map: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="Maps element IDs to their caption element IDs",
    )
    footnote_map: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="Maps element IDs to their footnote element IDs",
    )
    merge_map: Dict[int, List[int]] = Field(
        default_factory=dict,
        description="Maps element IDs to elements that should be merged with them",
    )
    image_width: int = Field(..., ge=1, description="Image width in pixels")
    image_height: int = Field(..., ge=1, description="Image height in pixels")
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the model/algorithm used",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def element_count(self) -> int:
        """Total number of ordered elements."""
        return len(self.ordered_elements)

    def get_full_text(self, separator: str = "\n\n") -> str:
        """
        Get concatenated text in reading order.

        Excludes page headers, footers, captions, and footnotes
        from main text flow.
        """
        main_elements = [
            e
            for e in self.ordered_elements
            if e.element_type
            not in (
                ElementType.PAGE_HEADER,
                ElementType.PAGE_FOOTER,
                ElementType.CAPTION,
                ElementType.FOOTNOTE,
            )
        ]
        return separator.join(e.text for e in main_elements if e.text)

    def get_elements_by_type(self, element_type: ElementType) -> List[OrderedElement]:
        """Filter elements by type."""
        return [e for e in self.ordered_elements if e.element_type == element_type]

    def get_captions_for(self, element_id: int) -> List[OrderedElement]:
        """Get caption elements for a given element ID."""
        caption_ids = self.caption_map.get(element_id, [])
        return [e for e in self.ordered_elements if e.original_id in caption_ids]

    def get_footnotes_for(self, element_id: int) -> List[OrderedElement]:
        """Get footnote elements for a given element ID."""
        footnote_ids = self.footnote_map.get(element_id, [])
        return [e for e in self.ordered_elements if e.original_id in footnote_ids]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "ordered_elements": [e.to_dict() for e in self.ordered_elements],
            "caption_map": self.caption_map,
            "footnote_map": self.footnote_map,
            "merge_map": self.merge_map,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "element_count": self.element_count,
        }

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "ReadingOrderOutput":
        """Load from JSON file."""
        path = Path(file_path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
