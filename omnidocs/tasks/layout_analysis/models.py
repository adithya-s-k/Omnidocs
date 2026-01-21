# Pydantic Models for type validation and serialization
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator


class LayoutBox(BaseModel):
    """Represents a single detected layout box with its properties.

    A LayoutBox contains information about a detected region in a document,
    including its classification label, bounding box coordinates, and
    optional confidence score.

    Attributes:
        label: The classification label of the detected box (e.g., "text",
            "table", "image").
        bbox: Bounding box coordinates as [x1, y1, x2, y2] in absolute pixels,
            where (x1, y1) is the top-left corner and (x2, y2) is the
            bottom-right corner.
        confidence: Optional confidence score of the detection, ranging
            from 0.0 to 1.0.

    Example:
        >>> box = LayoutBox(
        ...     label="table",
        ...     bbox=[100.0, 200.0, 500.0, 400.0],
        ...     confidence=0.95
        ... )
        >>> print(f"Found {box.label} with confidence {box.confidence}")
        Found table with confidence 0.95
    """
    label: str = Field(..., description="Classification label of the detected box")
    bbox: List[float] = Field(..., 
                            description="Bounding box coordinates [x1, y1, x2, y2]",
                            min_items=4, 
                            max_items=4)
    confidence: Optional[float] = Field(None, 
                                      description="Confidence score of detection",
                                      ge=0.0, 
                                      le=1.0)

    @validator('bbox')
    def validate_bbox(cls, v):
        """Validate that bounding box has exactly 4 numeric coordinates.

        Args:
            v: List of bounding box coordinates to validate.

        Returns:
            The validated list of coordinates.

        Raises:
            ValueError: If bbox doesn't have exactly 4 coordinates or if
                any coordinate is not numeric.
        """
        if len(v) != 4:
            raise ValueError('Bounding box must have exactly 4 coordinates')
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('All bbox coordinates must be numeric')
        return v

    def to_dict(self) -> Dict:
        """Convert the LayoutBox to a dictionary representation.

        Returns:
            Dictionary containing label, bbox, and confidence keys.

        Example:
            >>> box = LayoutBox(label="text", bbox=[0, 0, 100, 50], confidence=0.9)
            >>> box.to_dict()
            {'label': 'text', 'bbox': [0, 0, 100, 50], 'confidence': 0.9}
        """
        return {
            'label': self.label,
            'bbox': self.bbox,
            'confidence': self.confidence
        }

class LayoutOutput(BaseModel):
    """Container for all detected layout boxes in an image.

    LayoutOutput aggregates all detection results for a single image or
    document page, including the detected boxes and optional metadata.

    Attributes:
        bboxes: List of detected LayoutBox objects.
        page_number: Optional page number for multi-page documents (1-indexed).
        image_size: Optional tuple of (width, height) of the processed image
            in pixels.

    Example:
        >>> boxes = [
        ...     LayoutBox(label="title", bbox=[50, 10, 500, 50], confidence=0.98),
        ...     LayoutBox(label="text", bbox=[50, 60, 500, 300], confidence=0.95),
        ... ]
        >>> output = LayoutOutput(bboxes=boxes, page_number=1, image_size=(600, 800))
        >>> print(f"Page {output.page_number}: {len(output.bboxes)} elements")
        Page 1: 2 elements
    """
    bboxes: List[LayoutBox] = Field(default_factory=list,
                                   description="List of detected layout boxes")
    page_number: Optional[int] = Field(None, 
                                     description="Page number for multi-page documents",
                                     ge=1)
    image_size: Optional[Tuple[int, int]] = Field(None,
                                                 description="Size of the processed image (width, height)")

    def to_dict(self) -> Dict:
        """Convert the LayoutOutput to a dictionary representation.

        Returns:
            Dictionary containing bboxes (as list of dicts), page_number,
            and image_size.

        Example:
            >>> output = LayoutOutput(
            ...     bboxes=[LayoutBox(label="text", bbox=[0, 0, 100, 50])],
            ...     page_number=1
            ... )
            >>> result = output.to_dict()
            >>> print(result["page_number"])
            1
        """
        return {
            'bboxes': [box.to_dict() for box in self.bboxes],
            'page_number': self.page_number,
            'image_size': self.image_size
        }

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Save layout output to a JSON file.

        Serializes the LayoutOutput to JSON format and writes it to the
        specified file path.

        Args:
            output_path: Path where the JSON file will be saved. Can be a
                string or Path object.

        Example:
            >>> output = LayoutOutput(
            ...     bboxes=[LayoutBox(label="table", bbox=[100, 200, 400, 500])],
            ...     page_number=1
            ... )
            >>> output.save_json("layout_results.json")
        """
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)