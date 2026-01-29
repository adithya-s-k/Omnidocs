"""
Pydantic models for layout extraction outputs.

Defines standardized output types and label enums for layout detection.

Coordinate Systems:
    - **Absolute (default)**: Coordinates in pixels relative to original image size
    - **Normalized (0-1024)**: Coordinates scaled to 0-1024 range (virtual 1024x1024 canvas)

    Use `bbox.to_normalized(width, height)` or `output.get_normalized_bboxes()`
    to convert to normalized coordinates.

Example:
    >>> result = extractor.extract(image)  # Returns absolute pixel coordinates
    >>> normalized = result.get_normalized_bboxes()  # Returns 0-1024 normalized coords
"""

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from PIL import Image


# Normalization constant - all coordinates normalized to this range
NORMALIZED_SIZE = 1024


# ============= Standardized Layout Labels =============


class LayoutLabel(str, Enum):
    """
    Standardized layout labels used across all layout extractors.

    These provide a consistent vocabulary regardless of which model is used.
    """

    # Text elements
    TITLE = "title"
    TEXT = "text"
    LIST = "list"

    # Visual elements
    FIGURE = "figure"
    TABLE = "table"

    # Annotations
    CAPTION = "caption"
    FORMULA = "formula"
    FOOTNOTE = "footnote"

    # Page elements
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"

    # Other
    ABANDON = "abandon"  # Elements to ignore (watermarks, artifacts, etc.)
    UNKNOWN = "unknown"


# ============= Custom Label Definition =============


class CustomLabel(BaseModel):
    """
    Type-safe custom layout label definition for VLM-based models.

    VLM models like Qwen3-VL support flexible custom labels beyond the
    standard LayoutLabel enum. Use this class to define custom labels
    with validation.

    Example:
        >>> from omnidocs.tasks.layout_extraction import CustomLabel
        >>>
        >>> # Simple custom label
        >>> code_block = CustomLabel(name="code_block")
        >>>
        >>> # With metadata
        >>> sidebar = CustomLabel(
        ...     name="sidebar",
        ...     description="Secondary content panel",
        ...     color="#9B59B6",
        ... )
        >>>
        >>> # Use with QwenLayoutDetector
        >>> result = detector.extract(image, custom_labels=[code_block, sidebar])
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Label identifier (e.g., 'code_block', 'sidebar'). Must be non-empty and reasonably short.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Human-readable description of what this label represents.",
    )
    color: Optional[str] = Field(
        default=None,
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Visualization color as hex string (e.g., '#9B59B6'). Used by visualize() method if provided.",
    )
    detection_prompt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional custom prompt hint to improve detection accuracy.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __str__(self) -> str:
        """Return the label name as string."""
        return self.name

    def __hash__(self) -> int:
        """Make hashable for use in sets."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare by name."""
        if isinstance(other, CustomLabel):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented


# ============= Label Mapping Definitions =============


class LabelMapping:
    """
    Base class for model-specific label mappings.

    Each model maps its native labels to standardized LayoutLabel values.
    """

    def __init__(self, mapping: Dict[str, LayoutLabel]):
        """
        Initialize label mapping.

        Args:
            mapping: Dict mapping model-specific labels to LayoutLabel enum values
        """
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        self._reverse_mapping = {v: k for k, v in mapping.items()}

    def to_standard(self, model_label: str) -> LayoutLabel:
        """Convert model-specific label to standardized LayoutLabel."""
        return self._mapping.get(model_label.lower(), LayoutLabel.UNKNOWN)

    def from_standard(self, standard_label: LayoutLabel) -> Optional[str]:
        """Convert standardized LayoutLabel to model-specific label."""
        return self._reverse_mapping.get(standard_label)

    @property
    def supported_labels(self) -> List[str]:
        """Get list of supported model-specific labels."""
        return list(self._mapping.keys())

    @property
    def standard_labels(self) -> List[LayoutLabel]:
        """Get list of standard labels this mapping produces."""
        return list(set(self._mapping.values()))


# ============= Model-Specific Mappings =============


DOCLAYOUT_YOLO_MAPPING = LabelMapping(
    {
        "title": LayoutLabel.TITLE,
        "plain_text": LayoutLabel.TEXT,
        "abandon": LayoutLabel.ABANDON,
        "figure": LayoutLabel.FIGURE,
        "figure_caption": LayoutLabel.CAPTION,
        "table": LayoutLabel.TABLE,
        "table_caption": LayoutLabel.CAPTION,
        "table_footnote": LayoutLabel.FOOTNOTE,
        "isolate_formula": LayoutLabel.FORMULA,
        "formula_caption": LayoutLabel.CAPTION,
    }
)

DOCLAYOUT_YOLO_CLASS_NAMES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


# RT-DETR class names (HuggingPanda/docling-layout model)
# Note: Index 0 is background class, content labels start at index 1
RTDETR_CLASS_NAMES = {
    0: "background",
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title",
    # DLNv2 extended labels
    12: "Document Index",
    13: "Code",
    14: "Checkbox-Selected",
    15: "Checkbox-Unselected",
    16: "Form",
    17: "Key-Value Region",
}


RTDETR_MAPPING = LabelMapping(
    {
        # Background (typically filtered by confidence threshold)
        "background": LayoutLabel.ABANDON,
        # Core labels
        "caption": LayoutLabel.CAPTION,
        "footnote": LayoutLabel.FOOTNOTE,
        "formula": LayoutLabel.FORMULA,
        "list-item": LayoutLabel.LIST,
        "page-footer": LayoutLabel.PAGE_FOOTER,
        "page-header": LayoutLabel.PAGE_HEADER,
        "picture": LayoutLabel.FIGURE,
        "section-header": LayoutLabel.TITLE,
        "table": LayoutLabel.TABLE,
        "text": LayoutLabel.TEXT,
        "title": LayoutLabel.TITLE,
        # DLNv2 extended labels (map to closest standard label)
        "document index": LayoutLabel.LIST,
        "code": LayoutLabel.TEXT,
        "checkbox-selected": LayoutLabel.UNKNOWN,
        "checkbox-unselected": LayoutLabel.UNKNOWN,
        "form": LayoutLabel.UNKNOWN,
        "key-value region": LayoutLabel.TEXT,
    }
)


# ============= Bounding Box Model =============


class BoundingBox(BaseModel):
    """
    Bounding box coordinates in pixel space.

    Coordinates follow the convention: (x1, y1) is top-left, (x2, y2) is bottom-right.
    """

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
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] list."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return (self.x1, self.y1, self.width, self.height)

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        """Create from [x1, y1, x2, y2] list."""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def to_normalized(self, image_width: int, image_height: int) -> "BoundingBox":
        """
        Convert to normalized coordinates (0-1024 range).

        Scales coordinates from absolute pixel values to a virtual 1024x1024 canvas.
        This provides consistent coordinates regardless of original image size.

        Args:
            image_width: Original image width in pixels
            image_height: Original image height in pixels

        Returns:
            New BoundingBox with coordinates in 0-1024 range

        Example:
            >>> bbox = BoundingBox(x1=100, y1=50, x2=500, y2=300)
            >>> normalized = bbox.to_normalized(1000, 800)
            >>> # x: 100/1000*1024 = 102.4, y: 50/800*1024 = 64
        """
        return BoundingBox(
            x1=self.x1 / image_width * NORMALIZED_SIZE,
            y1=self.y1 / image_height * NORMALIZED_SIZE,
            x2=self.x2 / image_width * NORMALIZED_SIZE,
            y2=self.y2 / image_height * NORMALIZED_SIZE,
        )

    def to_absolute(self, image_width: int, image_height: int) -> "BoundingBox":
        """
        Convert from normalized (0-1024) to absolute pixel coordinates.

        Args:
            image_width: Target image width in pixels
            image_height: Target image height in pixels

        Returns:
            New BoundingBox with absolute pixel coordinates
        """
        return BoundingBox(
            x1=self.x1 / NORMALIZED_SIZE * image_width,
            y1=self.y1 / NORMALIZED_SIZE * image_height,
            x2=self.x2 / NORMALIZED_SIZE * image_width,
            y2=self.y2 / NORMALIZED_SIZE * image_height,
        )


# ============= Layout Box Model =============


class LayoutBox(BaseModel):
    """
    Single detected layout element with label, bounding box, and confidence.
    """

    label: LayoutLabel = Field(..., description="Standardized layout label")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: Optional[int] = Field(default=None, description="Model-specific class ID")
    original_label: Optional[str] = Field(default=None, description="Original model-specific label before mapping")

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "label": self.label.value,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "original_label": self.original_label,
        }

    def get_normalized_bbox(self, image_width: int, image_height: int) -> BoundingBox:
        """
        Get bounding box in normalized (0-1024) coordinates.

        Args:
            image_width: Original image width
            image_height: Original image height

        Returns:
            BoundingBox with normalized coordinates
        """
        return self.bbox.to_normalized(image_width, image_height)


# ============= Visualization Colors =============

LABEL_COLORS: Dict[LayoutLabel, str] = {
    LayoutLabel.TITLE: "#E74C3C",  # Red
    LayoutLabel.TEXT: "#3498DB",  # Blue
    LayoutLabel.LIST: "#2ECC71",  # Green
    LayoutLabel.FIGURE: "#9B59B6",  # Purple
    LayoutLabel.TABLE: "#F39C12",  # Orange
    LayoutLabel.CAPTION: "#1ABC9C",  # Teal
    LayoutLabel.FORMULA: "#E91E63",  # Pink
    LayoutLabel.FOOTNOTE: "#607D8B",  # Gray
    LayoutLabel.PAGE_HEADER: "#795548",  # Brown
    LayoutLabel.PAGE_FOOTER: "#795548",  # Brown
    LayoutLabel.ABANDON: "#BDC3C7",  # Light Gray
    LayoutLabel.UNKNOWN: "#95A5A6",  # Gray
}


# ============= Layout Output Model =============


class LayoutOutput(BaseModel):
    """
    Complete layout extraction results for a single image.
    """

    bboxes: List[LayoutBox] = Field(default_factory=list, description="Detected layout boxes")
    image_width: int = Field(..., ge=1, description="Image width in pixels")
    image_height: int = Field(..., ge=1, description="Image height in pixels")
    model_name: Optional[str] = Field(default=None, description="Name of the model used")

    model_config = ConfigDict(extra="forbid")

    @property
    def element_count(self) -> int:
        """Number of detected elements."""
        return len(self.bboxes)

    @property
    def labels_found(self) -> List[str]:
        """Unique labels found in detections."""
        return sorted(set(box.label.value for box in self.bboxes))

    def filter_by_label(self, label: LayoutLabel) -> List[LayoutBox]:
        """Filter boxes by label."""
        return [box for box in self.bboxes if box.label == label]

    def filter_by_confidence(self, min_confidence: float) -> List[LayoutBox]:
        """Filter boxes by minimum confidence."""
        return [box for box in self.bboxes if box.confidence >= min_confidence]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "bboxes": [box.to_dict() for box in self.bboxes],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
            "element_count": self.element_count,
            "labels_found": self.labels_found,
        }

    def sort_by_position(self, top_to_bottom: bool = True) -> "LayoutOutput":
        """
        Return a new LayoutOutput with boxes sorted by position.

        Args:
            top_to_bottom: If True, sort by y-coordinate (reading order)
        """
        sorted_boxes = sorted(self.bboxes, key=lambda b: (b.bbox.y1, b.bbox.x1), reverse=not top_to_bottom)
        return LayoutOutput(
            bboxes=sorted_boxes,
            image_width=self.image_width,
            image_height=self.image_height,
            model_name=self.model_name,
        )

    def get_normalized_bboxes(self) -> List[Dict]:
        """
        Get all bounding boxes in normalized (0-1024) coordinates.

        Returns:
            List of dicts with normalized bbox coordinates and metadata.

        Example:
            >>> result = extractor.extract(image)
            >>> normalized = result.get_normalized_bboxes()
            >>> for box in normalized:
            ...     print(f"{box['label']}: {box['bbox']}")  # coords in 0-1024 range
        """
        normalized = []
        for box in self.bboxes:
            norm_bbox = box.bbox.to_normalized(self.image_width, self.image_height)
            normalized.append(
                {
                    "label": box.label.value,
                    "bbox": norm_bbox.to_list(),
                    "confidence": box.confidence,
                    "class_id": box.class_id,
                    "original_label": box.original_label,
                }
            )
        return normalized

    def visualize(
        self,
        image: "Image.Image",
        output_path: Optional[Union[str, Path]] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        line_width: int = 3,
        font_size: int = 12,
    ) -> "Image.Image":
        """
        Visualize layout detection results on the image.

        Draws bounding boxes with labels and confidence scores on the image.
        Each layout category has a distinct color for easy identification.

        Args:
            image: PIL Image to draw on (will be copied, not modified)
            output_path: Optional path to save the visualization
            show_labels: Whether to show label text
            show_confidence: Whether to show confidence scores
            line_width: Width of bounding box lines
            font_size: Size of label text (note: uses default font)

        Returns:
            PIL Image with visualizations drawn

        Example:
            >>> result = extractor.extract(image)
            >>> viz = result.visualize(image, output_path="layout_viz.png")
            >>> viz.show()  # Display in notebook/viewer
        """
        from PIL import ImageDraw

        # Copy image to avoid modifying original
        viz_image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(viz_image)

        for box in self.bboxes:
            # Get color for this label
            color = LABEL_COLORS.get(box.label, "#95A5A6")

            # Draw bounding box
            coords = box.bbox.to_xyxy()
            draw.rectangle(coords, outline=color, width=line_width)

            # Build label text
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(box.label.value)
                if show_confidence:
                    label_parts.append(f"{box.confidence:.2f}")
                label_text = " ".join(label_parts)

                # Draw label background
                text_bbox = draw.textbbox((coords[0], coords[1] - 20), label_text)
                draw.rectangle(text_bbox, fill=color)

                # Draw label text
                draw.text(
                    (coords[0], coords[1] - 20),
                    label_text,
                    fill="white",
                )

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            viz_image.save(output_path)

        return viz_image

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "LayoutOutput":
        """
        Load a LayoutOutput instance from a JSON file.

        Reads a JSON file and deserializes its contents into a LayoutOutput object.
        Uses Pydantic's model_validate_json for proper handling of nested objects.

        Args:
            file_path: Path to JSON file containing serialized LayoutOutput data.
                      Can be string or pathlib.Path object.

        Returns:
            LayoutOutput: Deserialized layout output instance from file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            UnicodeDecodeError: If file cannot be decoded as UTF-8.
            ValueError: If file contents are not valid JSON.
            ValidationError: If JSON data doesn't match LayoutOutput schema.

        Example:
            >>> output = LayoutOutput.load_json('layout_results.json')
            >>> print(f"Found {output.element_count} elements")
            Found 5 elements
        """
        path = Path(file_path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save_json(self, file_path: Union[str, Path]) -> None:
        """
        Save LayoutOutput instance to a JSON file.

        Serializes the LayoutOutput object to JSON and writes it to a file.
        Automatically creates parent directories if they don't exist. Uses UTF-8
        encoding for compatibility and proper handling of special characters.

        Args:
            file_path: Path where JSON file should be saved. Can be string or
                      pathlib.Path object. Parent directories will be created
                      if they don't exist.

        Returns:
            None

        Raises:
            OSError: If file cannot be written due to permission or disk errors.
            TypeError: If file_path is not a string or Path object.

        Example:
            >>> output = LayoutOutput(bboxes=[], image_width=800, image_height=600)
            >>> output.save_json('results/layout_output.json')
            >>> # File is created at results/layout_output.json
            >>> # Parent 'results' directory is created if it didn't exist
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(), encoding="utf-8")
