"""
Pydantic models for OCR extraction outputs.

Defines standardized output types for OCR detection including text blocks
with bounding boxes, confidence scores, and granularity levels.

Key difference from Text Extraction:
- OCR returns text WITH bounding boxes (word/line/character level)
- Text Extraction returns formatted text (MD/HTML) WITHOUT bboxes

Coordinate Systems:
    - **Absolute (default)**: Coordinates in pixels relative to original image size
    - **Normalized (0-1024)**: Coordinates scaled to 0-1024 range (virtual 1024x1024 canvas)

    Use `bbox.to_normalized(width, height)` or `output.get_normalized_blocks()`
    to convert to normalized coordinates.

Example:
    >>> result = ocr.extract(image)  # Returns absolute pixel coordinates
    >>> normalized = result.get_normalized_blocks()  # Returns 0-1024 normalized coords
"""

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from PIL import Image


# Normalization constant - all coordinates normalized to this range
NORMALIZED_SIZE = 1024


class OCRGranularity(str, Enum):
    """
    OCR detection granularity levels.

    Different OCR engines return results at different granularity levels.
    This enum standardizes the options across all extractors.
    """

    CHARACTER = "character"  # Individual characters with boxes
    WORD = "word"  # Word-level boxes (most common)
    LINE = "line"  # Line-level boxes
    BLOCK = "block"  # Paragraph/block-level boxes


class BoundingBox(BaseModel):
    """
    Bounding box coordinates in pixel space.

    Coordinates follow the convention: (x1, y1) is top-left, (x2, y2) is bottom-right.
    For rotated text, use the polygon field in TextBlock instead.

    Example:
        >>> bbox = BoundingBox(x1=100, y1=50, x2=300, y2=80)
        >>> print(bbox.width, bbox.height)  # 200, 30
        >>> print(bbox.center)  # (200.0, 65.0)
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

    @classmethod
    def from_polygon(cls, polygon: List[List[float]]) -> "BoundingBox":
        """
        Create axis-aligned bounding box from polygon points.

        Args:
            polygon: List of [x, y] points (usually 4 for quadrilateral)

        Returns:
            BoundingBox that encloses all polygon points
        """
        if not polygon:
            raise ValueError("Polygon cannot be empty")

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return cls(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))

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


class TextBlock(BaseModel):
    """
    Single detected text element with text, bounding box, and confidence.

    This is the fundamental unit of OCR output - can represent a character,
    word, line, or block depending on the OCR model and configuration.

    Example:
        >>> block = TextBlock(
        ...     text="Hello",
        ...     bbox=BoundingBox(x1=100, y1=50, x2=200, y2=80),
        ...     confidence=0.95,
        ...     granularity=OCRGranularity.WORD,
        ... )
    """

    text: str = Field(..., description="Detected text content")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0-1)",
    )
    granularity: OCRGranularity = Field(
        default=OCRGranularity.WORD,
        description="Granularity level of this text block",
    )
    polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Original polygon coordinates for rotated text [[x1,y1], [x2,y2], ...]",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language code (ISO 639-1)",
    )

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "granularity": self.granularity.value,
            "polygon": self.polygon,
            "language": self.language,
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


class OCROutput(BaseModel):
    """
    Complete OCR extraction results for a single image.

    Contains all detected text blocks with their bounding boxes,
    plus metadata about the extraction.

    Example:
        >>> result = ocr.extract(image)
        >>> print(f"Found {result.block_count} blocks")
        >>> print(f"Full text: {result.full_text}")
        >>> for block in result.text_blocks:
        ...     print(f"'{block.text}' @ {block.bbox.to_list()}")
    """

    text_blocks: List[TextBlock] = Field(
        default_factory=list,
        description="Detected text blocks with bounding boxes",
    )
    full_text: str = Field(
        default="",
        description="Concatenated text from all blocks (reading order)",
    )
    image_width: int = Field(..., ge=1, description="Image width in pixels")
    image_height: int = Field(..., ge=1, description="Image height in pixels")
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the OCR model used",
    )
    languages_detected: Optional[List[str]] = Field(
        default=None,
        description="Languages detected in the document",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def block_count(self) -> int:
        """Number of detected text blocks."""
        return len(self.text_blocks)

    @property
    def word_count(self) -> int:
        """Approximate word count from full text."""
        return len(self.full_text.split())

    @property
    def average_confidence(self) -> float:
        """Average confidence across all text blocks."""
        if not self.text_blocks:
            return 0.0
        return sum(b.confidence for b in self.text_blocks) / len(self.text_blocks)

    def filter_by_confidence(self, min_confidence: float) -> List[TextBlock]:
        """Filter text blocks by minimum confidence."""
        return [b for b in self.text_blocks if b.confidence >= min_confidence]

    def filter_by_granularity(self, granularity: OCRGranularity) -> List[TextBlock]:
        """Filter text blocks by granularity level."""
        return [b for b in self.text_blocks if b.granularity == granularity]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "full_text": self.full_text,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
            "languages_detected": self.languages_detected,
            "block_count": self.block_count,
            "word_count": self.word_count,
            "average_confidence": self.average_confidence,
        }

    def sort_by_position(self, top_to_bottom: bool = True) -> "OCROutput":
        """
        Return a new OCROutput with blocks sorted by position.

        Args:
            top_to_bottom: If True, sort by y-coordinate (reading order)

        Returns:
            New OCROutput with sorted text blocks
        """
        sorted_blocks = sorted(
            self.text_blocks,
            key=lambda b: (b.bbox.y1, b.bbox.x1),
            reverse=not top_to_bottom,
        )
        # Regenerate full_text in sorted order
        full_text = " ".join(b.text for b in sorted_blocks)

        return OCROutput(
            text_blocks=sorted_blocks,
            full_text=full_text,
            image_width=self.image_width,
            image_height=self.image_height,
            model_name=self.model_name,
            languages_detected=self.languages_detected,
        )

    def get_normalized_blocks(self) -> List[Dict]:
        """
        Get all text blocks with normalized (0-1024) coordinates.

        Returns:
            List of dicts with normalized bbox coordinates and metadata.
        """
        normalized = []
        for block in self.text_blocks:
            norm_bbox = block.bbox.to_normalized(self.image_width, self.image_height)
            normalized.append(
                {
                    "text": block.text,
                    "bbox": norm_bbox.to_list(),
                    "confidence": block.confidence,
                    "granularity": block.granularity.value,
                    "language": block.language,
                }
            )
        return normalized

    def visualize(
        self,
        image: "Image.Image",
        output_path: Optional[Union[str, Path]] = None,
        show_text: bool = True,
        show_confidence: bool = False,
        line_width: int = 2,
        box_color: str = "#2ECC71",
        text_color: str = "#000000",
    ) -> "Image.Image":
        """
        Visualize OCR results on the image.

        Draws bounding boxes around detected text with optional labels.

        Args:
            image: PIL Image to draw on (will be copied, not modified)
            output_path: Optional path to save the visualization
            show_text: Whether to show detected text
            show_confidence: Whether to show confidence scores
            line_width: Width of bounding box lines
            box_color: Color for bounding boxes (hex)
            text_color: Color for text labels (hex)

        Returns:
            PIL Image with visualizations drawn

        Example:
            >>> result = ocr.extract(image)
            >>> viz = result.visualize(image, output_path="ocr_viz.png")
        """
        from PIL import ImageDraw, ImageFont

        # Copy image to avoid modifying original
        viz_image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(viz_image)

        # Try to get a font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except Exception:
                font = ImageFont.load_default()

        for block in self.text_blocks:
            coords = block.bbox.to_xyxy()

            # Draw polygon if available, otherwise draw rectangle
            if block.polygon:
                flat_polygon = [coord for point in block.polygon for coord in point]
                draw.polygon(flat_polygon, outline=box_color, width=line_width)
            else:
                draw.rectangle(coords, outline=box_color, width=line_width)

            # Build label text
            if show_text or show_confidence:
                label_parts = []
                if show_text:
                    # Truncate long text
                    text = block.text[:25] + "..." if len(block.text) > 25 else block.text
                    label_parts.append(text)
                if show_confidence:
                    label_parts.append(f"{block.confidence:.2f}")
                label_text = " | ".join(label_parts)

                # Position label below the box
                label_x = coords[0]
                label_y = coords[3] + 2  # Below bottom edge

                # Draw label with background
                text_bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                padding = 2
                draw.rectangle(
                    [
                        text_bbox[0] - padding,
                        text_bbox[1] - padding,
                        text_bbox[2] + padding,
                        text_bbox[3] + padding,
                    ],
                    fill="#FFFFFF",
                    outline=box_color,
                )
                draw.text((label_x, label_y), label_text, fill=text_color, font=font)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            viz_image.save(output_path)

        return viz_image

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "OCROutput":
        """
        Load an OCROutput instance from a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            OCROutput instance
        """
        path = Path(file_path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save_json(self, file_path: Union[str, Path]) -> None:
        """
        Save OCROutput instance to a JSON file.

        Args:
            file_path: Path where JSON file should be saved
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
