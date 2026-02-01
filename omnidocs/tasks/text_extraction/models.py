"""
Pydantic models for text extraction outputs.

Defines output types and format enums for text extraction.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OutputFormat(str, Enum):
    """
    Supported text extraction output formats.

    Each format has different characteristics:
        - HTML: Structured with div elements, preserves layout semantics
        - MARKDOWN: Portable, human-readable, good for documentation
        - JSON: Structured data with layout information (Dots OCR)
    """

    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"


class TextOutput(BaseModel):
    """
    Text extraction output from a document image.

    Contains the extracted text content in the requested format,
    along with optional raw output and plain text versions.

    Example:
        ```python
        result = extractor.extract(image, output_format="markdown")
        print(result.content)  # Clean markdown
        print(result.plain_text)  # Plain text without formatting
        ```
    """

    content: str = Field(
        ...,
        description="Extracted text content in the requested format (HTML or Markdown). "
        "This is the cleaned version with formatting artifacts removed.",
    )
    format: OutputFormat = Field(
        ...,
        description="The output format of the content.",
    )
    raw_output: Optional[str] = Field(
        default=None,
        description="Raw model output before cleaning. Includes bounding box annotations and other artifacts.",
    )
    plain_text: Optional[str] = Field(
        default=None,
        description="Plain text version without any formatting. Useful for text analysis and comparison.",
    )
    image_width: Optional[int] = Field(
        default=None,
        ge=1,
        description="Width of the source image in pixels.",
    )
    image_height: Optional[int] = Field(
        default=None,
        ge=1,
        description="Height of the source image in pixels.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the model used for extraction.",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def content_length(self) -> int:
        """Length of the extracted content in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Approximate word count of the plain text."""
        text = self.plain_text or self.content
        return len(text.split())


class LayoutElement(BaseModel):
    """
    Single layout element from document layout detection.

    Represents a detected region in the document with its bounding box,
    category label, and extracted text content.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2] (normalized to 0-1024)
        category: Layout category (e.g., "Text", "Title", "Table", "Formula")
        text: Extracted text content (None for pictures)
        confidence: Detection confidence score (optional)
    """

    bbox: List[int] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2] (normalized 0-1024)",
        min_length=4,
        max_length=4,
    )
    category: str = Field(..., description="Layout category (Text, Title, Formula, Table, etc.)")
    text: Optional[str] = Field(None, description="Extracted text content (None for Pictures)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence score")

    model_config = ConfigDict(extra="forbid")


class DotsOCRTextOutput(BaseModel):
    """
    Text extraction output from Dots OCR with layout information.

    Dots OCR provides structured output with:
    - Layout detection (11 categories)
    - Bounding boxes (normalized to 0-1024)
    - Multi-format text (Markdown/LaTeX/HTML)
    - Reading order preservation

    Layout Categories:
        Caption, Footnote, Formula, List-item, Page-footer, Page-header,
        Picture, Section-header, Table, Text, Title

    Text Formatting:
        - Text/Title/Section-header: Markdown
        - Formula: LaTeX
        - Table: HTML
        - Picture: (text omitted)

    Example:
        ```python
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        result = extractor.extract(image, include_layout=True)
        print(result.content)  # Full text with formatting
        for elem in result.layout:
                print(f"{elem.category}: {elem.bbox}")
        ```
    """

    content: str = Field(..., description="Extracted text in requested format (markdown/html/json)")
    format: OutputFormat = Field(..., description="Output format")
    layout: List[LayoutElement] = Field(
        default_factory=list, description="Layout elements with bounding boxes (if include_layout=True)"
    )
    has_layout: bool = Field(default=False, description="Whether layout information is included")
    layout_categories: List[str] = Field(
        default_factory=lambda: [
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
        ],
        description="Supported layout categories",
    )
    raw_output: Optional[str] = Field(None, description="Raw model output (for debugging)")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    truncated: bool = Field(default=False, description="Whether output was truncated (hit max tokens)")
    image_width: Optional[int] = Field(None, ge=1, description="Source image width in pixels")
    image_height: Optional[int] = Field(None, ge=1, description="Source image height in pixels")

    model_config = ConfigDict(extra="forbid")

    @property
    def num_layout_elements(self) -> int:
        """Number of detected layout elements."""
        return len(self.layout)

    @property
    def content_length(self) -> int:
        """Length of extracted content in characters."""
        return len(self.content)
