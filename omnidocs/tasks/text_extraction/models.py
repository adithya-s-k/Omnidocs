"""
Pydantic models for text extraction outputs.

Defines output types and format enums for text extraction.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class OutputFormat(str, Enum):
    """
    Supported text extraction output formats.

    Each format has different characteristics:
        - HTML: Structured with div elements, preserves layout semantics
        - MARKDOWN: Portable, human-readable, good for documentation
    """

    HTML = "html"
    MARKDOWN = "markdown"


class TextOutput(BaseModel):
    """
    Text extraction output from a document image.

    Contains the extracted text content in the requested format,
    along with optional raw output and plain text versions.

    Example:
        >>> result = extractor.extract(image, output_format="markdown")
        >>> print(result.content)  # Clean markdown
        >>> print(result.plain_text)  # Plain text without formatting
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
