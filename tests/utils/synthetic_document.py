"""
Synthetic Document Generation for OmniDocs Testing.

Creates test images with known content for validation and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextRegion:
    """A text region in a synthetic document."""

    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    font_size: int = 16
    font_color: str = "black"
    background_color: Optional[str] = None


@dataclass
class SyntheticDocument:
    """A synthetic document with known content."""

    image: Image.Image
    regions: List[TextRegion]
    width: int
    height: int

    @property
    def full_text(self) -> str:
        """Get all text in reading order (top to bottom, left to right)."""
        sorted_regions = sorted(self.regions, key=lambda r: (r.bbox[1], r.bbox[0]))
        return "\n".join(r.text for r in sorted_regions)


def create_synthetic_document(
    width: int = 800,
    height: int = 1000,
    background_color: str = "white",
    texts: Optional[List[str]] = None,
    include_table: bool = False,
    include_title: bool = True,
    margin: int = 50,
) -> SyntheticDocument:
    """
    Create a synthetic document image with known text content.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        background_color: Background color
        texts: List of text paragraphs to include
        include_table: Whether to include a simple table
        include_title: Whether to include a title
        margin: Margin from edges in pixels

    Returns:
        SyntheticDocument with image and region information
    """
    # Create base image
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    regions: List[TextRegion] = []
    y_offset = margin

    # Try to get a font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    # Add title
    if include_title:
        title_text = "Sample Document Title"
        title_bbox = draw.textbbox((margin, y_offset), title_text, font=title_font)
        draw.text((margin, y_offset), title_text, fill="black", font=title_font)

        regions.append(TextRegion(
            text=title_text,
            bbox=(margin, y_offset, title_bbox[2], title_bbox[3]),
            font_size=24,
        ))
        y_offset = title_bbox[3] + 30

    # Default texts if none provided
    if texts is None:
        texts = [
            "This is the first paragraph of the document. It contains some sample text "
            "that can be used for testing text extraction capabilities.",
            "The second paragraph provides additional content. Testing various scenarios "
            "helps ensure robust extraction across different document layouts.",
            "A third paragraph concludes the main text section. This demonstrates "
            "multi-paragraph document handling.",
        ]

    # Add paragraphs
    text_width = width - 2 * margin
    for text in texts:
        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            bbox = draw.textbbox((0, 0), test_line, font=body_font)
            if bbox[2] > text_width:
                current_line.pop()
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        # Draw the text
        para_start_y = y_offset
        for line in lines:
            draw.text((margin, y_offset), line, fill="black", font=body_font)
            line_bbox = draw.textbbox((margin, y_offset), line, font=body_font)
            y_offset = line_bbox[3] + 5

        regions.append(TextRegion(
            text=text,
            bbox=(margin, para_start_y, width - margin, y_offset),
            font_size=14,
        ))
        y_offset += 20

    # Add a simple table
    if include_table and y_offset + 150 < height:
        table_data = [
            ["Header 1", "Header 2", "Header 3"],
            ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
            ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"],
        ]

        table_x = margin
        table_y = y_offset + 20
        cell_width = (width - 2 * margin) // 3
        cell_height = 30

        for row_idx, row in enumerate(table_data):
            for col_idx, cell_text in enumerate(row):
                x1 = table_x + col_idx * cell_width
                y1 = table_y + row_idx * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height

                # Draw cell border
                draw.rectangle([x1, y1, x2, y2], outline="black", width=1)

                # Draw cell text
                text_x = x1 + 5
                text_y = y1 + 5
                draw.text((text_x, text_y), cell_text, fill="black", font=body_font)

                regions.append(TextRegion(
                    text=cell_text,
                    bbox=(x1, y1, x2, y2),
                    font_size=14,
                ))

    return SyntheticDocument(
        image=image,
        regions=regions,
        width=width,
        height=height,
    )


def create_simple_text_image(
    text: str,
    width: int = 400,
    height: int = 100,
    font_size: int = 16,
    background_color: str = "white",
    text_color: str = "black",
) -> Tuple[Image.Image, str]:
    """
    Create a simple image with a single text string.

    Args:
        text: Text to render
        width: Image width
        height: Image height
        font_size: Font size
        background_color: Background color
        text_color: Text color

    Returns:
        Tuple of (PIL Image, ground truth text)
    """
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=text_color, font=font)

    return image, text


def create_table_image(
    rows: int = 3,
    cols: int = 3,
    cell_width: int = 100,
    cell_height: int = 40,
    include_headers: bool = True,
) -> Tuple[Image.Image, List[List[str]]]:
    """
    Create a simple table image.

    Args:
        rows: Number of rows
        cols: Number of columns
        cell_width: Width of each cell
        cell_height: Height of each cell
        include_headers: Whether the first row is a header

    Returns:
        Tuple of (PIL Image, table data as 2D list)
    """
    width = cols * cell_width + 20
    height = rows * cell_height + 20

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    table_data = []
    margin = 10

    for row_idx in range(rows):
        row_data = []
        for col_idx in range(cols):
            if include_headers and row_idx == 0:
                cell_text = f"Col {col_idx + 1}"
            else:
                cell_text = f"R{row_idx}C{col_idx}"

            row_data.append(cell_text)

            x1 = margin + col_idx * cell_width
            y1 = margin + row_idx * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            # Draw cell
            draw.rectangle([x1, y1, x2, y2], outline="black", width=1)
            draw.text((x1 + 5, y1 + 10), cell_text, fill="black", font=font)

        table_data.append(row_data)

    return image, table_data
