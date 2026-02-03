"""
Pydantic models for table extraction outputs.

Provides structured table data with cells, spans, and multiple export formats
including HTML, Markdown, and Pandas DataFrame conversion.

Example:
    ```python
    result = extractor.extract(table_image)

    # Get HTML
    html = result.to_html()

    # Get Pandas DataFrame
    df = result.to_dataframe()

    # Access cells
    for cell in result.cells:
        print(f"[{cell.row},{cell.col}] {cell.text}")
    ```
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

# Normalization constant - all coordinates normalized to this range
NORMALIZED_SIZE = 1024


class CellType(str, Enum):
    """Type of table cell."""

    DATA = "data"  # Regular data cell
    COLUMN_HEADER = "column_header"  # Column header cell
    ROW_HEADER = "row_header"  # Row header cell
    SECTION_ROW = "section_row"  # Row spanning section header


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

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        """Create from [x1, y1, x2, y2] list."""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    @classmethod
    def from_ltrb(cls, left: float, top: float, right: float, bottom: float) -> "BoundingBox":
        """Create from left, top, right, bottom coordinates."""
        return cls(x1=left, y1=top, x2=right, y2=bottom)

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


class TableCell(BaseModel):
    """
    Single table cell with position, span, and content.

    The cell position uses 0-indexed row/column indices.
    Spans indicate how many rows/columns the cell occupies.
    """

    row: int = Field(..., ge=0, description="Row index (0-indexed)")
    col: int = Field(..., ge=0, description="Column index (0-indexed)")
    row_span: int = Field(default=1, ge=1, description="Number of rows this cell spans")
    col_span: int = Field(default=1, ge=1, description="Number of columns this cell spans")
    text: str = Field(default="", description="Cell text content")
    cell_type: CellType = Field(default=CellType.DATA, description="Type of cell")
    bbox: Optional[BoundingBox] = Field(default=None, description="Cell bounding box in image")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Detection confidence")

    model_config = ConfigDict(extra="forbid")

    @property
    def end_row(self) -> int:
        """Ending row index (exclusive)."""
        return self.row + self.row_span

    @property
    def end_col(self) -> int:
        """Ending column index (exclusive)."""
        return self.col + self.col_span

    @property
    def is_header(self) -> bool:
        """Check if cell is any type of header."""
        return self.cell_type in (CellType.COLUMN_HEADER, CellType.ROW_HEADER)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "text": self.text,
            "cell_type": self.cell_type.value,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "confidence": self.confidence,
        }


class TableOutput(BaseModel):
    """
    Complete table extraction result.

    Provides multiple export formats and utility methods for working
    with extracted table data.

    Example:
        ```python
        result = extractor.extract(table_image)

        # Basic info
        print(f"Table: {result.num_rows}x{result.num_cols}")

        # Export to HTML
        html = result.to_html()

        # Export to Pandas
        df = result.to_dataframe()

        # Export to Markdown
        md = result.to_markdown()

        # Access specific cell
        cell = result.get_cell(row=0, col=0)
        ```
    """

    cells: List[TableCell] = Field(default_factory=list, description="All table cells")
    num_rows: int = Field(..., ge=0, description="Total number of rows")
    num_cols: int = Field(..., ge=0, description="Total number of columns")
    image_width: Optional[int] = Field(default=None, description="Source image width")
    image_height: Optional[int] = Field(default=None, description="Source image height")
    model_name: Optional[str] = Field(default=None, description="Model used for extraction")
    otsl_sequence: Optional[List[str]] = Field(
        default=None,
        description="OTSL tag sequence (for TableFormer)",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def cell_count(self) -> int:
        """Number of cells in the table."""
        return len(self.cells)

    @property
    def has_headers(self) -> bool:
        """Check if table has header cells."""
        return any(c.is_header for c in self.cells)

    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """
        Get cell at specific position.

        Handles merged cells by returning the cell that covers the position.
        """
        for cell in self.cells:
            if cell.row <= row < cell.end_row and cell.col <= col < cell.end_col:
                return cell
        return None

    def get_row(self, row: int) -> List[TableCell]:
        """Get all cells in a specific row."""
        return [c for c in self.cells if c.row == row]

    def get_column(self, col: int) -> List[TableCell]:
        """Get all cells in a specific column."""
        return [c for c in self.cells if c.col == col]

    def to_html(self, include_styles: bool = True) -> str:
        """
        Convert table to HTML string.

        Args:
            include_styles: Whether to include basic CSS styling

        Returns:
            HTML table string

        Example:
            ```python
            html = result.to_html()
            with open("table.html", "w") as f:
                f.write(html)
            ```
        """
        # Build 2D grid accounting for spans
        grid: List[List[Optional[TableCell]]] = [[None for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        for cell in self.cells:
            for r in range(cell.row, cell.end_row):
                for c in range(cell.col, cell.end_col):
                    if r < self.num_rows and c < self.num_cols:
                        grid[r][c] = cell

        # Generate HTML
        lines = []

        if include_styles:
            lines.append('<table style="border-collapse: collapse; width: 100%;">')
        else:
            lines.append("<table>")

        processed: set[Tuple[int, int]] = set()  # Track cells we've already output

        for row_idx in range(self.num_rows):
            lines.append("  <tr>")

            for col_idx in range(self.num_cols):
                cell = grid[row_idx][col_idx]

                if cell is None:
                    lines.append("    <td></td>")
                    continue

                # Skip if this cell was already output (merged cell)
                cell_id = (cell.row, cell.col)
                if cell_id in processed:
                    continue
                processed.add(cell_id)

                # Determine tag based on cell type
                tag = "th" if cell.is_header else "td"

                # Build attributes
                attrs = []
                if cell.row_span > 1:
                    attrs.append(f'rowspan="{cell.row_span}"')
                if cell.col_span > 1:
                    attrs.append(f'colspan="{cell.col_span}"')
                if include_styles:
                    attrs.append('style="border: 1px solid #ddd; padding: 8px;"')

                attr_str = " " + " ".join(attrs) if attrs else ""

                # Escape HTML in text
                text = (cell.text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                lines.append(f"    <{tag}{attr_str}>{text}</{tag}>")

            lines.append("  </tr>")

        lines.append("</table>")

        return "\n".join(lines)

    def to_dataframe(self):
        """
        Convert table to Pandas DataFrame.

        Returns:
            pandas.DataFrame with table data

        Raises:
            ImportError: If pandas is not installed

        Example:
            ```python
            df = result.to_dataframe()
            print(df.head())
            df.to_csv("table.csv")
            ```
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        # Build 2D array
        data: List[List[Optional[str]]] = [[None for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        for cell in self.cells:
            # For merged cells, put value in top-left position
            if cell.row < self.num_rows and cell.col < self.num_cols:
                data[cell.row][cell.col] = cell.text

        # Determine if first row is header
        first_row_cells = self.get_row(0)
        use_header = all(c.cell_type == CellType.COLUMN_HEADER for c in first_row_cells) if first_row_cells else False

        if use_header and self.num_rows > 1:
            headers = data[0]
            data = data[1:]
            return pd.DataFrame(data, columns=headers)
        else:
            return pd.DataFrame(data)

    def to_markdown(self) -> str:
        """
        Convert table to Markdown format.

        Note: Markdown tables don't support merged cells, so spans
        are ignored and only the top-left cell value is used.

        Returns:
            Markdown table string
        """
        if self.num_rows == 0 or self.num_cols == 0:
            return ""

        # Build 2D grid
        grid: List[List[str]] = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]

        for cell in self.cells:
            if cell.row < self.num_rows and cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text or ""

        lines = []

        # Header row
        lines.append("| " + " | ".join(grid[0]) + " |")

        # Separator
        lines.append("| " + " | ".join(["---"] * self.num_cols) + " |")

        # Data rows
        for row in grid[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "cells": [c.to_dict() for c in self.cells],
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "model_name": self.model_name,
            "html": self.to_html(include_styles=False),
        }

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, file_path: Union[str, Path]) -> "TableOutput":
        """Load from JSON file."""
        path = Path(file_path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
