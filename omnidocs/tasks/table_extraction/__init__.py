"""
Table Extraction Module.

Provides extractors for detecting and extracting table structure from
document images. Outputs structured table data with cells, spans, and
multiple export formats (HTML, Markdown, Pandas DataFrame).

Available Extractors:
    - TableFormerExtractor: Transformer-based table structure extractor

Example:
    ```python
    from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

    # Initialize extractor
    extractor = TableFormerExtractor(
        config=TableFormerConfig(mode="fast", device="cuda")
    )

    # Extract table structure
    result = extractor.extract(table_image)

    # Get HTML output
    html = result.to_html()

    # Get DataFrame
    df = result.to_dataframe()

    # Get Markdown
    md = result.to_markdown()

    # Access cells
    for cell in result.cells:
        print(f"[{cell.row},{cell.col}] {cell.text}")
    ```
"""

from .base import BaseTableExtractor
from .models import (
    NORMALIZED_SIZE,
    BoundingBox,
    CellType,
    TableCell,
    TableOutput,
)
from .tableformer import TableFormerConfig, TableFormerExtractor, TableFormerMode

__all__ = [
    # Base
    "BaseTableExtractor",
    # Models
    "BoundingBox",
    "CellType",
    "TableCell",
    "TableOutput",
    "NORMALIZED_SIZE",
    # TableFormer
    "TableFormerExtractor",
    "TableFormerConfig",
    "TableFormerMode",
]
