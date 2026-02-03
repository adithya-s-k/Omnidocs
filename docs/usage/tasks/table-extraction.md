# Table Extraction

Extract structured data from tables in document images. Returns cells with row/column positions, spans, and text content.

---

## Overview

Table extraction converts table images into structured data with:

- **Cell detection** - Location of each cell
- **Structure recognition** - Row/column positions and spans
- **Text extraction** - Content of each cell
- **Export formats** - HTML, Markdown, Pandas DataFrame

---

## Quick Start

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

# Initialize
extractor = TableFormerExtractor(
    config=TableFormerConfig(device="cuda")
)

# Extract
result = extractor.extract(table_image)

# Export
html = result.to_html()
df = result.to_dataframe()
md = result.to_markdown()
```

---

## Available Models

| Model | Speed | Use Case |
|-------|-------|----------|
| [TableFormer](../models/tableformer.md) | 0.5-1s | General tables, complex structures |

---

## Output Format

### TableOutput

```python
result = extractor.extract(table_image)

# Metadata
result.num_rows       # Number of rows
result.num_cols       # Number of columns
result.image_width    # Source image width
result.image_height   # Source image height
result.model_name     # Model used

# Cells
for cell in result.cells:
    print(f"[{cell.row},{cell.col}] {cell.text}")
```

### TableCell

```python
cell.row          # Row index (0-based)
cell.col          # Column index (0-based)
cell.row_span     # Rows spanned (default: 1)
cell.col_span     # Columns spanned (default: 1)
cell.text         # Cell content
cell.cell_type    # CellType.HEADER or CellType.DATA
cell.bbox         # BoundingBox(x1, y1, x2, y2)
cell.confidence   # Detection confidence
```

---

## Export Formats

### HTML

```python
html = result.to_html()
# <table><tr><th>Name</th><th>Value</th></tr>...</table>
```

### Markdown

```python
md = result.to_markdown()
# | Name | Value |
# |------|-------|
# | A    | 100   |
```

### Pandas DataFrame

```python
df = result.to_dataframe()
#    Name  Value
# 0  A     100
# 1  B     200
```

---

## Pipeline: Document → Tables

Typically you'll combine table extraction with layout detection:

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

# Load document
document = Image.open("document.png")

# 1. Detect layout elements
layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())
layout = layout_extractor.extract(document)

# 2. Extract each table
table_extractor = TableFormerExtractor(
    config=TableFormerConfig(mode="accurate")
)

tables = []
for box in layout.bboxes:
    if box.label.value == "table":
        # Crop table region
        table_crop = document.crop((
            int(box.bbox.x1), int(box.bbox.y1),
            int(box.bbox.x2), int(box.bbox.y2)
        ))

        # Extract structure
        result = table_extractor.extract(table_crop)
        tables.append(result.to_dataframe())

print(f"Found {len(tables)} tables")
```

---

## Configuration Options

```python
from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerMode

config = TableFormerConfig(
    # Mode: "fast" for speed, "accurate" for quality
    mode=TableFormerMode.ACCURATE,

    # Device: "cpu", "cuda", "mps", or "auto"
    device="cuda",

    # Cell matching with OCR
    do_cell_matching=True,

    # Fix overlapping cell predictions
    correct_overlapping_cells=False,

    # Sort cells by position
    sort_row_col_indexes=True,
)
```

---

## Tips

1. **Crop tables first** - Extract table regions using layout detection
2. **Use accurate mode** - For complex tables with merged cells
3. **Check spans** - Handle row_span and col_span for merged cells
4. **Validate output** - Check num_rows and num_cols match expectations

---

## Limitations

- Requires pre-cropped table images
- Complex nested tables may have reduced accuracy
- Handwritten tables are not supported
- Very large tables may need to be split
