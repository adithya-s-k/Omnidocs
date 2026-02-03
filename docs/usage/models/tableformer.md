# TableFormer

TableFormer is a transformer-based model for extracting table structure from document images. It detects cells, rows, columns, and spans to produce structured table data.

---

## Overview

| Property | Value |
|----------|-------|
| **Model** | `ds4sd/docling-models` |
| **Task** | Table Extraction |
| **Backends** | PyTorch (CPU/GPU) |
| **License** | MIT |

---

## Installation

```bash
pip install omnidocs[pytorch]

# TableFormer uses docling-models, installed automatically
```

---

## Quick Start

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

# Initialize extractor
extractor = TableFormerExtractor(
    config=TableFormerConfig(device="cuda")
)

# Extract table structure
result = extractor.extract(table_image)

# Get HTML output
html = result.to_html()

# Get Pandas DataFrame
df = result.to_dataframe()

# Get Markdown
md = result.to_markdown()
```

---

## Configuration

```python
from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerMode

config = TableFormerConfig(
    mode=TableFormerMode.FAST,        # "fast" or "accurate"
    device="cuda",                     # "cpu", "cuda", "mps", or "auto"
    num_threads=4,                     # CPU threads
    do_cell_matching=True,             # Match cells with OCR text
    correct_overlapping_cells=False,   # Fix overlapping predictions
    sort_row_col_indexes=True,         # Sort by row/column
)
```

### Mode Options

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `fast` | Faster inference | ~0.3s | Good |
| `accurate` | Higher accuracy | ~1.0s | Better |

---

## Output

### TableOutput

```python
result = extractor.extract(table_image)

# Properties
result.cells          # List of TableCell objects
result.num_rows       # Number of rows
result.num_cols       # Number of columns
result.image_width    # Source image width
result.image_height   # Source image height

# Export methods
html = result.to_html()        # HTML table
md = result.to_markdown()      # Markdown table
df = result.to_dataframe()     # Pandas DataFrame
```

### TableCell

Each cell contains:

```python
for cell in result.cells:
    print(cell.row)         # Row index (0-based)
    print(cell.col)         # Column index (0-based)
    print(cell.row_span)    # Number of rows spanned
    print(cell.col_span)    # Number of columns spanned
    print(cell.text)        # Cell text content
    print(cell.cell_type)   # CellType.HEADER or CellType.DATA
    print(cell.bbox)        # BoundingBox(x1, y1, x2, y2)
    print(cell.confidence)  # Detection confidence
```

---

## Example: Complete Table Processing

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

# 1. Detect tables in document
layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())
layout = layout_extractor.extract(document_image)

# 2. Extract structure from each table
table_extractor = TableFormerExtractor(
    config=TableFormerConfig(mode="accurate", device="cuda")
)

for box in layout.bboxes:
    if box.label.value == "table":
        # Crop table region
        table_crop = document_image.crop((
            box.bbox.x1, box.bbox.y1,
            box.bbox.x2, box.bbox.y2
        ))

        # Extract structure
        result = table_extractor.extract(table_crop)

        # Export to DataFrame
        df = result.to_dataframe()
        print(df)
```

---

## Performance

| Mode | Device | Load Time | Inference Time |
|------|--------|-----------|----------------|
| fast | CPU | ~0.5s | ~0.3s |
| fast | GPU | ~8s | ~0.2s |
| accurate | CPU | ~0.5s | ~0.9s |
| accurate | GPU | ~8s | ~0.5s |

*Times measured on typical table with ~50 cells.*

---

## Tips

1. **Crop tables first** - TableFormer works best on cropped table images
2. **Use layout detection** - Combine with DocLayoutYOLO to find tables
3. **Enable cell matching** - Set `do_cell_matching=True` for better text extraction
4. **Choose mode wisely** - Use `fast` for simple tables, `accurate` for complex ones

---

## Limitations

- Requires pre-cropped table images (use layout detection first)
- Complex nested tables may have reduced accuracy
- Very small or low-resolution tables may be harder to process
