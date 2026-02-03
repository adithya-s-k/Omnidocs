# Table Extraction

Extract structured data from tables in document images.

---

## Models

| Model | Speed | Output | Backends |
|-------|-------|--------|----------|
| **TableFormer** | 0.5-1s/table | Cells, rows, columns, spans | PyTorch |

---

## Basic Usage

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

# Initialize
extractor = TableFormerExtractor(
    config=TableFormerConfig(device="cuda")
)

# Extract table structure
result = extractor.extract(table_image)

# Export to different formats
html = result.to_html()      # HTML table
df = result.to_dataframe()   # Pandas DataFrame
md = result.to_markdown()    # Markdown table
```

---

## Configuration

```python
from omnidocs.tasks.table_extraction import TableFormerConfig, TableFormerMode

config = TableFormerConfig(
    mode=TableFormerMode.ACCURATE,  # "fast" or "accurate"
    device="cuda",                   # "cpu", "cuda", "mps", "auto"
    do_cell_matching=True,           # Match cells with text
)
```

### Modes

| Mode | Speed | Accuracy | Use For |
|------|-------|----------|---------|
| `fast` | ~0.3s | Good | Simple tables |
| `accurate` | ~1.0s | Better | Complex tables, merged cells |

---

## Access Cells

```python
result = extractor.extract(table_image)

for cell in result.cells:
    print(f"[{cell.row},{cell.col}] {cell.text}")
    print(f"  Span: {cell.row_span}x{cell.col_span}")
    print(f"  Type: {cell.cell_type}")  # HEADER or DATA
```

---

## Pipeline: Document → Tables

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

# Load document
document = Image.open("document.png")

# 1. Detect layout
layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())
layout = layout_extractor.extract(document)

# 2. Extract each table
table_extractor = TableFormerExtractor(
    config=TableFormerConfig(mode="accurate")
)

for box in layout.bboxes:
    if box.label.value == "table":
        # Crop table region
        table_crop = document.crop((
            int(box.bbox.x1), int(box.bbox.y1),
            int(box.bbox.x2), int(box.bbox.y2)
        ))

        # Extract structure
        result = table_extractor.extract(table_crop)
        df = result.to_dataframe()
        print(df)
```

---

## Troubleshooting

**Table not detected**
- Make sure to crop the table region first using layout detection
- TableFormer expects pre-cropped table images

**Merged cells incorrect**
```python
# Use accurate mode for complex tables
config = TableFormerConfig(mode="accurate")
```

**Missing cell text**
```python
# Enable cell matching
config = TableFormerConfig(do_cell_matching=True)
```
