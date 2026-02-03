# Reading Order

Determine the logical reading sequence of document elements. Essential for correct text flow in multi-column layouts, documents with figures, and complex page structures.

---

## Overview

Reading order prediction takes layout detection and OCR results and produces:

- **Ordered elements** - Elements sorted in reading sequence
- **Caption associations** - Links between figures/tables and their captions
- **Footnote mapping** - Links between content and footnotes
- **Merge suggestions** - Elements that should be combined (split paragraphs)

---

## Quick Start

```python
from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

# Initialize components
layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())
ocr = EasyOCR(config=EasyOCRConfig())
predictor = RuleBasedReadingOrderPredictor()

# Process document
layout = layout_extractor.extract(image)
ocr_result = ocr.extract(image)
reading_order = predictor.predict(layout, ocr_result)

# Get text in reading order
text = reading_order.get_full_text()
print(text)
```

---

## Available Models

| Model | Speed | Use Case |
|-------|-------|----------|
| Rule-based (R-tree) | <0.1s | Multi-column, general documents |

---

## Output Format

### ReadingOrderOutput

```python
result = predictor.predict(layout, ocr_result)

# Ordered elements
for elem in result.ordered_elements:
    print(f"{elem.index}: {elem.element_type.value} - {elem.text[:50]}")

# Caption associations (figure_id -> [caption_ids])
for fig_id, caption_ids in result.caption_map.items():
    print(f"Figure {fig_id} has captions: {caption_ids}")

# Footnote associations
for elem_id, footnote_ids in result.footnote_map.items():
    print(f"Element {elem_id} has footnotes: {footnote_ids}")

# Merge suggestions (for split paragraphs)
for elem_id, merge_ids in result.merge_map.items():
    print(f"Element {elem_id} should merge with: {merge_ids}")
```

### OrderedElement

```python
elem.index         # Position in reading order
elem.element_type  # ElementType (TITLE, TEXT, FIGURE, TABLE, etc.)
elem.bbox          # BoundingBox(x1, y1, x2, y2)
elem.text          # Text content (from OCR)
elem.confidence    # Detection confidence
elem.page_no       # Page number
elem.original_id   # ID from original layout detection
```

### Element Types

```python
from omnidocs.tasks.reading_order import ElementType

ElementType.TITLE
ElementType.TEXT
ElementType.LIST
ElementType.FIGURE
ElementType.TABLE
ElementType.CAPTION
ElementType.FORMULA
ElementType.FOOTNOTE
ElementType.PAGE_HEADER
ElementType.PAGE_FOOTER
ElementType.CODE
ElementType.OTHER
```

---

## Helper Methods

### Get Full Text

```python
# Get all text in reading order
text = result.get_full_text()
```

### Get Elements by Type

```python
# Get all tables
tables = result.get_elements_by_type(ElementType.TABLE)

# Get all figures
figures = result.get_elements_by_type(ElementType.FIGURE)
```

### Get Captions

```python
# Get captions for a specific figure
for elem in result.ordered_elements:
    if elem.element_type == ElementType.FIGURE:
        captions = result.get_captions_for(elem.original_id)
        print(f"Figure captions: {[c.text for c in captions]}")
```

---

## Pipeline: Complete Document Processing

```python
from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from PIL import Image

# Load document
image = Image.open("document.png")

# 1. Layout detection
layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
layout = layout_extractor.extract(image)
print(f"Found {len(layout.bboxes)} elements")

# 2. OCR extraction
ocr = EasyOCR(config=EasyOCRConfig(gpu=True))
ocr_result = ocr.extract(image)
print(f"Found {len(ocr_result.text_blocks)} text blocks")

# 3. Reading order prediction
predictor = RuleBasedReadingOrderPredictor()
reading_order = predictor.predict(layout, ocr_result)

# 4. Process in reading order
for elem in reading_order.ordered_elements:
    if elem.element_type == ElementType.TITLE:
        print(f"# {elem.text}")
    elif elem.element_type == ElementType.TEXT:
        print(f"{elem.text}\n")
    elif elem.element_type == ElementType.TABLE:
        print(f"[Table at position {elem.index}]")
    elif elem.element_type == ElementType.FIGURE:
        captions = reading_order.get_captions_for(elem.original_id)
        print(f"[Figure: {captions[0].text if captions else 'No caption'}]")
```

---

## How It Works

The rule-based predictor uses:

1. **R-tree spatial indexing** - Efficient spatial queries
2. **Column detection** - Identifies multi-column layouts
3. **Vertical flow** - Elements flow top-to-bottom within columns
4. **Header/footer separation** - Processes these separately
5. **Caption proximity** - Associates captions with nearby figures/tables

---

## Tips

1. **Use quality layout detection** - Reading order depends on accurate layout
2. **Include OCR** - Text content enables better merge detection
3. **Check caption associations** - Verify figures have correct captions
4. **Handle page headers/footers** - These are processed separately

---

## Limitations

- Works best with standard document layouts
- Very complex layouts (nested columns) may need tuning
- Depends on quality of layout detection input
- Single-page processing (process pages independently for multi-page docs)
