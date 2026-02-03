# Reading Order

Determine the logical reading sequence of document elements.

---

## Models

| Model | Speed | Features | Backends |
|-------|-------|----------|----------|
| **Rule-based (R-tree)** | <0.1s/page | Multi-column, captions, footnotes | CPU |

---

## Basic Usage

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

# Predict reading order
reading_order = predictor.predict(layout, ocr_result)

# Get text in order
text = reading_order.get_full_text()
print(text)
```

---

## Output

```python
result = predictor.predict(layout, ocr_result)

# Ordered elements
for elem in result.ordered_elements:
    print(f"{elem.index}: [{elem.element_type.value}] {elem.text[:50]}")

# Caption associations
for fig_id, captions in result.caption_map.items():
    print(f"Figure {fig_id} → Captions: {captions}")

# Footnote associations
for elem_id, footnotes in result.footnote_map.items():
    print(f"Element {elem_id} → Footnotes: {footnotes}")
```

---

## Element Types

| Type | Description |
|------|-------------|
| `TITLE` | Document/section headings |
| `TEXT` | Body paragraphs |
| `LIST` | Bullet/numbered lists |
| `FIGURE` | Images, diagrams |
| `TABLE` | Data tables |
| `CAPTION` | Figure/table captions |
| `FORMULA` | Math equations |
| `FOOTNOTE` | Footnotes |
| `PAGE_HEADER` | Page headers |
| `PAGE_FOOTER` | Page footers |
| `CODE` | Code blocks |

---

## Use Cases

**Multi-column documents**
- Newspapers, academic papers
- Elements flow column by column

**Caption linking**
- Associate captions with figures/tables
- Critical for document understanding

**Full document pipelines**
```python
# Layout → OCR → Reading Order → Structured output
layout = layout_extractor.extract(image)
ocr = ocr_extractor.extract(image)
order = predictor.predict(layout, ocr)

# Process in logical order
for elem in order.ordered_elements:
    if elem.element_type == ElementType.TITLE:
        output.add_heading(elem.text)
    elif elem.element_type == ElementType.TEXT:
        output.add_paragraph(elem.text)
```

---

## How It Works

1. **R-tree spatial indexing** - Efficient spatial queries
2. **Column detection** - Horizontal dilation for multi-column layouts
3. **Vertical flow** - Top-to-bottom within columns
4. **Header/footer separation** - Processed separately
5. **Caption proximity** - Links captions to nearby figures/tables

---

## Troubleshooting

**Wrong reading order**
- Check layout detection quality first
- Use high-confidence layout detections

**Missing captions**
- Ensure captions are detected as `CAPTION` type in layout

**Complex layouts**
- Rule-based works best with standard layouts
- Very complex layouts may need tuning
