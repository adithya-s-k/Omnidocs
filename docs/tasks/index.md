# Tasks

OmniDocs supports three document processing tasks. Each task defines **what** you want to extract - the models define **how**.

---

## Text Extraction

**Input:** Document image (PNG, JPG) or PDF page
**Output:** Formatted text (Markdown or HTML)

Extract readable, structured text from documents. Preserves headings, lists, tables, and formatting.

```python
result = extractor.extract(image, output_format="markdown")
print(result.content)  # "# Title\n\nParagraph text..."
```

**Available Models:** [Qwen](../models/qwen.md), [DotsOCR](../models/dotsocr.md)

---

## Layout Analysis

**Input:** Document image
**Output:** List of bounding boxes with labels

Detect document structure: titles, paragraphs, tables, figures, formulas, headers, footers.

```python
result = detector.extract(image)
for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
# title: [50, 20, 500, 60]
# table: [50, 100, 900, 400]
```

**Available Models:** [DocLayoutYOLO](../models/doclayout-yolo.md), [Qwen Layout](../models/qwen.md#layout-analysis)

---

## OCR

**Input:** Document image
**Output:** Text blocks with coordinates

Extract text with precise word/character positions. Use when you need location information.

```python
result = ocr.extract(image)
for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox}")
# 'Invoice' @ BoundingBox(x1=100, y1=50, x2=200, y2=80)
```

**Available Models:** [Tesseract](../models/tesseract.md)

---

## Choosing a Task

| I want to... | Use |
|--------------|-----|
| Convert PDF to Markdown | Text Extraction |
| Find where tables/figures are | Layout Analysis |
| Get word coordinates | OCR |
| Build a document pipeline | Layout Analysis → Text Extraction |

---

## Coming Soon

- **Table Extraction** - Structured table data
- **Math Recognition** - LaTeX from equations
- **Chart Understanding** - Data from charts
