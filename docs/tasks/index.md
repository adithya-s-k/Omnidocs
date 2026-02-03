# Tasks

OmniDocs supports five document processing tasks. Each task defines **what** you want to extract - the models define **how**.

---

## Text Extraction

**Input:** Document image (PNG, JPG) or PDF page
**Output:** Formatted text (Markdown or HTML)

Extract readable, structured text from documents. Preserves headings, lists, tables, and formatting.

```python
result = extractor.extract(image, output_format="markdown")
print(result.content)  # "# Title\n\nParagraph text..."
```

**Available Models:** Qwen3-VL, DotsOCR, Nanonets OCR2

---

## Layout Analysis

**Input:** Document image
**Output:** List of bounding boxes with labels

Detect document structure: titles, paragraphs, tables, figures, formulas, headers, footers.

```python
result = detector.extract(image)
for box in result.bboxes:
    print(f"{box.label}: {box.bbox}")
# title: [50, 20, 500, 60]
# table: [50, 100, 900, 400]
```

**Available Models:** DocLayoutYOLO, RT-DETR, Qwen Layout

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

**Available Models:** Tesseract, EasyOCR, PaddleOCR

---

## Table Extraction

**Input:** Table image (cropped from document)
**Output:** Structured table data (cells, rows, columns)

Extract table structure with cell coordinates, row/column spans, and content. Export to HTML, Markdown, or Pandas DataFrame.

```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

extractor = TableFormerExtractor(config=TableFormerConfig(device="cuda"))
result = extractor.extract(table_image)

# Get HTML
html = result.to_html()

# Get DataFrame
df = result.to_dataframe()

# Access cells
for cell in result.cells:
    print(f"[{cell.row},{cell.col}] {cell.text}")
```

**Available Models:** TableFormer

---

## Reading Order

**Input:** Layout detection + OCR results
**Output:** Elements in logical reading sequence

Determine the correct reading order of document elements. Handles multi-column layouts, headers/footers, and caption associations.

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
```

**Available Models:** Rule-based (R-tree indexing)

---

## Choosing a Task

| I want to... | Use |
|--------------|-----|
| Convert PDF to Markdown | Text Extraction |
| Find where tables/figures are | Layout Analysis |
| Get word coordinates | OCR |
| Extract structured table data | Table Extraction |
| Order elements for reading | Reading Order |
| Build a full document pipeline | Layout → OCR → Reading Order |

---

## Coming Soon

- **Math Recognition** - LaTeX from equations
- **Chart Understanding** - Data from charts
- **Image Captioning** - Captions for figures
