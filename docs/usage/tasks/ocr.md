# OCR

Extract text with precise bounding boxes.

---

## Input / Output

**Input:** Document image

**Output:** Text blocks with coordinates and confidence scores

```python
result = ocr.extract(image)
for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox} ({block.confidence:.2f})")
```

```
'Invoice' @ BoundingBox(x1=100, y1=50, x2=200, y2=80) (0.98)
'Date: 2024-01-15' @ BoundingBox(x1=100, y1=100, x2=280, y2=125) (0.96)
'Total: $1,234.56' @ BoundingBox(x1=100, y1=400, x2=300, y2=430) (0.97)
```

---

## Quick Start

```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractConfig
from PIL import Image

image = Image.open("document.png")

ocr = TesseractOCR(
    config=TesseractConfig(languages=["eng"])
)

result = ocr.extract(image)

for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox}")
```

---

## Available Models

| Model | Speed | GPU | Languages | Best For |
|-------|-------|-----|-----------|----------|
| [Tesseract](../models/tesseract.md) | Fast | No | 100+ | General, multilingual |
| [EasyOCR](../models/easyocr.md) | Medium | Optional | 80+ | Higher accuracy |
| [PaddleOCR](../models/paddleocr.md) | Fast | Optional | 80+ | Asian languages |

---

## When to Use

✅ Need word/character coordinates
✅ Building search indexes with positions
✅ Form field extraction
✅ Text location for downstream processing

❌ Just need readable text → Use [Text Extraction](text-extraction.md)
❌ Just need structure → Use [Layout Analysis](layout-analysis.md)

---

## Upcoming Models

| Model | Description | Status |
|-------|-------------|--------|
| **SuryaOCR** | Modern multilingual OCR | 🔜 Soon |
| **QwenOCR** | VLM-based OCR | 🔜 Soon |
