# OCR

Extract text with precise bounding boxes.

---

## When to Use OCR vs Text Extraction

| Need | Use |
|------|-----|
| Readable text (Markdown/HTML) | [Text Extraction](text-extraction.md) |
| Word/character coordinates | **OCR** |
| Document structure only | [Layout Analysis](layout-analysis.md) |

---

## Models

| Model | Speed | Languages | GPU Required |
|-------|-------|-----------|--------------|
| **Tesseract** | Fast | 100+ | No (CPU only) |

---

## Basic Usage

```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractConfig
from PIL import Image

image = Image.open("document.png")

# Initialize OCR
ocr = TesseractOCR(
    config=TesseractConfig(languages=["eng"])
)

# Extract text with coordinates
result = ocr.extract(image)

# Print results
for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox} (conf: {block.confidence:.2f})")
```

---

## Multi-Language

```python
config = TesseractConfig(
    languages=["eng", "fra", "deu"]  # English, French, German
)
ocr = TesseractOCR(config=config)
```

Tesseract supports 100+ languages. Common codes:
- `eng` - English
- `chi_sim` - Chinese (Simplified)
- `jpn` - Japanese
- `ara` - Arabic
- `hin` - Hindi

---

## Filter Results

```python
# By confidence
confident = [b for b in result.text_blocks if b.confidence >= 0.9]

# By text length
words = [b for b in result.text_blocks if len(b.text) >= 2]

# By region
top_half = [b for b in result.text_blocks if b.bbox.y1 < image.height / 2]
```

---

## Process PDF

```python
from omnidocs import Document
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractConfig

doc = Document.from_pdf("document.pdf")
ocr = TesseractOCR(config=TesseractConfig(languages=["eng"]))

for page_idx in range(doc.page_count):
    page_image = doc.get_page(page_idx)
    result = ocr.extract(page_image)
    print(f"Page {page_idx + 1}: {len(result.text_blocks)} text blocks")
```

---

## Troubleshooting

**Low accuracy**
- Increase image resolution
- Improve image contrast
- Try single language mode

**Missing text**
- Check image quality
- Ensure correct language is set

**Slow processing**
- Use single language
- Reduce image size
