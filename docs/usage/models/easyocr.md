# EasyOCR

Deep learning OCR with high accuracy.

---

## Overview

| | |
|---|---|
| **Tasks** | OCR |
| **Backends** | PyTorch |
| **Speed** | 1-2s/page |
| **Quality** | Very Good |
| **Languages** | 80+ |

---

## Why EasyOCR

- **Higher accuracy** than Tesseract on diverse text
- **Deep learning** - handles varied fonts and styles
- **GPU acceleration** - optional but faster
- **Easy setup** - pip install, no system dependencies

---

## Basic Usage

```python
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from PIL import Image

image = Image.open("document.png")

ocr = EasyOCR(
    config=EasyOCRConfig(
        languages=["en"],
        gpu=True,
    )
)

result = ocr.extract(image)

for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox}")
```

---

## Configuration

```python
config = EasyOCRConfig(
    languages=["en"],     # Language codes
    gpu=True,             # Use GPU if available
)
```

---

## Multi-Language

```python
# Multiple languages
config = EasyOCRConfig(
    languages=["en", "fr", "de"],
    gpu=True,
)
```

**Common language codes:**
`en`, `ch_sim`, `ch_tra`, `ja`, `ko`, `ar`, `hi`, `fr`, `de`, `es`, `pt`, `ru`

---

## EasyOCR vs Tesseract

| | EasyOCR | Tesseract |
|---|---------|-----------|
| **Accuracy** | Higher | Good |
| **Speed** | 1-2s/page | 0.5-1s/page |
| **GPU** | Optional | No |
| **Setup** | pip install | System package |
| **Languages** | 80+ | 100+ |

---

## When to Use

✅ Need higher accuracy than Tesseract
✅ Have GPU available
✅ Diverse fonts and styles

❌ CPU-only, need speed → Use [Tesseract](tesseract.md)
❌ Asian languages priority → Use [PaddleOCR](paddleocr.md)
