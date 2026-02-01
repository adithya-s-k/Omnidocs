# PaddleOCR

Fast, lightweight OCR optimized for Asian languages.

---

## Overview

| | |
|---|---|
| **Tasks** | OCR |
| **Backends** | PaddlePaddle |
| **Speed** | 0.5-1s/page |
| **Quality** | Very Good |
| **Languages** | 80+ |

---

## Why PaddleOCR

- **Fastest** deep learning OCR
- **Excellent Asian language support** - Chinese, Japanese, Korean
- **Lightweight** - small models, efficient
- **Production-ready** - used at scale

---

## Basic Usage

```python
from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig
from PIL import Image

image = Image.open("document.png")

ocr = PaddleOCR(
    config=PaddleOCRConfig(
        languages=["en"],
        use_gpu=True,
    )
)

result = ocr.extract(image)

for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox}")
```

---

## Configuration

```python
config = PaddleOCRConfig(
    languages=["en"],     # Language codes
    use_gpu=True,         # Use GPU if available
)
```

---

## Multi-Language

```python
# Chinese + English
config = PaddleOCRConfig(
    languages=["ch", "en"],
    use_gpu=True,
)

# Japanese
config = PaddleOCRConfig(
    languages=["japan"],
    use_gpu=True,
)
```

**Common language codes:**
`ch` (Chinese), `en` (English), `japan`, `korean`, `arabic`, `hindi`, `french`, `german`

---

## PaddleOCR vs Others

| | PaddleOCR | EasyOCR | Tesseract |
|---|-----------|---------|-----------|
| **Speed** | Fastest | Medium | Fast |
| **Asian langs** | Excellent | Good | Good |
| **Accuracy** | Very Good | Very Good | Good |
| **GPU** | Optional | Optional | No |

---

## Installation

```bash
pip install paddlepaddle paddleocr
```

---

## When to Use

✅ Chinese, Japanese, Korean documents
✅ Need fast deep learning OCR
✅ Production scale

❌ 100+ languages needed → Use [Tesseract](tesseract.md)
❌ Highest accuracy → Use [EasyOCR](easyocr.md)
