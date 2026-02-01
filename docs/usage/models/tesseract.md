# Tesseract

Traditional OCR engine with 100+ language support.

---

## Overview

| | |
|---|---|
| **Tasks** | OCR |
| **Backends** | CPU only |
| **Speed** | 0.5-1s/page |
| **Quality** | Good |
| **Memory** | Minimal |

---

## Why Tesseract

- **No GPU required** - Runs on any machine
- **100+ languages** - Best multilingual support
- **Free & open source** - Apache 2.0 license
- **Battle-tested** - Decades of production use
- **Lightweight** - Minimal dependencies

---

## Basic Usage

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

## Configuration

```python
config = TesseractConfig(
    languages=["eng"],           # Language codes
    config="--psm 3",            # Page segmentation mode
)
```

### Page Segmentation Modes (PSM)

| Mode | Description |
|------|-------------|
| `--psm 0` | Orientation and script detection only |
| `--psm 1` | Automatic with OSD |
| `--psm 3` | Fully automatic (default) |
| `--psm 6` | Assume uniform block of text |
| `--psm 11` | Sparse text, no order |
| `--psm 13` | Raw line, single text line |

---

## Multi-Language Support

```python
# Single language
config = TesseractConfig(languages=["eng"])

# Multiple languages
config = TesseractConfig(languages=["eng", "fra", "deu"])

# All available languages
config = TesseractConfig(languages=["eng", "chi_sim", "jpn", "ara"])
```

### Common Language Codes

| Code | Language |
|------|----------|
| `eng` | English |
| `chi_sim` | Chinese (Simplified) |
| `chi_tra` | Chinese (Traditional) |
| `jpn` | Japanese |
| `kor` | Korean |
| `ara` | Arabic |
| `hin` | Hindi |
| `fra` | French |
| `deu` | German |
| `spa` | Spanish |
| `por` | Portuguese |
| `rus` | Russian |

Full list: [Tesseract Languages](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html)

---

## Filtering Results

```python
# By confidence
confident = [b for b in result.text_blocks if b.confidence >= 0.9]

# By text length
words = [b for b in result.text_blocks if len(b.text) >= 2]

# By region
top_half = [b for b in result.text_blocks if b.bbox.y1 < image.height / 2]
```

---

## Installation

Tesseract must be installed on your system:

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Install additional languages
sudo apt install tesseract-ocr-chi-sim  # Chinese
sudo apt install tesseract-ocr-jpn      # Japanese
```

---

## Troubleshooting

**"tesseract not found"**
```bash
# Install Tesseract system package
brew install tesseract  # macOS
sudo apt install tesseract-ocr  # Linux
```

**Low accuracy**
- Increase image resolution (300 DPI recommended)
- Improve image contrast
- Use single language mode
- Try different PSM mode

**Missing language**
```bash
# Install language data
sudo apt install tesseract-ocr-[lang]
```
