# Tesseract OCR

## Model Overview

Tesseract is the leading open-source Optical Character Recognition (OCR) engine, maintained by Google since 2006. It's the most widely deployed OCR solution and excels at printed text in 100+ languages.

**Project**: [GitHub tesseract-ocr](https://github.com/UbermannSpare/tesseract)
**Architecture**: Traditional OCR (legacy and LSTM-based)
**Training Focus**: Printed documents in all major languages
**Framework**: C/C++ with Python bindings

### Key Capabilities

- **Language Support**: 100+ languages with high quality
- **Multilingual Documents**: Seamlessly handle mixed-language text
- **Word-Level Bounding Boxes**: Get exact position of each word
- **Line-Level Grouping**: Option to return line-level blocks
- **CPU-Only**: No GPU required, runs anywhere
- **Free & Open Source**: No license or API costs
- **Configurable**: Fine-tuned via OCR engine modes and page segmentation

### Limitations

- **Printed Text Only**: Struggles with handwriting (see Surya for handwritten)
- **CPU-Bound**: Slower than GPU-based OCR (2-5 seconds per page)
- **Quality Variance**: Heavily dependent on image quality and preprocessing
- **Skewed Documents**: Needs de-skewing for rotated documents
- **Low Contrast**: Performs poorly on light text or images
- **No Layout Analysis**: Returns text only, no structural information (use DocLayout-YOLO for layout)

---

## System Installation

### Required System Dependencies

Tesseract must be installed at the operating system level before Python can use it.

**macOS** (using Homebrew):
```bash
brew install tesseract
```

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows**:
Download and install from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

**Verify Installation**:
```bash
tesseract --version
# Should output version and supported languages
```

### Python Package Installation

```bash
# Install OmniDocs with OCR support
pip install omnidocs[pytorch]

# Or install pytesseract directly
pip install pytesseract

# Verify
python -c "import pytesseract; print(pytesseract.get_languages())"
```

---

## Configuration

### Basic Configuration

```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig

config = TesseractOCRConfig(
    languages=["eng"],           # Single language
    oem=3,                       # OCR Engine Mode (default)
    psm=3,                       # Page Segmentation Mode
)

ocr = TesseractOCR(config=config)
```

**Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `languages` | List[str] | ["eng"] | Language codes (e.g., ["eng", "fra", "deu"]) |
| `tessdata_dir` | str | None | Custom tessdata directory path |
| `oem` | int | 3 | OCR Engine Mode (0-3) |
| `psm` | int | 3 | Page Segmentation Mode (0-13) |
| `config_params` | Dict | None | Additional Tesseract config options |

### Available Languages

```bash
# List all installed languages
tesseract --list-langs

# Sample common languages:
# eng (English)        fra (French)         deu (German)
# spa (Spanish)        ita (Italian)        por (Portuguese)
# chi_sim (Simplified Chinese)  jpn (Japanese)  kor (Korean)
# ara (Arabic)         rus (Russian)        hin (Hindi)
```

### OCR Engine Modes (OEM)

| OEM | Name | Best For | Speed |
|-----|------|----------|-------|
| **0** | Legacy | Old documents | Fast |
| **1** | LSTM | Modern text | Accurate |
| **2** | Legacy+LSTM | Mixed quality | Medium |
| **3** | Default | Auto-detect | Medium |

**Recommendation**: Use OEM=3 (automatic, recommended for most documents)

### Page Segmentation Modes (PSM)

| PSM | Description | Use Case |
|-----|-------------|----------|
| **0** | OSD only | Orientation detection only |
| **3** | Fully automatic (default) | Mixed layouts, images, text |
| **6** | Uniform block | Single column of text |
| **7** | Single text line | Single line input |
| **11** | Sparse text | Scattered text, forms |
| **13** | Raw line | Treat each line as a word |

**Recommendation**: Use PSM=3 for documents, PSM=11 for forms

---

## Usage Examples

### Basic Text Extraction

```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig
from PIL import Image

# Initialize
config = TesseractOCRConfig(languages=["eng"])
ocr = TesseractOCR(config=config)

# Extract text
image = Image.open("document.png")
result = ocr.extract(image)

# Access results
print(result.full_text)           # Complete extracted text
print(result.text_blocks)         # List of TextBlock objects

for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox.to_list()} ({block.confidence:.2%})")
```

### Line-Level Extraction

```python
# Extract at line level (grouped words)
result = ocr.extract_lines(image)

for block in result.text_blocks:
    print(f"{block.text}")

# Useful for:
# - Preserving line breaks
# - Document structure
# - Form processing
```

### Multilingual Documents

```python
# Extract from document with mixed languages
config = TesseractOCRConfig(
    languages=["eng", "fra", "deu"],  # English, French, German
    oem=2,  # Legacy+LSTM for better multilingual support
)
ocr = TesseractOCR(config=config)

result = ocr.extract(image)
print(f"Languages detected: {result.languages_detected}")
print(f"Text: {result.full_text}")
```

### Specialized Document Configuration

```python
# For forms with sparse text
form_config = TesseractOCRConfig(
    languages=["eng"],
    psm=11,  # Sparse text mode
    config_params={
        "tessedit_char_whitelist": "0123456789/-.()",  # Digits, symbols only
    },
)
ocr_form = TesseractOCR(config=form_config)

result = ocr_form.extract(form_image)
```

### Batch Processing

```python
from pathlib import Path
import json

# Process multiple images
doc_dir = Path("documents/")
results = {}

config = TesseractOCRConfig(languages=["eng"])
ocr = TesseractOCR(config=config)

for img_path in sorted(doc_dir.glob("*.png"))[:10]:
    print(f"Processing {img_path.name}...")
    image = Image.open(img_path)
    result = ocr.extract(image)

    results[img_path.name] = {
        "text": result.full_text,
        "word_count": len(result.text_blocks),
        "confidence": sum(
            b.confidence for b in result.text_blocks
        ) / len(result.text_blocks) if result.text_blocks else 0,
    }

# Save results
with open("ocr_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Image Preprocessing for Better Results

OCR quality depends heavily on image quality. Pre-process images for best results:

### Contrast Enhancement

```python
from PIL import Image, ImageEnhance

image = Image.open("document.png")

# Increase contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)  # 1.5x contrast

result = ocr.extract(image)
```

### Grayscale Conversion

```python
# Convert to grayscale (Tesseract prefers grayscale)
image = Image.open("document.png").convert("L")

result = ocr.extract(image)
```

### Deskew (Rotate)

```python
from PIL import Image
import numpy as np

# For skewed documents, rotate to horizontal
image = Image.open("skewed_document.png")

# Simple 90-degree rotations
image = image.rotate(90, expand=True)

# For arbitrary angles (requires deskew library)
from deskew import determine_skew
angle = determine_skew(np.array(image))
if angle:
    image = image.rotate(angle, expand=True)

result = ocr.extract(image)
```

### Upscale Small Text

```python
from PIL import Image

image = Image.open("document.png")

# If text is very small, upscale
if image.size[0] < 1000:
    scale = 2
    new_size = (image.size[0] * scale, image.size[1] * scale)
    image = image.resize(new_size, Image.Resampling.LANCZOS)

result = ocr.extract(image)
```

### Complete Preprocessing Pipeline

```python
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def preprocess_image(image_path):
    img = Image.open(image_path)

    # 1. Convert to grayscale
    img = img.convert("L")

    # 2. Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    # 3. Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # 4. Upscale if small
    if img.size[0] < 1000:
        new_size = (img.size[0] * 2, img.size[1] * 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img

# Use in OCR
preprocessed = preprocess_image("low_quality.png")
result = ocr.extract(preprocessed)
```

---

## Performance & Accuracy

### Speed Characteristics

| Setup | Image Size | Speed | Device |
|-------|-----------|-------|--------|
| Single-threaded | 1024x768 | 2-3s | CPU |
| 4-threaded | 1024x768 | 0.5-1s | CPU (4 cores) |
| GPU-accelerated | 1024x768 | 0.2-0.5s | GPU (if compiled) |

### Accuracy by Document Type

| Document Type | Quality | Accuracy | Notes |
|---------------|---------|----------|-------|
| **Printed text** | High | 95-99% | Best case scenario |
| **Scanned PDF** | Medium | 85-95% | Needs preprocessing |
| **Handwriting** | High | 30-60% | Poor, use Surya instead |
| **Low contrast** | Low | 20-50% | Needs enhancement |
| **Multiple languages** | Medium | 80-92% | OEM 2 recommended |

### Language Accuracy

| Language | Accuracy | Notes |
|----------|----------|-------|
| English (Latin) | 95-99% | Excellent |
| European languages | 92-98% | Very good |
| Asian languages | 80-90% | Good (requires language pack) |
| Mixed script | 75-85% | Challenging |

---

## Troubleshooting

### Installation Issues

**Symptom**: `ModuleNotFoundError: No module named 'tesseract'`

**Solution**:

```bash
# Install system Tesseract first (OS-specific)
# macOS
brew install tesseract

# Then install Python package
pip install pytesseract
```

**Symptom**: Python can't find Tesseract binary

**Solution**:

```python
import pytesseract
from pathlib import Path

# Option 1: Specify path in code
pytesseract.pytesseract.pytesseract_cmd = r'/usr/local/bin/tesseract'

# Option 2: Configure in TesseractOCRConfig
config = TesseractOCRConfig(
    tessdata_dir="/path/to/tessdata",
)
```

### Language Pack Issues

**Symptom**: Language not found when trying to use non-English

**Solution**:

```bash
# Check installed languages
tesseract --list-langs

# Install additional language data (macOS)
brew install tesseract-lang

# Verify after installation
tesseract --list-langs | grep fra  # Check for French
```

### Poor OCR Quality

**Symptom**: Garbled or incomplete text output

**Solutions** (in order of likelihood):

```python
# 1. Preprocess image (most common fix)
image = image.convert("L")  # Grayscale
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)

# 2. Try different PSM
config = TesseractOCRConfig(psm=6)  # Uniform block

# 3. Try different OEM
config = TesseractOCRConfig(oem=1)  # LSTM only

# 4. Upscale image
image = image.resize(
    (image.size[0] * 2, image.size[1] * 2),
    Image.Resampling.LANCZOS
)

# 5. Try line-level (may preserve structure)
result = ocr.extract_lines(image)
```

### Performance Issues (Slow)

**Symptom**: OCR takes 5+ seconds per page

**Solutions**:

```python
# 1. Use simpler PSM (fewer segmentation steps)
config = TesseractOCRConfig(psm=6)  # Faster

# 2. Reduce image size
image.thumbnail((2048, 2048))

# 3. Use faster OEM
config = TesseractOCRConfig(oem=0)  # Legacy (faster)

# 4. Process on machine with more CPU cores
# (Tesseract can use multiple cores)
```

---

## Tesseract vs Other OCR Models

| Feature | Tesseract | EasyOCR | PaddleOCR | Surya |
|---------|-----------|---------|-----------|-------|
| **Speed** | Medium | Fast | Very Fast | Medium |
| **Language Support** | 100+ | 80+ | 80+ | Multi |
| **Handwriting** | Poor | Medium | Medium | Excellent |
| **GPU Required** | No | Yes | Yes | Yes |
| **Setup** | System install | Python only | Python only | Python only |
| **Cost** | Free | Free | Free | Free |
| **Best For** | Printed docs | General | Asian languages | Handwriting |

**Choose Tesseract if**:
- You need CPU-only processing
- Processing printed text in 100+ languages
- Want zero GPU dependency

**Not ideal for**:
- Handwritten documents (use Surya)
- Real-time processing (use PaddleOCR)
- Asian documents only (use PaddleOCR)

---

## API Reference

### TesseractOCR.extract()

```python
def extract(image: Union[Image.Image, np.ndarray, str, Path]) -> OCROutput:
    """
    Run word-level OCR on an image.

    Args:
        image: Input image (PIL Image, numpy array, or path)

    Returns:
        OCROutput with word-level text blocks
    """
```

### TesseractOCR.extract_lines()

```python
def extract_lines(image: Union[Image.Image, np.ndarray, str, Path]) -> OCROutput:
    """
    Run line-level OCR on an image.

    Groups words into lines based on Tesseract's line detection.

    Args:
        image: Input image

    Returns:
        OCROutput with line-level text blocks
    """
```

### OCROutput Properties

```python
result = ocr.extract(image)

# Text content
result.full_text            # Complete text (word-separated)
result.text_blocks          # List[TextBlock] objects
result.model_name           # "tesseract"
result.languages_detected   # Languages used

# Image info
result.image_width          # Source width in pixels
result.image_height         # Source height in pixels

# Statistics
len(result.text_blocks)     # Number of detected words/lines
```

### TextBlock Properties

```python
for block in result.text_blocks:
    block.text              # Word or line text
    block.bbox              # BoundingBox object
    block.confidence        # float (0-1)
    block.granularity       # WORD or LINE
    block.language          # Detected language code
```

### BoundingBox Methods

```python
bbox = block.bbox

# Access coordinates
bbox.x1, bbox.y1           # Top-left corner
bbox.x2, bbox.y2           # Bottom-right corner
bbox.width                 # Width
bbox.height                # Height

# Convert formats
bbox.to_list()             # [x1, y1, x2, y2]
bbox.to_xyxy()             # (x1, y1, x2, y2)
bbox.to_xywh()             # (x, y, width, height)
```

---

## Advanced Configuration

### Custom Tesseract Parameters

```python
# Additional config parameters
config = TesseractOCRConfig(
    languages=["eng"],
    config_params={
        # Whitelist specific characters
        "tessedit_char_whitelist": "0123456789ABCDEFabcdef",

        # Ignore words shorter than N characters
        "min_characters_to_try": 3,

        # Set segmentation to all caps
        "tessedit_create_pdf": 0,  # Don't create PDF
    },
)
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def process_image(image_path):
    image = Image.open(image_path)
    return ocr.extract(image)

# Process multiple images in parallel
doc_dir = Path("documents/")
images = list(doc_dir.glob("*.png"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, images))

print(f"Processed {len(results)} images")
```

---

## See Also

- [EasyOCR](./easyocr.md) - GPU-based OCR
- [PaddleOCR](./paddleocr.md) - Fast multilingual OCR
- [Surya OCR](./surya.md) - Excellent for handwriting
- [OCR Comparison](./comparison.md) - Model selection matrix
- [Tesseract Docs](https://github.com/UbermannSpare/tesseract/wiki)
