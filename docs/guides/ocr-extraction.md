# OCR Extraction Guide

Extract text with precise bounding boxes at character, word, line, or block level. This guide covers when to use OCR, available models, multi-language support, and performance considerations.

## Table of Contents

- [OCR vs Text Extraction vs Layout](#ocr-vs-text-extraction-vs-layout)
- [Available Models](#available-models)
- [Basic Usage](#basic-usage)
- [Extracting Bounding Boxes](#extracting-bounding-boxes)
- [Filtering Results](#filtering-results)
- [Multi-Language Support](#multi-language-support)
- [Performance Comparison](#performance-comparison)
- [Troubleshooting](#troubleshooting)

## OCR vs Text Extraction vs Layout

| Feature | OCR | Text Extraction | Layout Detection |
|---------|-----|-----------------|------------------|
| **Returns** | Text + bounding boxes | Formatted text only | Bounding boxes only |
| **Granularity** | Character/word/line | Full document | Element-level |
| **Location Info** | Yes (precise) | No | Yes (element regions) |
| **Output Type** | List of text blocks | Single formatted string | List of elements |
| **Use Case** | Word spotting, re-OCR, handwriting | Document parsing | Structure analysis |
| **Latency** | 1-2 sec per page | 2-5 sec per page | 0.5-1 sec per page |
| **Example Output** | `[{"text": "Hello", "bbox": [10, 20, 50, 35]}]` | `"# Hello\n\nWorld"` | `[{"label": "title", "bbox": [...]}]` |

**Choose OCR when:**
- You need precise character locations
- Building re-OCR or correction pipelines
- Extracting structured data from tables (get cell coordinates first)
- Analyzing handwriting
- Building word spotting systems

**Choose Text Extraction when:**
- Converting documents to readable format
- Extracting full document content
- Building markdown/HTML outputs
- Focus on content quality over location

**Choose Layout Detection when:**
- Understanding document structure
- Filtering unwanted elements
- Multi-stage processing

## Available Models

### Model Comparison

| Model | Speed | Accuracy | Languages | GPU Req | Best For |
|-------|-------|----------|-----------|---------|----------|
| **Tesseract** | ⭐⭐⭐⭐⭐ (Fast) | ⭐⭐⭐ (Good) | 100+ | None | Legacy, CPU-only |
| **EasyOCR** | ⭐⭐⭐ (Medium) | ⭐⭐⭐⭐ (Very Good) | 80+ | Optional | Production use |
| **PaddleOCR** | ⭐⭐⭐⭐ (Very Fast) | ⭐⭐⭐⭐ (Very Good) | 11 | Optional | Speed-critical, Asian text |
| **CRAFT** | ⭐⭐⭐ (Medium) | ⭐⭐⭐⭐ (Very Good) | English | Optional | Scene text detection |

### 1. Tesseract (CPU-only)

Traditional OCR engine, excellent for clean printed text.

**Strengths:**
- No GPU required, CPU-only
- Extremely fast
- Supports 100+ languages
- Proven and reliable
- Opensource (Apache 2.0)

**Weaknesses:**
- Lower accuracy on complex layouts
- Struggles with handwriting
- Needs training data for custom fonts

**When to use:**
- CPU-only systems (Raspberry Pi, servers)
- Clean printed documents
- Cost-sensitive applications
- Multi-language documents

**Languages:** 100+ (English, Chinese, Arabic, Hindi, etc.)

### 2. EasyOCR (GPU-recommended)

Deep learning OCR with excellent accuracy.

**Strengths:**
- Very high accuracy on diverse text
- Supports 80+ languages
- Works with or without GPU
- Easy API
- Good on real-world documents

**Weaknesses:**
- Slower than PaddleOCR
- Higher memory usage
- Requires downloading large models

**When to use:**
- High accuracy needed
- Mixed language documents
- Production systems
- Irregular text layouts

**Languages:** English, Chinese, Japanese, Korean, Arabic, Hindi, etc. (80+ total)

### 3. PaddleOCR (Fastest with GPU)

Lightweight OCR optimized for speed.

**Strengths:**
- Fastest inference speed
- Small model size
- Excellent Asian language support
- Works on CPU and GPU
- Very efficient

**Weaknesses:**
- Fewer languages than EasyOCR
- Slightly lower accuracy on English
- Limited handwriting support

**When to use:**
- Performance-critical applications
- Asian language documents
- Resource-constrained environments
- High-throughput pipelines

**Languages:** English, Chinese, Japanese, Korean, Arabic (main languages)

## Basic Usage

### Example 1: Simple Word-Level OCR

Extract text with word-level bounding boxes.

```python
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from PIL import Image

image = Image.open("document_page.png")

# Initialize EasyOCR for high accuracy
config = EasyOCRConfig(
    languages=["en"],  # English only (faster)
    gpu=True,  # Use GPU if available
)
ocr = EasyOCR(config=config)

# Extract text with bounding boxes
result = ocr.extract(image)

print(f"Extracted {len(result.text_blocks)} text blocks")

# Access text and locations
for block in result.text_blocks:
    print(f"Text: '{block.text}'")
    print(f"Bbox: {block.bbox}")
    print(f"Confidence: {block.confidence:.2f}")
    print()
```

**Output Example:**
```
Extracted 5 text blocks
Text: 'Document'
Bbox: BoundingBox(x1=10, y1=5, x2=120, y2=30)
Confidence: 0.98

Text: 'Title'
Bbox: BoundingBox(x1=10, y1=35, x2=100, y2=55)
Confidence: 0.97

...
```

### Example 2: Fast CPU-Only OCR (Tesseract)

Use Tesseract for fast CPU-only extraction.

```python
from omnidocs.tasks.ocr_extraction import Tesseract, TesseractConfig
from PIL import Image

image = Image.open("document_page.png")

# Initialize Tesseract (CPU only, no GPU)
config = TesseractConfig(
    language="eng",  # Single language for speed
    config="--psm 3",  # Page segmentation mode
)
ocr = Tesseract(config=config)

# Extract
result = ocr.extract(image)

print(f"Found {len(result.text_blocks)} words")

# Display results with confidence
high_confidence = [b for b in result.text_blocks if b.confidence > 0.9]
print(f"High confidence blocks: {len(high_confidence)}")

# Get plain text
print("\nExtracted text:")
print(" ".join(block.text for block in result.text_blocks))
```

### Example 3: Multi-Language OCR

Extract from documents with multiple languages.

```python
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from PIL import Image

image = Image.open("multilingual_document.png")

# Support multiple languages
config = EasyOCRConfig(
    languages=["en", "zh", "ar"],  # English, Chinese, Arabic
    gpu=True,
)
ocr = EasyOCR(config=config)

result = ocr.extract(image)

# Group by detected language (if available)
for block in result.text_blocks:
    print(f"[{block.language}] {block.text}")
```

### Example 4: PDF with Character-Level Extraction

Extract at character granularity from PDF.

```python
from omnidocs import Document
from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig

# Load PDF
doc = Document.from_pdf("document.pdf")

# Initialize PaddleOCR for character-level extraction
config = PaddleOCRConfig(
    languages=["en", "ch"],  # English and Chinese
    gpu=True,
)
ocr = PaddleOCR(config=config)

# Process first page
page_image = doc.get_page(0)
result = ocr.extract(page_image, granularity="character")

# Access character-level data
char_count = len(result.text_blocks)
print(f"Extracted {char_count} characters")

# Find coordinates of specific character
for block in result.text_blocks:
    if block.text == "A":
        print(f"Found 'A' at {block.bbox}")
        break
```

## Extracting Bounding Boxes

### Get Text Blocks with Coordinates

```python
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from PIL import Image

image = Image.open("document.png")
config = EasyOCRConfig(languages=["en"], gpu=True)
ocr = EasyOCR(config=config)

result = ocr.extract(image)

# Print detailed block information
for block in result.text_blocks:
    x1, y1, x2, y2 = block.bbox.x1, block.bbox.y1, block.bbox.x2, block.bbox.y2
    width = x2 - x1
    height = y2 - y1

    print(f"'{block.text}' @ ({x1:.0f}, {y1:.0f}) "
          f"size: {width:.0f}x{height:.0f} "
          f"conf: {block.confidence:.2f}")
```

### Convert to Normalized Coordinates

Convert pixel coordinates to 0-1024 normalized range.

```python
# Normalize bounding boxes to 0-1024 range
image_width, image_height = image.size
normalized_blocks = result.get_normalized_blocks()

for block in normalized_blocks:
    # Coordinates now in 0-1024 range
    print(f"'{block.text}' @ {block.bbox} (normalized)")

# Manual normalization
NORM_SIZE = 1024

def normalize_bbox(bbox, image_size):
    """Convert pixel bbox to normalized 0-1024."""
    img_w, img_h = image_size
    x1 = int(bbox.x1 * NORM_SIZE / img_w)
    y1 = int(bbox.y1 * NORM_SIZE / img_h)
    x2 = int(bbox.x2 * NORM_SIZE / img_w)
    y2 = int(bbox.y2 * NORM_SIZE / img_h)
    return (x1, y1, x2, y2)

for block in result.text_blocks:
    norm_bbox = normalize_bbox(block.bbox, (image_width, image_height))
    print(f"Normalized: {norm_bbox}")
```

### Extract from Specific Regions

Get OCR results from a cropped region.

```python
# Crop image to specific region
region_bbox = (100, 100, 500, 400)  # x1, y1, x2, y2
cropped = image.crop(region_bbox)

# Run OCR on crop
result_crop = ocr.extract(cropped)

# Adjust bboxes back to original image coordinates
x1_offset, y1_offset = region_bbox[0], region_bbox[1]

for block in result_crop.text_blocks:
    # Shift coordinates
    adjusted_bbox = (
        block.bbox.x1 + x1_offset,
        block.bbox.y1 + y1_offset,
        block.bbox.x2 + x1_offset,
        block.bbox.y2 + y1_offset,
    )
    print(f"'{block.text}' @ {adjusted_bbox}")
```

## Filtering Results

### Filter by Confidence

Keep only high-confidence extractions.

```python
# Filter by confidence threshold
min_confidence = 0.85
confident_blocks = [
    b for b in result.text_blocks
    if b.confidence >= min_confidence
]

print(f"Original: {len(result.text_blocks)} blocks")
print(f"Filtered (conf >= {min_confidence}): {len(confident_blocks)} blocks")

# Display confidence distribution
confidences = [b.confidence for b in result.text_blocks]
print(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")
```

### Filter by Region

Extract OCR results from specific image regions.

```python
def is_in_region(bbox, region):
    """Check if bbox overlaps with region."""
    rx1, ry1, rx2, ry2 = region
    return not (bbox.x2 < rx1 or bbox.x1 > rx2 or
                bbox.y2 < ry1 or bbox.y1 > ry2)

# Top-left region
top_left = (0, 0, image.width//2, image.height//2)
top_left_blocks = [b for b in result.text_blocks if is_in_region(b.bbox, top_left)]

# Sidebar region
sidebar = (0, 0, 200, image.height)
sidebar_blocks = [b for b in result.text_blocks if is_in_region(b.bbox, sidebar)]

print(f"Top-left blocks: {len(top_left_blocks)}")
print(f"Sidebar blocks: {len(sidebar_blocks)}")
```

### Filter by Text Content

Find blocks matching patterns.

```python
import re

# Find numbers
number_blocks = [
    b for b in result.text_blocks
    if re.match(r'^\d+$', b.text.strip())
]

# Find email addresses
email_blocks = [
    b for b in result.text_blocks
    if re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', b.text.strip())
]

# Find specific phrases
phrase_blocks = [
    b for b in result.text_blocks
    if "important" in b.text.lower()
]

print(f"Numbers: {len(number_blocks)}")
print(f"Emails: {len(email_blocks)}")
print(f"'Important' mentions: {len(phrase_blocks)}")
```

### Filter by Size

Exclude very small or very large blocks.

```python
# Calculate block dimensions
def get_size(bbox):
    return (bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)

# Keep medium-sized blocks
medium_blocks = []
for b in result.text_blocks:
    width, height = get_size(b.bbox)
    if 30 < width < 500 and 10 < height < 100:
        medium_blocks.append(b)

print(f"Medium-sized blocks: {len(medium_blocks)}/{len(result.text_blocks)}")

# Analyze size distribution
sizes = [get_size(b.bbox) for b in result.text_blocks]
avg_width = sum(w for w, h in sizes) / len(sizes)
avg_height = sum(h for w, h in sizes) / len(sizes)
print(f"Average block size: {avg_width:.0f}x{avg_height:.0f}")
```

## Multi-Language Support

### Auto-Detect Language

EasyOCR can auto-detect language.

```python
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

# Auto-detect (leave empty or None)
config = EasyOCRConfig(
    languages=None,  # Auto-detect all languages
    gpu=True,
)
ocr = EasyOCR(config=config)

result = ocr.extract(image)

# Check detected languages
detected_langs = set()
for block in result.text_blocks:
    if hasattr(block, 'language'):
        detected_langs.add(block.language)

print(f"Detected languages: {detected_langs}")
```

### Process Mixed-Language Documents

Handle documents with multiple languages.

```python
# Support common languages
config = EasyOCRConfig(
    languages=["en", "zh", "ar", "hi", "ja"],  # English, Chinese, Arabic, Hindi, Japanese
    gpu=True,
)
ocr = EasyOCR(config=config)

result = ocr.extract(image)

# Group results by language
from collections import defaultdict
by_language = defaultdict(list)

for block in result.text_blocks:
    lang = getattr(block, 'language', 'unknown')
    by_language[lang].append(block)

for lang, blocks in by_language.items():
    print(f"\n{lang.upper()} ({len(blocks)} blocks):")
    for block in blocks[:3]:  # Show first 3
        print(f"  {block.text}")
```

### Language-Specific Optimization

Different languages need different models.

```python
# For English only (fastest)
config_en = EasyOCRConfig(languages=["en"], gpu=True)

# For Asian languages (use PaddleOCR)
from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig
config_cn = PaddleOCRConfig(languages=["ch"], gpu=True)  # Chinese

# For Arabic/Hebrew (right-to-left)
config_rtl = EasyOCRConfig(languages=["ar"], gpu=True)

# For handwriting
config_hw = EasyOCRConfig(languages=["en"], gpu=True)
# Note: Most OCR models struggle with handwriting
```

## Performance Comparison

### Speed Benchmarks

Processing a typical page (300 DPI, ~2000x3000px):

| Model | CPU | GPU | Latency | Memory |
|-------|-----|-----|---------|--------|
| Tesseract | 0.5-1.0s | N/A | Very Fast | ~100MB |
| PaddleOCR | 1-2s | 0.3-0.5s | Fast | ~500MB |
| EasyOCR | 2-4s | 0.5-1.0s | Medium | ~1GB |

### Choose by Speed Requirements

| Requirement | Model |
|-------------|-------|
| <200ms per page | Tesseract or PaddleOCR (GPU) |
| <500ms per page | PaddleOCR (GPU) or Tesseract |
| <1s per page | EasyOCR (GPU) or PaddleOCR (CPU) |
| 1-2s acceptable | EasyOCR (GPU) |
| Accuracy paramount | EasyOCR |

### Optimization for Speed

```python
import time

# Fast configuration
config_fast = PaddleOCRConfig(
    languages=["en"],  # Single language
    gpu=True,
)
ocr = PaddleOCR(config=config_fast)

# Benchmark
images = [Image.open(f"doc{i}.png") for i in range(5)]

start = time.time()
for img in images:
    result = ocr.extract(img)
elapsed = time.time() - start

print(f"Processed {len(images)} images in {elapsed:.1f}s")
print(f"Average: {elapsed/len(images):.2f}s per image")
```

## Troubleshooting

### Low Accuracy

**Problem:** OCR results have many errors.

**Solutions:**
1. Try different model (EasyOCR typically better)
2. Improve image quality
3. Try single language (faster + more accurate)

```python
# Solution 1: Use EasyOCR (more accurate)
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
config = EasyOCRConfig(languages=["en"], gpu=True)

# Solution 2: Improve image quality
from PIL import Image, ImageEnhance

img = Image.open("noisy_scan.png")

# Increase contrast
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.5)  # 50% more contrast

# Increase sharpness
enhancer = ImageEnhance.Sharpness(img)
img = enhancer.enhance(2.0)  # 2x sharpness

# Resize if too small
if img.width < 1024:
    img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)

result = ocr.extract(img)

# Solution 3: Single language
config = EasyOCRConfig(languages=["en"], gpu=True)  # Just English
```

### Missing Text

**Problem:** Some text not detected.

**Solutions:**
1. Check image quality
2. Try lower confidence threshold
3. Use different OCR model

```python
# Solution 1: Check image
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

# Solution 2: Lower confidence
all_blocks = result.text_blocks  # Includes low confidence

confidence_dist = [b.confidence for b in result.text_blocks]
print(f"Confidence range: {min(confidence_dist):.2f}-{max(confidence_dist):.2f}")

# Get even low-confidence blocks
low_conf_blocks = [b for b in result.text_blocks if b.confidence < 0.5]
print(f"Low confidence blocks: {len(low_conf_blocks)}")
```

### False Detections

**Problem:** Non-text detected as text.

**Solutions:**
1. Increase confidence threshold
2. Filter by text length
3. Manual post-processing

```python
# Solution 1: Increase confidence
high_conf = [b for b in result.text_blocks if b.confidence > 0.95]

# Solution 2: Filter short blocks (likely noise)
MIN_CHARS = 2
valid_blocks = [b for b in result.text_blocks if len(b.text) >= MIN_CHARS]

# Solution 3: Remove non-alphabetic text
import string
alpha_blocks = [
    b for b in result.text_blocks
    if any(c.isalpha() for c in b.text)
]
```

### Slow Performance

**Problem:** OCR taking too long.

**Solutions:**
1. Use faster model (PaddleOCR or Tesseract)
2. Reduce image resolution
3. Use GPU
4. Single language

```python
# Solution 1: Use PaddleOCR (faster)
from omnidocs.tasks.ocr_extraction import PaddleOCR
ocr = PaddleOCR(gpu=True)  # Fastest on GPU

# Solution 2: Reduce resolution
image = image.resize((image.width // 2, image.height // 2))

# Solution 3: Ensure GPU enabled
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Solution 4: Single language
config = PaddleOCRConfig(languages=["en"], gpu=True)
```

---

**Next Steps:**
- See [Text Extraction Guide](text-extraction.md) for formatted document output
- See [Layout Analysis Guide](layout-analysis.md) for document structure
- See [Batch Processing Guide](batch-processing.md) for processing many documents
