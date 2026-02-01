# OmniDocs Model Documentation

Complete reference documentation for all models available in OmniDocs, including configuration, usage, and performance characteristics.

## Quick Navigation

### Text Extraction
Models for extracting document text content with optional formatting.

- **[Qwen3-VL Text Extraction](./text-extraction/qwen.md)** - High-quality VLM with multi-backend support
  - Best for: Quality, multilingual documents, multiple backends
  - Variants: 2B, 4B, 8B, 32B
  - Backends: PyTorch, VLLM, MLX, API
  - Speed: 50-150 tok/s (PyTorch), 200-400 tok/s (VLLM)

- **[DotsOCR Text Extraction](./text-extraction/dotsocr.md)** - Layout-aware text extraction
  - Best for: Layout analysis with text, fast inference
  - Layout Categories: 11 fixed categories with bounding boxes
  - Backends: PyTorch, VLLM
  - Speed: 80-120 tok/s (PyTorch), 200-300 tok/s (VLLM)

### Layout Analysis
Models for detecting document structure and element regions.

- **[DocLayout-YOLO Layout Detection](./layout-analysis/doclayout-yolo.md)** - Fast YOLO-based detection
  - Best for: Speed, batch processing, fixed categories
  - Categories: 10 fixed layout types
  - Backend: PyTorch only
  - Speed: 0.1-0.2s per page (very fast)

- **[Qwen3-VL Layout Detection](./layout-analysis/qwen-layout.md)** - Flexible VLM-based detection
  - Best for: Custom categories, semantic understanding
  - Categories: Unlimited (define custom labels)
  - Backends: PyTorch, VLLM, MLX, API
  - Speed: 2-5s per page (PyTorch), 0.5-1.5s (VLLM)

### OCR Extraction
Models for character-level text recognition with bounding boxes.

- **[Tesseract OCR](./ocr-extraction/tesseract.md)** - Open-source OCR engine
  - Best for: Printed text, CPU-only deployment
  - Languages: 100+
  - Speed: 2-5s per page (CPU)
  - Accuracy: 95-99% for printed English

---

## Model Comparison

**Quick Comparison Table** - See full details in [Comparison Guide](./comparison.md)

| Model | Task | Speed | Quality | Backends | Memory |
|-------|------|-------|---------|----------|--------|
| **Qwen3-VL** | Text | Medium | Excellent | 4 | 4-40 GB |
| **DotsOCR** | Text + Layout | Fast | Very Good | 2 | 16 GB |
| **DocLayout-YOLO** | Layout | Very Fast | Good | 1 | 2-4 GB |
| **Qwen Layout** | Layout (Custom) | Medium | Excellent | 4 | 8-16 GB |
| **Tesseract** | OCR | Slow (CPU) | Excellent | CPU | Minimal |

---

## Choosing the Right Model

### By Task

**Text Extraction**
- High quality: **Qwen3-VL** (any variant)
- Fast + Layout: **DotsOCR**
- Multilingual: **Qwen3-VL**
- Apple Silicon: **Qwen3-VL** (MLX backend)

**Layout Detection**
- Speed: **DocLayout-YOLO** (0.1s/page)
- Accuracy: **RT-DETR** or **Qwen Layout**
- Custom categories: **Qwen Layout**
- Production pipeline: **DocLayout-YOLO**

**OCR (Character Recognition)**
- Printed text: **Tesseract** (free, CPU)
- Handwriting: **Surya**
- Speed: **PaddleOCR**
- Accuracy: **Surya**

### By Constraint

**Limited GPU Memory** (4 GB)
- DocLayout-YOLO (2-4 GB)
- Qwen3-VL-2B (4 GB)
- Tesseract (CPU)

**No GPU Available**
- Tesseract (CPU OCR)
- Qwen3-VL (API backend)

**Need Fast Processing** (sub-second)
- DocLayout-YOLO (0.1-0.2s)
- PaddleOCR (0.3-1s)

**Multilingual Documents**
- Qwen3-VL (25+ languages)
- Tesseract (100+ languages)
- PaddleOCR (80+ languages)

**Custom Layout Categories**
- Qwen Layout (unlimited custom labels)

---

## Recommended Pipelines

### Academic Paper Processing
```
1. Layout: DocLayoutYOLO (0.1-0.2s)
2. Text: Qwen3-VL-8B (2-4s)
Result: Structure + high-quality content
```

### Batch Document Processing (1000s)
```
1. Layout: DocLayoutYOLO (for structure)
2. Text: DotsOCR + VLLM (2-4 GPUs)
Result: 5-10k docs/hour
```

### Handwritten Document OCR
```
1. OCR: Surya (handwriting)
2. Layout: Qwen Layout (if needed)
Result: 85%+ handwriting accuracy
```

### Form Field Extraction
```
1. Layout: Qwen Layout (custom field labels)
2. OCR: Tesseract or EasyOCR per field
Result: Automated form parsing
```

### Cloud Deployment (No GPU)
```
1. Text: Qwen3-VL (API backend)
2. OCR: Tesseract (CPU)
Result: Serverless processing via API
```

---

## Performance Summary

### Speed Rankings (Fastest to Slowest)

**Text Extraction**
1. Nanonuts (200+ tok/s)
2. DotsOCR (120 tok/s)
3. Qwen3-VL-2B (150 tok/s)
4. Qwen3-VL-8B (100 tok/s)
5. Qwen3-VL-32B (40 tok/s)

**Layout Detection**
1. DocLayout-YOLO (0.1-0.2s/page)
2. RT-DETR (0.3-0.5s/page)
3. Qwen Layout (2-5s/page)

**OCR**
1. PaddleOCR (0.3-1s/page)
2. EasyOCR (1-2s/page)
3. Surya (1-3s/page)
4. Tesseract (2-5s/page, CPU)

### Memory Rankings (Smallest to Largest)

1. Tesseract (CPU, minimal)
2. DocLayout-YOLO (2-4 GB)
3. PaddleOCR (2-4 GB)
4. Qwen3-VL-2B (4 GB)
5. Qwen3-VL-8B (16 GB)
6. Qwen3-VL-32B (40 GB)

### Quality Rankings (Best to Good)

**Text**: Qwen3-VL-32B > Qwen3-VL-8B > DotsOCR > Nanonuts
**Layout**: Qwen Layout = RT-DETR > DocLayout-YOLO
**Handwriting**: Surya > PaddleOCR > EasyOCR > Tesseract
**Multilingual**: Qwen3-VL > Tesseract > PaddleOCR

---

## Installation & Setup

### Quick Start

```bash
# Text extraction with PyTorch (most common)
pip install omnidocs[pytorch]

# Layout detection (included with pytorch)
pip install omnidocs[pytorch]

# OCR with Tesseract
pip install omnidocs[pytorch]
brew install tesseract  # macOS, or apt-get on Linux

# All backends
pip install omnidocs[all]
```

### Verify Installation

```python
# Check available models
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.layout_extraction import DocLayoutYOLO
from omnidocs.tasks.ocr_extraction import TesseractOCR

print("✓ OmniDocs models available")
```

---

## FAQ

**Q: Which model should I use?**
A: See [Comparison Guide](./comparison.md) for detailed decision matrix

**Q: How do I handle documents in multiple languages?**
A: Use Qwen3-VL (25+ languages) or Tesseract (100+ languages)

**Q: Can I run without GPU?**
A: Yes - Tesseract (OCR) and API backends (text) work on CPU

**Q: What's the fastest option?**
A: DocLayout-YOLO (0.1s) for layout, PaddleOCR (0.3s) for OCR

**Q: How much GPU memory do I need?**
A: 2-4 GB for layout only, 8-16 GB for text extraction, 16+ GB for batch processing

**Q: Can I use custom layout categories?**
A: Yes - with Qwen Layout (define unlimited custom labels)

**Q: Which model is best for production?**
A: DocLayout-YOLO + Qwen3-VL-8B on 2x A10 GPU

---

## Documentation Structure

Each model documentation includes:
- **Model Overview** - Architecture, capabilities, limitations
- **Installation & Configuration** - Setup and config parameters
- **Usage Examples** - Copy-paste ready code for common tasks
- **Performance** - Speed, memory, accuracy characteristics
- **Troubleshooting** - Solutions for common issues
- **API Reference** - Complete function and parameter documentation

---

## Related Documentation

- [OmniDocs Architecture](../architecture.md)
- [Developer Guide](../developer-guide.md)
- [API Reference](../api)

---

## Version Information

Documentation updated: January 2026
Models supported:
- Qwen3-VL (all variants)
- DotsOCR (latest)
- DocLayout-YOLO-DocStructBench
- RT-DETR (via references)
- Tesseract (5.x+)

---

## Quick Links

- **[Text Extraction Qwen](./text-extraction/qwen.md)** - Start here for high-quality text
- **[Layout Detection DocLayout-YOLO](./layout-analysis/doclayout-yolo.md)** - Start here for speed
- **[Model Comparison](./comparison.md)** - Compare all models
- **[Tesseract OCR](./ocr-extraction/tesseract.md)** - Open-source OCR option
