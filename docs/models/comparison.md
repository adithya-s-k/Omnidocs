# Model Comparison & Selection Guide

A comprehensive comparison of all models available in OmniDocs to help you choose the right tool for your use case.

---

## Text Extraction Models

Models for extracting text content with optional formatting (Markdown/HTML/JSON).

### Feature Comparison

| Feature | Qwen3-VL | DotsOCR | Nanonuts |
|---------|----------|---------|----------|
| **Model Size** | 2B-32B | ~7B | ~7B |
| **Text Quality** | Excellent | Very Good | Very Good |
| **Layout Info** | Basic | Detailed (11 cats) | Not included |
| **Speed** | Medium | Fast | Fast |
| **Memory** | 4-40 GB | 16 GB | 12 GB |
| **Multilingual** | Yes (25+) | Limited | English-focused |
| **Backends** | PyTorch, VLLM, MLX, API | PyTorch, VLLM | PyTorch, VLLM |
| **Output Formats** | Markdown, HTML | Markdown (with JSON layout) | Markdown |
| **License** | Apache 2.0 | Open | Apache 2.0 |

### Decision Matrix: Text Extraction

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| **High-quality multilingual docs** | Qwen3-VL | Best text quality, many languages |
| **Need layout + text** | DotsOCR | Detailed layout categories with text |
| **Fast, English docs** | Nanonuts | Fastest, good quality for English |
| **Batch processing** | DotsOCR + VLLM | Good speed with detailed output |
| **Cloud/API deployment** | Qwen3-VL (API) | Only option with API backend |
| **Apple Silicon only** | Qwen3-VL (MLX) | Only VLM with MLX support |

### Performance Comparison

| Model | Speed (tok/s) | Quality | Cost |
|-------|---------------|---------|------|
| Qwen3-VL-2B | 100-150 | Good | Low (small) |
| Qwen3-VL-8B | 50-100 | Excellent | Medium |
| Qwen3-VL-32B | 20-40 | Outstanding | High |
| DotsOCR | 80-120 | Very Good | Medium |
| Nanonuts | 150-200 | Good | Medium |

---

## Layout Analysis Models

Models for detecting document structure and element regions.

### Feature Comparison

| Feature | DocLayout-YOLO | RT-DETR | Qwen Layout |
|---------|----------------|---------|------------|
| **Architecture** | YOLOv10 | DETR | Vision-Language |
| **Speed** | Very Fast | Fast | Medium |
| **Categories** | 10 (fixed) | 12+ (fixed) | Unlimited (custom) |
| **Accuracy** | Good | Excellent | Excellent |
| **Memory** | 2-4 GB | 4-8 GB | 8-16 GB |
| **Backends** | PyTorch | PyTorch | PyTorch, VLLM, MLX, API |
| **Custom Labels** | No | No | Yes |
| **GPU Required** | Yes | Yes | Yes (practical) |
| **Best For** | Speed | Accuracy | Flexibility |

### Fixed Categories Comparison

**DocLayout-YOLO (10)**:
Title, Plain text, Figure, Figure caption, Table, Table caption, Table footnote, Formula, Formula caption, Abandon

**RT-DETR (12+)**:
Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title, (+ Extended: Document Index, Code, Checkboxes, Forms)

**Qwen Layout**:
Standard labels (10) + unlimited custom labels per use case

### Decision Matrix: Layout Analysis

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| **Batch processing, speed critical** | DocLayout-YOLO | Fastest (0.1-0.2s/page) |
| **Academic papers, high precision** | RT-DETR | Excellent accuracy on papers |
| **Custom layout categories needed** | Qwen Layout | Only option for custom labels |
| **Web page layout** | Qwen Layout | Better understanding of semantic regions |
| **Form field detection** | Qwen Layout | Can detect custom field types |
| **Production pipeline** | DocLayout-YOLO | Proven, fast, stable |

### Speed Comparison

| Model | Per-Page Speed | Device |
|-------|----------------|--------|
| DocLayout-YOLO | 0.1-0.2s | Single A10 GPU |
| RT-DETR | 0.3-0.5s | Single A10 GPU |
| Qwen Layout | 2-5s | Single A10 GPU |
| Qwen Layout (VLLM) | 0.5-1.5s | 2x A10 GPU |

---

## OCR Models

Models for extracting text with character/word-level bounding boxes.

### Feature Comparison

| Feature | Tesseract | EasyOCR | PaddleOCR | Surya |
|---------|-----------|---------|-----------|-------|
| **Type** | Traditional | Deep Learning | Deep Learning | Deep Learning |
| **Speed** | Slow (CPU) | Medium (GPU) | Very Fast | Medium |
| **Languages** | 100+ | 80+ | 80+ | Multi |
| **Handwriting** | Poor | Medium | Medium | Excellent |
| **GPU Required** | No | Yes | Yes | Yes |
| **Memory** | CPU | 4-6 GB | 2-4 GB | 6-8 GB |
| **Setup** | System install | Python | Python | Python |
| **Accuracy** | High (printed) | Good | Excellent (CJK) | Best overall |

### Character Detection Accuracy

| Model | Latin | Asian | Handwriting |
|-------|-------|-------|------------|
| Tesseract | 95-99% | 70-80% | 30-50% |
| EasyOCR | 90-96% | 85-92% | 60-70% |
| PaddleOCR | 92-97% | 94-99% | 70-80% |
| Surya | 94-98% | 88-95% | 85-90% |

### Decision Matrix: OCR

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| **Printed English docs** | Tesseract | Fastest (CPU), excellent accuracy |
| **Mixed scripts/languages** | PaddleOCR | Best for Asian languages |
| **Handwritten documents** | Surya | Best handwriting support |
| **Cloud deployment** | EasyOCR or PaddleOCR | Easier setup than Tesseract |
| **No GPU available** | Tesseract | Only CPU option |
| **Real-time processing** | PaddleOCR | Fastest GPU inference |

### Performance Comparison

| Model | Speed | Accuracy | Cost |
|-------|-------|----------|------|
| Tesseract | 2-5s (CPU) | 95-99% (printed) | Free |
| EasyOCR | 1-2s (GPU) | 90-96% | Free |
| PaddleOCR | 0.3-1s (GPU) | 92-99% | Free |
| Surya | 1-3s (GPU) | 94-98% | Free |

---

## Task-Specific Recommendations

### Use Case: Academic Paper Processing

**Goal**: Extract text and layout from research papers

**Recommended Pipeline**:
1. **Layout**: DocLayout-YOLO (fast, accurate for papers)
2. **Text**: Qwen3-VL-8B (high quality, multilingual)
3. **Optional**: DotsOCR if detailed layout needed

**Configuration**:
```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Fast layout detection
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

# High-quality text extraction
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
)

# Process
layout_result = layout.extract(image)
text_result = extractor.extract(image)
```

**Estimated Performance**: 2-3 seconds per page, high accuracy

### Use Case: Document Batch Processing

**Goal**: Extract text from 1000s of documents quickly

**Recommended Pipeline**:
1. **Layout**: DocLayout-YOLO (for batching)
2. **Text**: DotsOCR with VLLM (good quality, fast)

**Configuration**:
```python
# VLLM for batching
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

extractor = DotsOCRTextExtractor(
    backend=DotsOCRVLLMConfig(
        tensor_parallel_size=2,  # 2 GPUs
        gpu_memory_utilization=0.85,
    )
)

# Process 100+ documents per hour
```

**Estimated Performance**: 5-10k documents/hour on 2x A10 GPU

### Use Case: Handwritten Document OCR

**Goal**: Extract text from handwritten documents

**Recommended Pipeline**:
1. **OCR**: Surya (best for handwriting)
2. **Layout**: Qwen Layout with custom labels (if needed)

**Configuration**:
```python
from omnidocs.tasks.ocr_extraction import SuryaOCR, SuryaOCRConfig

ocr = SuryaOCR(config=SuryaOCRConfig(
    languages=["en"],
    det_model="en",
))

result = ocr.extract(image)
```

**Estimated Performance**: 85%+ handwriting accuracy

### Use Case: Form Field Extraction

**Goal**: Extract text from form documents with custom field types

**Recommended Pipeline**:
1. **Layout**: Qwen Layout with custom labels for field types
2. **OCR**: Tesseract or EasyOCR per field

**Configuration**:
```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector, CustomLabel
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

custom_labels = [
    CustomLabel(name="text_field"),
    CustomLabel(name="checkbox"),
    CustomLabel(name="signature_line"),
]

detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(device="cuda")
)

result = detector.extract(image, custom_labels=custom_labels)
```

**Estimated Performance**: Form processing in 5-10 seconds

### Use Case: Multilingual Document Processing

**Goal**: Process documents in 20+ languages

**Recommended Pipeline**:
1. **Text**: Qwen3-VL-8B (25+ languages)
2. **Fallback**: PaddleOCR (80+ languages) for Asian scripts

**Configuration**:
```python
# Qwen handles most languages
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct")
)
```

**Supported**: English, French, German, Spanish, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Portuguese, Dutch, Polish, Turkish, Greek, Thai, Vietnamese, and more.

### Use Case: Real-Time Document Processing

**Goal**: Process documents with <1 second latency

**Recommended Pipeline**:
1. **Layout**: DocLayout-YOLO (0.1-0.2s)
2. **Text**: Fast OCR or small VLM

**Configuration**:
```python
# Fastest layout detection
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(
    device="cuda",
    img_size=768,  # Smaller for speed
))

result = extractor.extract(image)  # <200ms
```

**Estimated Performance**: 0.2-1 second per page with layout only

### Use Case: Cloud Deployment (No GPU)

**Goal**: Deploy document processing in serverless environment

**Recommended Pipeline**:
1. **Layout**: Not practical (needs GPU)
2. **Text**: Use API backend or Tesseract (CPU)
3. **OCR**: Tesseract (CPU-friendly)

**Configuration**:
```python
# API-based for cloud
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

extractor = QwenTextExtractor(
    backend=QwenTextAPIConfig(
        model="qwen3-vl-8b",
        api_key=os.getenv("QWEN_API_KEY"),
    )
)

# CPU-based OCR
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig

ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
```

**Estimated Cost**: $0.01-0.10 per document via API

---

## Performance Summary Table

### Text Extraction Speed (larger batch)

| Model | 1 GPU | 2 GPUs (VLLM) | Tokens/Sec |
|-------|-------|---------------|-----------|
| Qwen3-VL-2B | 100-150 | 250-350 | Per-doc |
| Qwen3-VL-8B | 50-100 | 150-250 | Per-doc |
| DotsOCR | 80-120 | 200-300 | Per-doc |
| Nanonuts | 150-200 | 400-500 | Per-doc |

### Layout Detection Speed

| Model | Speed | Device |
|-------|-------|--------|
| DocLayout-YOLO | 0.1-0.2s | A10 GPU |
| RT-DETR | 0.3-0.5s | A10 GPU |
| Qwen Layout (PyTorch) | 2-5s | A10 GPU |
| Qwen Layout (VLLM) | 0.5-1.5s | 2x A10 GPU |

### OCR Speed

| Model | Speed (1024x768) | Device |
|-------|------------------|--------|
| Tesseract | 2-3s | CPU |
| EasyOCR | 1-2s | GPU |
| PaddleOCR | 0.3-1s | GPU |
| Surya | 1-3s | GPU |

---

## Memory Requirements Summary

### VRAM Requirements

| Task | Minimal | Recommended | Optimal |
|------|---------|-------------|---------|
| **Text (Qwen-8B)** | 8 GB | 16 GB | 24 GB |
| **Layout (DocLayout)** | 2 GB | 4 GB | 8 GB |
| **Layout (Qwen)** | 8 GB | 16 GB | 24 GB |
| **OCR (GPU-based)** | 2 GB | 4 GB | 8 GB |
| **Multi-task pipeline** | 16 GB | 32 GB | 40 GB |

### CPU Requirements

| Model | CPU Load | Parallelization |
|-------|----------|-----------------|
| Tesseract | Medium | Thread-based (4+ cores) |
| EasyOCR | Light | Not parallelizable |
| DotsOCR | Light | GPU-bound |

---

## Cost Analysis

### Deployment Costs (per million documents)

| Strategy | GPU Cost | Model Cost | Total |
|----------|----------|-----------|-------|
| **Self-hosted (PyTorch)** | $500/month | Free | $6k/year |
| **Self-hosted (VLLM batch)** | $1000/month | Free | $12k/year |
| **API-based** | None | $1-2/doc | $1-2M |
| **Hybrid (API + cached)** | Minimal | $0.1-0.5/doc | $100k-500k |

### Development Time

| Task | Effort | Models Needed |
|------|--------|---------------|
| **Simple extraction** | 1 hour | 1 (any VLM) |
| **Layout + text** | 2-4 hours | 2 (layout + text) |
| **Custom layout** | 4-8 hours | Qwen layout + fine-tuning |
| **Production pipeline** | 1-2 weeks | 3+ with batching, caching |

---

## Frequently Asked Questions

### Q: Which model is fastest?
**A**: DocLayout-YOLO for layout (0.1-0.2s), PaddleOCR for OCR (0.3-1s), Nanonuts for text (50-80 tok/s)

### Q: Which is most accurate?
**A**: Qwen3-VL-32B for text, Surya for handwriting, RT-DETR for layout

### Q: Which requires least GPU?
**A**: DocLayout-YOLO (2-4 GB), Tesseract (CPU-only)

### Q: Which supports most languages?
**A**: Tesseract (100+), Qwen (25+), PaddleOCR (80+)

### Q: Which is cheapest to run?
**A**: Tesseract (free, CPU), DocLayout-YOLO (small GPU model)

### Q: Best for real-time (sub-second)?
**A**: DocLayout-YOLO for layout only, or PaddleOCR for OCR

### Q: Best for batch processing?
**A**: DotsOCR or Qwen with VLLM (2-4 GPUs)

### Q: Can I run without GPU?
**A**: Yes - Tesseract (OCR) and API backends (text)

### Q: Which is easiest to set up?
**A**: Qwen with PyTorch (single pip install)

### Q: Production recommendation?
**A**: DocLayout-YOLO + Qwen3-VL-8B on 2x A10 GPU

---

## Migration Guide

### From Tesseract to Modern OCR

```python
# Old: Tesseract only
from omnidocs.tasks.ocr_extraction import TesseractOCR

# New: Choose based on use case
from omnidocs.tasks.ocr_extraction import (
    TesseractOCR,  # Printed, many languages
    PaddleOCR,     # Speed, Asian languages
    SuryaOCR,      # Handwriting
)
```

### From Single-Model to Pipeline

```python
# Old: Text extraction only
text = extract_text(image)

# New: Layout + text pipeline
layout = detect_layout(image)  # Understand structure
text = extract_text(image)     # Extract content
# Combine results for better processing
```

### From CPU to GPU

```python
# Old: CPU-based
ocr = TesseractOCR()  # 2-3s per page

# New: GPU-accelerated
ocr = PaddleOCR()     # 0.3-1s per page (10x faster)
```

---

## See Also

- [Qwen Text Extraction](./text-extraction/qwen.md)
- [DotsOCR Text Extraction](./text-extraction/dotsocr.md)
- [DocLayout-YOLO](./layout-analysis/doclayout-yolo.md)
- [Qwen Layout Detection](./layout-analysis/qwen-layout.md)
- [Tesseract OCR](./ocr-extraction/tesseract.md)
