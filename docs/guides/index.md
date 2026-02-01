# OmniDocs Task-Oriented Guides

Practical, copy-paste-ready guides for common OmniDocs tasks. Each guide includes real-world examples, best practices, and troubleshooting.

## Quick Navigation

### By Task

**Extract text from documents:**
- [Text Extraction Guide](text-extraction.md) - Convert documents to Markdown/HTML (2000 words, 4 examples)
- Choose model: Qwen3-VL (recommended), DotsOCR (technical docs), Nanonets (coming soon)
- Output formats: Markdown, HTML, plain text
- Advanced: Custom prompts, include_layout, temperature control

**Detect document structure:**
- [Layout Analysis Guide](layout-analysis.md) - Find elements and their locations (2000 words, 4 examples)
- Choose model: DocLayoutYOLO (fast), RT-DETR (accurate), QwenLayoutDetector (flexible)
- Features: Fixed labels, custom labels, confidence filtering, visualization

**Extract text with locations:**
- [OCR Extraction Guide](ocr-extraction.md) - Get text + bounding boxes (1800 words, 4 examples)
- Choose model: Tesseract (CPU), EasyOCR (accurate), PaddleOCR (fast)
- Features: Multi-language, confidence filtering, region filtering
- Granularities: Character, word, line, block

**Process many documents:**
- [Batch Processing Guide](batch-processing.md) - Handle 100+ documents efficiently (1600 words)
- Patterns: Sequential, batched, PDF pages, parallel preprocessing
- Memory optimization: Streaming, garbage collection, monitoring
- Progress tracking with tqdm, ETA estimation

**Deploy to GPU cloud:**
- [Modal Deployment Guide](deployment-modal.md) - Scale with serverless GPUs (1800 words, 2 examples)
- Cost: $0.30-1.00 per hour of GPU
- Patterns: Single-GPU, multi-GPU, batch, scheduled, webhooks
- Multi-GPU: Tensor parallelism, load balancing

## Which Guide to Read First

### I want to...

**...extract text from a document**
1. Read [Text Extraction Guide](text-extraction.md) - Basic Usage section
2. Choose model (Table: Qwen3-VL recommended)
3. Copy Example 1: Simple Markdown Extraction
4. Run on your document

**...build a document processing pipeline**
1. Read [Layout Analysis Guide](layout-analysis.md) - Basic Usage
2. Read [Text Extraction Guide](text-extraction.md) - Advanced Features
3. Combine: Layout → Filter → Text extraction
4. See Batch Processing Guide for multiple documents

**...find precise text locations**
1. Read [OCR Extraction Guide](ocr-extraction.md) - Basic Usage
2. Choose model (Table: EasyOCR for accuracy, PaddleOCR for speed)
3. Copy Example 1: Simple Word-Level OCR
4. Filter results by confidence/region as needed

**...process 100+ documents**
1. Read [Batch Processing Guide](batch-processing.md) - Processing Patterns
2. Choose pattern (Sequential recommended for learning)
3. Add progress tracking (tqdm)
4. Monitor GPU memory
5. See [Deployment Guide](deployment-modal.md) for scale

**...deploy on GPU cloud**
1. Read [Deployment Guide](deployment-modal.md) - Standard Setup
2. Set up Modal (token, volume, secret)
3. Copy Example 1: Simple Text Extraction
4. Deploy with `modal run script.py`
5. Scale with Example 2: Multi-GPU VLLM

## Guide Comparison

| Guide | Focus | Typical Task | Time | Difficulty |
|-------|-------|--------------|------|------------|
| Text Extraction | Content conversion | "Convert PDF to Markdown" | 2-5s/page | Beginner |
| Layout Analysis | Structure detection | "Find all tables in doc" | 0.5-1s/page | Beginner |
| OCR Extraction | Text with locations | "Extract word coordinates" | 1-2s/page | Intermediate |
| Batch Processing | Scale & efficiency | "Process 1000 documents" | Variable | Intermediate |
| Deployment | Cloud infrastructure | "Run 24/7 API service" | Setup: 5min | Advanced |

## Common Workflows

### Workflow 1: Simple Document Parsing

Convert documents to readable Markdown.

**Tools:** Text Extraction only
**Steps:**
1. Load document with Document.from_pdf()
2. Use QwenTextExtractor with PyTorch backend
3. Extract to Markdown format
4. Save to file

**Guide:** [Text Extraction](text-extraction.md) - Example 3 (PDF with Multiple Pages)

### Workflow 2: Table and Figure Extraction

Find all tables and figures with their content.

**Tools:** Layout Analysis + Text Extraction
**Steps:**
1. Run layout detection (find all elements)
2. Filter for 'table' and 'figure' labels
3. Crop to bounding box
4. Extract text/tables from each region
5. Combine results

**Guides:**
- [Layout Analysis](layout-analysis.md) - Filtering and Analyzing Results
- [Text Extraction](text-extraction.md) - Basic Usage Example 2 (Layout Information)

### Workflow 3: Handwriting Localization

Find coordinates of all handwritten text.

**Tools:** OCR Extraction (EasyOCR or PaddleOCR)
**Steps:**
1. Load image
2. Run OCR extraction
3. Filter by confidence (handwriting has lower confidence)
4. Get bounding boxes for each word
5. Save coordinates

**Guide:** [OCR Extraction](ocr-extraction.md) - Extracting Bounding Boxes

### Workflow 4: Batch Document Processing

Process 100+ documents efficiently.

**Tools:** Text Extraction + Batch Processing
**Steps:**
1. Load all images from directory
2. Initialize extractor once
3. Process sequentially or batched
4. Stream results to disk (JSONL)
5. Track progress with tqdm

**Guides:**
- [Batch Processing](batch-processing.md) - Processing Patterns Example 1
- [Text Extraction](text-extraction.md) - Example 4 (Batch with Progress)

### Workflow 5: Scale to 1000+ Documents

Deploy batch processing on GPU cloud.

**Tools:** Batch Processing + Modal Deployment
**Steps:**
1. Write batch function
2. Deploy to Modal
3. Submit batch jobs
4. Monitor with logging
5. Optimize costs with spot instances

**Guides:**
- [Batch Processing](batch-processing.md) - All sections
- [Deployment](deployment-modal.md) - Production Patterns

## API Quick Reference

### Text Extraction

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

result = extractor.extract(image, output_format="markdown")
print(result.content)  # Formatted text
```

### Layout Analysis

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

config = DocLayoutYOLOConfig(device="cuda")
detector = DocLayoutYOLO(config=config)

result = detector.extract(image)
for elem in result.elements:
    print(f"{elem.label} @ {elem.bbox}")
```

### OCR Extraction

```python
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

config = EasyOCRConfig(languages=["en"], gpu=True)
ocr = EasyOCR(config=config)

result = ocr.extract(image)
for block in result.text_blocks:
    print(f"{block.text} @ {block.bbox}")
```

### Batch Processing

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from pathlib import Path

config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

for image_path in Path("images/").glob("*.png"):
    image = Image.open(image_path)
    result = extractor.extract(image)
    # Process...
```

### Modal Deployment

```python
import modal

@app.function(gpu="A10G:1")
def extract(image_bytes: bytes):
    # Extract text...
    return result

# Deploy: modal run script.py
# Or: modal deploy script.py
```

## Performance Expectations

### Per-Page Latency

| Task | Model | Device | Time |
|------|-------|--------|------|
| Text Extraction | Qwen3-VL-8B | A10G GPU | 2-3s |
| Text Extraction | Qwen3-VL-8B | CPU | 15-30s |
| Layout Detection | DocLayoutYOLO | A10G GPU | 0.5-1s |
| OCR | EasyOCR | A10G GPU | 1-2s |
| OCR | Tesseract | CPU | 0.5-1s |

### 100-Document Processing

| Setup | Tool | Time |
|-------|------|------|
| Single A10G | QwenTextExtractor | ~4-5 min |
| 2x A10G (VLLM) | QwenTextExtractor | ~2-3 min |
| CPU | Tesseract OCR | ~50-100 min |
| Modal batch | QwenTextExtractor | ~$0.30-0.50 cost |

## Troubleshooting Quick Links

**Model not found:**
- [Text Extraction Troubleshooting](text-extraction.md#troubleshooting) - Model Download Issues

**Out of memory:**
- [Text Extraction](text-extraction.md#troubleshooting) - OOM Errors
- [Batch Processing](batch-processing.md#troubleshooting) - OOM During Batch

**Slow inference:**
- [Text Extraction](text-extraction.md#performance-optimization) - Backend Optimization
- [Batch Processing](batch-processing.md#troubleshooting) - Slow Processing

**Poor accuracy:**
- [OCR Extraction](ocr-extraction.md#troubleshooting) - Low Accuracy
- [Layout Analysis](layout-analysis.md#troubleshooting) - Missing Elements

**Deployment issues:**
- [Modal Deployment](deployment-modal.md#troubleshooting) - All common issues

## Learn More

- **OmniDocs API Reference:** See `Omnidocs/omnidocs/` source code
- **Modal Documentation:** https://modal.com/docs
- **HuggingFace Models:** https://huggingface.co/models
- **GitHub Issues:** Report bugs or request features

## Examples Repository

All code examples in these guides are copy-paste ready. For working end-to-end examples with test data, see:
- `scripts/text_extract_omnidocs/` - Text extraction examples
- `scripts/layout_omnidocs/` - Layout analysis examples
- `scripts/ocr_omnidocs/` - OCR examples
- `Omnidocs/tests/` - Unit tests with examples

---

**Start with:** [Text Extraction Guide](text-extraction.md) → Example 1

**Happy extracting! 🚀**
