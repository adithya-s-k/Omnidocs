# Text Extraction Guide

Extract formatted text content (Markdown/HTML) from document images using vision-language models. This guide covers when to use text extraction, available models, output formats, and practical examples.

## Table of Contents

- [Quick Comparison: Text Extraction vs OCR vs Layout](#quick-comparison)
- [Available Models](#available-models)
- [Basic Usage](#basic-usage)
- [Output Formats](#output-formats)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Quick Comparison

| Feature | Text Extraction | OCR | Layout Detection |
|---------|-----------------|-----|------------------|
| **Output** | Formatted text (MD/HTML) | Text + bounding boxes | Element bounding boxes |
| **Use Case** | Document parsing, markdown export | Word/character localization | Document structure analysis |
| **Models** | Qwen3-VL, DotsOCR, Nanonets | Tesseract, EasyOCR, PaddleOCR | DocLayoutYOLO, Qwen-Layout |
| **Latency** | ~2-5 sec per page | ~1-2 sec per page | ~0.5-1 sec per page |
| **Output Type** | Single string | List of text blocks | List of bounding boxes |
| **Layout Info** | Optional (DotsOCR only) | No | Yes (with labels) |

**Choose Text Extraction when:**
- Converting documents to Markdown/HTML
- Extracting complete page content as formatted text
- Working with complex documents (multi-column, figures, tables)
- You need readable output for downstream processing

**Choose OCR when:**
- You need precise character/word locations
- Building re-OCR pipelines (e.g., for correction)
- Requiring character-level accuracy metrics

**Choose Layout Detection when:**
- You need document structure without text content
- Building advanced pipelines (layout + text)
- Analyzing document semantics

## Available Models

### 1. Qwen3-VL (Recommended for most cases)

High-quality general-purpose vision-language model.

**Strengths:**
- Best output quality across diverse documents
- Multi-backend support (PyTorch, VLLM, MLX, API)
- Consistent Markdown/HTML output
- Good at handling complex layouts

**Backends:**
- PyTorch: Local GPU inference (single GPU)
- VLLM: High-throughput serving (multiple GPUs)
- MLX: Apple Silicon (local)
- API: Hosted models (cloud)

**Model Variants:**
- `Qwen/Qwen3-VL-8B-Instruct`: Recommended (8B parameters)
- `Qwen/Qwen3-VL-32B-Instruct`: Higher quality (32B, slower, more VRAM)

### 2. DotsOCR (Best for technical documents)

Optimized for complex technical documents with precise layout preservation.

**Strengths:**
- Layout-aware extraction with bounding boxes
- Specialized formatting for tables (HTML) and formulas (LaTeX)
- Reading order preservation
- 11-category layout detection

**Weaknesses:**
- Slower than Qwen (requires layout analysis)
- Higher VRAM requirements

**Backends:**
- PyTorch: Local GPU inference
- VLLM: High-throughput serving
- API: Hosted models

**Output Types:**
- Structured JSON with layout information
- Markdown with coordinate annotations
- HTML with bbox attributes

### 3. Nanonets (Coming soon)

Specialized for OCR-quality text extraction.

## Basic Usage

### Example 1: Simple Markdown Extraction

Extract a document page to Markdown using PyTorch backend.

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from PIL import Image

# Load a single image
image = Image.open("document_page.png")

# Initialize extractor with PyTorch backend
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",  # or "cpu"
    torch_dtype="auto",  # Automatic dtype selection
)
extractor = QwenTextExtractor(backend=config)

# Extract text in Markdown format
result = extractor.extract(image, output_format="markdown")

# Access the extracted content
print(result.content)  # Formatted Markdown text
print(result.word_count)  # Number of words
print(f"Model: {result.model_name}")
```

### Example 2: Extract with Layout Information

Use DotsOCR to get text plus layout annotations.

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig
from PIL import Image
import json

image = Image.open("complex_document.png")

# Initialize DotsOCR with layout detection
config = DotsOCRPyTorchConfig(
    device="cuda",
    max_new_tokens=8192,  # Higher for complex documents
)
extractor = DotsOCRTextExtractor(backend=config)

# Extract with layout information
result = extractor.extract(image, include_layout=True)

# Access layout elements
print(f"Found {result.num_layout_elements} layout elements")
print(f"Content length: {result.content_length} characters")

# Iterate through layout elements
for element in result.layout:
    print(f"[{element.category}] @{element.bbox}: {element.text[:50]}...")

# Save layout information to JSON
layout_json = [elem.model_dump() for elem in result.layout]
with open("layout.json", "w") as f:
    json.dump(layout_json, f, indent=2)
```

### Example 3: Extract PDF Document

Process multiple pages of a PDF document.

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from pathlib import Path

# Load PDF document
doc = Document.from_pdf("multi_page_document.pdf")
print(f"Loaded PDF with {doc.page_count} pages")

# Initialize extractor
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
)
extractor = QwenTextExtractor(backend=config)

# Extract text from all pages
all_text = []
for page_idx in range(min(3, doc.page_count)):  # First 3 pages
    page_image = doc.get_page(page_idx)
    result = extractor.extract(page_image, output_format="markdown")
    all_text.append(result.content)
    print(f"Page {page_idx + 1}: {result.word_count} words")

# Combine results
full_document = "\n\n---\n\n".join(all_text)
print(f"\nTotal content: {len(full_document)} characters")

# Save to file
with open("extracted_document.md", "w") as f:
    f.write(full_document)
```

### Example 4: Batch Processing with Progress Tracking

Process multiple documents with progress reporting.

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from pathlib import Path
from PIL import Image
import time

# Find all image files
image_dir = Path("documents/")
image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
print(f"Found {len(image_files)} images to process")

# Initialize extractor
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    max_new_tokens=4096,
)
extractor = QwenTextExtractor(backend=config)

# Process with progress tracking
results = {}
start_time = time.time()

for idx, image_path in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...", end=" ")

    try:
        image = Image.open(image_path)
        result = extractor.extract(image, output_format="markdown")
        results[str(image_path)] = {
            "content_length": result.content_length,
            "word_count": result.word_count,
        }
        print(f"✓ ({result.word_count} words)")
    except Exception as e:
        print(f"✗ Error: {e}")
        results[str(image_path)] = {"error": str(e)}

# Summary
elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(image_files):.2f}s per image)")
print(f"Successful: {sum(1 for r in results.values() if 'error' not in r)}")
```

## Output Formats

### Markdown Format

Human-readable format with standard Markdown syntax. Best for documentation and web publishing.

```python
result = extractor.extract(image, output_format="markdown")
print(result.content)

# Example output:
# # Document Title
#
# This is the main content with **bold** and *italic* text.
#
# ## Section 1
#
# - Bullet point 1
# - Bullet point 2
#
# | Column 1 | Column 2 |
# |----------|----------|
# | Cell 1   | Cell 2   |
```

**Advantages:**
- Human-readable
- Git-friendly (version control)
- Easy to edit
- Good for documentation

**Limitations:**
- Loses some layout information
- Tables converted to Markdown tables (may lose formatting)
- No bounding box information

### HTML Format

Structured HTML with semantic tags. Better for preserving layout in web contexts.

```python
result = extractor.extract(image, output_format="html")
print(result.content)

# Example output:
# <div class="document">
#   <h1>Document Title</h1>
#   <p>This is the main content with <b>bold</b> and <i>italic</i> text.</p>
#   <h2>Section 1</h2>
#   <ul>
#     <li>Bullet point 1</li>
#     <li>Bullet point 2</li>
#   </ul>
#   <table>...</table>
# </div>
```

**Advantages:**
- Structured and semantic
- Better layout preservation
- Good for web rendering
- Supports nested elements

**Limitations:**
- More verbose
- Requires HTML parser for processing
- Layout information may still be approximate

### Plain Text (Fallback)

Extract plain text without any formatting.

```python
# Get plain text version
plain_text = result.plain_text
print(plain_text)

# Also available as property:
from omnidocs.tasks.text_extraction import QwenTextExtractor
# ... after extraction ...
print(result.plain_text)  # No formatting, just raw text
```

### DotsOCR JSON Format

Structured JSON with layout information (DotsOCR only).

```python
result = extractor.extract(image, output_format="json", include_layout=True)

# Result includes:
# {
#   "content": "Full text...",
#   "layout": [
#     {
#       "bbox": [100, 50, 400, 80],
#       "category": "Title",
#       "text": "Document Title"
#     },
#     ...
#   ]
# }
```

## Advanced Features

### Custom Prompts

Override the default extraction prompt for specialized use cases.

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from PIL import Image

image = Image.open("document.png")
config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)

# Custom prompt for extractive summarization
custom_prompt = """
Extract the most important information from this document image.
Focus on key facts, numbers, and action items.
Format as a concise Markdown list.
"""

result = extractor.extract(
    image,
    output_format="markdown",
    custom_prompt=custom_prompt,
)

print(result.content)
```

### Temperature Control (PyTorch only)

Adjust model creativity/determinism via temperature parameter.

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Lower temperature = more deterministic (better for factual extraction)
config = QwenTextPyTorchConfig(
    device="cuda",
    temperature=0.1,  # Default: 0.1 (deterministic)
)

# Higher temperature = more creative (for summarization, etc.)
config_creative = QwenTextPyTorchConfig(
    device="cuda",
    temperature=0.7,
)
```

### Backend Switching

Easily switch between backends without changing extraction code.

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,
    QwenTextVLLMConfig,
    QwenTextMLXConfig,
    QwenTextAPIConfig,
)
from PIL import Image

image = Image.open("document.png")

# Use PyTorch for single-GPU inference
pytorch_extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)
result1 = pytorch_extractor.extract(image, output_format="markdown")

# Use VLLM for high-throughput inference
vllm_extractor = QwenTextExtractor(
    backend=QwenTextVLLMConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=1,
    )
)
result2 = vllm_extractor.extract(image, output_format="markdown")

# Use MLX for Apple Silicon
mlx_extractor = QwenTextExtractor(
    backend=QwenTextMLXConfig(device="gpu")
)
result3 = mlx_extractor.extract(image, output_format="markdown")

# Use API for hosted models
api_extractor = QwenTextExtractor(
    backend=QwenTextAPIConfig(
        model="qwen3-vl-8b",
        api_key="your-api-key",
        base_url="https://api.example.com/v1",
    )
)
result4 = api_extractor.extract(image, output_format="markdown")

print(f"PyTorch: {result1.word_count} words")
print(f"VLLM: {result2.word_count} words")
print(f"MLX: {result3.word_count} words")
print(f"API: {result4.word_count} words")
```

## Performance Optimization

### Model Selection

| Model | Latency | Quality | VRAM | Speed |
|-------|---------|---------|------|-------|
| Qwen3-VL-8B | 2-3 sec | Excellent | 16GB | Fast |
| Qwen3-VL-32B | 5-8 sec | Outstanding | 32GB | Slow |
| DotsOCR | 3-5 sec | Very Good (technical) | 20GB | Medium |

**Recommendation:** Start with Qwen3-VL-8B (best quality/speed tradeoff).

### Backend Optimization

**PyTorch (Single GPU):**
- Best for development and small batches
- Load time: ~2-3 seconds
- Per-image latency: ~2-3 seconds

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype="auto",  # Let PyTorch choose optimal dtype
    max_new_tokens=4096,  # Reduce for faster inference
)
```

**VLLM (Multi-GPU):**
- Best for batch processing / high throughput
- Load time: ~5-8 seconds (slightly slower but amortizes)
- Throughput: 2-4x better than PyTorch for multiple requests

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.9,  # Use 90% of VRAM
    max_tokens=4096,
)
```

**MLX (Apple Silicon):**
- Best for MacBook development
- No GPU-related issues
- Slightly slower than VRAM-constrained models

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

config = QwenTextMLXConfig(
    model="Qwen/Qwen3-VL-8B-Instruct-MLX",
    device="gpu",
    quantization="4bit",  # Quantization reduces VRAM
)
```

### Batch Processing Strategy

For processing many documents, batch requests to amortize model loading.

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
from pathlib import Path
from PIL import Image
import time

# Initialize once (expensive)
config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_tokens=4096,
)
extractor = QwenTextExtractor(backend=config)

# Process many documents (cheap)
image_paths = list(Path("documents/").glob("*.png"))
results = []

start = time.time()
for image_path in image_paths:
    image = Image.open(image_path)
    result = extractor.extract(image, output_format="markdown")
    results.append(result)

elapsed = time.time() - start
print(f"Processed {len(results)} images in {elapsed:.1f}s")
print(f"Average: {elapsed/len(results):.2f}s per image")
```

### Token Limit Tuning

Adjust `max_new_tokens` based on expected output length.

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# For short documents (< 1000 words)
config_short = QwenTextPyTorchConfig(
    device="cuda",
    max_new_tokens=2048,  # Faster
)

# For medium documents (1000-5000 words)
config_medium = QwenTextPyTorchConfig(
    device="cuda",
    max_new_tokens=4096,  # Default
)

# For long documents (> 5000 words)
config_long = QwenTextPyTorchConfig(
    device="cuda",
    max_new_tokens=8192,  # Slower but handles longer docs
)
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Problem:** CUDA out of memory during inference.

**Solutions:**
1. Reduce `max_new_tokens`
2. Use smaller model variant (8B instead of 32B)
3. Switch to VLLM with `tensor_parallel_size > 1`
4. Use quantization (if available)

```python
# Option 1: Reduce max_new_tokens
config = QwenTextPyTorchConfig(
    device="cuda",
    max_new_tokens=2048,  # Reduced from 4096
)

# Option 2: Smaller model
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",  # Instead of 32B
    device="cuda",
)

# Option 3: VLLM with tensor parallelism
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
config = QwenTextVLLMConfig(
    tensor_parallel_size=2,  # Distribute across 2 GPUs
    max_tokens=4096,
)
```

### Slow Inference

**Problem:** Text extraction takes too long.

**Solutions:**
1. Check GPU utilization (should be >80%)
2. Reduce `max_new_tokens`
3. Use VLLM instead of PyTorch
4. Use VLLM tensor parallelism

```python
import subprocess

# Check GPU usage during extraction
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
    capture_output=True,
    text=True
)
print(f"GPU Utilization: {result.stdout.strip()}%")

# If <50%, increase batch size or use VLLM
```

### Incorrect or Garbled Output

**Problem:** Extracted text is incomplete or corrupted.

**Solutions:**
1. Check image quality (min 1024px width recommended)
2. Verify model downloaded correctly
3. Try with explicit output format

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from PIL import Image

image = Image.open("document.png")

# Check image size
print(f"Image size: {image.size}")  # Should be at least (1024, 768)

# Resize if too small
if image.width < 1024:
    image = image.resize((image.width * 2, image.height * 2))

# Try extraction
config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)
result = extractor.extract(image, output_format="markdown")

# Check result
if len(result.content) < 10:
    print("Warning: Very short output, may indicate extraction failure")
    print(f"Raw output: {result.raw_output}")
```

### Model Download Issues

**Problem:** Model fails to download or load.

**Solutions:**
1. Check internet connection
2. Verify HuggingFace token
3. Set custom cache directory

```python
import os

# Set HuggingFace token
os.environ["HF_TOKEN"] = "your-token-here"

# Set custom cache directory
os.environ["HF_HOME"] = "/large/disk/hf_cache"

# Verify download by loading model explicitly
from transformers import AutoTokenizer, AutoModel

model_id = "Qwen/Qwen3-VL-8B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"✓ Model {model_id} loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
```

### API Backend Timeouts

**Problem:** API requests timeout or fail.

**Solutions:**
1. Increase timeout value
2. Check API credentials
3. Reduce batch size

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

config = QwenTextAPIConfig(
    model="qwen3-vl-8b",
    api_key="your-api-key",
    base_url="https://api.example.com/v1",
    timeout=60,  # Increase timeout
    rate_limit=5,  # Reduce concurrent requests
)
extractor = QwenTextExtractor(backend=config)
```

---

**Next Steps:**
- See [Batch Processing Guide](batch-processing.md) for processing many documents
- See [Deployment Guide](deployment-modal.md) for scaling on Modal
- See [Layout Analysis Guide](layout-analysis.md) for structure-aware extraction
