# Concepts

Understanding OmniDocs architecture, config patterns, backend system, and document model.

---

## Architecture Overview

### Core Flow

```
User Code
    ↓
Document.from_pdf() → PIL Images
    ↓
Extractor(config) → Load model with backend
    ↓
extractor.extract(image) → Pydantic output
```

### Design Principles

| Principle | What It Means |
|-----------|---------------|
| **Unified API** | `.extract()` for all tasks |
| **Class imports** | `from omnidocs.tasks.x import Model` (no string factories) |
| **Type-safe configs** | Pydantic validation, IDE autocomplete |
| **Stateless Document** | Document = source data, not results |
| **Config = capability** | Available configs show supported backends |
| **Init vs Extract** | Config sets hardware, extract sets task params |

### Component Architecture

```
omnidocs/
├── document.py           # Document class (stateless)
├── tasks/
│   ├── text_extraction/  # Text → Markdown/HTML
│   ├── layout_analysis/  # Structure detection
│   ├── ocr_extraction/   # Text + bounding boxes
│   └── ...
└── inference/
    ├── pytorch.py        # HuggingFace/torch
    ├── vllm.py           # High-throughput
    ├── mlx.py            # Apple Silicon
    └── api.py            # LiteLLM
```

---

## Config Pattern

### Single-Backend Models

Models that support only one backend (typically PyTorch):

```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

# Pattern: {Model}Config → config= parameter
layout = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(
        device="cuda",
        confidence=0.25,
    )
)
```

### Multi-Backend Models

Models that support multiple backends:

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenPyTorchConfig,   # Local GPU
    QwenVLLMConfig,      # High throughput
    QwenMLXConfig,       # Apple Silicon
    QwenAPIConfig,       # Cloud API
)

# Pattern: {Model}{Backend}Config → backend= parameter
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(device="cuda")
)
```

### Config Naming

| Model Type | Naming Pattern | Parameter |
|------------|----------------|-----------|
| Single-backend | `{Model}Config` | `config=` |
| Multi-backend | `{Model}{Backend}Config` | `backend=` |

### What Goes Where

**Init (config/backend)**
- Model name/path
- Device (cuda, cpu, mps)
- Quantization, dtype
- Backend-specific settings

**Extract (method params)**
- Output format (markdown, html)
- Custom prompts
- Task-specific options
- Per-call settings

```python
# Init: hardware/model setup
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen3-VL-8B",
        device="cuda",
        torch_dtype="bfloat16",
    )
)

# Extract: task parameters
result = extractor.extract(
    image,
    output_format="markdown",
    include_layout=True,
)
```

---

## Backend System

### Backend Comparison

| Backend | Use Case | Requirements |
|---------|----------|--------------|
| **PyTorch** | Development, local GPU | CUDA 12+ or CPU |
| **VLLM** | Production, high throughput | NVIDIA GPU 24GB+ |
| **MLX** | Apple Silicon development | M1/M2/M3 Mac |
| **API** | No GPU, cloud-first | API key + internet |

### Backend Selection

```python
# PyTorch - development default
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig
backend = QwenPyTorchConfig(device="cuda", torch_dtype="bfloat16")

# VLLM - production throughput
from omnidocs.tasks.text_extraction.qwen import QwenVLLMConfig
backend = QwenVLLMConfig(tensor_parallel_size=2, gpu_memory_utilization=0.9)

# MLX - Apple Silicon
from omnidocs.tasks.text_extraction.qwen import QwenMLXConfig
backend = QwenMLXConfig(quantization="4bit")

# API - cloud
from omnidocs.tasks.text_extraction.qwen import QwenAPIConfig
backend = QwenAPIConfig(api_key="sk-...", base_url="https://...")
```

### Switching Backends

OmniDocs makes it easy to switch - only the config changes:

```python
# Development: PyTorch
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(device="cuda")
)

# Production: switch to VLLM
extractor = QwenTextExtractor(
    backend=QwenVLLMConfig(tensor_parallel_size=2)
)

# Same .extract() API works for both
result = extractor.extract(image, output_format="markdown")
```

### Discoverability

Available backends = importable config classes:

```python
# Check what backends a model supports
from omnidocs.tasks.text_extraction.qwen import (
    QwenPyTorchConfig,  # ✓ PyTorch supported
    QwenVLLMConfig,     # ✓ VLLM supported
    QwenMLXConfig,      # ✓ MLX supported
    QwenAPIConfig,      # ✓ API supported
)

# If import fails → backend not supported for that model
```

---

## Document Model

### Design: Stateless

Document contains source data only, not analysis results.

**Why?**
- Clean separation of concerns
- User controls caching strategy
- Memory efficient
- Works with any workflow

```python
doc = Document.from_pdf("file.pdf")  # Just loads PDF
result = extractor.extract(doc.get_page(0))  # User manages result
```

### Loading Methods

```python
from omnidocs import Document

# From file
doc = Document.from_pdf("file.pdf", dpi=150)

# From URL
doc = Document.from_url("https://example.com/doc.pdf")

# From bytes
doc = Document.from_bytes(pdf_bytes, filename="doc.pdf")

# From images
doc = Document.from_image("page.png")
doc = Document.from_images(["p1.png", "p2.png"])
```

### Lazy Loading

Pages render on demand, then cache:

```python
doc = Document.from_pdf("large.pdf")  # Fast: no rendering yet

page = doc.get_page(0)  # Renders now (~200ms)
page = doc.get_page(0)  # Cached: instant
```

### Memory Management

```python
# Efficient iteration (one page at a time)
for page in doc.iter_pages():
    result = extractor.extract(page)
    save(result)

# Clear cache for large documents
doc.clear_cache()        # All pages
doc.clear_cache(page=0)  # Specific page

# Context manager
with Document.from_pdf("file.pdf") as doc:
    # Use doc
    pass  # Auto-closes
```

### Metadata

```python
doc.page_count              # Number of pages
doc.metadata.source_type    # "file", "url", "bytes"
doc.metadata.file_name      # Filename
doc.metadata.file_size      # Size in bytes
doc.metadata.format         # "pdf", "png", etc.
doc.to_dict()               # Serialize metadata
```

---

## Key Patterns

### Pattern 1: Single Page

```python
doc = Document.from_pdf("paper.pdf")
result = extractor.extract(doc.get_page(0))
```

### Pattern 2: All Pages

```python
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page)
    save(f"page_{i}.md", result.content)
```

### Pattern 3: Memory Control

```python
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page)
    save(result)
    if i % 10 == 0:
        doc.clear_cache()  # Free every 10 pages
```

### Pattern 4: Environment-Based Backend

```python
import os

if os.getenv("USE_VLLM"):
    backend = QwenVLLMConfig(tensor_parallel_size=2)
elif os.getenv("USE_API"):
    backend = QwenAPIConfig(api_key=os.getenv("API_KEY"))
else:
    backend = QwenPyTorchConfig(device="cuda")

extractor = QwenTextExtractor(backend=backend)
```

---

## Trade-offs

| Choice | Option A | Option B |
|--------|----------|----------|
| Speed vs Quality | 2B model (fast) | 8B+ model (accurate) |
| Setup vs Throughput | PyTorch (simple) | VLLM (10x faster) |
| Privacy vs Convenience | Local (private) | API (no setup) |
| Memory vs Speed | Lazy loading | Load all pages |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Architecture** | Image → Extractor → Pydantic output |
| **Configs** | Single-backend: `config=`, Multi-backend: `backend=` |
| **Backends** | PyTorch (dev), VLLM (prod), MLX (Mac), API (cloud) |
| **Document** | Stateless, lazy-loaded, user manages results |
