# OmniDocs Architecture Overview

> **Core Philosophy**: Unified interface, type-safe configurations, multi-backend support, and stateless documents.

## Table of Contents

1. [System Design Principles](#system-design-principles)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Core Design Decisions](#core-design-decisions)
5. [How It Works (Nothing Magic)](#how-it-works-nothing-magic)
6. [Example: Qwen Text Extraction Flow](#example-qwen-text-extraction-flow)

---

## System Design Principles

OmniDocs is built on six core principles that shape every design decision:

### 1. Unified API

**Problem Solved**: Different document processing tasks (layout analysis, OCR, text extraction) come from different libraries, each with different APIs.

**Solution**: All extractors implement the same `.extract()` method.

```python
# All these follow the same pattern
layout_result = layout_extractor.extract(image)
ocr_result = ocr_extractor.extract(image)
text_result = text_extractor.extract(image, output_format="markdown")
```

This consistency makes it trivial to swap models or add new tasks - the user code doesn't change.

### 2. Type-Safe Configurations

**Problem Solved**: Magic strings and untyped dictionaries make it impossible to catch configuration errors early.

**Solution**: Pydantic-based config classes with IDE autocomplete and validation.

```python
# Config is validated at creation time
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype="bfloat16",  # Wrong! IDE catches this
    device="cuda",
)
# Error: 'torch_dtype' should be "float16", "bfloat16", "float32", or "auto"
```

### 3. Multi-Backend Support

**Problem Solved**: Not all users have the same hardware. A researcher might use PyTorch locally, while a startup deploys on VLLM, and another team uses MLX on Apple Silicon.

**Solution**: The same model can run on multiple backends. Backend selection is explicit via config classes.

```python
# Switch backends by changing the config type - same extractor
extractor = QwenTextExtractor(backend=QwenTextPyTorchConfig(...))
extractor = QwenTextExtractor(backend=QwenTextVLLMConfig(...))
extractor = QwenTextExtractor(backend=QwenTextMLXConfig(...))
```

### 4. Stateless Document

**Problem Solved**: Models that store extraction results create confusion about what's source data vs. analysis output, and make it hard to run multiple tasks on the same document.

**Solution**: The `Document` class only holds source data (PDF bytes, pages). Users manage task results.

```python
# Document doesn't know about extraction results
doc = Document.from_pdf("paper.pdf")
page = doc.get_page(0)  # Returns PIL Image

# Different extractors process the same page independently
layout = layout_extractor.extract(page)
ocr = ocr_extractor.extract(page)
text = text_extractor.extract(page)

# Users combine results as needed
combined = {
    "layout": layout,
    "ocr": ocr,
    "text": text,
}
```

### 5. Separation of Init vs Extract

**Problem Solved**: It's unclear what configuration belongs to model setup vs. task parameters.

**Solution**: Clear separation in method signatures.

```python
# Init (via config) = Model setup, happens once
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",  # Model choice
        device="cuda",                      # Hardware
        torch_dtype="bfloat16",             # Quantization
    )
)

# Extract (method call) = Task parameters, changes per call
result = extractor.extract(
    image,
    output_format="markdown",   # Task parameter
    include_layout=True,        # Task-specific option
)
```

### 6. Backend Discoverability

**Problem Solved**: Users don't know which backends a model supports.

**Solution**: If you can import a config class, that backend is supported.

```python
# These imports work = backends are supported
from omnidocs.tasks.text_extraction.qwen import (
    QwenTextPyTorchConfig,  ✓ PyTorch backend
    QwenTextVLLMConfig,     ✓ VLLM backend
    QwenTextMLXConfig,      ✓ MLX backend
    QwenTextAPIConfig,      ✓ API backend
)

# This import fails = backend not implemented yet
from omnidocs.tasks.text_extraction.qwen import QwenTextONNXConfig  # ImportError
```

---

## Component Architecture

OmniDocs has four main components:

### 1. Document Class

The entry point for loading and accessing document data.

```
Document (stateless)
├── from_pdf(path)           → Load PDF file
├── from_url(url)            → Download and load PDF
├── from_bytes(data)         → Load from memory
├── from_image(path)         → Load single image
└── from_images(paths)       → Load multi-page images

Document properties:
├── page_count: int          → Number of pages
├── metadata: Metadata       → Source info, DPI, etc.
├── get_page(i): Image       → Get single page (0-indexed)
├── iter_pages(): Iterator   → Iterate pages (memory efficient)
└── text: str                → Full document text (cached)
```

**Key Design**: Lazy page rendering. Pages are NOT rendered until accessed. This enables:
- Fast document loading (no rendering upfront)
- Memory efficiency (only rendered pages stay in RAM)
- Page-level caching (rendered pages cached automatically)

### 2. Extractors

Task-specific processors. All inherit from a base class and implement `.extract()`.

```
BaseTextExtractor (abstract)
├── _load_model() → Load model into memory
└── extract(image, ...) → Process image

Implementations:
├── QwenTextExtractor (multi-backend)
├── DotsOCRTextExtractor (multi-backend)
└── Other models...
```

Each extractor handles:
- Model loading and device placement
- Input preprocessing
- Inference
- Output formatting

### 3. Config Classes

Pydantic models that specify how to initialize extractors.

**Single-Backend Model**:
```python
DocLayoutYOLOConfig
└── Parameters for PyTorch inference only

QwenTextPyTorchConfig
└── Parameters for PyTorch inference of Qwen
```

**Multi-Backend Model**:
```python
QwenTextPyTorchConfig    → PyTorch backend
QwenTextVLLMConfig       → VLLM backend
QwenTextMLXConfig        → MLX backend
QwenTextAPIConfig        → API backend
```

Configs drive:
- Which backend to use (via `isinstance` checks in `_load_model()`)
- Model hyperparameters
- Device placement
- Quantization settings

### 4. Backend Inference

Low-level inference implementations. Hidden from users (internal).

```
Backend Inference Hierarchy:

For each model + backend combination:
├── QwenTextExtractor receives QwenTextPyTorchConfig
│   └── _load_model() detects it's PyTorch
│       └── Initialize PyTorch model, processor
│       └── Store for use in extract()
│
├── QwenTextExtractor receives QwenTextVLLMConfig
│   └── _load_model() detects it's VLLM
│       └── Initialize VLLM server, engine
│       └── Store for use in extract()
│
└── etc for MLX, API...
```

Users never interact with backend code directly - it's all encapsulated in the extractor.

---

## Data Flow

### Single-Page Processing

```
┌─────────────────────────────────────────────────────┐
│  User Code                                          │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        ┌────────────────────┐
        │   Document         │
        │ .from_pdf("x.pdf") │
        └──────────┬─────────┘
                   │
                   ▼
     ┌─────────────────────────────┐
     │ document.get_page(0)         │
     │ Returns: PIL.Image (RGB)     │
     └──────────┬──────────────────┘
                │
                ▼
    ┌────────────────────────────┐
    │  Extractor                 │
    │  (e.g., QwenTextExtractor) │
    │  .extract(image)           │
    └──────────┬─────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Backend _load_model()      │
    │  (detects config type)      │
    │  (loads model once)         │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Image Preprocessing        │
    │  PIL → numpy or model input │
    │  Normalization, resizing    │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Model Inference            │
    │  (GPU/CPU/API)              │
    │  Returns: model output      │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  Post-processing            │
    │  Parse model output         │
    │  Format as structured data  │
    └──────────┬──────────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │  Return TextOutput         │
    │  (Pydantic model)          │
    │  with content, metadata    │
    └────────────────────────────┘
```

### Multi-Page Processing (Memory Efficient)

```
for page in doc.iter_pages():
    ┌──────────────────────┐
    │  Load page (lazy)    │ ← Only rendered when accessed
    │  Render to image     │
    │  Send to extractor   │
    │  Get result          │
    │  Store result        │ ← User manages results
    │  Clear page cache    │ ← Memory freed
    └──────────────────────┘
```

---

## Core Design Decisions

### Why Pydantic for Configs?

**Benefits**:
1. **Validation at creation time**: Errors caught immediately
2. **IDE autocomplete**: All parameters visible in editor
3. **Documentation**: Field descriptions in docstrings and type hints
4. **Serialization**: Configs can be saved/loaded as JSON
5. **Immutability**: `frozen=True` prevents accidental changes

```python
# IDE shows all valid parameters with descriptions
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",  # ← IDE autocomplete
    torch_dtype="bfloat16",              # ← Type validation
    device="cuda",
)

# Typo caught immediately (Pydantic validation)
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype="float64",               # ERROR: Invalid literal
)
```

### Why Stateless Document?

**Problem with Stateful Document**:
```python
# BAD: Document stores results
doc = Document.from_pdf("paper.pdf")
doc.extract_layout()  # Stores layout internally
doc.extract_text()    # Stores text internally

# Now what is doc.get_page(0)?
# Is it the page, or the extracted data?
# Can I access original pixels?
```

**Solution: Stateless Document**:
```python
# GOOD: Document is just source data
doc = Document.from_pdf("paper.pdf")
page = doc.get_page(0)  # ← Always returns PIL Image

# Extractors are independent
layout = layout_extractor.extract(page)
text = text_extractor.extract(page)

# User combines results as needed
result = {"layout": layout, "text": text}
```

**Benefits**:
- Clear separation: Document = source, Extractors = analysis
- Reuse: Run multiple tasks on same pages
- Composition: Easy to build pipelines
- Caching: User controls what to cache

### Why Backend Selection via Config Type?

**Alternative (Magic String)**:
```python
# BAD: Which backends are available?
extractor = QwenTextExtractor(backend="pytorch")
# No IDE support, typos not caught, unclear what's available
```

**Solution (Config Type)**:
```python
# GOOD: IDE shows available configs
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(...)  # IDE autocomplete shows all options
)
extractor = QwenTextExtractor(
    backend=QwenTextVLLMConfig(...)     # Obvious which are available
)
```

**How It Works**:
```python
def _load_model(self) -> None:
    """Load appropriate backend based on config type."""
    config_type = type(self.backend_config).__name__

    if config_type == "QwenTextPyTorchConfig":
        # Load PyTorch model
    elif config_type == "QwenTextVLLMConfig":
        # Load VLLM engine
    elif config_type == "QwenTextMLXConfig":
        # Load MLX model
    # etc
```

---

## How It Works (Nothing Magic)

### Load Model (First Time Only)

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
)

extractor = QwenTextExtractor(backend=config)
# This calls __init__ → _load_model()
```

What happens in `_load_model()`:

1. **Detect config type**
   ```python
   config_type = type(self.backend_config).__name__
   # "QwenTextPyTorchConfig"
   ```

2. **Load model code**
   ```python
   from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
   model = Qwen2VLForConditionalGeneration.from_pretrained(
       "Qwen/Qwen3-VL-8B-Instruct",
       torch_dtype=torch.bfloat16,
       device_map="auto",
       trust_remote_code=True,
   )
   processor = AutoProcessor.from_pretrained(...)
   ```

3. **Store for later use**
   ```python
   self._model = model
   self._processor = processor
   ```

### Extract (Every Time)

```python
doc = Document.from_pdf("paper.pdf")
page = doc.get_page(0)  # PIL Image

result = extractor.extract(page, output_format="markdown")
```

What happens in `extract()`:

1. **Prepare image**
   ```python
   if isinstance(page, Image.Image):
       image = page.convert("RGB")  # Ensure RGB
   ```

2. **Process with vision encoder**
   ```python
   inputs = self._processor(
       text="Extract text in markdown format",
       images=image,
       return_tensors="pt",
   )
   ```

3. **Run model inference**
   ```python
   with torch.no_grad():
       outputs = self._model.generate(**inputs)
   ```

4. **Decode and clean**
   ```python
   raw_text = self._processor.decode(outputs[0])
   cleaned = _clean_markdown_output(raw_text)
   ```

5. **Return structured output**
   ```python
   return TextOutput(
       content=cleaned,
       format="markdown",
       metadata={"model": "Qwen3-VL-8B"},
   )
   ```

---

## Example: Qwen Text Extraction Flow

Let's trace a complete example from start to finish.

### Setup Phase

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# 1. Create config (validated immediately)
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
    max_new_tokens=8192,
)
# ✓ Pydantic validates all fields
# ✓ No typos possible (IDE catches them)

# 2. Create extractor (loads model)
extractor = QwenTextExtractor(backend=config)
# → __init__ called
# → _load_model() called
# → Detects "QwenTextPyTorchConfig"
# → Imports transformers, loads Qwen3-VL model
# → Loads processor for vision encoding
# → Stores both in self._model, self._processor
```

### Load Document Phase

```python
# 3. Load document (lazy, no page rendering yet)
doc = Document.from_pdf("research_paper.pdf")
# → Reads PDF file bytes
# → Creates pypdfium2 document object
# → Creates metadata (page count, file size, etc.)
# → Creates LazyPage wrappers (not rendered)
# → Returns Document

print(doc.page_count)  # 12 pages
print(doc.metadata.file_name)  # "research_paper.pdf"
```

### Process Pages Phase

```python
# 4. Process first page
page = doc.get_page(0)
# → Accesses LazyPage[0]
# → Renders page to PIL Image (150 DPI default)
# → Caches image in LazyPage
# → Returns PIL Image

# 5. Extract text from page
result = extractor.extract(
    page,
    output_format="markdown",
)
# → Validates image input
# → Processes with processor
# → Runs model inference on GPU
# → Cleans markdown output
# → Returns TextOutput(content=..., format="markdown")

print(result.content[:200])  # "# Research Paper Title\n\n## Abstract\n..."
```

### Scale to Multiple Pages Phase

```python
# 6. Process all pages efficiently
results = []
for i, page in enumerate(doc.iter_pages()):
    # Each iteration:
    # - Loads next page (lazy)
    # - Renders to image (cached)
    # - Extracts text via Qwen
    # - Clears cache to free memory
    result = extractor.extract(page, output_format="markdown")
    results.append(result)

    if i > 0:
        doc.clear_cache(i - 1)  # Free previous page from memory

# Results list: [TextOutput, TextOutput, ...]
full_content = "\n\n".join([r.content for r in results])
```

### Key Points in This Flow

| Step | Component | Purpose |
|------|-----------|---------|
| Config creation | Pydantic | Validate parameters upfront |
| Model loading | Extractor._load_model() | One-time initialization |
| Document loading | Document.from_pdf() | Lazy, fast, no rendering |
| Page access | doc.get_page() | Renders on demand, caches |
| Extraction | extractor.extract() | Runs model on page |
| Memory management | doc.clear_cache() | Frees rendered pages |

### Why This Design Matters

```python
# Without lazy loading
doc = Document.from_pdf("500_page_book.pdf")
# Would need to render all 500 pages in memory = SLOW, HIGH MEMORY

# With lazy loading + caching
for page in doc.iter_pages():
    result = extractor.extract(page)
    doc.clear_cache()  # Free each page
# Only 1 page in memory at a time = FAST, LOW MEMORY
```

---

## Summary

OmniDocs architecture is built on six principles:

1. **Unified API** - All extractors implement `.extract()`
2. **Type-Safe Configs** - Pydantic validates at creation time
3. **Multi-Backend Support** - Same model, different hardware
4. **Stateless Document** - Source data separate from analysis
5. **Separation of Init vs Extract** - Clear parameter ownership
6. **Backend Discoverability** - Imports reveal what's available

The result is a system that:
- **Catches errors early** (type validation)
- **Scales efficiently** (lazy loading, memory management)
- **Remains flexible** (swap models or backends easily)
- **Stays understandable** (clear data flow, no magic)

---

## Next Steps

- See [Document Model](./document-model.md) for details on loading and accessing documents
- See [Backend System](./backend-system.md) for understanding multi-backend support
- See [Config Pattern](./config-pattern.md) for designing extractors

