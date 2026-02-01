# OmniDocs Concepts

> **Deep-dive documentation explaining the "why" behind OmniDocs architecture.**

This section explains the fundamental concepts and design decisions that power OmniDocs. Rather than "how to use," these documents explain "how it works" and "why it's designed this way."

## Documentation Files

### 1. [Architecture Overview](./architecture-overview.md) - ~2100 words

**What you'll learn**:
- Six core design principles that shape every decision
- Component architecture (Document, Extractors, Configs, Backends)
- Data flow from user code to inference result
- Why stateless documents matter
- Complete Qwen text extraction walkthrough

**Best for**: Understanding the big picture, how components fit together, design rationale

**Key Concepts**:
- Unified API via `.extract()` method
- Type-safe Pydantic configurations
- Multi-backend support with config-driven selection
- Stateless Document class
- Separation of init (model setup) vs extract (task parameters)
- Backend discoverability through imports

### 2. [Document Model](./document-model.md) - ~2400 words

**What you'll learn**:
- Why Document class is stateless (clean separation of concerns)
- How lazy page loading works (memory efficiency)
- All Document methods and their use cases
- DocumentMetadata structure
- Memory management and caching strategies
- Common processing patterns

**Best for**: Understanding document loading, page access, memory efficiency

**Key Concepts**:
- LazyPage wrapper for efficient rendering
- Multiple source formats (PDF, URL, bytes, images)
- Page caching and explicit memory control
- Iterator interface for large documents
- Page-level and full-text caching
- When to use Document vs direct image input

### 3. [Backend System](./backend-system.md) - ~2000 words

**What you'll learn**:
- Four backend options: PyTorch, VLLM, MLX, API
- How backend selection works at runtime
- Config classes drive backend choice via `isinstance` checks
- Lazy imports to avoid unnecessary dependencies
- Adding new backends (extension points)
- Trade-offs: speed, complexity, cost, privacy

**Best for**: Understanding multi-backend support, adding new backends, choosing the right backend

**Key Concepts**:
- PyTorch for development, VLLM for production, MLX for Apple Silicon, API for managed
- Type name detection in `_load_model()`
- Lazy imports prevent dependency bloat
- Each backend has separate config file
- Backend discoverability through importable configs
- Decision matrix for backend selection

### 4. [Config Pattern](./config-pattern.md) - ~2000 words

**What you'll learn**:
- Single-backend vs multi-backend models (how to tell the difference)
- Pydantic config class structure and validation
- Clear separation: init (config) vs extract (task parameters)
- What parameters go where and why
- Real examples from codebase (DocLayoutYOLO, QwenTextExtractor)
- Extending configs with custom validation

**Best for**: Understanding config design, building new extractors, parameter organization

**Key Concepts**:
- Single-backend: `{Model}Config` with `config=` parameter
- Multi-backend: `{Model}{Backend}Config` with `backend=` parameter
- Pydantic validation with Field() and validators
- Init parameters: model, device, quantization, hardware
- Extract parameters: format, custom prompts, task options
- Decision tree for where parameters belong

---

## How to Use These Docs

### I'm new to OmniDocs

1. Start with [Architecture Overview](./architecture-overview.md) to understand the system design
2. Read [Document Model](./document-model.md) to learn how to load and access documents
3. Skim [Config Pattern](./config-pattern.md) to understand parameter organization

### I want to add a new model

1. Read [Architecture Overview](./architecture-overview.md) for design principles
2. Check [Config Pattern](./config-pattern.md) for how to structure configs
3. Refer to [Backend System](./backend-system.md) if supporting multiple backends

### I'm deploying to production

1. Review [Backend System](./backend-system.md) for trade-offs
2. Check the comparison table to choose VLLM or API
3. See configuration options in [Config Pattern](./config-pattern.md)

### I'm optimizing memory usage

1. Read [Document Model](./document-model.md) for lazy loading and caching
2. Check memory management strategies for large documents
3. Use `iter_pages()` and `clear_cache()` patterns shown in examples

---

## Quick Reference

### Single-Backend Model

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

config = DocLayoutYOLOConfig(device="cuda")
extractor = DocLayoutYOLO(config=config)
result = extractor.extract(image)
```

**Pattern**: `{Model}Config` → `config=` parameter

### Multi-Backend Model

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(device="cuda")
extractor = QwenTextExtractor(backend=config)
result = extractor.extract(image, output_format="markdown")
```

**Pattern**: `{Model}{Backend}Config` → `backend=` parameter

### Config Validation

```python
from pydantic import BaseModel, Field, ConfigDict

class MyConfig(BaseModel):
    param1: str = Field(..., description="Required parameter")
    param2: int = Field(default=10, ge=1, le=100)
    param3: Literal["a", "b"] = Field(default="a")

    model_config = ConfigDict(extra="forbid")

# Validated at creation time
config = MyConfig(param1="value")
```

### Document Loading

```python
from omnidocs import Document

# Multiple source formats
doc = Document.from_pdf("file.pdf")
doc = Document.from_url("https://example.com/doc.pdf")
doc = Document.from_bytes(pdf_bytes)
doc = Document.from_image("page.png")
doc = Document.from_images(["p1.png", "p2.png"])

# Memory efficient processing
for page in doc.iter_pages():
    result = extractor.extract(page)
    doc.clear_cache()
```

### Backend Selection

```python
# PyTorch: Development
QwenTextPyTorchConfig(device="cuda", torch_dtype="bfloat16")

# VLLM: Production high-throughput
QwenTextVLLMConfig(tensor_parallel_size=4, gpu_memory_utilization=0.9)

# MLX: Apple Silicon
QwenTextMLXConfig(quantization="4bit")

# API: Managed services
QwenTextAPIConfig(api_key="sk-...", base_url="https://...")
```

---

## Key Design Decisions

| Decision | Benefit |
|----------|---------|
| Unified `.extract()` API | Easy to swap models, consistent interface |
| Pydantic configs | Type validation, IDE autocomplete, documentation |
| Multi-backend support | Choose infrastructure (GPU, Apple Silicon, API) |
| Stateless Document | Clear separation: source data vs. analysis |
| Config-driven backend | Backend selection explicit and discoverable |
| Lazy page loading | Memory efficient for large documents |
| Separation of init vs extract | Model setup vs. task parameters clear |

---

## Understanding the Trade-Offs

### Memory vs Speed

- **Render all pages at once** (fast access) vs **lazy render** (low memory)
  - Solution: Lazy render with caching (best of both)

### Setup Complexity vs Throughput

- **PyTorch** (simple setup, single GPU) vs **VLLM** (complex setup, high throughput)
  - Solution: Choose based on deployment scenario

### Privacy vs Convenience

- **Local inference** (private data, controlled) vs **API** (managed infrastructure, simple)
  - Solution: Support all options, user chooses

### Flexibility vs Discoverability

- **Magic strings** (flexible, unclear) vs **config types** (clear, discoverable)
  - Solution: Config types make available options obvious

---

## Common Patterns

### Pattern: Single Page Processing

```python
doc = Document.from_pdf("paper.pdf")
page = doc.get_page(0)
result = extractor.extract(page)
```

### Pattern: Multi-Page with Memory Control

```python
doc = Document.from_pdf("large_book.pdf")
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page)
    if i > 0:
        doc.clear_cache(i - 1)  # Free previous page
```

### Pattern: Conditional Backend Selection

```python
import os

backend = os.environ.get("BACKEND", "pytorch")
if backend == "vllm":
    config = QwenTextVLLMConfig(...)
else:
    config = QwenTextPyTorchConfig(...)

extractor = QwenTextExtractor(backend=config)
```

### Pattern: Batch Processing with Different Outputs

```python
extractor = QwenTextExtractor(backend=QwenTextVLLMConfig(...))

results = []
for page in doc.iter_pages():
    # Same extractor, different outputs
    markdown = extractor.extract(page, output_format="markdown")
    html = extractor.extract(page, output_format="html")
    results.append({"markdown": markdown, "html": html})
```

---

## Document Statistics

- **Total**: ~8,600 words across 4 documents
- **Architecture Overview**: ~2,100 words (system design, principles, flow)
- **Document Model**: ~2,400 words (lazy loading, memory, methods)
- **Backend System**: ~2,000 words (four backends, selection, trade-offs)
- **Config Pattern**: ~2,000 words (single vs multi-backend, validation)

---

## Relationship to Other Documentation

- **CLAUDE.md**: Development workflow and implementation guide (how to add features)
- **docs/concepts/**: Architecture and design (why it's built this way)
- **docs/guides/**: Practical usage (how to use OmniDocs)
- **docs/api/**: API reference (what each class does)

---

## Next Steps

- **For Usage**: See `docs/guides/` for practical examples
- **For API Reference**: See `docs/api/` for method signatures
- **For Development**: See `CLAUDE.md` for contribution workflow
- **For Theory**: Continue reading concept documents

---

**Last Updated**: February 2026
**Maintained By**: Adithya S Kolavi
**Status**: Complete v1.0

