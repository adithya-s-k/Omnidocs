# OmniDocs - Final Developer Experience Design

> **Status**: ✅ Design Complete - Ready for Implementation
> **Last Updated**: January 20, 2026
> **Version**: 2.0.0

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Core Design Principles](#core-design-principles)
3. [Backend Configuration](#backend-configuration)
4. [Architecture](#architecture)
5. [Task Distinctions](#task-distinctions)
6. [Layout Detection: Fixed vs Flexible Models](#layout-detection-fixed-vs-flexible-models)
7. [Document Loading](#document-loading)
8. [Usage Patterns](#usage-patterns)
9. [Complete Examples](#complete-examples)
10. [Import Reference](#import-reference)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

**OmniDocs** is a unified Python toolkit for visual document processing that provides a consistent API across multiple models and tasks.

### Core Philosophy

**Input Standardization**: `Image → Model → Pydantic Output`

All tasks follow this pattern regardless of:
- Which model is used (specialized vs VLM)
- Which backend runs inference (PyTorch, VLLM, MLX, API)
- Task complexity

### Supported Tasks

1. **Layout Analysis** - Detect document structure (headings, paragraphs, figures, tables)
2. **OCR Extraction** - Extract text **with bounding boxes** from images
3. **Text Extraction** - Export document to **Markdown/HTML** (10+ specialized VLM models)
4. **Table Extraction** - Extract tables and convert to structured formats
5. **Math Expression Recognition** - Convert math to LaTeX
6. **Reading Order Detection** - Order layout elements in reading sequence
7. **Image Captioning** - Caption figures and images
8. **Chart Understanding** - Convert charts to data + metadata
9. **Structured Output Extraction** - Extract structured data with schemas

---

## Core Design Principles

### ✅ Final Decisions

1. **Class-Based Imports** - No string-based factory pattern
   ```python
   from omnidocs.tasks.layout_analysis import DocLayoutYOLO  # ✅ YES
   layout = LayoutAnalysis(model="doclayout-yolo")           # ❌ NO
   ```

2. **Unified Method Name** - `.extract()` for ALL tasks (including layout)
   ```python
   layout_result = layout.extract(image)
   ocr_result = ocr.extract(image)
   text_result = text.extract(image, output_format="markdown")
   ```

3. **Model-Specific Configs** - Each model defines its own config classes
   ```python
   # Single-backend model
   from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

   layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

   # Multi-backend model - import config for desired backend
   from omnidocs.tasks.text_extraction import QwenTextExtractor
   from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig, QwenAPIConfig

   extractor = QwenTextExtractor(backend=QwenPyTorchConfig(model="Qwen/Qwen2-VL-7B"))
   ```

4. **Separation of Init vs Extract**:
   - **`__init__` (via config)** = Model initialization, download, verification
     - Which model to use
     - Which backend (PyTorch/VLLM/MLX/API)
     - Model loading settings (device, dtype, quantization)
     - Download and cache paths
   - **`.extract()`** = Runtime task parameters
     - Output format (markdown/html)
     - Custom prompts
     - Task-specific options (include_layout, custom_labels)
     - Per-call inference settings

5. **Stateless Document** - Document is source data only, does NOT store task results
   ```python
   doc = Document.from_pdf("file.pdf")  # Just loads the data
   result = layout.extract(doc.get_page(0))  # User manages results
   ```

6. **Discoverability** - Available backends = available config classes
   ```python
   # Multi-backend model - see what configs exist
   from omnidocs.tasks.text_extraction.qwen import (
       QwenPyTorchConfig,  # ✓ PyTorch supported
       QwenVLLMConfig,     # ✓ VLLM supported
       QwenMLXConfig,      # ✓ MLX supported
       QwenAPIConfig,      # ✓ API supported
   )

   # Single-backend model - only one config
   from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig
   ```

7. **Separation of Concerns**:
   - **Document Loading** = Internal (pypdfium2, PyMuPDF) - NOT separate extractors
   - **OCR Extraction** = Text + bounding boxes from images
   - **Text Extraction** = Markdown/HTML export (specialized VLMs)

---

## Backend Configuration

### Design: Model-Specific Config Classes

Each model has config classes specific to its supported backends. This provides:
- **IDE autocomplete** with only relevant parameters
- **Type safety** with Pydantic validation
- **Clear discoverability** of supported backends

### Single-Backend Models

Models that only support one backend (e.g., DocLayoutYOLO = PyTorch only):

```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

# Config has model-specific parameters
layout = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(
        device="cuda",
        model_path=None,        # Optional custom weights
        img_size=1024,          # Model-specific
    )
)

result = layout.extract(image)
```

### Multi-Backend Models

Models that support multiple backends (e.g., Qwen = PyTorch, VLLM, MLX, API):

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenMLXConfig,
    QwenAPIConfig,
)

# ─────────────────────────────────────
# Option 1: PyTorch (local HuggingFace)
# ─────────────────────────────────────
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda",
        trust_remote_code=True,
        torch_dtype="bfloat16",
    )
)

# ─────────────────────────────────────
# Option 2: VLLM (high-throughput)
# ─────────────────────────────────────
extractor = QwenTextExtractor(
    backend=QwenVLLMConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=False,
    )
)

# ─────────────────────────────────────
# Option 3: MLX (Apple Silicon)
# ─────────────────────────────────────
extractor = QwenTextExtractor(
    backend=QwenMLXConfig(
        model="Qwen/Qwen2-VL-7B-Instruct-MLX",
        quantization="4bit",
    )
)

# ─────────────────────────────────────
# Option 4: API (hosted or proxy)
# ─────────────────────────────────────
extractor = QwenTextExtractor(
    backend=QwenAPIConfig(
        model="qwen2-vl-7b",
        api_key="YOUR_API_KEY",
        base_url="https://api.provider.com/v1",  # Custom endpoint
        rate_limit=20,
        timeout=30,
    )
)

# Task parameters in .extract()
result = extractor.extract(
    image,
    output_format="markdown",
    include_layout=True,
    custom_prompt=None,
)
```

### Config Class Naming Convention

| Model Type | Config Naming | Example |
|------------|---------------|---------|
| Single-backend | `{Model}Config` | `DocLayoutYOLOConfig` |
| Multi-backend PyTorch | `{Model}PyTorchConfig` | `QwenPyTorchConfig` |
| Multi-backend VLLM | `{Model}VLLMConfig` | `QwenVLLMConfig` |
| Multi-backend MLX | `{Model}MLXConfig` | `QwenMLXConfig` |
| Multi-backend API | `{Model}APIConfig` | `QwenAPIConfig` |

### Model-Backend Support Matrix

| Model | PyTorch | VLLM | MLX | API |
|-------|---------|------|-----|-----|
| **Layout Analysis** |
| DocLayoutYOLO | ✅ | ❌ | ❌ | ❌ |
| RTDETRLayoutDetector | ✅ | ❌ | ❌ | ❌ |
| SuryaLayoutDetector | ✅ | ❌ | ❌ | ❌ |
| QwenLayoutDetector | ✅ | ✅ | ✅ | ✅ |
| VLMLayoutDetector | ❌ | ❌ | ❌ | ✅ |
| **Text Extraction** |
| QwenTextExtractor | ✅ | ✅ | ✅ | ✅ |
| DotsOCRTextExtractor | ✅ | ✅ | ✅ | ❌ |
| ChandraTextExtractor | ✅ | ✅ | ✅ | ❌ |
| GemmaTextExtractor | ✅ | ✅ | ✅ | ✅ |
| VLMTextExtractor | ❌ | ❌ | ❌ | ✅ |
| **OCR Extraction** |
| TesseractOCR | ✅ | ❌ | ❌ | ❌ |
| SuryaOCR | ✅ | ❌ | ❌ | ❌ |
| QwenOCR | ✅ | ✅ | ✅ | ✅ |

---

## Architecture

### System Overview

```
┌─────────────────────────────────────┐
│    Document Loading (Internal)      │
│  pypdfium2, PyMuPDF, pdfplumber     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         Task Layer                  │
│  Layout, OCR, Text, Table, Math...  │
│  (Each model has its own configs)   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Inference Layer                │
│  PyTorch, VLLM, MLX, LiteLLM        │
└─────────────────────────────────────┘
```

### Directory Structure

```
omnidocs/
├── __init__.py                 # Export Document
├── document.py                 # Document class (stateless)
│
├── tasks/
│   ├── layout_analysis/
│   │   ├── __init__.py         # Export models + configs
│   │   ├── base.py             # BaseLayoutExtractor
│   │   ├── models.py           # LayoutBox, LayoutOutput (Pydantic)
│   │   │
│   │   ├── doc_layout_yolo.py  # DocLayoutYOLO + DocLayoutYOLOConfig
│   │   ├── rtdetr.py           # RTDETRLayoutDetector + RTDETRConfig
│   │   ├── surya.py            # SuryaLayoutDetector + SuryaLayoutConfig
│   │   │
│   │   ├── qwen.py             # QwenLayoutDetector
│   │   └── qwen/               # Qwen backend configs
│   │       ├── __init__.py
│   │       ├── pytorch.py      # QwenPyTorchConfig
│   │       ├── vllm.py         # QwenVLLMConfig
│   │       ├── mlx.py          # QwenMLXConfig
│   │       └── api.py          # QwenAPIConfig
│   │
│   ├── ocr_extraction/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── models.py           # OCROutput, TextBlock (Pydantic)
│   │   │
│   │   ├── tesseract.py        # TesseractOCR + TesseractConfig
│   │   ├── paddle.py           # PaddleOCR + PaddleOCRConfig
│   │   ├── easyocr.py          # EasyOCR + EasyOCRConfig
│   │   ├── surya.py            # SuryaOCR + SuryaOCRConfig
│   │   │
│   │   ├── qwen.py             # QwenOCR
│   │   └── qwen/               # Qwen backend configs
│   │       └── ...
│   │
│   ├── text_extraction/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── models.py           # TextOutput (Pydantic)
│   │   │
│   │   ├── vlm_extractor.py    # VLMTextExtractor + VLMTextConfig (API-only)
│   │   │
│   │   ├── qwen.py             # QwenTextExtractor
│   │   ├── qwen/               # Qwen backend configs
│   │   │   ├── __init__.py
│   │   │   ├── pytorch.py      # QwenPyTorchConfig
│   │   │   ├── vllm.py         # QwenVLLMConfig
│   │   │   ├── mlx.py          # QwenMLXConfig
│   │   │   └── api.py          # QwenAPIConfig
│   │   │
│   │   ├── dotsocr.py          # DotsOCRTextExtractor
│   │   ├── dotsocr/            # DotsOCR backend configs
│   │   │   ├── pytorch.py      # DotsOCRPyTorchConfig
│   │   │   ├── vllm.py         # DotsOCRVLLMConfig
│   │   │   └── mlx.py          # DotsOCRMLXConfig (no API)
│   │   │
│   │   ├── chandra.py          # ChandraTextExtractor
│   │   ├── gemma.py            # GemmaTextExtractor
│   │   ├── granite.py          # GraniteDoclingOCR
│   │   ├── hunyuan.py          # HunyuanTextExtractor
│   │   ├── lighton.py          # LightOnOCRExtractor
│   │   ├── mineru.py           # MinerUOCRExtractor
│   │   ├── nanonuts.py         # NanonutsOCRExtractor
│   │   ├── olmo.py             # OlmOCRExtractor
│   │   └── paddle.py           # PaddleTextExtractor
│   │
│   ├── table_extraction/
│   │   ├── table_transformer.py
│   │   ├── surya_table.py
│   │   ├── qwen.py
│   │   └── vlm_extractor.py
│   │
│   ├── math_expression_extraction/
│   │   ├── unimernet.py
│   │   ├── qwen.py
│   │   └── vlm_extractor.py
│   │
│   └── structured_output_extraction/
│       └── vlm_extractor.py
│
├── inference/
│   ├── __init__.py
│   ├── base.py                 # Base backend classes
│   ├── pytorch.py              # PyTorch inference utilities
│   ├── vllm.py                 # VLLM inference utilities
│   ├── mlx.py                  # MLX inference utilities
│   └── api.py                  # LiteLLM/API utilities
│
├── workflows/
│   └── document_workflow.py
│
└── utils/
    ├── visualization.py
    └── export.py
```

---

## Task Distinctions

### ⚠️ Critical Clarifications

| Component | Role | Output | Examples |
|-----------|------|--------|----------|
| **Document Loading** | Load PDFs/images | PIL Images + Metadata | `Document.from_pdf()` |
| **OCR Extraction** | Text + bounding boxes | `OCROutput(text_blocks=[...])` | TesseractOCR, SuryaOCR, QwenOCR |
| **Text Extraction** | Markdown/HTML export | `TextOutput(content, format)` | QwenTextExtractor, DotsOCRTextExtractor |
| **Layout Analysis** | Detect structure | `LayoutOutput(bboxes=[...])` | DocLayoutYOLO, QwenLayoutDetector |

**Important**:
- **PyMuPDF, PDFPlumber, pypdfium2** are internal to Document - NOT separate extractors
- **OCR** returns text WITH bounding boxes
- **Text Extraction** returns formatted text (MD/HTML) WITHOUT bboxes

### OCR vs Text Extraction

```python
# ═══════════════════════════════════════════════════════════
# OCR Extraction - Text + Bounding Boxes
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.ocr_extraction import SuryaOCR, SuryaOCRConfig

ocr = SuryaOCR(config=SuryaOCRConfig(device="cuda"))
result = ocr.extract(image)

# Output: OCROutput
for text_block in result.text_blocks:
    print(f"Text: {text_block.text}")
    print(f"BBox: {text_block.bbox}")
    print(f"Confidence: {text_block.confidence}")


# ═══════════════════════════════════════════════════════════
# Text Extraction - Markdown/HTML Export
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig, QwenAPIConfig

# Local inference with PyTorch
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda",
    )
)

# OR API inference
extractor = QwenTextExtractor(
    backend=QwenAPIConfig(
        model="qwen2-vl-7b",
        api_key="YOUR_API_KEY",
        base_url="https://api.provider.com/v1",
    )
)

# Task parameters in .extract()
result = extractor.extract(
    image,
    output_format="markdown",      # "markdown" or "html"
    include_layout=True,           # Include layout information
    custom_prompt=None,            # Override default prompt
)

# Output: TextOutput
print(result.content)      # Full markdown/html
print(result.format)       # "markdown" or "html"
```

---

## Layout Detection: Fixed vs Flexible Models

### Model Categories

OmniDocs layout detectors fall into two categories:

| Category | Models | Label Support | Use Case |
|----------|--------|---------------|----------|
| **Fixed Labels** | DocLayoutYOLO, RT-DETR, Surya | Predefined only | Fast, specialized detection |
| **Flexible VLM** | Qwen, Florence-2, VLMLayoutDetector | Custom labels via prompting | Adaptable to any document type |

### Fixed Label Models

**Examples**: DocLayoutYOLO, RTDETRLayoutDetector, SuryaLayoutDetector

These models are trained on specific label sets and **cannot detect custom elements**.

```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)

# Fixed labels only:
# - title
# - text
# - list
# - table
# - figure
# - caption
# - formula
```

**Characteristics**:
- ✅ Fast inference
- ✅ Highly accurate on standard elements
- ❌ Cannot detect custom elements (code blocks, sidebars, etc.)
- ❌ Fixed label set (no flexibility)

### Flexible VLM Models

**Examples**: QwenLayoutDetector, Florence2LayoutDetector, VLMLayoutDetector

These models use vision-language prompting and **can detect ANY custom layout elements**.

#### Basic Usage (Default Labels)

```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector
from omnidocs.tasks.layout_analysis.qwen import QwenPyTorchConfig

layout = QwenLayoutDetector(
    backend=QwenPyTorchConfig(model="Qwen/Qwen2-VL-7B")
)

# Standard labels (same as fixed models)
result = layout.extract(image)
# Returns: title, text, table, figure, etc.
```

#### Custom Labels (Simple Strings)

```python
# Detect custom elements via simple strings
result = layout.extract(
    image,
    custom_labels=["code_block", "sidebar", "pull_quote", "diagram"]
)

for box in result.bboxes:
    print(f"{box.label}: {box.bbox}")
    # code_block: [x1, y1, x2, y2]
    # sidebar: [x1, y1, x2, y2]
```

#### Custom Labels (Structured)

For advanced use cases, use `CustomLabel` with metadata:

```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector, CustomLabel
from omnidocs.tasks.layout_analysis.qwen import QwenPyTorchConfig

layout = QwenLayoutDetector(
    backend=QwenPyTorchConfig(model="Qwen/Qwen2-VL-7B")
)

# Structured labels with metadata
result = layout.extract(
    image,
    custom_labels=[
        CustomLabel(
            name="code_block",
            description="Programming source code areas",
            detection_prompt="Regions with monospace text and syntax highlighting",
            color="#2ecc71",
        ),
        CustomLabel(
            name="sidebar",
            description="Sidebar or callout content",
            detection_prompt="Boxed regions with supplementary information",
            color="#3498db",
        ),
        CustomLabel(
            name="pull_quote",
            description="Highlighted quotations",
            detection_prompt="Large formatted quotes in different font/color",
            color="#e74c3c",
        ),
    ]
)

# Access metadata
for box in result.bboxes:
    print(f"Label: {box.label.name}")
    print(f"Description: {box.label.description}")
    print(f"Color: {box.label.color}")
```

### CustomLabel Type Definition

```python
from pydantic import BaseModel, Field
from typing import Optional

class CustomLabel(BaseModel):
    """Custom layout label definition for flexible VLM models."""

    name: str = Field(..., description="Label identifier (e.g., 'code_block')")

    description: Optional[str] = Field(
        default=None,
        description="Human-readable description"
    )

    detection_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt hint for detection"
    )

    color: Optional[str] = Field(
        default=None,
        description="Visualization color (hex or name)"
    )

    class Config:
        extra = "allow"  # Users can add custom fields
```

### Reusable Label Sets

```python
from omnidocs.tasks.layout_analysis import CustomLabel

class TechnicalDocLabels:
    """Reusable labels for technical documentation."""

    CODE_BLOCK = CustomLabel(
        name="code_block",
        description="Source code listings",
        color="#2ecc71"
    )

    API_REFERENCE = CustomLabel(
        name="api_reference",
        description="API documentation tables",
        color="#3498db"
    )

    DIAGRAM = CustomLabel(
        name="diagram",
        description="Architecture diagrams",
        color="#9b59b6"
    )

    @classmethod
    def all(cls):
        return [cls.CODE_BLOCK, cls.API_REFERENCE, cls.DIAGRAM]

# Use across projects
result = layout.extract(image, custom_labels=TechnicalDocLabels.all())
```

### User Extensions

Users can extend `CustomLabel` with custom fields:

```python
from omnidocs.tasks.layout_analysis import CustomLabel

class MyLabel(CustomLabel):
    priority: int = 1          # Custom field
    requires_ocr: bool = True  # Custom field

result = layout.extract(
    image,
    custom_labels=[
        MyLabel(
            name="important_section",
            description="High-priority content",
            priority=10,
            requires_ocr=True,
        )
    ]
)

# Access custom fields
for box in result.bboxes:
    print(f"Priority: {box.label.priority}")
    print(f"Requires OCR: {box.label.requires_ocr}")
```

### Comparison

| Feature | Fixed Models | Flexible VLMs |
|---------|--------------|---------------|
| Speed | ⚡ Fast | 🐢 Slower |
| Accuracy (standard) | ⭐⭐⭐ High | ⭐⭐ Good |
| Custom labels | ❌ No | ✅ Yes |
| String labels | ❌ No | ✅ Yes |
| Structured labels | ❌ No | ✅ Yes (CustomLabel) |
| Label metadata | ❌ No | ✅ Yes |
| Detection prompts | ❌ No | ✅ Yes |
| Use case | Standard docs | Any document type |

---

## Document Loading

### Design Decision: Stateless Document

**Document is SOURCE DATA only** - it does NOT store task results.

**Rationale**:
- Clean separation: Document = loaded PDF/images, Tasks = analysis results
- Memory efficient: Document doesn't grow with analysis
- User control: Users decide what to cache and how
- Flexibility: Works with any caching strategy

### Document API

```python
from omnidocs import Document

# Load from various sources
doc = Document.from_pdf("file.pdf", dpi=150, page_range=(0, 4))
doc = Document.from_url("https://example.com/doc.pdf")
doc = Document.from_bytes(pdf_bytes, filename="doc.pdf")
doc = Document.from_image("page.png")
doc = Document.from_images(["page1.png", "page2.png"])

# Properties (metadata only)
doc.page_count          # Number of pages
doc.metadata            # DocumentMetadata object
doc.pages               # List[Image.Image] - all pages
doc.text                # Full text (lazy extraction, cached)

# Access specific pages
page_img = doc.get_page(0)              # 0-indexed
page_text = doc.get_page_text(1)        # 1-indexed
page_size = doc.get_page_size(0)        # Dimensions

# Iterate (memory efficient)
for page_img in doc.iter_pages():
    process(page_img)

# Utilities
doc.save_images("output/", prefix="page", format="PNG")
doc.to_dict()
doc.clear_cache()       # Free cached page images
```

---

## Usage Patterns

### Pattern 1: Single-Backend Model (Simple)

```python
from omnidocs import Document
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

# Load document
doc = Document.from_pdf("paper.pdf")

# Single-backend model - just use config=
layout = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(
        device="cuda",
        img_size=1024,
    )
)

# Process
for i in range(doc.page_count):
    page = doc.get_page(i)
    result = layout.extract(page)

    for box in result.bboxes:
        print(f"{box.label}: {box.bbox}")
```

### Pattern 2: Multi-Backend Model (Flexible)

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import (
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenAPIConfig,
)

doc = Document.from_pdf("paper.pdf")

# Choose backend based on environment
import os

if os.getenv("USE_VLLM"):
    backend = QwenVLLMConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        tensor_parallel_size=2,
    )
elif os.getenv("USE_API"):
    backend = QwenAPIConfig(
        model="qwen2-vl-7b",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("API_BASE_URL"),
    )
else:
    backend = QwenPyTorchConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda",
    )

extractor = QwenTextExtractor(backend=backend)

# Process with task params in extract()
for i in range(doc.page_count):
    page = doc.get_page(i)
    result = extractor.extract(
        page,
        output_format="markdown",
        include_layout=True,
    )
    print(result.content)
```

### Pattern 3: API-Only Models (VLMTextExtractor)

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import VLMTextExtractor, VLMTextConfig

doc = Document.from_pdf("file.pdf")

# Generic VLM extractor for API-only models (Gemini, GPT-4, Claude)
extractor = VLMTextExtractor(
    config=VLMTextConfig(
        model="gemini-1.5-flash",      # or "gpt-4o", "claude-3-sonnet"
        api_key="YOUR_API_KEY",
        base_url=None,                  # Optional custom endpoint
        rate_limit=20,
    )
)

result = extractor.extract(
    doc.get_page(0),
    output_format="markdown",
    custom_prompt="Extract all text preserving structure.",
)
```

### Pattern 4: Mixed Pipeline

```python
from omnidocs import Document
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig
from omnidocs.tasks.table_extraction import TableTransformer, TableTransformerConfig

doc = Document.from_pdf("research_paper.pdf")

# Different models for different tasks
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

text = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        device="cuda",
    )
)

table = TableTransformer(config=TableTransformerConfig(device="cuda"))

# Process based on detected layout
page = doc.get_page(0)
layout_result = layout.extract(page)

for box in layout_result.bboxes:
    region = page.crop(box.bbox)

    if box.label == "text":
        result = text.extract(region, output_format="markdown")
    elif box.label == "table":
        result = table.extract(region)

    print(f"{box.label}: {result}")
```

---

## Complete Examples

### Example 1: Sanskrit Document Processing

```python
from omnidocs import Document
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenAPIConfig

# Load document
doc = Document.from_pdf(
    "Mayavada_khandanam.pdf",
    dpi=150,
    page_range=(0, 4)
)

# Setup extractors
layout = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(device="cuda", confidence=0.25)
)

text_extractor = QwenTextExtractor(
    backend=QwenAPIConfig(
        model="qwen2-vl-72b",
        api_key="YOUR_API_KEY",
        rate_limit=10,
    )
)

# Process each page
all_results = {}

for page_num in range(doc.page_count):
    page = doc.get_page(page_num)

    # Detect layout
    layout_result = layout.extract(page)

    # Extract text from text regions
    page_results = []
    for box in layout_result.bboxes:
        if box.label == "text":
            region = page.crop(box.bbox)
            text_result = text_extractor.extract(
                region,
                output_format="markdown",
                custom_prompt="Extract Sanskrit/Hindi text accurately.",
            )
            page_results.append({
                "bbox": box.bbox,
                "text": text_result.content,
            })

    all_results[f"page_{page_num}"] = page_results
```

### Example 2: High-Throughput with VLLM

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenVLLMConfig

# VLLM for batch processing
extractor = QwenTextExtractor(
    backend=QwenVLLMConfig(
        model="Qwen/Qwen2-VL-7B-Instruct",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
)

# Process many documents efficiently
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

for doc_path in documents:
    doc = Document.from_pdf(doc_path)

    for i in range(doc.page_count):
        result = extractor.extract(
            doc.get_page(i),
            output_format="markdown",
        )
        # Save result...
```

### Example 3: Apple Silicon with MLX

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenMLXConfig

# MLX for Apple Silicon
extractor = QwenTextExtractor(
    backend=QwenMLXConfig(
        model="Qwen/Qwen2-VL-7B-Instruct-MLX",
        quantization="4bit",
    )
)

doc = Document.from_pdf("document.pdf")

result = extractor.extract(
    doc.get_page(0),
    output_format="markdown",
)
```

### Example 4: Structured Output Extraction

```python
from omnidocs import Document
from omnidocs.tasks.structured_output_extraction import VLMStructuredExtractor, VLMStructuredConfig
from pydantic import BaseModel
from typing import List

# Define schema
class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[dict]

# Setup extractor
extractor = VLMStructuredExtractor(
    config=VLMStructuredConfig(
        model="gpt-4o",
        api_key="YOUR_API_KEY",
    )
)

doc = Document.from_pdf("invoice.pdf")

# Extract with schema
result = extractor.extract(
    doc.get_page(0),
    output_model=Invoice,
)

# Typed, validated output
print(f"Vendor: {result.data.vendor}")
print(f"Total: ${result.data.total_amount}")
```

---

## Import Reference

### Complete Import Guide

```python
# ═══════════════════════════════════════════════════════════
# Document Loading
# ═══════════════════════════════════════════════════════════
from omnidocs import Document


# ═══════════════════════════════════════════════════════════
# Layout Analysis
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.layout_analysis import (
    # Single-backend models (config included)
    DocLayoutYOLO, DocLayoutYOLOConfig,
    RTDETRLayoutDetector, RTDETRConfig,
    SuryaLayoutDetector, SuryaLayoutConfig,

    # Multi-backend model
    QwenLayoutDetector,

    # API-only
    VLMLayoutDetector, VLMLayoutConfig,

    # Custom label support
    CustomLabel,
)

# Qwen layout backend configs
from omnidocs.tasks.layout_analysis.qwen import (
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenMLXConfig,
    QwenAPIConfig,
)


# ═══════════════════════════════════════════════════════════
# OCR Extraction (text + bboxes)
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.ocr_extraction import (
    # Single-backend models
    TesseractOCR, TesseractConfig,
    PaddleOCR, PaddleOCRConfig,
    EasyOCR, EasyOCRConfig,
    SuryaOCR, SuryaOCRConfig,

    # Multi-backend model
    QwenOCR,

    # API-only
    VLMOCRExtractor, VLMOCRConfig,
)

# Qwen OCR backend configs
from omnidocs.tasks.ocr_extraction.qwen import (
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenMLXConfig,
    QwenAPIConfig,
)


# ═══════════════════════════════════════════════════════════
# Text Extraction (MD/HTML)
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.text_extraction import (
    # Multi-backend models
    QwenTextExtractor,
    DotsOCRTextExtractor,
    ChandraTextExtractor,
    GemmaTextExtractor,
    GraniteDoclingOCR,
    HunyuanTextExtractor,
    LightOnOCRExtractor,
    MinerUOCRExtractor,
    NanonutsOCRExtractor,
    OlmOCRExtractor,
    PaddleTextExtractor,

    # API-only
    VLMTextExtractor, VLMTextConfig,
)

# Qwen text extraction backend configs
from omnidocs.tasks.text_extraction.qwen import (
    QwenPyTorchConfig,
    QwenVLLMConfig,
    QwenMLXConfig,
    QwenAPIConfig,
)

# DotsOCR backend configs (no API)
from omnidocs.tasks.text_extraction.dotsocr import (
    DotsOCRPyTorchConfig,
    DotsOCRVLLMConfig,
    DotsOCRMLXConfig,
)


# ═══════════════════════════════════════════════════════════
# Table Extraction
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.table_extraction import (
    TableTransformer, TableTransformerConfig,
    SuryaTable, SuryaTableConfig,
    QwenTableExtractor,
    VLMTableExtractor, VLMTableConfig,
)


# ═══════════════════════════════════════════════════════════
# Math Expression Extraction
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.math_expression_extraction import (
    UniMERNet, UniMERNetConfig,
    QwenMathExtractor,
    VLMMathExtractor, VLMMathConfig,
)


# ═══════════════════════════════════════════════════════════
# Structured Output Extraction
# ═══════════════════════════════════════════════════════════
from omnidocs.tasks.structured_output_extraction import (
    VLMStructuredExtractor, VLMStructuredConfig,
)


# ═══════════════════════════════════════════════════════════
# Workflows (Optional)
# ═══════════════════════════════════════════════════════════
from omnidocs.workflows import DocumentWorkflow
```

---

## Implementation Roadmap

### Phase 1: Core Infrastructure

**Goals**: Base classes and config system

- [ ] Base extractor classes with `.extract()` method
- [ ] Pydantic config classes pattern
- [ ] Pydantic output models (LayoutOutput, OCROutput, TextOutput)
- [ ] Document class (stateless)

**Deliverables**:
- `omnidocs/document.py`
- `omnidocs/tasks/*/base.py`
- `omnidocs/tasks/*/models.py`

### Phase 2: Single-Backend Models

**Goals**: Implement models with single backend

- [ ] DocLayoutYOLO + DocLayoutYOLOConfig
- [ ] SuryaOCR + SuryaOCRConfig
- [ ] UniMERNet + UniMERNetConfig
- [ ] TableTransformer + TableTransformerConfig

### Phase 3: Multi-Backend Models

**Goals**: Implement models with multiple backends

- [ ] QwenTextExtractor + all backend configs
- [ ] DotsOCRTextExtractor + backend configs
- [ ] Backend-specific inference utilities

### Phase 4: API-Only Models

**Goals**: Generic VLM wrappers

- [ ] VLMTextExtractor for Gemini, GPT-4, Claude
- [ ] VLMStructuredExtractor with schema support
- [ ] LiteLLM integration

### Phase 5: Testing & Documentation

**Goals**: Comprehensive testing and docs

- [ ] Unit tests for all extractors
- [ ] Integration tests
- [ ] API documentation
- [ ] Tutorial notebooks

---

## Summary

### ✅ Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Import Pattern** | Class-based | Direct, explicit, type-safe |
| **Method Name** | `.extract()` for all | Consistent, predictable |
| **Config Style** | Model-specific | IDE autocomplete, clear discoverability |
| **Init vs Extract** | Config at init, task params at extract | Clear separation |
| **Document Design** | Stateless | Separation of concerns |
| **Backend Discovery** | Config classes exist = supported | Obvious, no guessing |

### Config Parameter Naming

| Model Type | Parameter | Example |
|------------|-----------|---------|
| Single-backend | `config=` | `DocLayoutYOLO(config=...)` |
| Multi-backend | `backend=` | `QwenTextExtractor(backend=...)` |
| API-only | `config=` | `VLMTextExtractor(config=...)` |

---

**Last Updated**: January 20, 2026
**Status**: ✅ Design Complete - Ready for Implementation
**Maintainer**: Adithya S Kolavi
