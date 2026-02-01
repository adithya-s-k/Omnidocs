<p align="center">
  <img src="assets/omnidocs_banner.png" alt="OmniDocs Banner" width="100%">
</p>

<p align="center">
  <strong>Unified Python toolkit for visual document processing</strong>
</p>

<p align="center">
  Extract text, detect layouts, and run OCR on documents with a clean, type-safe API.
</p>

---

## Quick Install

```bash
pip install omnidocs[pytorch]
```

## Quick Start

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Load a PDF
doc = Document.from_pdf("document.pdf")

# Extract text as Markdown
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda"
    )
)

result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Unified API** | Single `.extract()` method for all tasks |
| **Type-Safe** | Pydantic configs with full IDE autocomplete |
| **Multi-Backend** | PyTorch, VLLM, MLX, API - choose what fits |
| **Production Ready** | GPU-accelerated, battle-tested |

---

## What Can You Do?

<div class="grid cards" markdown>

-   :material-text-box-outline: **Text Extraction**

    ---

    Convert documents to Markdown or HTML

    [:octicons-arrow-right-24: Guide](guides/text-extraction.md)

-   :material-view-grid-outline: **Layout Analysis**

    ---

    Detect structure: titles, tables, figures

    [:octicons-arrow-right-24: Guide](guides/layout-analysis.md)

-   :material-format-text: **OCR Extraction**

    ---

    Get text with bounding box coordinates

    [:octicons-arrow-right-24: Guide](guides/ocr-extraction.md)

-   :material-rocket-launch: **Batch Processing**

    ---

    Process hundreds of documents efficiently

    [:octicons-arrow-right-24: Guide](guides/batch-processing.md)

</div>

---

## Choose Your Path

### :material-speedometer: Just Get Started (5 min)

1. [Installation](getting-started/installation.md) - Set up your environment
2. [Quick Start](getting-started/quickstart.md) - Your first extraction
3. [Text Extraction Guide](guides/text-extraction.md) - Learn the basics

### :material-school: Understand the System (30 min)

1. [Architecture Overview](concepts/architecture-overview.md) - How it works
2. [Backend System](concepts/backend-system.md) - PyTorch vs VLLM vs MLX
3. [Config Pattern](concepts/config-pattern.md) - Configuration design

### :material-server: Deploy to Production (1 hour)

1. [Choosing Backends](getting-started/choosing-backends.md) - Pick the right one
2. [Modal Deployment](guides/deployment-modal.md) - Cloud GPU setup
3. [Batch Processing](guides/batch-processing.md) - Scale efficiently

---

## Supported Models

### Text Extraction
| Model | Backends | Best For |
|-------|----------|----------|
| [Qwen3-VL](models/text-extraction/qwen.md) | PyTorch, VLLM, MLX, API | General purpose |
| [DotsOCR](models/text-extraction/dotsocr.md) | PyTorch, VLLM | Layout-aware extraction |

### Layout Analysis
| Model | Backends | Best For |
|-------|----------|----------|
| [DocLayoutYOLO](models/layout-analysis/doclayout-yolo.md) | PyTorch | Fast detection |
| [Qwen Layout](models/layout-analysis/qwen-layout.md) | PyTorch, VLLM, MLX, API | Custom labels |

### OCR
| Model | Backends | Best For |
|-------|----------|----------|
| [Tesseract](models/ocr-extraction/tesseract.md) | CPU | Free, offline |

[:octicons-arrow-right-24: Full Model Comparison](models/comparison.md)

---

## Installation Options

```bash
# PyTorch (recommended for most users)
pip install omnidocs[pytorch]

# VLLM (high-throughput production)
pip install omnidocs[vllm]

# Apple Silicon
pip install omnidocs[mlx]

# API-only (no local GPU)
pip install omnidocs[api]

# Everything
pip install omnidocs[all]
```

---

## FAQ

??? question "Text Extraction vs OCR - What's the difference?"

    **Text Extraction** gives you formatted Markdown/HTML - use when you want document content.

    **OCR** gives you text WITH coordinates - use when you need to know WHERE text is located.

??? question "Which model should I use?"

    - **Text Extraction**: Start with Qwen, try DotsOCR for layout-aware
    - **Layout Detection**: DocLayoutYOLO for speed, Qwen for custom labels
    - **OCR**: Tesseract for free/CPU, PaddleOCR for Asian languages

??? question "Which backend should I use?"

    - **PyTorch**: Development, local GPU (recommended start)
    - **VLLM**: Production, high throughput
    - **MLX**: Apple Silicon only
    - **API**: No GPU needed

??? question "What are the hardware requirements?"

    - **PyTorch**: NVIDIA GPU (CUDA 12+) or CPU
    - **VLLM**: NVIDIA GPU required
    - **MLX**: Apple M1/M2/M3+
    - **API**: No GPU needed

---

## Performance

| Task | Model | Speed | Memory |
|------|-------|-------|--------|
| Text Extraction | Qwen3-VL-8B | 2-5 sec/page | 16GB |
| Layout Detection | DocLayoutYOLO | 0.5 sec/page | 4GB |
| OCR | Tesseract | 0.2 sec/page | 100MB |

*Benchmarks on A10G GPU with 1024x1280 images*

---

## Licensing

**OmniDocs**: Apache 2.0

**Models**: Each model has its own license - check the [Model Card](https://huggingface.co/models) on Hugging Face before use.

---

## Get Help

- [:material-book-open-variant: Documentation](getting-started/installation.md)
- [:material-github: GitHub Issues](https://github.com/adithya-s-k/OmniDocs/issues)
- [:material-account-group: Contributing](CONTRIBUTING.md)

---

<div style="text-align: center; margin-top: 2rem;">
    <a href="getting-started/quickstart/" class="md-button md-button--primary">Get Started →</a>
</div>
