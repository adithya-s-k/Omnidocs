# OmniDocs

![OmniDocs Banner](./assets/omnidocs_banner.png)

<p align="center">
  <b>Unified toolkit for visual document understanding</b><br>
  <a href="https://pypi.org/project/omnidocs/"><img src="https://img.shields.io/pypi/v/omnidocs.svg" alt="PyPI version"></a>
  <a href="https://github.com/adithya-s-k/OmniDocs/blob/main/LICENSE"><img src="https://img.shields.io/github/license/adithya-s-k/OmniDocs" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

**OmniDocs** is your all-in-one Python toolkit for extracting layout, tables, text, math, and OCR from PDFs and images, powered by classic libraries and state-of-the-art deep learning models. Build robust document workflows with a single, consistent API.

- Unified, production-ready API for all tasks
- Fast, GPU-accelerated, and easy to extend
- Type-safe with Pydantic models

---

## Quick Start

```python
from omnidocs import Document
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

# Load a PDF document
doc = Document.from_pdf("paper.pdf")

# Initialize layout detector
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

# Extract layout from a page
result = layout.extract(doc.get_page(0))

# Access detected elements
for box in result.bboxes:
    print(f"{box.label.value}: {box.confidence:.2f}")

# Visualize results
result.visualize(doc.get_page(0), output_path="layout.png")
```

### Text Extraction with VLM

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Load document
doc = Document.from_pdf("report.pdf")

# Initialize VLM text extractor
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
    )
)

# Extract as markdown
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

---

## Installation

Choose your preferred method:

- **PyPI (Recommended):**
  ```bash
  pip install omnidocs
  ```

- **uv (Fastest):**
  ```bash
  uv pip install omnidocs
  ```

- **From Source:**
  ```bash
  git clone https://github.com/adithya-s-k/Omnidocs.git
  cd Omnidocs
  uv sync
  ```

---

## Features

### Document Loading
- **Multiple Sources**: PDF files, URLs, bytes, images
- **Lazy Loading**: Pages rendered only when accessed
- **Memory Efficient**: Page caching with manual control

```python
from omnidocs import Document

# From file
doc = Document.from_pdf("file.pdf", page_range=(0, 9))

# From URL
doc = Document.from_url("https://arxiv.org/pdf/1706.03762")

# From images
doc = Document.from_images(["p1.png", "p2.png"])
```

### Layout Extraction
Detect document structure with multiple backends:

| Model | Description | Backend |
|-------|-------------|---------|
| **DocLayoutYOLO** | YOLO-based detector (fast, accurate) | PyTorch |
| **RT-DETR** | Transformer-based detector | PyTorch |
| **QwenLayoutDetector** | VLM-based with custom labels | PyTorch, VLLM, MLX, API |

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

# DocLayoutYOLO (fast, fixed labels)
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)

# Filter by label
tables = result.filter_by_label(LayoutLabel.TABLE)
figures = result.filter_by_label(LayoutLabel.FIGURE)
```

#### Custom Labels with Qwen-VL
Define your own document element categories:

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector, CustomLabel
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

# Initialize with PyTorch backend
detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
    )
)

# Define custom labels for your domain
custom_labels = [
    CustomLabel(name="chart", description="Bar charts, line charts, pie charts"),
    CustomLabel(name="code_block", description="Source code snippets"),
    CustomLabel(name="sidebar", description="Secondary content panels"),
]

# Detect with custom labels
result = detector.extract(image, custom_labels=custom_labels)
```

### Text Extraction
Convert document images to Markdown/HTML with VLM-powered extraction:

| Model | Description | Backend |
|-------|-------------|---------|
| **QwenTextExtractor** | High-accuracy VLM text extraction | PyTorch, VLLM, MLX, API |

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Initialize extractor
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
    )
)

# Extract text as markdown
result = extractor.extract(image, output_format="markdown")
print(result.content)  # Full markdown output

# Or as HTML
result = extractor.extract(image, output_format="html")
```

### Multi-Backend Support
All VLM-based extractors support multiple inference backends:

```python
# PyTorch (default, local GPU)
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
config = QwenTextPyTorchConfig(model="Qwen/Qwen3-VL-8B-Instruct", device="cuda")

# VLLM (high-throughput serving)
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig
config = QwenTextVLLMConfig(model="Qwen/Qwen3-VL-8B-Instruct", tensor_parallel_size=2)

# MLX (Apple Silicon)
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig
config = QwenTextMLXConfig(model="mlx-community/Qwen2.5-VL-7B-Instruct-8bit")

# API (OpenRouter, etc.)
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig
config = QwenTextAPIConfig(model="qwen/qwen3-vl-8b-instruct", api_key="...")
```

### OCR Extraction
Extract text with bounding boxes using multiple OCR engines:

| Model | Description | Backend |
|-------|-------------|---------|
| **TesseractOCR** | Google's open-source OCR (100+ languages) | System |
| **EasyOCR** | PyTorch-based OCR (80+ languages) | PyTorch |
| **PaddleOCR** | PaddlePaddle OCR (CJK optimized) | PaddlePaddle |

```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig
from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig

# Tesseract OCR
ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
result = ocr.extract(image)

# EasyOCR (GPU accelerated)
ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=True))
result = ocr.extract(image)

# PaddleOCR (excellent for Chinese/Japanese/Korean)
ocr = PaddleOCR(config=PaddleOCRConfig(lang="ch", device="gpu"))
result = ocr.extract(image)

# Access results
for block in result.text_blocks:
    print(f"'{block.text}' @ {block.bbox.to_list()} (conf: {block.confidence:.2f})")
```

### Coming Soon
- Table Extraction (TableTransformer, Camelot, Tabula)
- Math Extraction (UniMERNet, SuryaMath)
- Structured Output Extraction (Pydantic schemas)

---

## How It Works

**OmniDocs** organizes document processing tasks into modular components:

1. **Unified Interface**: Consistent `.extract()` method across all tasks
2. **Model Independence**: Switch between models effortlessly
3. **Pipeline Flexibility**: Combine components to create custom workflows
4. **Type Safety**: Pydantic models for all configs and outputs

## Architecture

```
Document (source data) → Task Extractor → Structured Output
                              ↓
                    Backend (PyTorch/VLLM/MLX/API)
```

See [Design Documents](docs/) for full architecture details.

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run fast tests only
uv run pytest tests/ -v -m "not slow"

# Lint and format
uv run ruff check .
uv run ruff format .

# Build docs
uv run mkdocs serve
```

---

## Roadmap

- [x] Layout Extraction (DocLayoutYOLO, RT-DETR)
- [x] VLM Layout Detection with Custom Labels (Qwen-VL)
- [x] Text Extraction to Markdown/HTML (Qwen-VL)
- [x] Multi-backend support (PyTorch, VLLM, MLX, API)
- [x] OCR Extraction module (Tesseract, EasyOCR, PaddleOCR)
- [ ] Table Extraction module
- [ ] Math Expression Extraction module
- [ ] Reading Order Detection
- [ ] Structured Output Extraction (Pydantic schemas)
- [ ] CLI support for batch processing

---

## Contributing

We welcome contributions to **OmniDocs**! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For a complete example of the contribution workflow, see:
- [Issue #18](https://github.com/adithya-s-k/Omnidocs/issues/18) - Feature request
- [PR #19](https://github.com/adithya-s-k/Omnidocs/pull/19) - Implementation

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Support

If you find **OmniDocs** helpful, please give us a star on GitHub!

- [Documentation](https://adithya-s-k.github.io/OmniDocs/)
- [Issues](https://github.com/adithya-s-k/OmniDocs/issues)
- [PyPI](https://pypi.org/project/omnidocs/)
- Email: adithyaskolavi@gmail.com
