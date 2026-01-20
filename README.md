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

| Model | Description | Classes |
|-------|-------------|---------|
| **DocLayoutYOLO** | YOLO-based detector (fast, accurate) | 10 |
| **RT-DETR** | Transformer-based detector | 17+ |

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, RTDETRLayoutExtractor

# DocLayoutYOLO (fast)
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

# RT-DETR (more categories)
layout = RTDETRLayoutExtractor(config=RTDETRConfig(device="cuda"))

result = layout.extract(image)

# Filter by label
tables = result.filter_by_label(LayoutLabel.TABLE)
figures = result.filter_by_label(LayoutLabel.FIGURE)

# Get normalized coordinates (0-1024 range)
normalized = result.get_normalized_bboxes()
```

### Coming Soon
- OCR Extraction (PaddleOCR, Tesseract, EasyOCR, SuryaOCR)
- Text Extraction (PyMuPDF, PDFPlumber, Surya)
- Table Extraction (TableTransformer, Camelot, Tabula)
- Math Extraction (UniMERNet, SuryaMath)

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

- [ ] OCR Extraction module
- [ ] Text Extraction module
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
