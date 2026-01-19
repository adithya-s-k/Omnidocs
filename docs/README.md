# OmniDocs

> **Unified Python toolkit for visual document processing** - Think Transformers for document AI

[![PyPI version](https://badge.fury.io/py/omnidocs.svg)](https://badge.fury.io/py/omnidocs)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Status**: 🚧 v0.2 - Document Loading Complete | Task Extractors In Progress

## Overview

OmniDocs provides a consistent, type-safe API across multiple document processing models and tasks:
- ✅ **Document Loading** - Lazy-loaded PDFs and images with metadata (v0.2)
- 🚧 **Layout Analysis** - Coming soon
- 🚧 **OCR Extraction** - Coming soon
- 🚧 **Text Extraction** - Coming soon
- 🚧 **Table Extraction** - Coming soon

## Installation

```bash
pip install omnidocs
```

Or for development:

```bash
git clone https://github.com/adithya-s-k/OmniDocs.git
cd OmniDocs/Omnidocs
uv sync
```

## Quick Start

```python
from omnidocs import Document

# Load PDF with lazy rendering
doc = Document.from_pdf("paper.pdf", dpi=150)

# Access pages (rendered on demand)
page = doc.get_page(0)  # PIL Image
text = doc.get_page_text(1)  # 1-indexed

# Memory efficient iteration
for page in doc.iter_pages():
    # Process each page
    pass

# Full document text (cached)
full_text = doc.text

# Metadata
print(f"Pages: {doc.page_count}")
print(f"Size: {doc.metadata.file_size}")
```

## Features

### Document Loading ✅

- **Multiple Sources**: PDF files, URLs, bytes, images
- **Lazy Loading**: Pages rendered only when accessed
- **MIT/Apache Licensed**: pypdfium2 (Apache 2.0) + pdfplumber (MIT)
- **Type-Safe**: Pydantic models for configs and outputs
- **Memory Efficient**: Page caching with manual control

```python
# From file
doc = Document.from_pdf("file.pdf", page_range=(0, 9))

# From URL
doc = Document.from_url("https://arxiv.org/pdf/1706.03762")

# From bytes
doc = Document.from_bytes(pdf_bytes)

# From images
doc = Document.from_image("page.png")
doc = Document.from_images(["p1.png", "p2.png"])
```

## Architecture

OmniDocs follows a clean, stateless design:
- **Document** = Source data only (doesn't store task results)
- **Tasks** = Analysis operations (layout, OCR, text extraction)
- **Backends** = Inference engines (PyTorch, VLLM, MLX, API)

See Design Documents for full architecture details.

## Development

Run tests:

```bash
uv run pytest tests/ -v
```

Run fast tests only:

```bash
uv run pytest tests/ -v -m "not slow"
```

Build docs:

```bash
uv run mkdocs serve
```

## Requirements

- Python 3.10 - 3.11
- Dependencies managed with [uv](https://github.com/astral-sh/uv)

## License

Apache 2.0 - See [LICENSE](LICENSE.md) for details

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Links

- 📚 [Documentation](https://adithya-s-k.github.io/OmniDocs/)
- 🐛 [Issues](https://github.com/adithya-s-k/OmniDocs/issues)
- 📦 [PyPI](https://pypi.org/project/omnidocs/)
- 📝 [Changelog](https://github.com/adithya-s-k/OmniDocs/releases)
