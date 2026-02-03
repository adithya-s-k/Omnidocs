<p align="center">
  <img src="assets/omnidocs_banner.png" alt="OmniDocs Banner" width="100%">
</p>

<p align="center">
  <strong>Unified Python toolkit for visual document processing</strong>
</p>

---

## Install

```bash
pip install omnidocs[pytorch]
```

## Extract Text in 4 Lines

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

doc = Document.from_pdf("document.pdf")
extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

---

## What OmniDocs Does

| Task | What You Get | Example Models |
|------|--------------|----------------|
| **Text Extraction** | Markdown/HTML from documents | Qwen3-VL, DotsOCR, Nanonets OCR2 |
| **Layout Analysis** | Bounding boxes for titles, tables, figures | DocLayoutYOLO, RT-DETR, Qwen Layout |
| **OCR** | Text + coordinates | Tesseract, EasyOCR, PaddleOCR |
| **Table Extraction** | Structured table data (rows, columns, cells) | TableFormer |
| **Reading Order** | Logical reading sequence | Rule-based R-tree |

---

## Core Design

```
Image → Extractor.extract() → Pydantic Output
```

- **One API**: `.extract()` for every task
- **Type-Safe**: Pydantic configs with IDE autocomplete
- **Multi-Backend**: PyTorch, VLLM, MLX, API
- **Stateless**: Document loads data, you manage results

---

## Choose Your Backend

| Backend | Install | Best For |
|---------|---------|----------|
| **PyTorch** | `pip install omnidocs[pytorch]` | Development, single GPU |
| **VLLM** | `pip install omnidocs[vllm]` | Production, high throughput |
| **MLX** | `pip install omnidocs[mlx]` | Apple Silicon (M1/M2/M3) |
| **API** | `pip install omnidocs[api]` | No GPU, cloud-based |

---

## What's Available

| Model | Task | PyTorch | VLLM | MLX | API |
|-------|------|---------|------|-----|-----|
| **Qwen3-VL** | Text, Layout | ✅ | ✅ | ✅ | ✅ |
| **DotsOCR** | Text | ✅ | ✅ | -- | ✅ |
| **Nanonets OCR2** | Text | ✅ | ✅ | ✅ | -- |
| **DocLayoutYOLO** | Layout | ✅ | -- | -- | -- |
| **RT-DETR** | Layout | ✅ | -- | -- | -- |
| **TableFormer** | Table | ✅ | -- | -- | -- |
| **Tesseract** | OCR | ✅ | -- | -- | -- |
| **EasyOCR** | OCR | ✅ | -- | -- | -- |
| **PaddleOCR** | OCR | ✅ | -- | -- | -- |
| **Rule-based** | Reading Order | ✅ | -- | -- | -- |

## Coming Soon

| Model | Task | Status |
|-------|------|--------|
| Granite Docling | Text | 🔜 Scripts ready |
| MinerU VL | Text | 🔜 Scripts ready |
| Surya | OCR, Layout | 🔜 Planned |

See [Roadmap](ROADMAP.md) for full tracking.

---

## Documentation

<div class="grid cards" markdown>

-   **[Getting Started](getting-started.md)**

    Install, configure, and run your first extraction

-   **[Concepts](concepts.md)**

    Architecture, configs, backends, and design decisions

-   **[Usage](usage/index.md)**

    Tasks, models, batch processing, and deployment

</div>

---

## Quick Reference

### Single-Backend Model (e.g., DocLayoutYOLO)
```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)
```

### Multi-Backend Model (e.g., Qwen)
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig  # or VLLMConfig, MLXConfig, APIConfig

extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))
result = extractor.extract(image, output_format="markdown")
```

---

## Links

- [GitHub](https://github.com/adithya-s-k/OmniDocs)
- [Issues](https://github.com/adithya-s-k/OmniDocs/issues)
- [Contributing](contributing/index.md)

---

<div style="text-align: center; margin-top: 2rem;">
    <a href="getting-started" class="md-button md-button--primary">Get Started</a>
</div>
