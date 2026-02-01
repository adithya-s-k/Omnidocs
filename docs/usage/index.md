# Usage

Everything you need to use OmniDocs in your projects.

---

## Tasks & Models

### [Text Extraction](tasks/text-extraction.md)
Convert documents to Markdown/HTML.

| Model | Speed | Backends |
|-------|-------|----------|
| [Qwen](models/qwen.md) | 2-3s/page | PyTorch, VLLM, MLX, API |
| [DotsOCR](models/dotsocr.md) | 3-5s/page | PyTorch, VLLM, API |

### [Layout Analysis](tasks/layout-analysis.md)
Detect structure (titles, tables, figures).

| Model | Speed | Labels |
|-------|-------|--------|
| [DocLayoutYOLO](models/doclayout-yolo.md) | 0.1-0.2s/page | Fixed (11) |
| [RT-DETR](models/rtdetr.md) | 0.3-0.5s/page | Fixed (11) |
| Qwen Layout | 2-3s/page | Custom |

### [OCR](tasks/ocr.md)
Extract text with coordinates.

| Model | Speed | Languages |
|-------|-------|-----------|
| [Tesseract](models/tesseract.md) | 0.5-1s/page | 100+ |
| [EasyOCR](models/easyocr.md) | 1-2s/page | 80+ |
| [PaddleOCR](models/paddleocr.md) | 0.5-1s/page | 80+ |

---

## Workflows

- [Batch Processing](batch-processing.md) - Process multiple documents
- [Deployment](deployment.md) - Deploy on Modal GPUs

---

## Upcoming

**Tasks:** Table Extraction, Math Recognition, Chart Understanding

**Models:** Chandra, LightOnOCR-2, MinerU, SuryaOCR, SuryaLayout

See [Roadmap](../ROADMAP.md) for full tracking.
