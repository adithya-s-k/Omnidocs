# Models

Complete reference for all models available in OmniDocs.

---

## Quick Navigation

### Text Extraction
| Model | Backends | Best For |
|-------|----------|----------|
| [Qwen3-VL](text-extraction/qwen.md) | PyTorch, VLLM, MLX, API | General purpose, multilingual |
| [DotsOCR](text-extraction/dotsocr.md) | PyTorch, VLLM | Layout-aware extraction |

### Layout Analysis
| Model | Backends | Best For |
|-------|----------|----------|
| [DocLayoutYOLO](layout-analysis/doclayout-yolo.md) | PyTorch | Fast detection |
| [Qwen Layout](layout-analysis/qwen-layout.md) | PyTorch, VLLM, MLX, API | Custom labels |

### OCR
| Model | Backends | Best For |
|-------|----------|----------|
| [Tesseract](ocr-extraction/tesseract.md) | CPU | Free, offline, 100+ languages |

---

## Comparison

| Model | Speed | Quality | Memory | Backends |
|-------|-------|---------|--------|----------|
| Qwen3-VL-2B | Fast | Good | 4GB | 4 |
| Qwen3-VL-8B | Medium | Excellent | 16GB | 4 |
| DotsOCR | Fast | Very Good | 8GB | 2 |
| DocLayoutYOLO | Very Fast | Good | 2-4GB | 1 |
| Tesseract | Slow (CPU) | Good | Minimal | 1 |

See [full comparison](comparison.md) for detailed benchmarks.

---

## Choosing a Model

### By Task

**Text Extraction**
- General: Qwen3-VL (any size)
- Layout-aware: DotsOCR
- Multilingual: Qwen3-VL

**Layout Detection**
- Speed: DocLayoutYOLO (0.1s/page)
- Custom labels: Qwen Layout
- Accuracy: Qwen Layout

**OCR (with coordinates)**
- Free/CPU: Tesseract
- Accuracy: Surya (coming soon)

### By Constraint

**Limited GPU (4GB)**
- Qwen3-VL-2B
- DocLayoutYOLO

**No GPU**
- Tesseract (CPU)
- Qwen3-VL (API backend)

**Apple Silicon**
- Qwen3-VL (MLX)
- Granite-Docling (coming soon)

**Production Scale**
- DotsOCR + VLLM
- Qwen3-VL + VLLM

---

## Quick Examples

### Text Extraction
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(device="cuda")
)
result = extractor.extract(image, output_format="markdown")
```

### Layout Detection
```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)
```

### OCR
```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractConfig

ocr = TesseractOCR(config=TesseractConfig(languages=["eng"]))
result = ocr.extract(image)
```

---

## Coming Soon

See [Roadmap](../roadmap.md) for upcoming models:
- LightOnOCR-2 (1B, fastest OCR)
- Chandra (9B, best accuracy)
- olmOCR-2 (7B, tables/math)
- MinerU2.5 (1.2B, MLX support)
- Granite-Docling (258M, edge deployment)
