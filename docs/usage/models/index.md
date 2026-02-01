# Models

All supported models and their configurations.

---

## Available Models

### Text Extraction

| Model | Speed | Backends | Status |
|-------|-------|----------|--------|
| [Qwen](qwen.md) | 2-3s/page | PyTorch, VLLM, MLX, API | ✅ Ready |
| [DotsOCR](dotsocr.md) | 3-5s/page | PyTorch, VLLM, API | ✅ Ready |

### Layout Analysis

| Model | Speed | Backends | Status |
|-------|-------|----------|--------|
| [DocLayoutYOLO](doclayout-yolo.md) | 0.1-0.2s/page | PyTorch | ✅ Ready |
| [RT-DETR](rtdetr.md) | 0.3-0.5s/page | PyTorch | ✅ Ready |
| [Qwen Layout](qwen.md#layout-analysis) | 2-3s/page | PyTorch, VLLM, MLX, API | ✅ Ready |

### OCR

| Model | Speed | Backends | Status |
|-------|-------|----------|--------|
| [Tesseract](tesseract.md) | 0.5-1s/page | CPU | ✅ Ready |
| [EasyOCR](easyocr.md) | 1-2s/page | PyTorch | ✅ Ready |
| [PaddleOCR](paddleocr.md) | 0.5-1s/page | PaddlePaddle | ✅ Ready |

---

## By Backend

| Backend | Models |
|---------|--------|
| **PyTorch** | Qwen, DotsOCR, DocLayoutYOLO, RT-DETR, EasyOCR |
| **VLLM** | Qwen, DotsOCR |
| **MLX** | Qwen |
| **API** | Qwen, DotsOCR |
| **CPU** | Tesseract, PaddleOCR |

---

## Upcoming Models

### Text Extraction
| Model | Parameters | Description | Status |
|-------|------------|-------------|--------|
| **Chandra** | 9B | High accuracy text extraction | 🔜 Soon |
| **LightOnOCR-2** | 1B | Fast text extraction | 🔜 Soon |
| **MinerU** | 1.2B | Layout-aware extraction | 🔜 Soon |
| **Granite-Docling** | 258M | Edge deployment | 🔜 Planned |
| **OlmOCR** | 7B | Tables and math | 🔜 Planned |

### Layout Analysis
| Model | Description | Status |
|-------|-------------|--------|
| **SuryaLayout** | Modern layout detection | 🔜 Soon |
| **VLMLayout** | API-based (GPT-4V, Gemini) | 🔜 Planned |

### OCR
| Model | Description | Status |
|-------|-------------|--------|
| **SuryaOCR** | Modern multilingual OCR | 🔜 Soon |
| **QwenOCR** | VLM-based OCR with coordinates | 🔜 Planned |

### New Tasks
| Task | Models | Status |
|------|--------|--------|
| **Table Extraction** | TableTransformer, Surya | 🔜 Soon |
| **Math Recognition** | UniMERNet, Qwen | 🔜 Soon |
| **Structured Output** | VLM (GPT-4V, Gemini) | 🔜 Planned |

See [Roadmap](../../ROADMAP.md) for full tracking.
