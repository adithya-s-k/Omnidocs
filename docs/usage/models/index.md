# Models

All supported models and their configurations.

---

## Available Models

### Text Extraction

| Model | Speed | Backends | Status |
|-------|-------|----------|--------|
| [Qwen](qwen.md) | 2-3s/page | PyTorch, VLLM, MLX, API | ✅ Ready |
| [DotsOCR](dotsocr.md) | 3-5s/page | PyTorch, VLLM, API | ✅ Ready |
| [Nanonets OCR2](nanonets.md) | 2-4s/page | PyTorch, VLLM, MLX | ✅ Ready |

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

### Table Extraction

| Model | Speed | Backends | Status |
|-------|-------|----------|--------|
| [TableFormer](tableformer.md) | 0.5-1s/table | PyTorch | ✅ Ready |

### Reading Order

| Model | Speed | Backends | Status |
|-------|-------|----------|--------|
| Rule-based | <0.1s/page | CPU | ✅ Ready |

---

## By Backend

| Backend | Models |
|---------|--------|
| **PyTorch** | Qwen, DotsOCR, Nanonets, DocLayoutYOLO, RT-DETR, EasyOCR, TableFormer |
| **VLLM** | Qwen, DotsOCR, Nanonets |
| **MLX** | Qwen, Nanonets |
| **API** | Qwen, DotsOCR |
| **CPU** | Tesseract, PaddleOCR, Rule-based Reading Order |

---

## Upcoming Models

### Text Extraction
| Model | Parameters | Description | Status |
|-------|------------|-------------|--------|
| **Granite Docling** | 258M | Edge deployment, fast inference | 🔜 Scripts ready |
| **MinerU VL** | 1.2B | Layout-aware extraction | 🔜 Scripts ready |
| **Chandra** | 9B | High accuracy text extraction | 🔜 Planned |

### Layout Analysis
| Model | Description | Status |
|-------|-------------|--------|
| **SuryaLayout** | Modern layout detection | 🔜 Planned |

### OCR
| Model | Description | Status |
|-------|-------------|--------|
| **SuryaOCR** | Modern multilingual OCR | 🔜 Planned |

### New Tasks
| Task | Models | Status |
|------|--------|--------|
| **Math Recognition** | UniMERNet, Qwen | 🔜 Planned |
| **Structured Output** | VLM (GPT-4V, Gemini) | 🔜 Planned |

See [Roadmap](../../ROADMAP.md) for full tracking.
