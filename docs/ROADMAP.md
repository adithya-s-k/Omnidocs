# Roadmap

**Status**: ✅ Done | 🚧 WIP | 🔜 Soon | ❌ N/A

---

## Text Extraction Models

| Model | Params | PyTorch | VLLM | MLX | API |
|-------|--------|---------|------|-----|-----|
| **Qwen3-VL** | 2-32B | ✅ | ✅ | ✅ | ✅ |
| **DotsOCR** | 1.7B | ✅ | ✅ | ❌ | ✅ |
| **Nanonets OCR2** | 3B | ✅ | ✅ | ✅ | ❌ |
| Granite-Docling | 258M | 🔜 | 🔜 | ❌ | ❌ |
| MinerU VL | 1.2B | 🔜 | 🔜 | 🔜 | ❌ |
| LightOnOCR-2 | 1B | 🔜 | 🔜 | ❌ | ❌ |
| Chandra | 9B | 🔜 | 🔜 | ❌ | ❌ |
| olmOCR-2 | 7B | 🔜 | 🔜 | ❌ | 🔜 |
| DeepSeek-OCR-2 | 3B | 🔜 | 🔜 | 🔜 | 🔜 |

---

## Layout Analysis Models

| Model | Params | PyTorch | VLLM | MLX | API |
|-------|--------|---------|------|-----|-----|
| **DocLayoutYOLO** | - | ✅ | ❌ | ❌ | ❌ |
| **RT-DETR** | - | ✅ | ❌ | ❌ | ❌ |
| **Qwen Layout** | 2-32B | ✅ | ✅ | ✅ | ✅ |
| Surya Layout | - | 🔜 | ❌ | ❌ | ❌ |
| Florence-2 | - | 🔜 | ❌ | ❌ | 🔜 |

---

## OCR Models

| Model | Params | PyTorch | VLLM | MLX | API |
|-------|--------|---------|------|-----|-----|
| **Tesseract** | - | ✅ | ❌ | ❌ | ❌ |
| **EasyOCR** | - | ✅ | ❌ | ❌ | ❌ |
| **PaddleOCR** | - | ✅ | ❌ | ❌ | ❌ |
| Surya OCR | - | 🔜 | ❌ | ❌ | ❌ |
| GOT-OCR2 | 700M | 🔜 | ❌ | ❌ | ❌ |

---

## Table Extraction Models

| Model | Params | PyTorch | VLLM | MLX | API |
|-------|--------|---------|------|-----|-----|
| **TableFormer** | - | ✅ | ❌ | ❌ | ❌ |
| TableTransformer | - | 🔜 | ❌ | ❌ | ❌ |
| Surya-Table | - | 🔜 | ❌ | ❌ | ❌ |

---

## Reading Order Models

| Model | Params | PyTorch | VLLM | MLX | API |
|-------|--------|---------|------|-----|-----|
| **Rule-based (R-tree)** | - | ✅ | ❌ | ❌ | ❌ |

---

## Tasks

| Task | Status |
|------|--------|
| Document Loading | ✅ |
| Text Extraction | ✅ |
| Layout Analysis | ✅ |
| OCR Extraction | ✅ |
| Table Extraction | ✅ |
| Reading Order | ✅ |
| Math Recognition | 🔜 |
| Structured Output | 🔜 |
| Chart Understanding | 🔜 |

---

## Infrastructure

| Component | Status |
|-----------|--------|
| Document class | ✅ |
| Pydantic configs | ✅ |
| Multi-backend | ✅ |
| Batch processing | ✅ |
| Modal deployment | ✅ |
| Testing framework | ✅ |
| Benchmarking suite | ✅ |

---

## Scripts Ready (Pending Integration)

These models have working scripts but aren't yet integrated into OmniDocs:

| Model | Task | Scripts |
|-------|------|---------|
| Granite Docling | Text | VLLM, HF, MLX |
| MinerU VL | Text | VLLM, MLX |

---

## Under Consideration

Models being evaluated for future integration:

### High Priority
| Model | Use Case | Why |
|-------|----------|-----|
| **Marker** | Full pipeline | Uses Surya, good tables |
| **Granite Vision 3.3** | Document understanding | IBM, good charts |
| **Surya** | OCR + Layout | Multi-language, modern |

### Specialized Use Cases
| Use Case | Models Under Review |
|----------|---------------------|
| Handwriting | TrOCR, Surya |
| Scientific Papers | Nougat, Marker |
| Asian Languages | PaddleOCR-VL (109 langs) |
| Edge/Mobile | Granite-Docling (258M) |
| Forms & Receipts | DeepSeek-OCR-2 |

### Framework Integration
| Framework | Status |
|-----------|--------|
| deepdoctection | 🔜 |
| Docling | 🔜 |
| Marker | 🔜 |

---

## Benchmarks Reference

### OlmOCR-Bench (Higher = Better)
| Model | Score | Params |
|-------|-------|--------|
| LightOnOCR-2 | 83.2 | 1B |
| Chandra | 83.1 | 9B |
| olmOCR-2 | 82.4 | 7B |
| DotsOCR | 79.1 | 1.7B |
| Nanonets OCR2 | ~78 | 3B |
| DeepSeek-OCR | 75.4 | 3B |

### Speed (Pages/Second on H100)
| Model | Speed |
|-------|-------|
| LightOnOCR-2 | 5.7 |
| PaddleOCR-VL | 3.3 |
| DeepSeek-OCR | 2.3 |

---

## Next Up

1. 🔜 **Granite Docling** - Integration from scripts
2. 🔜 **MinerU VL** - Integration from scripts
3. 🔜 **Math Recognition** - UniMERNet or Qwen
4. 🔜 **Surya** - Multi-language OCR + Layout

---

For detailed model specs, see [ROADMAP_DETAILED.md](ROADMAP_DETAILED.md).

**Last Updated**: February 2026
