# OmniDocs Development Roadmap


## 📦 Target Model Support

> **Research Date**: January 2026
> **Status**: Comprehensive model research completed
> **Models Ordered By**: Release date (newest first within each size category)

---

## 🎯 Quick Reference: Model Capabilities & Backend Support

### Comprehensive Model Comparison Table

| Model | Size | PyTorch | VLLM | MLX | OpenAI API | Tasks | Release |
|-------|------|---------|------|-----|------------|-------|---------|
| **Granite-Docling-258M** | 258M | ✅ | ⚠️ | ✅ | ❌ | T, L, Tab, F | Dec 2024 |
| **GOT-OCR2.0** | 700M | ✅ | ❌ | ❌ | ❌ | T, O, F, Tab | Sep 2024 |
| **PaddleOCR-VL** | 900M | ✅ | ⚠️ | ❌ | ❌ | T, L, O, Tab, F | Oct 2025 |
| **LightOnOCR-1B** | 1B | ✅ | ❌ | ❌ | ❌ | T, O | Oct 2025 |
| **LightOnOCR-2-1B** | 1B | ✅ | ✅ | ❌ | ❌ | T, O | Jan 2025 |
| **LightOnOCR-2-1B-bbox** | 1B | ✅ | ✅ | ❌ | ❌ | T, L, O | Jan 2025 |
| **MinerU2.5-2509-1.2B** | 1.2B | ✅ | ✅ | ✅ | ❌ | T, L, Tab, F, O | Sep 2024 |
| **dots.ocr** | 1.7B | ✅ | ✅ | ❌ | ❌ | T, L, Tab, F, O | Dec 2024 |
| **Qwen3-VL-2B** | 2B | ✅ | ✅ | ✅ | ✅ | T, L, S, O, Tab | Oct 2025 |
| **DeepSeek-OCR** | 3B | ✅ | ✅ | ✅ | ✅ | T, O, Tab | Oct 2024 |
| **Qwen2.5-VL-3B** | 3B | ✅ | ✅ | ✅ | ✅ | T, L, S, O | 2024 |
| **Nanonets-OCR2-3B** | 3B | ✅ | ✅ | ✅ | ❌ | T, F, O | 2024 |
| **Qwen3-VL-4B** | 4B | ✅ | ✅ | ✅ | ❌ | T, L, S, O, Tab | Oct 2025 |
| **Gemma-3-4B-IT** | 4B | ✅ | ❌ | ❌ | ✅ | T, S, O | 2025 |
| **olmOCR-2-7B** | 7B | ✅ | ✅ | ❌ | ✅ | T, O, Tab, F | Oct 2025 |
| **Qwen2.5-VL-7B** | 7B | ✅ | ✅ | ✅ | ✅ | T, L, S, O, Tab | 2024 |
| **Qwen3-VL-8B** | 8B | ✅ | ✅ | ✅ | ✅ | T, L, S, O, Tab, F | Oct 2025 |
| **Chandra** | 9B | ✅ | ✅ | ❌ | ❌ | T, L, O, Tab, F | 2024 |
| **Qwen3-VL-32B** | 32B | ✅ | ✅ | ✅ | ✅ | T, L, S, O, Tab, F | Oct 2025 |
| **Qwen2.5-VL-32B** | 32B | ✅ | ✅ | ✅ | ✅ | T, L, S, O, Tab | 2024 |

**Legend:**
- **Tasks**: T=Text Extract, L=Layout, O=OCR, S=Structured, Tab=Table, F=Formula
- **✅** = Fully supported | **⚠️** = Limited/Partial support | **❌** = Not supported

---

### Backend Details

#### PyTorch Support
- **All models** support PyTorch via HuggingFace Transformers
- Primary development backend for all models
- Requirements: `transformers>=4.46`, `torch>=2.0`

#### VLLM Support (High-Throughput Production)
**Fully Supported** (✅):
- Qwen3-VL Series (vllm>=0.11.0)
- Qwen2.5-VL Series
- DeepSeek-OCR (official upstream)
- dots.ocr (recommended, vllm>=0.9.1)
- MinerU2.5
- olmOCR-2 (via olmOCR toolkit)
- Chandra
- LightOnOCR-2-1B (vllm>=0.11.1)
- Nanonets-OCR2-3B

**Limited Support** (⚠️):
- Granite-Docling-258M (untied weights required)
- PaddleOCR-VL (possible but not officially confirmed)

**Not Supported** (❌):
- GOT-OCR2.0
- Gemma-3-4B-IT
- LightOnOCR-1B (legacy)

#### MLX Support (Apple Silicon M1/M2/M3+)
**Fully Supported** via mlx-community (✅):
- **Qwen3-VL Series** - [Collection](https://huggingface.co/collections/mlx-community/qwen3-vl)
  - 2B, 4B, 8B, 32B (4-bit, 8-bit variants)
- **Qwen2.5-VL Series** - [Collection](https://huggingface.co/collections/mlx-community/qwen25-vl)
  - 3B, 7B, 32B, 72B (4-bit, 8-bit variants)
- **DeepSeek-OCR** - [4-bit](https://huggingface.co/mlx-community/DeepSeek-OCR-4bit), [8-bit](https://huggingface.co/mlx-community/DeepSeek-OCR-8bit)
- **Granite-Docling-258M** - [Official MLX](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
- **MinerU2.5** - [bf16](https://huggingface.co/mlx-community/MinerU2.5-2509-1.2B-bf16)
- **Nanonets-OCR2-3B** - [4-bit](https://huggingface.co/mlx-community/Nanonets-OCR2-3B-4bit)

**Usage**:
```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --prompt "Extract text from this document" --image doc.png
```

#### OpenAI-Compatible API Providers

**OpenRouter** ([openrouter.ai](https://openrouter.ai)):
- ✅ Qwen3-VL-235B-A22B ($0.45/$3.50 per M tokens)
- ✅ Qwen3-VL-30B-A3B
- ✅ Qwen2.5-VL-3B (SOTA visual understanding)
- ✅ Qwen2.5-VL-32B (structured outputs, math)
- ✅ Qwen2.5-VL-72B (best overall)

**Novita AI** ([novita.ai](https://novita.ai)):
- ✅ DeepSeek-OCR ([Model Page](https://novita.ai/models/model-detail/deepseek-deepseek-ocr))
- ✅ Qwen2.5-VL-72B (OCR + scientific reasoning)
- ✅ Qwen3-VL-8B ($0.08/$0.50 per M tokens)

**Together AI** ([together.ai](https://www.together.ai/models)):
- ✅ Various vision-language models
- ✅ Lightweight models with multilingual support

**Replicate** ([replicate.com](https://replicate.com/collections/vision-models)):
- ✅ Vision models collection
- ✅ Pay-per-use inference

**Others**:
- DeepInfra: olmOCR-2-7B
- Parasail: olmOCR-2-7B
- Cirrascale: olmOCR-2-7B

**API Integration Example**:
```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenAPIConfig

# OpenRouter
extractor = QwenTextExtractor(
    backend=QwenAPIConfig(
        model="qwen/qwen3-vl-8b-instruct",
        api_key="YOUR_OPENROUTER_KEY",
        base_url="https://openrouter.ai/api/v1"
    )
)

# Novita AI
extractor = QwenTextExtractor(
    backend=QwenAPIConfig(
        model="novita/qwen3-vl-8b-instruct",
        api_key="YOUR_NOVITA_KEY",
        base_url="https://api.novita.ai/v3/openai"
    )
)
```

---

### Task Capability Matrix

| Task | Description | Model Count | Top Models |
|------|-------------|-------------|------------|
| **Text Extract** (T) | Document → Markdown/HTML | 18 | LightOnOCR-2, Chandra, Qwen3-VL-8B |
| **Layout** (L) | Structure detection with bboxes | 8 | Qwen3-VL-8B, Chandra, MinerU2.5 |
| **OCR** (O) | Text + bbox coordinates | 15 | LightOnOCR-2, olmOCR-2, Chandra |
| **Structured** (S) | Schema-based extraction | 5 | Qwen3-VL (all), Qwen2.5-VL (all), Gemma-3 |
| **Table** (Tab) | Table detection/extraction | 12 | Qwen3-VL-8B, DeepSeek-OCR, olmOCR-2 |
| **Formula** (F) | Math expression recognition | 8 | Nanonets-OCR2, Qwen3-VL-8B, GOT-OCR2.0 |

---

## Model Overview by Task Capability

### Task Categories

| Task | Description | Model Count |
|------|-------------|-------------|
| **text_extract** | Document to Markdown/HTML conversion | 18 |
| **layout** | Document structure detection with bounding boxes | 8 |
| **ocr** | Text extraction with bbox coordinates | 6 |
| **structured** | Schema-based data extraction | 5 |
| **table** | Table detection and extraction | 4 |
| **formula** | Mathematical expression recognition | 3 |

---

## 🎯 Core Models (By Size & Release Date)

### Ultra-Compact Models (<1B Parameters)

#### 1. IBM Granite-Docling-258M
**Released**: December 2024 | **Parameters**: 258M | **License**: Apache 2.0

**HuggingFace**: [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M)

**Description**: Ultra-compact vision-language model (VLM) for converting documents to machine-readable formats while fully preserving layout, tables, equations, and lists. Built on Idefics3 architecture with siglip2-base-patch16-512 vision encoder and Granite 165M LLM.

**Key Features**:
- End-to-end document understanding at 258M parameters
- Handles inline/floating math, code, table structure
- Rivals systems several times its size
- Extremely cost-effective

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ MLX (Apple Silicon) - [ibm-granite/granite-docling-258M-mlx](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
- ✅ WebGPU - [Demo Space](https://huggingface.co/spaces/ibm-granite/granite-docling-258M-WebGPU)

**Integration**:
```bash
pip install docling  # Automatically downloads model
```

**Dependencies**: `transformers`, `torch`, `pillow`, `docling`

**Tasks**: `text_extract`, `layout`, `table`, `formula`

**Links**:
- [Model Card](https://huggingface.co/ibm-granite/granite-docling-258M)
- [MLX Version](https://huggingface.co/ibm-granite/granite-docling-258M-mlx)
- [Demo Space](https://huggingface.co/spaces/ibm-granite/granite-docling-258m-demo)
- [Official Docs](https://www.ibm.com/granite/docs/models/docling)
- [Collection](https://huggingface.co/collections/ibm-granite/granite-docling)

---

#### 2. stepfun-ai GOT-OCR2.0
**Released**: September 2024 | **Parameters**: 700M | **License**: Apache 2.0

**HuggingFace**: [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0)

**Description**: General OCR Theory model for multilingual OCR on plain documents, scene text, formatted documents, tables, charts, mathematical formulas, geometric shapes, molecular formulas, and sheet music.

**Key Features**:
- Interactive OCR with region-specific recognition (coordinate or color-based)
- Plain text OCR + formatted text OCR (markdown, LaTeX)
- Multi-page document processing
- Wide range of specialized content types

**Model Variations**:
- **stepfun-ai/GOT-OCR2_0** - Original with custom code
- **stepfun-ai/GOT-OCR-2.0-hf** - HuggingFace-native transformers integration

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ Custom inference pipeline

**Dependencies**: `transformers`, `torch`, `pillow`

**Tasks**: `text_extract`, `ocr`, `formula`, `table`

**Links**:
- [Model Card](https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- [HF-Native Version](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)

---

### Compact Models (1-2B Parameters)

#### 3. rednote-hilab dots.ocr
**Released**: December 2024 | **Parameters**: 1.7B | **License**: MIT

**HuggingFace**: [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)

**Description**: Multilingual documents parsing model based on 1.7B LLM with SOTA performance. Provides faster inference than many high-performing models based on larger foundations.

**Key Features**:
- Task switching via prompt alteration only
- Competitive detection vs traditional models (DocLayout-YOLO)
- Built-in VLLM support for high throughput
- Released with paper [arXiv:2512.02498](https://huggingface.co/papers/2512.02498)

**Model Variations**:
- **rednote-hilab/dots.ocr** - Full model
- **rednote-hilab/dots.ocr.base** - Base variant

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM (Recommended for production) - vLLM 0.9.1+

**Dependencies**: `transformers`, `torch`, `vllm>=0.9.1` (recommended)

**Tasks**: `text_extract`, `layout`, `table`, `formula`, `ocr`

**Links**:
- [Model Card](https://huggingface.co/rednote-hilab/dots.ocr)
- [GitHub](https://github.com/rednote-hilab/dots.ocr)
- [Live Demo](https://dotsocr.xiaohongshu.com)
- [Paper](https://huggingface.co/papers/2512.02498)
- [Collection](https://huggingface.co/collections/rednote-hilab/dotsocr)

---

#### 4. PaddlePaddle PaddleOCR-VL
**Released**: October 2025 | **Parameters**: 900M | **License**: Apache 2.0

**HuggingFace**: [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)

**Description**: Ultra-compact multilingual documents parsing VLM with SOTA performance. Integrates NaViT-style dynamic resolution visual encoder with ERNIE-4.5-0.3B language model.

**Key Features**:
- Supports 109 languages
- Excels in recognizing complex elements (text, tables, formulas, charts)
- Minimal resource consumption
- Fast inference speeds
- SOTA in page-level parsing and element-level recognition

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers - officially integrated)
- ✅ PaddlePaddle framework

**Dependencies**: `transformers`, `torch`, `paddlepaddle`

**Tasks**: `text_extract`, `layout`, `ocr`, `table`, `formula`

**Links**:
- [Model Card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [Online Demo](https://huggingface.co/spaces/PaddlePaddle/PaddleOCR-VL_Online_Demo)
- [Collection](https://huggingface.co/collections/PaddlePaddle/paddleocr-vl)
- [Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/paddleocr_vl)
- [GitHub - PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

#### 5. LightOn AI LightOnOCR Series
**Released**: January 2025 (v2), October 2025 (v1) | **Parameters**: 1B | **License**: Apache 2.0

**HuggingFace Models**:
- [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) - **Recommended** for OCR
- [lightonai/LightOnOCR-2-1B-bbox](https://huggingface.co/lightonai/LightOnOCR-2-1B-bbox) - Best localization
- [lightonai/LightOnOCR-2-1B-bbox-soup](https://huggingface.co/lightonai/LightOnOCR-2-1B-bbox-soup) - Balanced OCR + bbox
- [lightonai/LightOnOCR-1B-1025](https://huggingface.co/lightonai/LightOnOCR-1B-1025) - Legacy v1

**Description**: Compact, end-to-end vision-language model for OCR and document understanding. State-of-the-art accuracy in its weight class while being several times faster than larger VLMs.

**Key Features**:
- **LightOnOCR-2-1B**: SOTA on OlmOCR-Bench (83.2 ± 0.9), outperforms Chandra-9B
- **Performance**: 3.3× faster than Chandra, 1.7× faster than OlmOCR, 5× faster than dots.ocr
- **Variants**: OCR-only, bbox-capable (figure/image localization), and balanced checkpoints
- Paper: [arXiv:2601.14251](https://arxiv.org/html/2601.14251)

**Model Comparison**:
| Model | Use Case | Bbox Support |
|-------|----------|--------------|
| LightOnOCR-2-1B | Default for PDF→Text/Markdown | ❌ |
| LightOnOCR-2-1B-bbox | Best localization of figures/images | ✅ Best |
| LightOnOCR-2-1B-bbox-soup | Balanced OCR + localization | ✅ Balanced |

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers - upstream support)
- ⚠️ Requires transformers from source for v2 (not yet in stable release)

**Quantized Versions**:
- [GGUF format](https://huggingface.co/Mungert/LightOnOCR-1B-1025-GGUF)

**Dependencies**: `transformers>=4.48` (from source for v2), `torch`, `pillow`

**Tasks**: `text_extract`, `ocr`, `layout` (bbox variants only)

**Links**:
- [LightOnOCR-2 Blog](https://huggingface.co/blog/lightonai/lightonocr-2)
- [LightOnOCR-1 Blog](https://huggingface.co/blog/lightonai/lightonocr)
- [Demo Space](https://huggingface.co/spaces/lightonai/LightOnOCR-2-1B-Demo)
- [Paper (arXiv)](https://arxiv.org/html/2601.14251)
- [Organization](https://huggingface.co/lightonai)

---

#### 6. opendatalab MinerU2.5
**Released**: September 2024 | **Parameters**: 1.2B | **License**: Apache 2.0

**HuggingFace**: [opendatalab/MinerU2.5-2509-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)

**Description**: Decoupled vision-language model for efficient high-resolution document parsing with state-of-the-art accuracy and low computational overhead.

**Key Features**:
- Two-stage parsing: global layout analysis on downsampled images → fine-grained content recognition on native-resolution crops
- Outperforms Gemini-2.5 Pro, Qwen2.5-VL-72B, GPT-4o, MonkeyOCR, dots.ocr, PP-StructureV3
- Large-scale diverse data engine for pretraining/fine-tuning
- New performance records in text, formula, table recognition, and reading order

**Model Variations**:
- **opendatalab/MinerU2.5-2509-1.2B** - Official model
- **mlx-community/MinerU2.5-2509-1.2B-bf16** - MLX for Apple Silicon
- **Mungert/MinerU2.5-2509-1.2B-GGUF** - GGUF quantized

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM (with OpenAI API specs)
- ✅ MLX (Apple Silicon)

**Dependencies**: `transformers`, `torch`, `vllm` (optional)

**Tasks**: `text_extract`, `layout`, `table`, `formula`, `ocr`

**Links**:
- [Model Card](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)
- [Paper (arXiv:2509.22186)](https://arxiv.org/html/2509.22186v2)
- [MLX Version](https://huggingface.co/mlx-community/MinerU2.5-2509-1.2B-bf16)
- [GGUF Version](https://huggingface.co/Mungert/MinerU2.5-2509-1.2B-GGUF)

---

### Small Models (2-4B Parameters)

#### 7. Qwen3-VL-2B-Instruct
**Released**: October 2025 | **Parameters**: 2B | **License**: Apache 2.0

**HuggingFace**: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

**Description**: Multimodal LLM from Alibaba Cloud's Qwen team with comprehensive upgrades: superior text understanding/generation, deeper visual perception/reasoning, extended context, and stronger agent interaction.

**Key Features**:
- Dense and MoE architectures that scale from edge to cloud
- Instruct and reasoning-enhanced "Thinking" editions
- Enhanced spatial and video dynamics comprehension
- Part of Qwen3-VL multimodal retrieval framework (arXiv:2601.04720, 2026)

**Model Variations**:
- **Qwen/Qwen3-VL-2B-Instruct** - Instruction-tuned
- **Qwen/Qwen3-VL-2B-Thinking** - Reasoning-enhanced
- **Qwen/Qwen3-VL-2B-Instruct-GGUF** - Quantized GGUF

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM
- ✅ MLX (via mlx-community)
- ✅ API (via cloud providers)

**Dependencies**: `transformers>=4.46`, `torch`, `qwen-vl-utils`

**Tasks**: `text_extract`, `layout`, `structured`, `ocr`, `table`

**Links**:
- [Model Card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Collection](https://huggingface.co/collections/Qwen/qwen3-vl)
- [GGUF Version](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)

---

#### 8. DeepSeek-OCR
**Released**: October 2024 | **Parameters**: ~3B | **License**: MIT

**HuggingFace**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

**Description**: High-accuracy OCR model from DeepSeek-AI for extracting text from complex visual inputs (documents, screenshots, receipts, natural scenes).

**Key Features**:
- Built for real-world documents: PDFs, forms, tables, handwritten/noisy text
- Outputs clean, structured Markdown
- VLLM support upstream
- ~2500 tokens/s on A100 with vLLM
- Paper: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)

**Model Variations**:
- **deepseek-ai/DeepSeek-OCR** - Official BF16 (~6.7 GB)
- **NexaAI/DeepSeek-OCR-GGUF** - Quantized GGUF

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM (officially supported)

**Requirements**:
- Python 3.12.9 + CUDA 11.8
- `torch==2.6.0`, `transformers==4.46.3`, `flash-attn==2.7.3`
- L4 / A100 GPUs (≥16 GB VRAM)

**Dependencies**: `transformers`, `torch`, `vllm`, `flash-attn`, `einops`

**Tasks**: `text_extract`, `ocr`, `table`

**Links**:
- [Model Card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
- [GGUF Version](https://huggingface.co/NexaAI/DeepSeek-OCR-GGUF)
- [Demo Space](https://huggingface.co/spaces/merterbak/DeepSeek-OCR-Demo)

---

#### 9. Nanonets-OCR2-3B
**Released**: 2024 | **Parameters**: 3B | **License**: Apache 2.0

**HuggingFace**: [nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B)

**Description**: State-of-the-art image-to-markdown OCR model that transforms documents into structured markdown with intelligent content recognition and semantic tagging, optimized for LLM downstream processing.

**Key Features**:
- LaTeX equation recognition (inline $...$ and display $$...$$)
- Intelligent image description with structured tags (logos, charts, graphs)
- 125K context window
- ~7.53 GB model size

**Model Variations**:
- **nanonets/Nanonets-OCR2-3B** - Full BF16
- **Mungert/Nanonets-OCR2-3B-GGUF** - GGUF quantized
- **mlx-community/Nanonets-OCR2-3B-4bit** - MLX 4-bit
- **yasserrmd/Nanonets-OCR2-3B** - Ollama format

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ MLX (Apple Silicon)
- ✅ Ollama

**Dependencies**: `transformers`, `torch`, `pillow`

**Tasks**: `text_extract`, `formula`, `ocr`

**Links**:
- [Model Card](https://huggingface.co/nanonets/Nanonets-OCR2-3B)
- [GGUF Version](https://huggingface.co/Mungert/Nanonets-OCR2-3B-GGUF)
- [MLX 4-bit](https://huggingface.co/mlx-community/Nanonets-OCR2-3B-4bit)
- [Ollama](https://ollama.com/yasserrmd/Nanonets-OCR2-3B)

---

#### 10. Qwen3-VL-4B-Instruct
**Released**: October 2025 | **Parameters**: 4B | **License**: Apache 2.0

**HuggingFace**: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

**Description**: Mid-size Qwen3-VL model with balanced performance and efficiency. Part of comprehensive multimodal model series with text understanding, visual reasoning, and agent capabilities.

**Model Variations**:
- **Qwen/Qwen3-VL-4B-Instruct** - Instruction-tuned
- **Qwen/Qwen3-VL-4B-Thinking** - Reasoning-enhanced

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM
- ✅ MLX (via mlx-community)
- ✅ API (via cloud providers)

**Dependencies**: `transformers>=4.46`, `torch`, `qwen-vl-utils`

**Tasks**: `text_extract`, `layout`, `structured`, `ocr`, `table`

**Links**:
- [Collection](https://huggingface.co/collections/Qwen/qwen3-vl)
- [GitHub](https://github.com/QwenLM/Qwen3-VL)

---

#### 11. Google Gemma-3-4B-IT
**Released**: 2025 | **Parameters**: 4B | **License**: Gemma License

**HuggingFace**: [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

**Description**: Lightweight, state-of-the-art multimodal model from Google built from same research/technology as Gemini. Handles text and image input, generates text output.

**Key Features**:
- 128K context window
- Multilingual support (140+ languages)
- SigLIP image encoder (896×896 square images)
- Gemma-3-4B-IT beats Gemma-2-27B-IT on benchmarks

**Model Variations**:
- **google/gemma-3-4b-it** - Instruction-tuned (vision-capable)
- **google/gemma-3-4b-pt** - Pre-trained base
- **google/gemma-3-4b-it-qat-q4_0-gguf** - Quantized GGUF
- **bartowski/google_gemma-3-4b-it-GGUF** - Community GGUF

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ Google AI SDK
- ✅ API (Google AI Studio)

**Dependencies**: `transformers>=4.46`, `torch`, `pillow`

**Tasks**: `text_extract`, `structured`, `ocr`

**Links**:
- [Model Card](https://huggingface.co/google/gemma-3-4b-it)
- [Blog Post](https://huggingface.co/blog/gemma3)
- [Transformers Docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3)
- [Google Docs](https://ai.google.dev/gemma/docs/huggingface_inference)
- [DeepMind Page](https://deepmind.google/models/gemma/gemma-3/)

---

### Medium Models (7-9B Parameters)

#### 12. allenai olmOCR-2-7B-1025
**Released**: October 2025 | **Parameters**: 7B | **License**: Apache 2.0

**HuggingFace**: [allenai/olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025)

**Description**: State-of-the-art OCR for English-language digitized print documents. Fine-tuned from Qwen2.5-VL-7B-Instruct using olmOCR-mix-1025 dataset + GRPO RL training.

**Key Features**:
- 82.4 points on olmOCR-Bench (SOTA for real-world documents)
- Substantial improvements where OCR often fails (math equations, tables, tricky cases)
- Boosted via reinforcement learning (GRPO)

**Model Variations**:
- **allenai/olmOCR-2-7B-1025** - Full BF16 version
- **allenai/olmOCR-2-7B-1025-FP8** - **Recommended** FP8 quantized (practical use except fine-tuning)
- **bartowski/allenai_olmOCR-2-7B-1025-GGUF** - GGUF quantized
- **richardyoung/olmOCR-2-7B-1025-GGUF** - Alternative GGUF

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM (recommended via olmOCR toolkit)
- ✅ API (DeepInfra, Parasail, Cirrascale)

**Best Usage**: Via olmOCR toolkit with VLLM for efficient inference at scale (millions of documents).

**Dependencies**: `transformers`, `torch`, `vllm`, `olmocr` (toolkit
**Tasks**: `text_extract`, `ocr`, `table`, `formula`

**Links**:
- [Model Card](https://huggingface.co/allenai/olmOCR-2-7B-1025)
- [FP8 Version](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)
- [Blog Post](https://allenai.org/blog/olmocr-2)
- [GGUF (bartowski)](https://huggingface.co/bartowski/allenai_olmOCR-2-7B-1025-GGUF)

---

#### 13. Qwen3-VL-8B-Instruct
**Released**: October 2025 | **Parameters**: 8B | **License**: Apache 2.0

**HuggingFace**: [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

**Description**: Primary model in Qwen3-VL series with optimal balance of performance and efficiency. Enhanced document parsing over Qwen2.5-VL with improved visual perception, text understanding, and advanced reasoning.

**Key Features**:
- Custom layout label support (flexible VLM)
- Extended context length
- Enhanced spatial and video comprehension
- Stronger agent interaction capabilities

**Model Variations**:
- **Qwen/Qwen3-VL-8B-Instruct** - Instruction-tuned
- **Qwen/Qwen3-VL-8B-Thinking** - Reasoning-enhanced

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM
- ✅ MLX (Apple Silicon) - [mlx-community/Qwen3-VL-8B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit)
- ✅ API (Novita AI, OpenRouter, etc.)

**API Providers**:
- **Novita AI**: Context 131K tokens, Max output 33K tokens
  - Pricing: $0.08/M input tokens, $0.50/M output tokens

**Dependencies**: `transformers>=4.46`, `torch`, `qwen-vl-utils`, `vllm` (optional)

**Tasks**: `text_extract`, `layout`, `structured`, `ocr`, `table`, `formula`

**Links**:
- [Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Collection](https://huggingface.co/collections/Qwen/qwen3-vl)
- [GitHub](https://github.com/QwenLM/Qwen3-VL)
- [MLX 4-bit](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit)

---

#### 14. datalab-to Chandra
**Released**: 2024 | **Parameters**: 9B | **License**: Apache 2.0

**HuggingFace**: [datalab-to/chandra](https://huggingface.co/datalab-to/chandra)

**Description**: OCR model handling complex tables, forms, and handwriting with full layout preservation. Uses Qwen3VL for document understanding.

**Key Features**:
- 83.1 ± 0.9 overall on OlmOCR benchmark (outperforms DeepSeek OCR, dots.ocr, olmOCR)
- Strong grounding capabilities
- Supports 40+ languages
- Layout-aware output with bbox coordinates for every text block, table, and image
- Outputs in HTML, Markdown, and JSON with detailed layout

**Use Cases**:
- Handwritten forms
- Mathematical notation
- Multi-column layouts
- Complex tables

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM (production throughput)

**Installation**:
```bash
pip install chandra-ocr
```

**Model Variations**:
- **datalab-to/chandra** - Official model
- **noctrex/Chandra-OCR-GGUF** - GGUF quantized

**Dependencies**: `transformers`, `torch`, `vllm` (optional), `chandra-ocr`

**Tasks**: `text_extract`, `layout`, `ocr`, `table`, `formula`

**Links**:
- [Model Card](https://huggingface.co/datalab-to/chandra)
- [GitHub](https://github.com/datalab-to/chandra)
- [Blog Post](https://www.datalab.to/blog/introducing-chandra)
- [DeepWiki Docs](https://deepwiki.com/datalab-to/chandra)
- [GGUF Version](https://huggingface.co/noctrex/Chandra-OCR-GGUF)

---

### Large Models (32B+ Parameters)

#### 15. Qwen3-VL-32B-Instruct
**Released**: October 2025 | **Parameters**: 32B | **License**: Apache 2.0

**HuggingFace**: [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)

**Description**: Largest Qwen3-VL model with maximum performance for complex document understanding and multimodal reasoning tasks.

**Key Features**:
- Superior performance on complex documents
- Extended context length
- Enhanced reasoning capabilities
- Production-grade for demanding applications

**Model Variations**:
- **Qwen/Qwen3-VL-32B-Instruct** - Instruction-tuned
- **Qwen/Qwen3-VL-32B-Thinking** - Reasoning-enhanced

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM (recommended for production)
- ✅ API (cloud providers)

**GPU Requirements**: A100 40GB+ or multi-GPU setup

**Dependencies**: `transformers>=4.46`, `torch`, `qwen-vl-utils`, `vllm`

**Tasks**: `text_extract`, `layout`, `structured`, `ocr`, `table`, `formula`

**Links**:
- [Model Card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- [Collection](https://huggingface.co/collections/Qwen/qwen3-vl)
- [GitHub](https://github.com/QwenLM/Qwen3-VL)

---

### Specialized Models

#### 16. docling-project/docling-models
**Released**: 2024 | **Parameters**: Various | **License**: Apache 2.0

**HuggingFace**: [docling-project/docling-models](https://huggingface.co/docling-project/docling-models)

**Description**: Collection of models powering the Docling PDF document conversion package. Includes layout detection (RT-DETR) and table structure recognition (TableFormer).

**Models Included**:
1. **Layout Model**: RT-DETR for detecting document components
   - Labels: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title
2. **TableFormer Model**: Table structure identification from images

**Note**: Superseded by **granite-docling-258M** for end-to-end document conversion (receives updates and support).

**Backends Supported**:
- ✅ PyTorch (via Docling library)

**Integration**:
```bash
pip install docling
```

**Dependencies**: `docling`, `transformers`, `torch`

**Tasks**: `layout`, `table`

**Links**:
- [Model Card](https://huggingface.co/docling-project/docling-models)
- [Vision Models Docs](https://docling-project.github.io/docling/usage/vision_models/)
- [SmolDocling (legacy)](https://huggingface.co/docling-project/SmolDocling-256M-preview)

---

## 📦 Optional Models (Legacy/Alternative)

### Qwen 2.5-VL Series (Previous Generation)

#### Qwen2.5-VL-3B-Instruct
**Released**: 2024 | **Parameters**: 3B | **License**: Apache 2.0

**HuggingFace**: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

**Description**: Previous generation Qwen VLM with strong visual understanding, agentic capabilities, video understanding (1+ hour), and structured outputs.

**Key Features**:
- Analyzes texts, charts, icons, graphics, layouts
- Visual agent capabilities (computer use, phone use)
- Video comprehension with temporal segment pinpointing
- ViT architecture with SwiGLU and RMSNorm
- Dynamic resolution + dynamic FPS sampling

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM
- ✅ MLX
- ✅ API

**Dependencies**: `transformers`, `torch`, `qwen-vl-utils`

**Tasks**: `text_extract`, `layout`, `structured`, `ocr`

**Links**:
- [Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Collection](https://huggingface.co/collections/Qwen/qwen25-vl)

---

#### Qwen2.5-VL-7B-Instruct
**Released**: 2024 | **Parameters**: 7B | **License**: Apache 2.0

**HuggingFace**: [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

**Description**: Mid-size Qwen2.5-VL model with same capabilities as 3B variant but enhanced performance.

**Model Variations**:
- **Qwen/Qwen2.5-VL-7B-Instruct** - Official
- **unsloth/Qwen2.5-VL-7B-Instruct-GGUF** - GGUF quantized
- **nvidia/Qwen2.5-VL-7B-Instruct-NVFP4** - NVIDIA FP4 optimized

**Backends Supported**:
- ✅ PyTorch (HuggingFace Transformers)
- ✅ VLLM
- ✅ MLX
- ✅ API

**Dependencies**: `transformers`, `torch`, `qwen-vl-utils`

**Tasks**: `text_extract`, `layout`, `structured`, `ocr`, `table`

**Links**:
- [Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Collection](https://huggingface.co/collections/Qwen/qwen25-vl)
- [GGUF Version](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF)

---

## 📊 Model Comparison Summary

### By Release Date (2024-2026)

| Model | Release | Params | Benchmark Score |
|-------|---------|--------|-----------------|
| LightOnOCR-2-1B | Jan 2025 | 1B | 83.2 (OlmOCR) |
| dots.ocr | Dec 2024 | 1.7B | 79.1 (OlmOCR) |
| Granite-Docling-258M | Dec 2024 | 258M | N/A |
| Chandra | 2024 | 9B | 83.1 (OlmOCR) |
| Qwen3-VL Series | Oct 2025 | 2-32B | SOTA |
| PaddleOCR-VL | Oct 2025 | 900M | SOTA |
| olmOCR-2-7B | Oct 2025 | 7B | 82.4 (OlmOCR) |
| DeepSeek-OCR | Oct 2024 | 3B | 75.4 (OlmOCR) |
| GOT-OCR2.0 | Sep 2024 | 700M | N/A |
| MinerU2.5 | Sep 2024 | 1.2B | SOTA |

### By Performance (OlmOCR-Bench)

| Rank | Model | Score | Params |
|------|-------|-------|--------|
| 1 | LightOnOCR-2-1B | 83.2 ± 0.9 | 1B |
| 2 | Chandra | 83.1 ± 0.9 | 9B |
| 3 | olmOCR-2-7B | 82.4 | 7B |
| 4 | dots.ocr | 79.1 | 1.7B |
| 5 | olmOCR (v1) | 78.5 | 7B |
| 6 | DeepSeek-OCR | 75.4 ± 1.0 | 3B |

### By Speed (Relative Performance)

| Model | Speed Multiplier | Params |
|-------|------------------|--------|
| LightOnOCR-2-1B | **Fastest baseline** | 1B |
| PaddleOCR-VL | 1.73× slower | 900M |
| DeepSeek-OCR (vLLM) | 1.73× slower | 3B |
| olmOCR-2 | 1.7× slower | 7B |
| Chandra | 3.3× slower | 9B |
| dots.ocr | 5× slower | 1.7B |

---

## 🔧 Backend Support Matrix

| Model | PyTorch | VLLM | MLX | API | GGUF |
|-------|---------|------|-----|-----|------|
| Granite-Docling-258M | ✅ | ❌ | ✅ | ❌ | ❌ |
| dots.ocr | ✅ | ✅ | ❌ | ❌ | ❌ |
| GOT-OCR2.0 | ✅ | ❌ | ❌ | ❌ | ❌ |
| PaddleOCR-VL | ✅ | ❌ | ❌ | ❌ | ❌ |
| MinerU2.5 | ✅ | ✅ | ✅ | ❌ | ✅ |
| LightOnOCR-2-1B | ✅ | ❌ | ❌ | ❌ | ✅ |
| Qwen3-VL (all) | ✅ | ✅ | ✅ | ✅ | ✅ |
| DeepSeek-OCR | ✅ | ✅ | ❌ | ❌ | ✅ |
| Nanonets-OCR2-3B | ✅ | ❌ | ✅ | ❌ | ✅ |
| Gemma-3-4B-IT | ✅ | ❌ | ❌ | ✅ | ✅ |
| olmOCR-2-7B | ✅ | ✅ | ❌ | ✅ | ✅ |
| Chandra | ✅ | ✅ | ❌ | ❌ | ✅ |
| Qwen2.5-VL (all) | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 📚 Recommended Model Selection Guide

### By Use Case

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Edge/Mobile Deployment** | Granite-Docling-258M | Ultra-compact (258M), MLX support |
| **Fast OCR (CPU)** | LightOnOCR-2-1B | Fastest in class, SOTA accuracy |
| **Multilingual Documents** | PaddleOCR-VL | 109 languages, minimal resources |
| **High-Throughput Serving** | dots.ocr + VLLM | Built for VLLM, fast inference |
| **Best Accuracy (English)** | LightOnOCR-2-1B or Chandra | SOTA on OlmOCR-Bench |
| **Custom Layout Detection** | Qwen3-VL-8B | Flexible VLM with prompt-based labels |
| **Production Balanced** | Qwen3-VL-8B or olmOCR-2-7B | Performance + reliability |
| **Complex Documents** | Chandra or Qwen3-VL-32B | Handles tables, forms, handwriting |
| **Apple Silicon (M1/M2/M3)** | Granite-Docling-258M (MLX) | Native MLX optimization |
| **Cost-Effective API** | Qwen3-VL-8B (Novita) | $0.08/M tokens input |

---

## 🚀 Quick Start Examples

### Ultra-Compact (258M) - Granite-Docling
```python
from omnidocs.tasks.text_extraction import GraniteDoclingOCR, GraniteDoclingConfig

extractor = GraniteDoclingOCR(
    config=GraniteDoclingConfig(device="cuda")
)
result = extractor.extract(image, output_format="markdown")
```

### Fastest OCR (1B) - LightOnOCR-2
```python
from omnidocs.tasks.text_extraction import LightOnOCRExtractor, LightOnOCRConfig

extractor = LightOnOCRExtractor(
    config=LightOnOCRConfig(
        model="lightonai/LightOnOCR-2-1B",
        device="cuda"
    )
)
result = extractor.extract(image, output_format="markdown")
```

### High-Throughput (1.7B) - dots.ocr + VLLM
```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

extractor = DotsOCRTextExtractor(
    backend=DotsOCRVLLMConfig(
        model="rednote-hilab/dots.ocr",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
)
result = extractor.extract(image, output_format="markdown")
```

### Best Accuracy (7-9B) - olmOCR-2 or Chandra
```python
from omnidocs.tasks.text_extraction import OlmOCRExtractor, ChandraTextExtractor
from omnidocs.tasks.text_extraction.olm import OlmOCRVLLMConfig
from omnidocs.tasks.text_extraction.chandra import ChandraPyTorchConfig

# Option 1: olmOCR-2-7B with VLLM
extractor = OlmOCRExtractor(
    backend=OlmOCRVLLMConfig(
        model="allenai/olmOCR-2-7B-1025-FP8",
        tensor_parallel_size=1
    )
)

# Option 2: Chandra-9B
extractor = ChandraTextExtractor(
    backend=ChandraPyTorchConfig(
        model="datalab-to/chandra",
        device="cuda"
    )
)
```

### Flexible Custom Layouts (8B) - Qwen3-VL
```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenPyTorchConfig

layout = QwenLayoutDetector(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda"
    )
)

result = layout.extract(
    image,
    custom_labels=["code_block", "sidebar", "diagram"]
)
```

---

## 🎯 Current Focus: Layout Analysis Models

### Phase 1: Multi-Backend VLM Integration

#### 1. Qwen3-VL-8B-Instruct Integration

**Status**: 🟡 In Progress

Integrate Qwen3-VL-8B-Instruct for flexible layout detection with custom label support across all backends.

**Key Features**:
- Enhanced document parsing over Qwen2.5-VL
- Improved visual perception and text understanding
- Advanced reasoning capabilities
- Custom layout label support

##### Implementation Checklist:

- [ ] **HuggingFace/PyTorch Backend** (`QwenLayoutDetector` + `QwenPyTorchConfig`)

  **Model**: `Qwen/Qwen3-VL-8B-Instruct`

  **Config Class**: `omnidocs/tasks/layout_analysis/qwen/pytorch.py`
  ```python
  class QwenPyTorchConfig(BaseModel):
      model: str = "Qwen/Qwen3-VL-8B-Instruct"
      device: str = "cuda"
      torch_dtype: Literal["auto", "float16", "bfloat16"] = "auto"
      attn_implementation: Optional[str] = None  # "flash_attention_2" if available
      cache_dir: Optional[str] = None
  ```

  **Dependencies**:
  - `torch`, `transformers`
  - `qwen-vl-utils` (model-specific utility)

  **Reference Implementation**: See `scripts/layout/modal_qwen3_vl_layout.py` in the repository

  **Testing**:
  - Validate on synthetic document images
  - Compare detection accuracy with ground truth
  - Test custom label support

- [ ] **VLLM Backend** (`QwenVLLMConfig`)

  **Model**: `Qwen/Qwen3-VL-8B-Instruct`

  **Config Class**: `omnidocs/tasks/layout_analysis/qwen/vllm.py`
  ```python
  class QwenVLLMConfig(BaseModel):
      model: str = "Qwen/Qwen3-VL-8B-Instruct"
      tensor_parallel_size: int = 1
      gpu_memory_utilization: float = 0.9
      max_model_len: Optional[int] = None
      trust_remote_code: bool = True
  ```

  **Dependencies**:
  - `vllm>=0.4.0`
  - `torch>=2.0`

  **Use Case**: High-throughput batch processing (10+ documents/second)

  **Modal Config**:
  - GPU: `A10G:1` (minimum), `A100:1` (recommended for production)
  - Image: VLLM GPU Image with flash-attn

  **Testing**:
  - Benchmark throughput vs PyTorch
  - Validate output consistency
  - Test batch processing

- [ ] **MLX Backend** (`QwenMLXConfig`)

  **Model**: `mlx-community/Qwen3-VL-8B-Instruct-4bit`

  **Config Class**: `omnidocs/tasks/layout_analysis/qwen/mlx.py`
  ```python
  class QwenMLXConfig(BaseModel):
      model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
      quantization: Literal["4bit", "8bit"] = "4bit"
      max_tokens: int = 4096
  ```

  **Dependencies**:
  - `mlx>=0.10`
  - `mlx-lm>=0.10`

  **Platform**: Apple Silicon only (M1/M2/M3+)

  **Use Case**: Local development and testing on macOS

  **Note**: ⚠️ DO NOT deploy MLX to Modal - local development only

- [ ] **API Backend** (`QwenAPIConfig`)

  **Model**: `qwen3-vl-8b-instruct`

  **Config Class**: `omnidocs/tasks/layout_analysis/qwen/api.py`
  ```python
  class QwenAPIConfig(BaseModel):
      model: str = "novita/qwen3-vl-8b-instruct"
      api_key: str
      base_url: Optional[str] = None
      max_tokens: int = 4096
      temperature: float = 0.1
  ```

  **Provider**: Novita AI
  - **Context Length**: 131K tokens
  - **Max Output**: 33K tokens
  - **Pricing**:
    - Input: $0.08/M tokens
    - Output: $0.50/M tokens

  **Dependencies**:
  - `litellm>=1.30`
  - `openai>=1.0`

  **Use Case**:
  - Serverless deployments
  - No GPU infrastructure required
  - Cost-effective for low-volume processing

- [ ] **Main Extractor Class** (`omnidocs/tasks/layout_analysis/qwen.py`)

  Implement unified `QwenLayoutDetector` class:
  ```python
  from typing import Union, List, Optional
  from PIL import Image
  from .base import BaseLayoutExtractor
  from .models import LayoutOutput
  from .qwen import (
      QwenPyTorchConfig,
      QwenVLLMConfig,
      QwenMLXConfig,
      QwenAPIConfig,
  )

  QwenBackendConfig = Union[
      QwenPyTorchConfig,
      QwenVLLMConfig,
      QwenMLXConfig,
      QwenAPIConfig,
  ]

  class QwenLayoutDetector(BaseLayoutExtractor):
      """Flexible VLM-based layout detector with custom label support."""

      def __init__(self, backend: QwenBackendConfig):
          self.backend_config = backend
          self._backend = self._create_backend()

      def extract(
          self,
          image: Image.Image,
          custom_labels: Optional[List[str]] = None,
      ) -> LayoutOutput:
          """
          Detect layout elements with optional custom labels.

          Args:
              image: PIL Image
              custom_labels: Optional custom layout categories
                  Default: ["title", "paragraph", "table", "figure",
                           "caption", "formula", "list"]

          Returns:
              LayoutOutput with detected bounding boxes
          """
          # Implementation...
  ```

- [ ] **Integration Tests**

  Test suite covering:
  - All backend configurations
  - Custom label functionality
  - Cross-backend output consistency
  - Edge cases (empty images, single elements, complex layouts)

- [ ] **Documentation**

  - API reference with examples for each backend
  - Performance comparison table (PyTorch vs VLLM vs MLX vs API)
  - Migration guide from Qwen2.5-VL
  - Custom label usage examples

- [ ] **Modal Deployment Script**

  Create production-ready deployment:
  - `scripts/layout_omnidocs/modal_qwen_layout_vllm_online.py`
  - Web endpoint for layout detection API
  - Batch processing support
  - Monitoring and logging

---

### Phase 2: Additional Layout Models

#### 2. RT-DETR Layout Detector

- [ ] **Single-Backend Implementation** (PyTorch only)
  - Model: `RT-DETR` (Facebook AI)
  - Fixed label support (COCO-based)
  - Real-time detection optimization

#### 3. Surya Layout Detector

- [ ] **Single-Backend Implementation** (PyTorch only)
  - Model: `vikp/surya_layout`
  - Multi-language document support
  - Optimized for speed

#### 4. Florence-2 Layout Detector

- [ ] **Multi-Backend Implementation**
  - HuggingFace/PyTorch backend
  - API backend (Microsoft Azure)
  - Object detection + dense captioning

---

## 🔮 Future Phases

Additional task categories will be added after layout analysis is complete:

- **OCR Extraction**: Surya-OCR, PaddleOCR, Qwen-OCR
- **Text Extraction**: VLM-based Markdown/HTML extraction
- **Table Extraction**: Table Transformer, Surya-Table
- **Math Expression Extraction**: UniMERNet, Surya-Math
- **Advanced Features**: Reading order, image captioning, chart understanding
- **Package & Distribution**: PyPI publishing, comprehensive documentation

---

## 🎯 Success Metrics (Layout Analysis)

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Layout Detection Accuracy (mAP) | >90% | TBD |
| Inference Speed (PyTorch) | <2s per page | TBD |
| Inference Speed (VLLM) | <0.5s per page | TBD |
| Custom Label Support | 100% functional | TBD |

### Quality Targets

- [ ] Type hints coverage: 100%
- [ ] Docstring coverage: 100%
- [ ] Test coverage: >80%
- [ ] All backends tested on production data
- [ ] Cross-backend output consistency validated

---

## 🔧 Infrastructure

### Modal Deployment Standards

**Consistency Requirements** (as per CLAUDE.md):

- Volume Name: `omnidocs`
- Secret Name: `adithya-hf-wandb`
- CUDA Version: `12.4.0-devel-ubuntu22.04`
- Python Version: `3.11` (3.12 for Qwen3-VL)
- Cache Directory: `/data/.cache` (HuggingFace)
- Model Cache: `/data/omnidocs_models`
- Dependency Management: `.uv_pip_install()` (NO version pinning)

### GPU Configurations

| GPU | Use Case | Cost (est.) |
|-----|----------|-------------|
| `A10G:1` | Development & Testing | $0.60/hr |
| `A100:1` | Production Inference | $3.00/hr |
| `A100:2` | High-Throughput VLLM | $6.00/hr |

---

## 📚 References

### Design Documents

- **Backend Architecture** - Core design principles (see `IMPLEMENTATION_PLAN/BACKEND_ARCHITECTURE.md`)
- **Developer Experience (DevEx)** - API design and patterns (see `IMPLEMENTATION_PLAN/DEVEX.md`)
- **Claude Development Guide** - Implementation standards (see `CLAUDE.md` in repo root)

### External Resources

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL MLX (4bit)](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit)
- [Modal Documentation](https://modal.com/docs)
- [UV Package Manager](https://github.com/astral-sh/uv)

---

## 📝 Notes

### Implementation Order Rationale

1. **Qwen3-VL Priority**: Multi-backend support demonstrates v2.0 architecture
2. **RT-DETR**: Fast fixed-label detection for production use
3. **Surya**: Multi-language support and speed optimization
4. **Florence-2**: Microsoft's advanced VLM capabilities

### Breaking Changes from v1.0

- String-based factory pattern removed (use class imports)
- Document class is now stateless (doesn't store results)
- Config classes are model-specific (not generic)
- Backend selection via config type (not string parameter)

---

**Last Updated**: January 21, 2026
**Maintainer**: Adithya S Kolavi
**Version**: 2.0.0-dev