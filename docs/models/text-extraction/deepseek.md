# DeepSeek-OCR Text Extraction

## Model Overview

DeepSeek-OCR and DeepSeek-OCR-2 are compact (~3B parameter) OCR models from DeepSeek AI, built for high-accuracy extraction of complex real-world documents — including dense PDFs, forms, handwritten text, and noisy scans. The model uses a hybrid vision encoder + causal text decoder to produce structured Markdown output directly, without post-processing bounding boxes.

| Version | Release | arXiv | License | Architecture |
|---------|---------|-------|---------|--------------|
| **DeepSeek-OCR-2** | Jan 2026 | 2601.20552 | Apache 2.0 | Visual Causal Flow |
| **DeepSeek-OCR** | Oct 2024 | 2510.18234 | MIT | Hybrid Vision + Causal LM |

**Default model**: `deepseek-ai/DeepSeek-OCR-2`
**Repository (v2)**: [deepseek-ai/DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)

### Key Capabilities

- **High-Accuracy OCR**: Handles dense, noisy, and handwritten documents
- **Markdown Output**: Structured output from model directly (no post-processing)
- **Adaptive Tiling**: `crop_mode=True` splits dense pages into overlapping tiles for small fonts
- **Multi-Prompt**: Supports markdown, OCR, plain text, and figure parsing modes
- **Production Throughput**: ~2500 tok/s on A100 via VLLM (official upstream support for v1)

### Limitations

- No layout categories or bounding boxes (use DotsOCR for that)
- PyTorch backend requires `transformers==4.46.3` (pinned version)
- MLX variants currently only available for v1; check mlx-community for v2
- VLLM support for v2 may require a nightly build (see official repo)
- `crop_mode` (PyTorch) writes temporary `.mmd` files to disk

---

## Supported Backends

| Backend | Use Case | Performance | Notes |
|---------|----------|-------------|-------|
| **PyTorch** | Local GPU inference | 80–120 tok/s | Requires pinned `transformers==4.46.3` |
| **VLLM** | High-throughput production | ~2500 tok/s | Official support for v1; v2 may need nightly |
| **MLX** | Apple Silicon | 20–40 tok/s | v1 only (4bit/8bit quantized) |
| **API** | Cloud / no GPU | Variable | Via Novita AI (litellm) |

---

## Installation & Configuration

### Basic Installation

```bash
# PyTorch backend
pip install omnidocs[pytorch]
pip install "transformers==4.46.3" einops addict easydict  # Required extras

# VLLM backend
pip install omnidocs[vllm]

# MLX backend
pip install omnidocs[mlx]

# API backend (included in base install)
pip install omnidocs
```

### PyTorch Backend Configuration

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

config = DeepSeekOCRTextPyTorchConfig(
    model="deepseek-ai/DeepSeek-OCR-2",
    device="cuda",
    torch_dtype="bfloat16",        # Required per official README
    use_flash_attention=False,     # True requires flash-attn==2.7.3
    trust_remote_code=True,
    base_size=1024,
    image_size=768,
    crop_mode=True,                # Adaptive tiling for dense pages
)
extractor = DeepSeekOCRTextExtractor(backend=config)
```

**PyTorch Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `deepseek-ai/DeepSeek-OCR-2` | HuggingFace model ID |
| `device` | str | `cuda` | Device: "cuda" or "cpu" (MPS not tested) |
| `torch_dtype` | str | `bfloat16` | BF16 required per official README |
| `use_flash_attention` | bool | `False` | Flash Attention 2 (needs flash-attn==2.7.3) |
| `trust_remote_code` | bool | `True` | Required — custom model code |
| `base_size` | int | `1024` | Visual encoder canvas size (512–2048) |
| `image_size` | int | `768` | Tile resize target (256–1024) |
| `crop_mode` | bool | `True` | Adaptive tiling ("Gundam mode") for dense pages |
| `cache_dir` | str | `None` | Override model cache directory |

### VLLM Backend Configuration

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

config = DeepSeekOCRTextVLLMConfig(
    model="deepseek-ai/DeepSeek-OCR",  # v1 has stable VLLM support
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    max_tokens=8192,
    temperature=0.0,
    enable_prefix_caching=False,       # Must be False for v1
    mm_processor_cache_gb=0,           # Must be 0 for v1
)
extractor = DeepSeekOCRTextExtractor(backend=config)
```

**VLLM Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `deepseek-ai/DeepSeek-OCR` | HuggingFace model ID |
| `tensor_parallel_size` | int | `1` | Number of GPUs |
| `gpu_memory_utilization` | float | `0.9` | GPU memory fraction |
| `max_model_len` | int | `8192` | Max context length |
| `temperature` | float | `0.0` | 0.0 for greedy decoding |
| `enforce_eager` | bool | `False` | Disable CUDA graph (useful for Modal cold start) |
| `enable_prefix_caching` | bool | `False` | **Must be False** for v1 |
| `mm_processor_cache_gb` | float | `0` | **Must be 0** for v1 |
| `ngram_size` | int | `30` | NGram window size for logits processor |
| `ngram_window_size` | int | `90` | NGram context window |

### MLX Backend Configuration (Apple Silicon)

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

config = DeepSeekOCRTextMLXConfig(
    model="mlx-community/DeepSeek-OCR-4bit",  # or "DeepSeek-OCR-8bit"
    max_tokens=8192,
    temperature=0.0,
)
extractor = DeepSeekOCRTextExtractor(backend=config)
```

**MLX Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `mlx-community/DeepSeek-OCR-4bit` | MLX HuggingFace model ID |
| `max_tokens` | int | `8192` | Max tokens to generate |
| `temperature` | float | `0.0` | Sampling temperature |
| `cache_dir` | str | `None` | Override cache (sets HF_HOME) |

### API Backend Configuration

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

config = DeepSeekOCRTextAPIConfig(
    model="novita/deepseek/deepseek-ocr",
    api_key=None,     # reads NOVITA_API_KEY from env
    api_base=None,    # override URL if using a different provider
    max_tokens=8192,
    temperature=0.0,
    timeout=180,
)
extractor = DeepSeekOCRTextExtractor(backend=config)
```

**API Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `novita/deepseek/deepseek-ocr` | litellm model string |
| `api_key` | str | `None` | API key (reads from env if None) |
| `api_base` | str | `None` | Override provider base URL |
| `max_tokens` | int | `8192` | Max tokens to generate |
| `temperature` | float | `0.0` | Sampling temperature |
| `timeout` | int | `180` | Request timeout in seconds |

---

## Usage Examples

### Basic Text Extraction

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig
from PIL import Image

extractor = DeepSeekOCRTextExtractor(
    backend=DeepSeekOCRTextVLLMConfig()
)

image = Image.open("document.png")
result = extractor.extract(image)

print(result.content)        # Markdown text
print(result.model_name)     # "DeepSeek-OCR (model, backend)"
```

### Dense Document with Crop Mode (PyTorch)

```python
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

config = DeepSeekOCRTextPyTorchConfig(
    device="cuda",
    crop_mode=True,   # Adaptive tiling for small fonts or dense layouts
    base_size=1024,
    image_size=768,
)
extractor = DeepSeekOCRTextExtractor(backend=config)
result = extractor.extract(image)
```

### Batch Processing with VLLM

```python
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig
from PIL import Image

config = DeepSeekOCRTextVLLMConfig(
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
)
extractor = DeepSeekOCRTextExtractor(backend=config)

images = [Image.open(f"doc_{i}.png") for i in range(20)]
results = [extractor.extract(img) for img in images]
```

### Using v1 vs v2

```python
# v2 (default — Apache 2.0, Jan 2026)
config = DeepSeekOCRTextPyTorchConfig(model="deepseek-ai/DeepSeek-OCR-2")

# v1 (MIT, Oct 2024 — stable VLLM support)
config = DeepSeekOCRTextPyTorchConfig(model="deepseek-ai/DeepSeek-OCR")
```

---

## Performance Characteristics

### VRAM Requirements

| Backend | Min VRAM |
|---------|----------|
| PyTorch | 16 GB |
| VLLM (1 GPU) | 20 GB |
| VLLM (2 GPU tensor parallel) | 20 GB/GPU |

### Inference Speed

| Backend | Device | Speed |
|---------|--------|-------|
| PyTorch | A100-40G | 80–120 tok/s |
| VLLM | A100-40G | ~2500 tok/s |
| MLX (4bit) | M3 Max (48GB) | 20–40 tok/s |
| API | Novita AI | Variable |

---

## Troubleshooting

### Empty Output (PyTorch)

PyTorch backend writes `.mmd` files to a temp directory. If output is empty:

```python
# 1. Disable crop_mode on simple/small images
config = DeepSeekOCRTextPyTorchConfig(crop_mode=False)

# 2. Verify image is large enough
print(image.size)  # Should be > 256x256
```

### VLLM ImportError: `NGramPerReqLogitsProcessor`

```bash
pip install "vllm>=0.11.1"
```

### OOM on PyTorch

```python
config = DeepSeekOCRTextPyTorchConfig(
    base_size=768,   # reduce from 1024
    image_size=512,  # reduce from 768
)
```

### Slow PyTorch Inference

Install Flash Attention:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

```python
config = DeepSeekOCRTextPyTorchConfig(use_flash_attention=True)
```

Or switch to VLLM for significantly higher throughput.

### `transformers` Version Conflicts

DeepSeek-OCR requires `transformers==4.46.3`. If other libraries conflict, use a dedicated environment:

```bash
python -m venv .venv-deepseek
source .venv-deepseek/bin/activate
pip install omnidocs[pytorch] "transformers==4.46.3" einops addict easydict
```

---

## Model Selection Guide

### When to Use DeepSeek-OCR

**Best for**:
- High-throughput production pipelines (VLLM backend)
- Dense, complex, or noisy real-world documents
- Handwritten text
- Scenarios where raw Markdown output is sufficient

**Not ideal for**:
- Documents needing per-element bounding boxes → use DotsOCR
- Multilingual (25+ language) support → use Qwen3-VL
- Apple Silicon with latest model → check mlx-community for v2 availability

### DeepSeek-OCR vs Qwen3-VL vs DotsOCR

| Feature | DeepSeek-OCR-2 | Qwen3-VL-8B | DotsOCR |
|---------|----------------|-------------|---------|
| **Throughput (VLLM)** | ~2500 tok/s | 200–300 tok/s | 150–200 tok/s |
| **Layout info** | None | Basic | Detailed (11 cats) |
| **Multilingual** | Limited | Excellent (25+) | Limited |
| **Backends** | 4 | 4 | 2 |
| **Model size** | 3B | 2B–32B | ~7B |

---

## API Reference

### DeepSeekOCRTextExtractor.extract()

```python
def extract(
    image: Union[Image.Image, np.ndarray, str, Path],
    output_format: Literal["html", "markdown"] = "markdown",
) -> TextOutput:
    """
    Extract text from a document image using DeepSeek-OCR.

    DeepSeek-OCR always outputs Markdown. output_format is accepted for
    API compatibility but does not change the output type.

    Args:
        image: Input image (PIL Image, numpy array, or file path)
        output_format: Accepted for compatibility (default: "markdown")

    Returns:
        TextOutput with extracted Markdown content
    """
```

### TextOutput Properties

```python
result = extractor.extract(image)

result.content        # Extracted Markdown text
result.plain_text     # Plain text (same as content for DeepSeek-OCR)
result.format         # OutputFormat.MARKDOWN
result.image_width    # Source image width
result.image_height   # Source image height
result.model_name     # "DeepSeek-OCR (model, backend)"
result.raw_output     # Raw model output
```

---

## See Also

- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [Novita AI — API Hosting](https://novita.ai/models/model-detail/deepseek-deepseek-ocr)
- [DotsOCR](./dotsocr.md) — Layout-aware extraction with bounding boxes
- [Qwen3-VL](./qwen.md) — Multilingual, multi-backend text extraction
- [Comparison Guide](../comparison.md) — Full model selection matrix
