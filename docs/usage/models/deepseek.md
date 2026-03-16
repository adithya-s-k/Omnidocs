# DeepSeek OCR

High-accuracy document OCR using DeepSeek-OCR and DeepSeek-OCR-2.

---

## Overview

| Property | Value |
|----------|-------|
| **Model (default)** | `deepseek-ai/DeepSeek-OCR-2` |
| **Parameters** | ~3B |
| **Task** | Text Extraction |
| **Backends** | PyTorch, VLLM, MLX, API |
| **License** | Apache 2.0 (v2), MIT (v1) |

Two generations of DeepSeek OCR are supported:

| Version | Release | License | Architecture |
|---------|---------|---------|--------------|
| **DeepSeek-OCR-2** (default) | Jan 2026 | Apache 2.0 | Visual Causal Flow |
| **DeepSeek-OCR** | Oct 2024 | MIT | Hybrid Vision + Causal LM |

Both share the same inference interface — swap the model string to switch.

---

## Installation

```bash
# PyTorch backend
pip install omnidocs[pytorch]

# VLLM backend (recommended for production — ~2500 tok/s on A100)
pip install omnidocs[vllm]

# MLX backend (Apple Silicon)
pip install omnidocs[mlx]

# API backend (no GPU — included in base install)
pip install omnidocs
```

!!! note "Extra dependencies for PyTorch backend"
    DeepSeek-OCR requires `transformers==4.46.3`, `einops`, `addict`, and `easydict`.
    Optionally install `flash-attn==2.7.3` with `--no-build-isolation` for faster inference.

    ```bash
    pip install "transformers==4.46.3" einops addict easydict
    ```

---

## Quick Start

### PyTorch Backend

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig
from PIL import Image

image = Image.open("document.png")

extractor = DeepSeekOCRTextExtractor(
    backend=DeepSeekOCRTextPyTorchConfig(device="cuda")
)
result = extractor.extract(image)
print(result.content)
```

### VLLM Backend (Recommended for Production)

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

extractor = DeepSeekOCRTextExtractor(
    backend=DeepSeekOCRTextVLLMConfig(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
)
result = extractor.extract(image)
print(result.content)
```

### MLX Backend (Apple Silicon)

```python
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

extractor = DeepSeekOCRTextExtractor(
    backend=DeepSeekOCRTextMLXConfig(
        model="mlx-community/DeepSeek-OCR-4bit"
    )
)
result = extractor.extract(image)
```

### API Backend (Novita AI)

```python
import os
from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

extractor = DeepSeekOCRTextExtractor(
    backend=DeepSeekOCRTextAPIConfig(
        model="novita/deepseek/deepseek-ocr",
        api_key=os.getenv("NOVITA_API_KEY"),
    )
)
result = extractor.extract(image)
```

---

## Configuration

### PyTorch Config

```python
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

config = DeepSeekOCRTextPyTorchConfig(
    model="deepseek-ai/DeepSeek-OCR-2",   # or "deepseek-ai/DeepSeek-OCR"
    device="cuda",                          # "cuda" or "cpu" (MPS not tested)
    torch_dtype="bfloat16",                # Required per official README
    use_flash_attention=False,             # True requires flash-attn==2.7.3
    trust_remote_code=True,               # Required — custom model code
    base_size=1024,                        # Visual encoder canvas size
    image_size=768,                        # Tile resize target
    crop_mode=True,                        # Adaptive tiling for dense pages
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `deepseek-ai/DeepSeek-OCR-2` | HuggingFace model ID |
| `device` | `cuda` | Inference device |
| `torch_dtype` | `bfloat16` | BF16 required per official README |
| `use_flash_attention` | `False` | Enable Flash Attention 2 (needs flash-attn==2.7.3) |
| `crop_mode` | `True` | Adaptive tiling for dense/small-font pages ("Gundam mode") |
| `base_size` | `1024` | Visual encoder canvas size (512–2048) |
| `image_size` | `768` | Tile resize target (256–1024) |

### VLLM Config

```python
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

config = DeepSeekOCRTextVLLMConfig(
    model="deepseek-ai/DeepSeek-OCR",       # v1 has official VLLM support
    tensor_parallel_size=1,                  # GPUs for parallelism
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    max_tokens=8192,
    temperature=0.0,                         # Greedy decoding recommended
    enable_prefix_caching=False,             # Must be False for v1
    mm_processor_cache_gb=0,                 # Must be 0 for v1
)
```

!!! warning "VLLM v1 constraints"
    For DeepSeek-OCR v1, `enable_prefix_caching` **must** be `False` and `mm_processor_cache_gb` **must** be `0`. These are required for the NGram logits processor to work correctly.

    DeepSeek-OCR-2 VLLM support: check the [official repo](https://github.com/deepseek-ai/DeepSeek-OCR-2) for updated setup instructions — may require a nightly VLLM build.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `deepseek-ai/DeepSeek-OCR` | HuggingFace model ID |
| `tensor_parallel_size` | `1` | Number of GPUs |
| `gpu_memory_utilization` | `0.9` | GPU memory fraction |
| `max_model_len` | `8192` | Max context length |
| `temperature` | `0.0` | 0.0 = greedy decoding |
| `enable_prefix_caching` | `False` | Must be False for v1 |
| `mm_processor_cache_gb` | `0` | Must be 0 for v1 |

### MLX Config

```python
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

config = DeepSeekOCRTextMLXConfig(
    model="mlx-community/DeepSeek-OCR-4bit",  # or "DeepSeek-OCR-8bit"
    max_tokens=8192,
    temperature=0.0,
)
```

!!! note "MLX model availability"
    MLX quantized variants are currently available for DeepSeek-OCR v1 (`mlx-community/DeepSeek-OCR-4bit`, `mlx-community/DeepSeek-OCR-8bit`). Check [mlx-community on HuggingFace](https://huggingface.co/mlx-community) for DeepSeek-OCR-2 variants as they are published.

### API Config

```python
from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

config = DeepSeekOCRTextAPIConfig(
    model="novita/deepseek/deepseek-ocr",  # litellm format
    api_key=None,                           # reads NOVITA_API_KEY from env
    api_base=None,                          # override provider URL if needed
    max_tokens=8192,
    temperature=0.0,
    timeout=180,
)
```

Set the environment variable for Novita AI:

```bash
export NOVITA_API_KEY=your_key_here
```

---

## Prompt Modes

DeepSeek-OCR supports four built-in prompt modes. The extractor uses `markdown` by default.

| Mode | Prompt | Best For |
|------|--------|----------|
| `markdown` | `<\|grounding\|>Convert the document to markdown.` | Structured documents (default) |
| `ocr` | `<\|grounding\|>OCR this image.` | General image OCR |
| `free` | `Free OCR.` | Plain text, no layout |
| `figure` | `Parse the figure.` | Figures and diagrams |

---

## Output

```python
result = extractor.extract(image)

print(result.content)        # Extracted Markdown text
print(result.plain_text)     # Plain text (same as content for DeepSeek)
print(result.model_name)     # "DeepSeek-OCR (model, backend)"
print(result.image_width)    # Source image width
print(result.image_height)   # Source image height
print(result.raw_output)     # Raw model output
```

DeepSeek-OCR always outputs Markdown. The `output_format` parameter is accepted for API compatibility but does not change the output type.

---

## Performance

| Backend | Device | Speed | Notes |
|---------|--------|-------|-------|
| PyTorch | A100-40G | ~80–120 tok/s | BF16, `crop_mode=True` |
| VLLM | A100-40G | ~2500 tok/s | Official upstream support for v1 |
| MLX | M3 Max (48GB) | ~20–40 tok/s | 4-bit quantized |
| API | Novita AI | Variable | No GPU required |

**VRAM requirements:**

| Backend | Min VRAM |
|---------|----------|
| PyTorch | 16 GB |
| VLLM (1 GPU) | 20 GB |
| VLLM (2 GPU) | 20 GB/GPU |

---

## Comparison with Other Models

| Model | Speed | Layout Info | Multilingual | Backends |
|-------|-------|-------------|--------------|----------|
| **DeepSeek-OCR-2** | Fast (VLLM) | No | Limited | PyTorch, VLLM, MLX, API |
| Qwen3-VL-8B | Medium | Basic | Yes (25+) | PyTorch, VLLM, MLX, API |
| DotsOCR | Medium | Yes (11 cats) | Limited | PyTorch, VLLM |
| Nanonets OCR2 | Fast | No | Limited | PyTorch, VLLM, MLX |

**Choose DeepSeek-OCR if:**
- You need maximum throughput (VLLM ~2500 tok/s on A100)
- You're processing dense, complex real-world documents
- You need handwritten or noisy document support

**Choose Qwen3-VL if:** You need multilingual support or an API backend with broader provider coverage.

**Choose DotsOCR if:** You need bounding boxes and layout categories alongside the text.

---

## Troubleshooting

**Empty or incomplete output (PyTorch)**

The PyTorch backend writes output to `.mmd` files on disk. If the output directory is empty, the model may have failed silently. Try:

```python
# Ensure image is large enough
from PIL import Image
image = Image.open("document.png")
print(image.size)  # Should be > 256x256

# Disable crop_mode for small/simple images
config = DeepSeekOCRTextPyTorchConfig(crop_mode=False)
```

**VLLM ImportError: `NGramPerReqLogitsProcessor`**

This requires `vllm>=0.11.1`. Update with:

```bash
pip install "vllm>=0.11.1"
```

**OOM on PyTorch**

```python
# Reduce tile size
config = DeepSeekOCRTextPyTorchConfig(
    base_size=768,   # down from 1024
    image_size=512,  # down from 768
)
```

**Slow inference on PyTorch**

Switch to VLLM, or install Flash Attention:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

Then enable it:

```python
config = DeepSeekOCRTextPyTorchConfig(use_flash_attention=True)
```

**API 401 / authentication errors**

```bash
# Verify your Novita API key is set
echo $NOVITA_API_KEY

# Or pass it directly
config = DeepSeekOCRTextAPIConfig(
    model="novita/deepseek/deepseek-ocr",
    api_key="your_key_here",
)
```

---

## See Also

- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [Novita AI Hosting](https://novita.ai/models/model-detail/deepseek-deepseek-ocr)
- [DotsOCR](./dotsocr.md) — layout-aware alternative
- [Qwen](./qwen.md) — multilingual alternative
- [Model Comparison](../../models/comparison.md)
