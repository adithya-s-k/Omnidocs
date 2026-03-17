# LightOn Text Extraction

## Model Overview

LightOn OCR is a self-hosted OCR model optimised for document text extraction with multi-lingual support. It uses a conditional generation architecture and is available in multiple inference backends.

**Model ID**: `lightonai/LightOnOCR-2-1B`
**Repository**: [lightonai/LightOnOCR-2-1B on HuggingFace](https://huggingface.co/lightonai/LightOnOCR-2-1B)
**Architecture**: LightOnOcrForConditionalGeneration (1B parameters)

### Key Capabilities

- **Fully self-hosted**: No external API calls — weights downloaded from HuggingFace and run locally
- **Multi-lingual**: Strong multi-lingual document extraction
- **Multi-backend**: PyTorch, VLLM, and MLX
- **Lightweight**: 1B parameters, efficient on a single GPU

### Limitations

- Requires `transformers>=5.0.0` — conflicts with `docling-ibm-models` which pins `<5.0.0`
- No API backend (self-hosted only)
- MLX requires Apple Silicon

---

## Supported Backends

| Backend | Use Case | Performance | Setup |
|---------|----------|-------------|-------|
| **PyTorch** | Local GPU inference | Fast | Single GPU |
| **VLLM** | High-throughput batching | Very fast | Requires GPU cluster |
| **MLX** | Apple Silicon (native) | Good on M-series | macOS M1/M2/M3+ only |

---

## Installation
```bash
# PyTorch backend
pip install omnidocs[pytorch]

# VLLM backend
pip install omnidocs[vllm]

# LightOn requires transformers>=5.0.0
pip install "transformers>=5.0.0"
```

> **Note**: If `docling-ibm-models` is installed it pins `transformers<5.0.0`. Use an override:
> ```bash
> uv pip install omnidocs[pytorch] --override <(echo 'transformers>=5.0.0')
> ```

---

## Configuration

### PyTorch Backend
```python
from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

config = LightOnTextPyTorchConfig(
    model="lightonai/LightOnOCR-2-1B",
    device="auto",              # Automatically selects cuda/mps/cpu
    torch_dtype="bfloat16",
    device_map="auto",
    use_flash_attention=False,  # Set True if flash-attn installed
    max_new_tokens=4096,
)
extractor = LightOnTextExtractor(backend=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"lightonai/LightOnOCR-2-1B"` | HuggingFace model ID |
| `device` | str | `"auto"` | `"cuda"`, `"mps"`, `"cpu"`, `"auto"` |
| `torch_dtype` | str | `"bfloat16"` | `"float16"`, `"bfloat16"`, `"float32"`, `"auto"` |
| `device_map` | str | `"auto"` | Model parallelism strategy |
| `use_flash_attention` | bool | `False` | Flash Attention 2 (requires flash-attn); uses SDPA by default |
| `trust_remote_code` | bool | `True` | Trust remote code from HuggingFace |
| `cache_dir` | str | `None` | Model cache directory |
| `max_new_tokens` | int | `4096` | Max tokens to generate (256–16384) |

### VLLM Backend
```python
from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextVLLMConfig

config = LightOnTextVLLMConfig(
    model="lightonai/LightOnOCR-2-1B",
    gpu_memory_utilization=0.85,
    max_model_len=16384,
    enforce_eager=True,
    max_tokens=4096,
)
extractor = LightOnTextExtractor(backend=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel_size` | int | `1` | Number of GPUs |
| `gpu_memory_utilization` | float | `0.85` | GPU memory fraction |
| `max_model_len` | int | `16384` | Maximum sequence length |
| `enforce_eager` | bool | `True` | Disable CUDA graphs (faster startup) |
| `max_tokens` | int | `4096` | Max tokens to generate |
| `download_dir` | str | `None` | Model download directory |

### MLX Backend (Apple Silicon)
```python
from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextMLXConfig

config = LightOnTextMLXConfig(
    model="lightonai/LightOnOCR-2-1B",
    max_tokens=4096,
)

extractor = LightOnTextExtractor(backend=config)
```

---

## Usage Examples

### Basic Extraction
```python
from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig
from PIL import Image

extractor = LightOnTextExtractor(
    backend=LightOnTextPyTorchConfig(device="cuda")
)
image = Image.open("document.png")

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

### HTML Output
```python
result = extractor.extract(image, output_format="html")
print(result.content)  # <div>...</div>
```

### Apple Silicon
```python
from omnidocs.tasks.text_extraction import LightOnTextExtractor
from omnidocs.tasks.text_extraction.lighton import LightOnTextMLXConfig

extractor = LightOnTextExtractor(backend=LightOnTextMLXConfig())
result = extractor.extract(image)
print(result.content)
```

---

## Performance

### Memory Requirements

| Backend | VRAM |
|---------|------|
| PyTorch bfloat16 | ~3 GB |
| VLLM | ~4–5 GB |
| MLX bf16 | ~2.5 GB unified memory |

---

## Troubleshooting

### `ImportError: transformers>=5.0.0 required`
```bash
uv pip install "transformers>=5.0.0"
```

### Model not found / slow download
```python
import os
os.environ["HF_HOME"] = "/path/to/cache"  # or set cache_dir in config

config = LightOnTextPyTorchConfig(cache_dir="/path/to/cache")
```

---

## Comparison

| Feature | LightOn | GLM-OCR | Qwen3-VL-8B |
|---------|---------|---------|-------------|
| **Size** | 1B | 0.9B | 8B |
| **VRAM** | ~3 GB | ~3 GB | ~16 GB |
| **Backends** | PyTorch, VLLM, MLX | PyTorch, VLLM, MLX, API | PyTorch, VLLM, MLX, API |
| **Multilingual** | Strong | Good | Excellent (25+ langs) |
| **Self-hosted** | Yes (only) | Yes + API | Yes + API |

**Choose LightOn if**: You need a fully self-hosted 1B OCR model with multi-lingual support.

---

## See Also

- [lightonai/LightOnOCR-2-1B on HuggingFace](https://huggingface.co/lightonai/LightOnOCR-2-1B)
- [GLM-OCR Text Extraction](./glmocr.md)
- [Qwen3-VL Text Extraction](./qwen.md)
- [Comparison Guide](./comparison.md)