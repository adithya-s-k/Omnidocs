# GLM-OCR Text Extraction

## Model Overview

GLM-OCR is a purpose-built OCR model from ZhipuAI, ranked #1 on OmniDocBench V1.5. At only 0.9B parameters it is significantly faster and cheaper than general-purpose VLMs while delivering excellent text extraction quality.

**Model ID**: `zai-org/GLM-OCR`
**Repository**: [zai-org/GLM-OCR on HuggingFace](https://huggingface.co/zai-org/GLM-OCR)
**Architecture**: Vision Encoder + Language Model (0.9B parameters)
**Release**: February 2026

### Key Capabilities

- **#1 on OmniDocBench V1.5**: State-of-the-art OCR benchmark performance
- **Lightweight**: 0.9B parameters — runs on a single A10G or consumer GPU (~3GB VRAM)
- **Fast**: Significantly faster than 7B+ general-purpose VLMs
- **Clean Output**: No `<think>` tokens, no `<|begin_of_box|>` artifacts
- **Multi-backend**: PyTorch, VLLM, MLX, and API (self-hosted vLLM)

### Limitations

- Requires `transformers>=5.3.0` — conflicts with `docling-ibm-models` which pins `<5.0.0`
- Optimised for pure OCR — not a general-purpose VLM
- No built-in layout/bounding box output

---

## Supported Backends

| Backend | Use Case | Performance | Setup |
|---------|----------|-------------|-------|
| **PyTorch** | Local GPU inference | Fast (0.9B is small) | Single GPU, easy |
| **VLLM** | High-throughput batching | Very fast with MTP | Requires vllm>=0.17.0 |
| **MLX** | Apple Silicon (native) | Good on M-series | macOS M1/M2/M3+ only |
| **API** | Self-hosted vLLM server | Variable | OpenAI-compatible endpoint |

---

## Installation
```bash
# PyTorch backend
pip install omnidocs[pytorch]

# VLLM backend
pip install omnidocs[vllm]

# GLM-OCR requires transformers>=5.3.0
pip install "transformers>=5.3.0"
```

> **Note**: If `docling-ibm-models` is installed it pins `transformers<5.0.0`. Use an override:
> ```bash
> uv pip install omnidocs[pytorch] --override <(echo 'transformers>=5.3.0')
> ```

---

## Configuration

### PyTorch Backend
```python
from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

config = GLMOCRPyTorchConfig(
    model="zai-org/GLM-OCR",
    device="cuda",
    torch_dtype="bfloat16",
    device_map="auto",
    use_flash_attention=False,  # Set True if flash-attn installed
    max_new_tokens=4096,
    temperature=0.0,            # Greedy decoding recommended for OCR
)
extractor = GLMOCRTextExtractor(backend=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"zai-org/GLM-OCR"` | HuggingFace model ID |
| `device` | str | `"cuda"` | `"cuda"`, `"mps"`, `"cpu"` |
| `torch_dtype` | str | `"bfloat16"` | `"float16"`, `"bfloat16"`, `"float32"`, `"auto"` |
| `device_map` | str | `"auto"` | Model parallelism strategy |
| `use_flash_attention` | bool | `False` | Flash Attention 2 (requires flash-attn) |
| `cache_dir` | str | `None` | Model cache directory |
| `max_new_tokens` | int | `4096` | Max tokens to generate (256–16384) |
| `temperature` | float | `0.0` | Sampling temperature (0.0 = greedy) |

### VLLM Backend

GLM-OCR supports MTP (Multi-Token Prediction) speculative decoding in VLLM for significantly higher throughput.
```python
from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

config = GLMOCRVLLMConfig(
    model="zai-org/GLM-OCR",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    temperature=0.0,
    repetition_penalty=1.05,    # Prevents looping at temperature=0.0
)
extractor = GLMOCRTextExtractor(backend=config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor_parallel_size` | int | `1` | Number of GPUs |
| `gpu_memory_utilization` | float | `0.85` | GPU memory fraction |
| `max_model_len` | int | `8192` | Maximum sequence length |
| `enforce_eager` | bool | `False` | Disable CUDA graph optimisation |
| `max_tokens` | int | `4096` | Max tokens to generate |
| `repetition_penalty` | float | `1.05` | Penalty to prevent output loops |

### MLX Backend (Apple Silicon)
```python
from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRMLXConfig

config = GLMOCRMLXConfig(
    model="mlx-community/GLM-OCR-bf16",  # or "mlx-community/GLM-OCR-6bit"
    max_tokens=4096,
)
extractor = GLMOCRTextExtractor(backend=config)
```

| Model | Size | Notes |
|-------|------|-------|
| `mlx-community/GLM-OCR-bf16` | ~2.2 GB | Full precision, default |
| `mlx-community/GLM-OCR-6bit` | Smaller | Quantized |

### API Backend (Self-hosted vLLM)
```python
from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

config = GLMOCRAPIConfig(
    model="openai/glm-ocr",             # litellm format matching --served-model-name
    api_base="http://localhost:8000/v1",
    api_key="token-abc",
    max_tokens=4096,
    repetition_penalty=1.05,
)
extractor = GLMOCRTextExtractor(backend=config)
```

Start a local vLLM server:
```bash
vllm serve zai-org/GLM-OCR \
    --served-model-name glm-ocr \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 \
    --enforce-eager
```

---

## Usage Examples

### Basic Extraction
```python
from omnidocs.tasks.text_extraction import GLMOCRTextExtractor
from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig
from PIL import Image

extractor = GLMOCRTextExtractor(backend=GLMOCRPyTorchConfig(device="cuda"))
image = Image.open("document.png")

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

### HTML Output
```python
result = extractor.extract(image, output_format="html")
print(result.content)  # <div>...</div>
```

---

## Performance

### Memory Requirements

| Backend | VRAM |
|---------|------|
| PyTorch bfloat16 | ~3 GB |
| VLLM | ~4 GB |
| MLX bf16 | ~2.2 GB unified memory |
| MLX 6bit | ~1.5 GB unified memory |

---

## Troubleshooting

### `ImportError: transformers>=5.3.0 required`
```bash
uv pip install "transformers>=5.3.0"
```

### VLLM output looping

At `temperature=0.0` output can loop. Increase the repetition penalty:
```python
config = GLMOCRVLLMConfig(repetition_penalty=1.1)
```

---

## Comparison

| Feature | GLM-OCR | Qwen3-VL-8B | DotsOCR |
|---------|---------|-------------|---------|
| **Size** | 0.9B | 8B | 8B |
| **OCR Quality** | #1 OmniDocBench | Excellent | Very Good |
| **VRAM** | ~3 GB | ~16 GB | ~16 GB |
| **Backends** | PyTorch, VLLM, MLX, API | PyTorch, VLLM, MLX, API | PyTorch, VLLM |
| **Multilingual** | Good | Excellent (25+ langs) | Limited |
| **Layout info** | No | Basic | Yes (11 categories) |

**Choose GLM-OCR if**: You need fast, accurate OCR with minimal VRAM.

---

## See Also

- [zai-org/GLM-OCR on HuggingFace](https://huggingface.co/zai-org/GLM-OCR)
- [LightOn Text Extraction](./lighton.md)
- [Qwen3-VL Text Extraction](./qwen.md)
- [Comparison Guide](./comparison.md)