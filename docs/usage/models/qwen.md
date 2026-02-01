# Qwen

Vision-language model for text extraction and layout analysis.

---

## Overview

| | |
|---|---|
| **Tasks** | Text Extraction, Layout Analysis |
| **Backends** | PyTorch, VLLM, MLX, API |
| **Speed** | 2-3s/page |
| **Quality** | Excellent |
| **VRAM** | 8-16GB (8B model) |

---

## Text Extraction

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

doc = Document.from_pdf("document.pdf")

extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(device="cuda")
)

result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

---

## Layout Analysis

```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector
from omnidocs.tasks.layout_analysis.qwen import QwenLayoutPyTorchConfig

detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(device="cuda")
)

result = detector.extract(image)
for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
```

### Custom Labels

```python
from omnidocs.tasks.layout_analysis import CustomLabel

custom_labels = [
    CustomLabel(name="code_block", description="Code snippets"),
    CustomLabel(name="sidebar", description="Sidebar content"),
]

result = detector.extract(image, custom_labels=custom_labels)
```

---

## Backend Configs

### PyTorch (Local GPU)
```python
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

config = QwenPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",              # "cuda", "cpu", "mps"
    torch_dtype="bfloat16",
)
```

### VLLM (High Throughput)
```python
from omnidocs.tasks.text_extraction.qwen import QwenVLLMConfig

config = QwenVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=1,     # GPUs to use
    gpu_memory_utilization=0.9,
)
```

### MLX (Apple Silicon)
```python
from omnidocs.tasks.text_extraction.qwen import QwenMLXConfig

config = QwenMLXConfig(
    model="Qwen/Qwen3-VL-2B-Instruct",
    quantization="4bit",
)
```

### API (Cloud)
```python
from omnidocs.tasks.text_extraction.qwen import QwenAPIConfig

config = QwenAPIConfig(
    model="qwen3-vl-8b",
    api_key="YOUR_API_KEY",
    base_url="https://api.provider.com/v1",
)
```

---

## Model Variants

| Model | Parameters | VRAM | Quality | Speed |
|-------|------------|------|---------|-------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | 4GB | Good | Fast |
| `Qwen/Qwen3-VL-8B-Instruct` | 8B | 16GB | Excellent | Medium |
| `Qwen/Qwen3-VL-32B-Instruct` | 32B | 64GB | Outstanding | Slow |

**Recommendation:** Start with 8B for best quality/speed balance.

---

## Troubleshooting

**CUDA out of memory**
```python
# Use smaller model
config = QwenPyTorchConfig(model="Qwen/Qwen3-VL-2B-Instruct")
```

**Slow inference**
```python
# Use VLLM backend
config = QwenVLLMConfig(tensor_parallel_size=1)
```

**No GPU**
```python
# Use API backend
config = QwenAPIConfig(api_key="...", base_url="...")
```
