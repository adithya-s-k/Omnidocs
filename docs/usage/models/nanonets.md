# Nanonets OCR2

Nanonets OCR2-3B is a Vision-Language Model optimized for document text extraction with excellent accuracy on diverse document types.

---

## Overview

| Property | Value |
|----------|-------|
| **Model** | `nanonets/Nanonets-OCR-s` |
| **Parameters** | 3B |
| **Task** | Text Extraction |
| **Backends** | PyTorch, VLLM, MLX |
| **License** | Apache 2.0 |

---

## Installation

```bash
# PyTorch backend
pip install omnidocs[pytorch]

# VLLM backend (high throughput)
pip install omnidocs[vllm]

# MLX backend (Apple Silicon)
pip install omnidocs[mlx]
```

---

## Quick Start

### PyTorch Backend

```python
from omnidocs.tasks.text_extraction import NanonetsTextExtractor
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

extractor = NanonetsTextExtractor(
    backend=NanonetsTextPyTorchConfig(device="cuda")
)
result = extractor.extract(image, output_format="markdown")
print(result.content)
```

### VLLM Backend (High Throughput)

```python
from omnidocs.tasks.text_extraction import NanonetsTextExtractor
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextVLLMConfig

extractor = NanonetsTextExtractor(
    backend=NanonetsTextVLLMConfig(
        gpu_memory_utilization=0.85,
        max_model_len=8192,
    )
)
result = extractor.extract(image)
```

### MLX Backend (Apple Silicon)

```python
from omnidocs.tasks.text_extraction import NanonetsTextExtractor
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextMLXConfig

extractor = NanonetsTextExtractor(
    backend=NanonetsTextMLXConfig()
)
result = extractor.extract(image)
```

---

## Configuration

### PyTorch Config

```python
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

config = NanonetsTextPyTorchConfig(
    model="nanonets/Nanonets-OCR-s",  # Model ID
    device="cuda",                     # "cuda", "cpu", or "mps"
    torch_dtype="bfloat16",           # "float16", "bfloat16", "float32"
    attn_implementation="flash_attention_2",  # or "sdpa", "eager"
)
```

### VLLM Config

```python
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextVLLMConfig

config = NanonetsTextVLLMConfig(
    model="nanonets/Nanonets-OCR-s",
    gpu_memory_utilization=0.85,  # GPU memory fraction
    max_model_len=8192,           # Max sequence length
    tensor_parallel_size=1,       # Multi-GPU parallelism
)
```

### MLX Config

```python
from omnidocs.tasks.text_extraction.nanonets import NanonetsTextMLXConfig

config = NanonetsTextMLXConfig(
    model="nanonets/Nanonets-OCR-s",
    max_tokens=4096,
)
```

---

## Output

```python
result = extractor.extract(image, output_format="markdown")

# Access content
print(result.content)       # Extracted Markdown text
print(result.model_name)    # "nanonets/Nanonets-OCR-s"
print(result.output_format) # "markdown"
```

---

## Performance

| Backend | Device | Load Time | Inference Time |
|---------|--------|-----------|----------------|
| PyTorch | L4 GPU | ~44s | ~6.3s |
| VLLM | L4 GPU | ~194s | ~8.4s |
| MLX | M1/M2/M3 | ~8s | ~12s |

*Times measured on single-page document with default settings.*

---

## Comparison with Other Models

| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| Nanonets OCR2 | Fast | High | 6-8 GB |
| Qwen3-VL | Medium | High | 8-16 GB |
| DotsOCR | Medium | High | 6-8 GB |

---

## Use Cases

- **Document digitization** - Convert scanned documents to editable text
- **Invoice processing** - Extract text from invoices and receipts
- **Form processing** - Extract text from forms and applications
- **OCR pipelines** - High-throughput batch processing

---

## Tips

1. **Use VLLM for batch processing** - Higher throughput for multiple documents
2. **Use MLX on Mac** - Native performance on Apple Silicon
3. **Set output_format** - Use `"markdown"` for formatted output, `"text"` for plain text
