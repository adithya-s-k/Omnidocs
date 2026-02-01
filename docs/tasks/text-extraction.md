# Text Extraction

Convert document images to Markdown or HTML.

---

## Models

| Model | Speed | Quality | Backends |
|-------|-------|---------|----------|
| **Qwen3-VL** | 2-3s/page | Excellent | PyTorch, VLLM, MLX, API |
| **DotsOCR** | 3-5s/page | Very Good | PyTorch, VLLM |

**Recommendation:** Start with Qwen3-VL-8B for most use cases.

---

## Basic Usage

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Load document
doc = Document.from_pdf("document.pdf")

# Initialize extractor
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen3-VL-8B-Instruct",
        device="cuda",
    )
)

# Extract text
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

---

## Output Formats

```python
# Markdown (default)
result = extractor.extract(page, output_format="markdown")
# Output: # Heading\n\nParagraph text...

# HTML
result = extractor.extract(page, output_format="html")
# Output: <h1>Heading</h1><p>Paragraph text...</p>
```

---

## Backend Configs

### PyTorch (Local GPU)
```python
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

config = QwenPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",              # "cuda", "cpu", or "mps"
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

## Process Multiple Pages

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

doc = Document.from_pdf("document.pdf")
extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))

# Process all pages
for i in range(doc.page_count):
    result = extractor.extract(doc.get_page(i), output_format="markdown")
    with open(f"page_{i+1}.md", "w") as f:
        f.write(result.content)
```

---

## DotsOCR (Layout-Aware)

DotsOCR includes layout information with extracted text.

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

config = DotsOCRPyTorchConfig(device="cuda")
extractor = DotsOCRTextExtractor(backend=config)

result = extractor.extract(image, include_layout=True)

# Access layout elements
for element in result.layout:
    print(f"[{element.category}] {element.text[:50]}...")
```

---

## Troubleshooting

**CUDA out of memory**
```python
# Use smaller model
config = QwenPyTorchConfig(model="Qwen/Qwen3-VL-2B-Instruct")
```

**Slow inference**
```python
# Switch to VLLM
config = QwenVLLMConfig(tensor_parallel_size=1)
```

**No GPU available**
```python
# Use API backend
config = QwenAPIConfig(api_key="...", base_url="...")
```
