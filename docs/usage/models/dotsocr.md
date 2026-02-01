# DotsOCR

Layout-aware text extraction with bounding boxes.

---

## Overview

| | |
|---|---|
| **Tasks** | Text Extraction |
| **Backends** | PyTorch, VLLM |
| **Speed** | 3-5s/page |
| **Quality** | Very Good |
| **VRAM** | 8-12GB |

---

## What Makes It Different

DotsOCR extracts text **with layout information**. Each text block includes:
- Text content
- Bounding box coordinates
- Element category (title, text, table, etc.)

Best for technical documents where structure matters.

---

## Basic Usage

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig
from PIL import Image

image = Image.open("document.png")

extractor = DotsOCRTextExtractor(
    backend=DotsOCRPyTorchConfig(device="cuda")
)

result = extractor.extract(image, output_format="markdown")
print(result.content)
```

---

## With Layout Information

```python
result = extractor.extract(image, include_layout=True)

# Access layout elements
for element in result.layout:
    print(f"[{element.category}] {element.bbox}: {element.text[:50]}...")
```

**Output:**
```
[title] [50, 20, 500, 60]: Introduction to Machine Learning
[text] [50, 80, 900, 200]: Machine learning is a subset of...
[table] [50, 220, 900, 450]: | Model | Accuracy | Speed |...
[figure] [50, 470, 400, 700]: [Figure caption text]
```

---

## Backend Configs

### PyTorch
```python
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

config = DotsOCRPyTorchConfig(
    device="cuda",
    max_new_tokens=8192,  # Increase for long documents
)
```

### VLLM
```python
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

config = DotsOCRVLLMConfig(
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)
```

---

## Layout Categories

DotsOCR detects 11 element types:

| Category | Description |
|----------|-------------|
| `title` | Document/section headings |
| `text` | Body paragraphs |
| `list` | Bullet/numbered lists |
| `table` | Data tables |
| `figure` | Images, diagrams |
| `caption` | Figure/table captions |
| `formula` | Math equations |
| `footnote` | Footnotes |
| `header` | Page headers |
| `footer` | Page footers |
| `abstract` | Abstract sections |

---

## When to Use DotsOCR vs Qwen

| Use Case | Model |
|----------|-------|
| General text extraction | Qwen |
| Need bounding boxes | DotsOCR |
| Technical documents | DotsOCR |
| Tables with coordinates | DotsOCR |
| Fastest extraction | Qwen |
| MLX / API support | Qwen |

---

## Troubleshooting

**Truncated output**
```python
# Increase token limit
config = DotsOCRPyTorchConfig(max_new_tokens=16384)
```

**Missing layout elements**
```python
# Ensure include_layout=True
result = extractor.extract(image, include_layout=True)
```
