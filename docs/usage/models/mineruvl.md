# MinerU VL

Vision-language model for layout-aware document extraction with specialized table and equation recognition.

---

## Overview

| | |
|---|---|
| **Tasks** | Text Extraction, Layout Analysis |
| **Backends** | PyTorch, VLLM, MLX, API |
| **Speed** | 3-6s/page |
| **Quality** | Excellent (especially tables/equations) |
| **VRAM** | 3-4GB |

MinerU VL performs two-step extraction:

1. **Layout Detection** - Detects 22+ element types (text, tables, equations, figures, code, etc.)
2. **Content Recognition** - Extracts text/table/equation content from each region

---

## Text Extraction

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import MinerUVLTextExtractor
from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

doc = Document.from_pdf("document.pdf")

extractor = MinerUVLTextExtractor(
    backend=MinerUVLTextPyTorchConfig(device="cuda")
)

result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

### With Detailed Blocks

```python
# Get both text output and detailed block information
result, blocks = extractor.extract_with_blocks(image, output_format="markdown")

for block in blocks:
    print(f"{block.type}: {block.bbox}")
    print(f"  Content: {block.content[:50]}...")
```

---

## Layout Analysis

```python
from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector
from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

detector = MinerUVLLayoutDetector(
    backend=MinerUVLLayoutPyTorchConfig(device="cuda")
)

result = detector.extract(image)
for box in result.bboxes:
    print(f"{box.label}: {box.bbox}")
```

### Detected Element Types

MinerU VL detects 22+ element types:

| Category | Types |
|----------|-------|
| **Text** | text, title, header, footer, page_number |
| **Tables** | table, table_caption, table_footnote |
| **Math** | equation, equation_block |
| **Code** | code, algorithm, code_caption |
| **Images** | image, image_caption, image_footnote |
| **Other** | list, ref_text, aside_text, phonetic |

---

## Model Caching

MinerU VL supports intelligent model caching. When you create both a text extractor and layout detector, they share the same underlying model:

```python
from omnidocs import get_cache_info

# First extractor loads the model (~4s)
text_extractor = MinerUVLTextExtractor(backend=MinerUVLTextMLXConfig())

# Second extractor reuses cached model (instant)
layout_detector = MinerUVLLayoutDetector(backend=MinerUVLLayoutMLXConfig())

# Check cache status
print(get_cache_info())
# {'num_entries': 1, 'entries': {..., 'ref_count': 2}}
```

Configure cache behavior:

```python
from omnidocs import set_cache_config

# Limit cached models (default: 10)
set_cache_config(max_entries=5)

# Clear cache to free memory
from omnidocs import clear_cache
clear_cache()
```

---

## Backend Configs

### PyTorch (Local GPU)

```python
from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

config = MinerUVLTextPyTorchConfig(
    model="opendatalab/MinerU2.5-2509-1.2B",
    device="cuda",              # "cuda", "cpu", "auto"
    torch_dtype="float16",      # "float16", "bfloat16", "float32"
    use_flash_attention=False,  # Use SDPA by default
)
```

### VLLM (High Throughput)

```python
from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

config = MinerUVLTextVLLMConfig(
    model="opendatalab/MinerU2.5-2509-1.2B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)
```

### MLX (Apple Silicon)

```python
from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

config = MinerUVLTextMLXConfig(
    model="opendatalab/MinerU2.5-2509-1.2B",
    max_tokens=4096,
)
```

### API (VLLM Server)

```python
from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextAPIConfig

config = MinerUVLTextAPIConfig(
    server_url="http://localhost:8000",
    model_name="opendatalab/MinerU2.5-2509-1.2B",
    max_tokens=4096,
)
```

---

## Output Formats

### Markdown (Default)

```python
result = extractor.extract(image, output_format="markdown")
# # Title
#
# This is paragraph text.
#
# | Header 1 | Header 2 |
# |----------|----------|
# | Cell 1   | Cell 2   |
#
# $$E = mc^2$$
```

### HTML

```python
result = extractor.extract(image, output_format="html")
# <h1>Title</h1>
# <p>This is paragraph text.</p>
# <table>...</table>
# <math>E = mc^2</math>
```

---

## Comparison with Qwen

| Feature | MinerU VL | Qwen VL |
|---------|-----------|---------|
| **Model Size** | 1.2B | 2B-32B |
| **VRAM** | 3-4GB | 4-64GB |
| **Table Quality** | Excellent (OTSL format) | Good |
| **Equation Quality** | Excellent (LaTeX) | Good |
| **General Text** | Good | Excellent |
| **Speed** | 3-6s/page | 2-3s/page |

**Recommendation:** Use MinerU VL for documents with complex tables and equations. Use Qwen for general-purpose text extraction.

---

## Troubleshooting

**CUDA out of memory**
```python
# MinerU VL is already small (1.2B), try reducing batch size
# or use MLX on Apple Silicon
config = MinerUVLTextMLXConfig()
```

**Tables not rendering correctly**
```python
# Use HTML output format for better table rendering
result = extractor.extract(image, output_format="html")
```

**Slow inference**
```python
# Use VLLM backend for production
config = MinerUVLTextVLLMConfig(
    gpu_memory_utilization=0.9,
    enforce_eager=True,
)
```

---

## Attribution

MinerU VL utilities include code adapted from [mineru-vl-utils](https://github.com/opendatalab/mineru-vl-utils), licensed under AGPL-3.0.
