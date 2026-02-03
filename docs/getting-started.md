# Getting Started

Everything you need to install OmniDocs, choose a backend, and extract your first document.

---

## 1. Install

```bash
# PyTorch (recommended for most users)
pip install omnidocs[pytorch]

# Or with uv (faster)
uv pip install omnidocs[pytorch]
```

**Other backends:**
```bash
pip install omnidocs[vllm]   # High-throughput production
pip install omnidocs[mlx]    # Apple Silicon
pip install omnidocs[api]    # Cloud-based, no GPU
pip install omnidocs[all]    # Everything
```

**Requirements:**
- Python 3.10, 3.11, or 3.12
- 4GB RAM minimum (8GB+ recommended)
- GPU optional but recommended

**Verify:**
```python
from omnidocs import Document
print("Ready!")
```

---

## 2. Choose Your Backend

```
Mac with Apple Silicon? → MLX
Processing 100+ docs/day? → VLLM
Have NVIDIA GPU? → PyTorch
No GPU? → API
```

| Backend | Speed | Cost | GPU | Best For |
|---------|-------|------|-----|----------|
| **PyTorch** | 1-2s/page | Free | Optional | Development |
| **VLLM** | 0.1s/page | Free | Required | Production scale |
| **MLX** | 2-3s/page | Free | No (Mac) | Apple Silicon |
| **API** | 3-5s/page | $0.01-0.10/doc | No | Quick testing |

---

## 3. Your First Extraction

### Load a Document

```python
from omnidocs import Document

# From PDF
doc = Document.from_pdf("document.pdf")

# From image
doc = Document.from_image("page.png")

# From URL
doc = Document.from_url("https://example.com/doc.pdf")

print(f"Pages: {doc.page_count}")
```

### Extract Text

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Initialize extractor (downloads model on first run)
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",  # Small, fast
        device="cuda",  # or "cpu", "mps"
    )
)

# Extract first page
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

### Complete Example

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Load
doc = Document.from_pdf("research_paper.pdf")

# Initialize once
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(device="cuda")
)

# Process all pages
for i in range(doc.page_count):
    result = extractor.extract(doc.get_page(i), output_format="markdown")
    with open(f"page_{i+1}.md", "w") as f:
        f.write(result.content)
    print(f"Page {i+1} done")
```

---

## 4. Backend Configuration

### PyTorch
```python
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

config = QwenPyTorchConfig(
    model="Qwen/Qwen3-VL-2B-Instruct",  # 2B for speed, 8B for quality
    device="cuda",                       # "cuda", "cpu", "mps"
    torch_dtype="bfloat16",             # "auto", "float16", "bfloat16"
)
```

### VLLM
```python
from omnidocs.tasks.text_extraction.qwen import QwenVLLMConfig

config = QwenVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=1,      # GPUs to use
    gpu_memory_utilization=0.9,  # Memory fraction
)
```

### MLX (Mac only)
```python
from omnidocs.tasks.text_extraction.qwen import QwenMLXConfig

config = QwenMLXConfig(
    model="Qwen/Qwen3-VL-2B-Instruct",
    quantization="4bit",  # "4bit", "8bit", or None
)
```

### API
```python
from omnidocs.tasks.text_extraction.qwen import QwenAPIConfig

config = QwenAPIConfig(
    model="qwen3-vl-8b",
    api_key="YOUR_API_KEY",
    base_url="https://api.provider.com/v1",
)
```

---

## 5. Working with Documents

### Document Class

```python
from omnidocs import Document

doc = Document.from_pdf("file.pdf", dpi=150)  # dpi: 72-300

# Properties
doc.page_count        # Number of pages
doc.metadata          # DocumentMetadata object
doc.text              # Full text (lazy, cached)

# Access pages
page = doc.get_page(0)           # Single page (0-indexed)
for page in doc.iter_pages():    # Memory efficient iteration
    process(page)

# Memory management
doc.clear_cache()     # Free rendered pages
doc.close()           # Release resources
```

### Large Documents

```python
# Load specific pages only
doc = Document.from_pdf("huge.pdf", page_range=(0, 50))

# Process with memory control
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page)
    save_result(result)

    if i % 10 == 0:
        doc.clear_cache()  # Free memory every 10 pages
```

---

## 6. Output Formats

```python
# Markdown (default)
result = extractor.extract(page, output_format="markdown")
# Output: # Heading\n\nParagraph text...

# HTML
result = extractor.extract(page, output_format="html")
# Output: <h1>Heading</h1><p>Paragraph text...</p>

# Access result
print(result.content)   # The extracted text
print(result.format)    # "markdown" or "html"
```

---

## 7. Quick Troubleshooting

**"CUDA out of memory"**
```python
# Use smaller model
config = QwenPyTorchConfig(model="Qwen/Qwen3-VL-2B-Instruct")
```

**"Model download slow"**
- First run downloads ~4-16GB. Cached after.

**"No GPU"**
```bash
pip install omnidocs[api]
# Then use QwenAPIConfig
```

**"Page out of range"**
```python
if page_num < doc.page_count:
    page = doc.get_page(page_num)
```

---

## 8. Other Tasks

OmniDocs supports more than just text extraction:

### Layout Analysis
```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

detector = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
layout = detector.extract(page)

for box in layout.bboxes:
    print(f"{box.label.value}: {box.confidence:.2f}")
```

### Table Extraction
```python
from omnidocs.tasks.table_extraction import TableFormerExtractor, TableFormerConfig

extractor = TableFormerExtractor(config=TableFormerConfig(device="cuda"))
result = extractor.extract(table_image)
df = result.to_dataframe()
```

### Reading Order
```python
from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor

predictor = RuleBasedReadingOrderPredictor()
reading_order = predictor.predict(layout, ocr_result)
text = reading_order.get_full_text()
```

---

## Next Steps

- **[Concepts](concepts.md)** - Understand the architecture
- **[Tasks](usage/tasks/index.md)** - All available tasks (text, layout, OCR, tables, reading order)
- **[Models](usage/models/index.md)** - All available models and their configurations
- **[Usage](usage/index.md)** - Tasks, models, and workflows
