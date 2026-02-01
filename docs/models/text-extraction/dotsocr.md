# DotsOCR Text Extraction

## Model Overview

DotsOCR (Deep Object Text Segmentation OCR) is a specialized Vision-Language Model designed specifically for document understanding with built-in layout analysis. Unlike general-purpose VLMs, DotsOCR outputs structured information about document layout while extracting text content.

**Model ID**: rednote-hilab/dots.ocr
**Repository**: [DotsOCR on HuggingFace](https://huggingface.co/rednote-hilab/dots.ocr)
**Architecture**: Vision Encoder + Language Model
**Training Focus**: Academic papers, technical documents, PDFs

### Key Capabilities

- **Layout-Aware Extraction**: Detects 11 document element categories with bounding boxes
- **Multi-Format Text**: Different formats per category (Markdown, LaTeX, HTML)
- **Fast Inference**: 50-100% faster than general-purpose VLMs
- **Normalized Coordinates**: All bboxes in 0-1024 range (scale-independent)
- **Reading Order**: Maintains document reading order
- **Format-Specific Output**:
  - Text/Title/Section-header: Markdown
  - Formula: LaTeX
  - Table: HTML
  - Picture: Bounding box only (no text)

### Limitations

- PyTorch and VLLM backends only (no MLX, no API)
- Optimized for academic/technical documents (less good for forms, invoices)
- Fixed layout categories (cannot add custom categories)
- Requires GPU (minimum 16GB VRAM for 8B variant)
- Output is JSON-focused (not raw markdown like Qwen)

---

## Supported Backends

DotsOCR supports **2 inference backends**:

| Backend | Use Case | Performance | Setup |
|---------|----------|-------------|-------|
| **PyTorch** | Single document, development | 50-100 tok/s | Simple GPU setup |
| **VLLM** | Batch processing, production | 150-300 tok/s | Multi-GPU cluster |

No MLX or API backends available.

---

## Installation & Configuration

### Basic Installation

```bash
# Install with PyTorch backend
pip install omnidocs[pytorch]

# Or with VLLM for batching
pip install omnidocs[vllm]
```

### PyTorch Backend Configuration

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

config = DotsOCRPyTorchConfig(
    model="rednote-hilab/dots.ocr",
    device="cuda",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="flash_attention_2",  # Recommended
)

extractor = DotsOCRTextExtractor(backend=config)
```

**PyTorch Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "rednote-hilab/dots.ocr" | HuggingFace model ID |
| `device` | str | "cuda" | Device: "cuda", "mps", "cpu" |
| `torch_dtype` | str | "bfloat16" | Data type: "float16", "bfloat16", "float32" |
| `trust_remote_code` | bool | True | Allow custom model code from HuggingFace |
| `device_map` | str | "auto" | Model parallelism: "auto", "balanced", "sequential" |
| `attn_implementation` | str | "flash_attention_2" | Attention type: "eager", "flash_attention_2", "sdpa" |

### VLLM Backend Configuration

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

config = DotsOCRVLLMConfig(
    model="rednote-hilab/dots.ocr",
    tensor_parallel_size=1,  # Use 2+ for large models
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)

extractor = DotsOCRTextExtractor(backend=config)
```

**VLLM Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | HuggingFace model ID |
| `tensor_parallel_size` | int | 1 | Number of GPUs for parallelism |
| `gpu_memory_utilization` | float | 0.85 | GPU memory usage (0.1-1.0) |
| `max_model_len` | int | None | Max context length in tokens |

---

## Layout Categories (11 Fixed)

DotsOCR recognizes exactly 11 layout element categories:

| Category | Description | Text Format | Typical Content |
|----------|-------------|-------------|-----------------|
| **Title** | Document/section title | Markdown | "Introduction", "Chapter 2" |
| **Section-header** | Subsection heading | Markdown | "3.1 Method Overview" |
| **Text** | Body paragraph | Markdown | Main content paragraphs |
| **List-item** | Bulleted/numbered item | Markdown | "1. First point", "• Item" |
| **Table** | Tabular data | HTML | `<table><tr><td>...</td>...` |
| **Formula** | Mathematical equation | LaTeX | `$E=mc^2$` or display math |
| **Figure** | Image/figure/diagram | None | Bounding box only |
| **Caption** | Figure/table caption | Markdown | "Fig 1: System Overview" |
| **Footnote** | Footer note | Markdown | Explanatory footnotes |
| **Page-header** | Page header text | Markdown | Page number, document title |
| **Page-footer** | Page footer text | Markdown | Page number, author name |

---

## Usage Examples

### Basic Layout-Aware Extraction

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig
from PIL import Image

# Initialize extractor
config = DotsOCRPyTorchConfig(
    model="rednote-hilab/dots.ocr",
    device="cuda",
)
extractor = DotsOCRTextExtractor(backend=config)

# Load document
image = Image.open("paper.png")

# Extract with layout information
result = extractor.extract(
    image,
    include_layout=True,  # Returns DotsOCRTextOutput
)

# Access layout elements
print(f"Found {result.num_layout_elements} layout elements")
for elem in result.layout:
    print(f"  {elem.category} @ {elem.bbox}: {elem.text[:50]}...")
```

### Output Format Examples

```python
# Default: DotsOCRTextOutput with layout
result = extractor.extract(image)

# Access structured layout
for elem in result.layout:
    category = elem.category  # "Title", "Text", "Table", etc.
    bbox = elem.bbox          # [x1, y1, x2, y2] (0-1024 normalized)
    text = elem.text          # Content (formatted per category)
    confidence = elem.confidence  # Detection confidence

print(result.content)        # Full text (Markdown)
print(result.format)         # "markdown" (fixed)
print(result.has_layout)     # True
print(result.content_length) # Total character count
print(result.image_width)    # Source image width
print(result.image_height)   # Source image height
```

### Category-Specific Processing

```python
# Extract only formulas (as LaTeX)
formulas = [
    elem for elem in result.layout
    if elem.category == "Formula"
]

for formula in formulas:
    print(f"Formula @ {formula.bbox}:")
    print(formula.text)  # LaTeX format
    print()

# Extract tables (as HTML)
tables = [
    elem for elem in result.layout
    if elem.category == "Table"
]

for table in tables:
    print(f"Table @ {table.bbox}:")
    print(table.text)  # HTML table
    print()

# Extract all text content (non-figure)
text_elements = [
    elem for elem in result.layout
    if elem.category not in ["Figure", "Page-header", "Page-footer"]
]

full_text = "\n".join(elem.text for elem in text_elements)
print(full_text)  # Cleaned text without layout markers
```

### Bounding Box Operations

```python
# Access normalized bounding boxes (0-1024 scale)
for elem in result.layout:
    x1, y1, x2, y2 = elem.bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height

    print(f"{elem.category}: {width}x{height} at ({x1}, {y1})")

# Filter elements by region (e.g., top half of page)
top_half = [
    elem for elem in result.layout
    if elem.bbox[1] < 512  # y1 < midpoint
]

# Filter by size
large_elements = [
    elem for elem in result.layout
    if (elem.bbox[2] - elem.bbox[0]) * (elem.bbox[3] - elem.bbox[1]) > 102400
]
```

### Batch Processing with VLLM

```python
from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig
from PIL import Image
import json

# Initialize with VLLM
config = DotsOCRVLLMConfig(
    model="rednote-hilab/dots.ocr",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8,
)
extractor = DotsOCRTextExtractor(backend=config)

# Process multiple documents
documents = ["doc1.png", "doc2.png", "doc3.png"]
results = []

for doc_path in documents:
    image = Image.open(doc_path)
    result = extractor.extract(image, include_layout=True)

    results.append({
        "file": doc_path,
        "elements": len(result.layout),
        "content_length": result.content_length,
        "layout": [
            {
                "category": elem.category,
                "bbox": elem.bbox,
                "text_length": len(elem.text) if elem.text else 0,
            }
            for elem in result.layout
        ]
    })

# Save results
with open("extraction_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Performance Characteristics

### Memory Requirements

| Model | Framework | VRAM | Batch Size |
|-------|-----------|------|-----------|
| DotsOCR | PyTorch | 16 GB | 1 (single doc) |
| DotsOCR | VLLM | 20 GB | 2-4 |
| DotsOCR | VLLM (2-GPU) | 20 GB (per GPU) | 6-10 |

### Inference Speed

| Setup | Speed | Throughput |
|-------|-------|-----------|
| PyTorch (single A10) | 50-80 tok/s | ~400-600 chars/s |
| VLLM (single A10) | 150-200 tok/s | ~1200-1600 chars/s |
| VLLM (2x A10) | 250-350 tok/s | ~2000-2800 chars/s |

### Typical Processing Times

| Document | Tokens | Time (PyTorch) | Time (VLLM) |
|----------|--------|----------------|------------|
| Single page | 1000-2000 | 12-25s | 5-10s |
| 5 pages | 5000-10000 | 60-130s | 20-40s |
| 10 pages | 10000-20000 | 130-260s | 40-80s |

---

## Troubleshooting

### Memory Errors

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# 1. Use CPU (slow but works)
config = DotsOCRPyTorchConfig(device="cpu")

# 2. Reduce image size before processing
from PIL import Image
image = Image.open("document.png")
image.thumbnail((2048, 2048))  # Resize if larger

# 3. Use VLLM with memory management
config = DotsOCRVLLMConfig(
    gpu_memory_utilization=0.7,  # Reduced from 0.85
    max_model_len=2048,  # Reduced from 4096
)
```

### Layout Parsing Errors

**Symptom**: `ValueError: Invalid layout JSON structure`

**Solution**:

```python
# Check raw output for issues
result = extractor.extract(image, include_layout=True)

if result.error:
    print(f"Extraction error: {result.error}")
    print(f"Raw output: {result.raw_output[:500]}...")

# Ensure image is valid
if image.size[0] < 256 or image.size[1] < 256:
    print("Image too small for reliable layout detection")
```

### Missing Layout Categories

**Symptom**: Some expected elements not detected

**Solutions**:

```python
# Check what was detected
detected_categories = set(
    elem.category for elem in result.layout
)
print(f"Found: {detected_categories}")

# Element may be below confidence threshold
# Access raw output to see low-confidence detections
print(result.raw_output)

# Try with different preprocessing
from PIL import ImageEnhance
image = Image.open("document.png")
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.3)
result = extractor.extract(image)
```

---

## DotsOCR vs Other Models

### DotsOCR vs Qwen3-VL

| Feature | DotsOCR | Qwen3-VL |
|---------|---------|----------|
| **Layout Info** | Detailed (11 cats) | Basic |
| **Output Format** | JSON + Markdown | Markdown/HTML |
| **Speed** | Fast | Medium |
| **Text Quality** | Good | Excellent |
| **Multilingual** | Limited | Excellent (25+ langs) |
| **Backends** | PyTorch, VLLM | PyTorch, VLLM, MLX, API |
| **Best For** | Layout analysis | Text quality |

**Choose DotsOCR if**: You need precise layout information for post-processing
**Choose Qwen if**: You need high-quality text in multiple languages

### When to Use DotsOCR

**Ideal scenarios**:
- Academic papers with structured layouts
- Technical documents with formulas and tables
- Batch processing with layout analysis
- When you need bounding boxes for each element

**Not ideal for**:
- Handwritten documents (use Surya)
- Forms with complex fields (use specialized form parser)
- Real-time single-document processing (overhead > benefit)
- Custom layout categories needed

---

## API Reference

### DotsOCRTextExtractor.extract()

```python
def extract(
    image: Union[Image.Image, np.ndarray, str, Path],
    include_layout: bool = True,
    output_format: str = "markdown",
) -> DotsOCRTextOutput:
    """
    Extract text with layout from document image.

    Args:
        image: Input image (PIL Image, numpy array, or path)
        include_layout: Include layout elements with bboxes (default: True)
        output_format: "markdown" or "json" (fixed)

    Returns:
        DotsOCRTextOutput with layout elements and text
    """
```

### DotsOCRTextOutput Properties

```python
result = extractor.extract(image)

# Layout information
result.layout                   # List[LayoutElement]
result.has_layout              # True
result.num_layout_elements     # int

# Text content
result.content                 # Full text (Markdown)
result.format                  # "markdown"
result.content_length          # Characters

# Element categories
result.layout_categories       # List of 11 categories

# Metadata
result.image_width            # Source image width
result.image_height           # Source image height
result.truncated              # Output hit max tokens
result.error                  # Error message if any
result.raw_output             # Raw model JSON
```

### LayoutElement Properties

```python
for elem in result.layout:
    elem.category        # "Title", "Text", "Table", etc.
    elem.bbox            # [x1, y1, x2, y2] (0-1024)
    elem.text            # Content (Markdown/LaTeX/HTML)
    elem.confidence      # float (0-1) - detection confidence
```

---

## Advanced Usage

### Post-Processing: Extract Figures

```python
# Get all figures with their captions
figures = {}
for elem in result.layout:
    if elem.category == "Figure":
        bbox = elem.bbox
        figures[str(bbox)] = {
            "bbox": bbox,
            "caption": None,
        }
    elif elem.category == "Caption":
        # Find nearest figure
        # (could implement spatial matching here)
        pass

for fig_bbox, fig_data in figures.items():
    print(f"Figure @ {fig_data['bbox']}")
    print(f"  Caption: {fig_data['caption']}")
```

### Export to Structured Format

```python
import json
from dataclasses import asdict

# Convert to JSON-serializable format
output_data = {
    "document": {
        "width": result.image_width,
        "height": result.image_height,
    },
    "elements": [
        {
            "category": elem.category,
            "bbox": {
                "x1": elem.bbox[0],
                "y1": elem.bbox[1],
                "x2": elem.bbox[2],
                "y2": elem.bbox[3],
            },
            "text": elem.text,
            "confidence": elem.confidence,
        }
        for elem in result.layout
    ]
}

# Save
with open("layout_analysis.json", "w") as f:
    json.dump(output_data, f, indent=2)
```

### Visualization with Bounding Boxes

```python
from PIL import Image, ImageDraw

# Load original image
image = Image.open("document.png")
img_w, img_h = image.size

# Create visualization
viz = image.copy()
draw = ImageDraw.Draw(viz)

# Color map for categories
colors = {
    "Title": "red",
    "Text": "blue",
    "Table": "orange",
    "Formula": "purple",
    "Figure": "green",
}

# Draw bounding boxes
for elem in result.layout:
    # Convert from 0-1024 to pixel coordinates
    bbox = [
        (elem.bbox[0] / 1024) * img_w,
        (elem.bbox[1] / 1024) * img_h,
        (elem.bbox[2] / 1024) * img_w,
        (elem.bbox[3] / 1024) * img_h,
    ]

    color = colors.get(elem.category, "gray")
    draw.rectangle(bbox, outline=color, width=3)
    draw.text((bbox[0], bbox[1] - 15), elem.category, fill=color)

# Save
viz.save("layout_visualization.png")
```

---

## See Also

- [Qwen3-VL Text Extraction](./qwen.md) - For pure text quality
- [DotsOCR Repository](https://github.com/rednote-hilab/dots.ocr)
- [Comparison Guide](./comparison.md) - Model selection matrix
