# Qwen3-VL Layout Detection

## Model Overview

Qwen3-VL is a Vision-Language Model that can perform flexible layout detection beyond fixed label sets. Unlike DocLayout-YOLO or RT-DETR, Qwen supports **custom layout categories** while maintaining high accuracy for standard layout analysis tasks.

**Model Family**: Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B, Qwen3-VL-32B
**Repository**: [Qwen/Qwen3-VL](https://huggingface.co/Qwen)
**Architecture**: Vision Encoder + Language Model
**Key Feature**: Flexible custom labels for domain-specific layout detection

### Key Capabilities

- **Custom Labels**: Define unlimited layout categories beyond standard types
- **VLM-Based**: Understands semantic meaning of regions
- **High Accuracy**: Better handling of complex/unusual layouts
- **Multi-Backend**: PyTorch, VLLM, MLX, API support
- **Confidence Scores**: Per-detection confidence for filtering
- **Multilingual**: Works with documents in any language

### Limitations

- **Slower than YOLO**: 5-10x slower than DocLayout-YOLO
- **Requires GPU**: No CPU inference practical
- **Memory intensive**: 8-16 GB VRAM minimum
- **Less standardized**: Labels are user-defined, not fixed enum

---

## Supported Backends

Qwen layout detection supports **4 inference backends**:

| Backend | Use Case | Speed | Setup |
|---------|----------|-------|-------|
| **PyTorch** | Single document, development | 20-50 tok/s | Easy |
| **VLLM** | Batch processing | 80-150 tok/s | Multi-GPU |
| **MLX** | Apple Silicon | 10-30 tok/s | macOS M1/M3+ |
| **API** | Cloud inference | Variable | Hosted |

---

## Installation & Configuration

### Basic Installation

```bash
# Install with PyTorch backend (most common)
pip install omnidocs[pytorch]

# Or with all backends
pip install omnidocs[all]
```

### PyTorch Backend Configuration

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

config = QwenLayoutPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
    device_map="auto",
    max_new_tokens=4096,
    temperature=0.1,
)

detector = QwenLayoutDetector(backend=config)
```

**PyTorch Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "Qwen/Qwen3-VL-8B-Instruct" | HuggingFace model ID |
| `device` | str | "cuda" | Device: "cuda", "mps", "cpu" |
| `torch_dtype` | str | "auto" | Data type: "float16", "bfloat16", "float32", "auto" |
| `device_map` | str | "auto" | Model parallelism strategy |
| `use_flash_attention` | bool | False | Use Flash Attention 2 (if available) |
| `max_new_tokens` | int | 4096 | Max tokens to generate |
| `temperature` | float | 0.1 | Sampling temperature (deterministic output) |

### VLLM Backend Configuration

```python
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

config = QwenLayoutVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

detector = QwenLayoutDetector(backend=config)
```

### MLX Backend Configuration (Apple Silicon)

```python
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutMLXConfig

config = QwenLayoutMLXConfig(
    model="Qwen/Qwen3-VL-8B-Instruct-MLX",
    quantization="4bit",
    max_tokens=4096,
)

detector = QwenLayoutDetector(backend=config)
```

### API Backend Configuration

```python
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

config = QwenLayoutAPIConfig(
    model="qwen3-vl-8b",
    api_key="your-api-key",
    base_url="https://api.provider.com/v1",
)

detector = QwenLayoutDetector(backend=config)
```

---

## Standard Label Detection

### Using Fixed Standard Labels

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector, LayoutLabel
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
from PIL import Image

# Initialize
config = QwenLayoutPyTorchConfig(device="cuda")
detector = QwenLayoutDetector(backend=config)

# Load image
image = Image.open("document.png")

# Detect standard layout (no custom labels)
result = detector.extract(image)

# Access standard labels
print(f"Found {result.element_count} elements")
for box in result.bboxes:
    print(f"  {box.label.value}: confidence={box.confidence:.2f}")

# Filter by standard label type
titles = result.filter_by_label(LayoutLabel.TITLE)
text_blocks = result.filter_by_label(LayoutLabel.TEXT)
figures = result.filter_by_label(LayoutLabel.FIGURE)
tables = result.filter_by_label(LayoutLabel.TABLE)

print(f"Titles: {len(titles)}")
print(f"Text blocks: {len(text_blocks)}")
print(f"Figures: {len(figures)}")
print(f"Tables: {len(tables)}")
```

**Standard Layout Labels**:

| Label | Description |
|-------|-------------|
| `LayoutLabel.TITLE` | Document/section title |
| `LayoutLabel.TEXT` | Body text paragraph |
| `LayoutLabel.LIST` | Bulleted/numbered list |
| `LayoutLabel.FIGURE` | Image, diagram, plot |
| `LayoutLabel.TABLE` | Tabular data |
| `LayoutLabel.CAPTION` | Figure/table caption |
| `LayoutLabel.FORMULA` | Mathematical equation |
| `LayoutLabel.FOOTNOTE` | Footer note |
| `LayoutLabel.PAGE_HEADER` | Page header |
| `LayoutLabel.PAGE_FOOTER` | Page footer |

---

## Custom Label Detection

### Define Custom Labels

```python
from omnidocs.tasks.layout_extraction import CustomLabel

# Simple custom labels
code_block = CustomLabel(name="code_block")
sidebar = CustomLabel(name="sidebar")
annotation = CustomLabel(name="annotation")

# Labels with metadata
abstract = CustomLabel(
    name="abstract",
    description="Document abstract or summary",
    color="#E8F4F8",
    detection_prompt="Look for abstract sections, usually after title",
)

related_work = CustomLabel(
    name="related_work",
    description="Related work or background section",
    color="#FFF3CD",
)

# Create list of custom labels
custom_labels = [code_block, sidebar, abstract, related_work]
```

### Extract with Custom Labels

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
from PIL import Image

detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(device="cuda")
)

image = Image.open("document.png")

# Detect with custom labels
result = detector.extract(
    image,
    custom_labels=[code_block, sidebar, abstract],
)

# Access detections
for box in result.bboxes:
    print(f"Label: {box.label}")  # Custom label name
    print(f"Bbox: {box.bbox.to_list()}")
    print(f"Confidence: {box.confidence}")
    print()
```

### Mixed Standard and Custom Labels

```python
from omnidocs.tasks.layout_extraction import LayoutLabel, CustomLabel

# Combine standard and custom labels
standard_labels = [LayoutLabel.TITLE, LayoutLabel.TEXT]
custom_labels = [
    CustomLabel(name="code_example"),
    CustomLabel(name="warning_box"),
]

# Detect with both
result = detector.extract(
    image,
    custom_labels=custom_labels,  # Standard labels always included
)

# All labels present in result
for box in result.bboxes:
    print(f"Detected: {box.label}")
    # Could be: title, text, code_example, warning_box
```

---

## Usage Examples

### Example 1: Academic Paper Layout

```python
from omnidocs.tasks.layout_extraction import CustomLabel
from PIL import Image

# Define academic paper custom labels
custom_labels = [
    CustomLabel(
        name="abstract",
        description="Abstract section",
        detection_prompt="Find the abstract after the title",
    ),
    CustomLabel(
        name="methodology",
        description="Methods and experimental setup",
    ),
    CustomLabel(
        name="results_table",
        description="Results presented as table",
    ),
    CustomLabel(
        name="reference",
        description="Bibliography and references",
    ),
]

# Detect layout
result = detector.extract(
    image,
    custom_labels=custom_labels,
)

# Extract sections
for section in custom_labels:
    elements = [
        box for box in result.bboxes
        if box.label == section.name
    ]
    if elements:
        print(f"Found {section.name}: {len(elements)} element(s)")
        for elem in elements:
            print(f"  Position: {elem.bbox.to_list()}")
```

### Example 2: Website Layout Analysis

```python
# For web page screenshots
custom_labels = [
    CustomLabel(name="header", description="Top navigation bar"),
    CustomLabel(name="sidebar", description="Left/right sidebar"),
    CustomLabel(name="main_content", description="Primary content area"),
    CustomLabel(name="advertisement", description="Ad placement"),
    CustomLabel(name="footer", description="Footer section"),
]

result = detector.extract(image, custom_labels=custom_labels)

# Map to regions
regions = {label.name: [] for label in custom_labels}
for box in result.bboxes:
    if box.label in regions:
        regions[box.label].append(box)

for region_name, boxes in regions.items():
    print(f"{region_name}: {len(boxes)} element(s)")
```

### Example 3: Form Field Detection

```python
# For forms and structured documents
form_labels = [
    CustomLabel(name="text_field", description="Text input field"),
    CustomLabel(name="checkbox", description="Checkbox option"),
    CustomLabel(name="radio_button", description="Radio button"),
    CustomLabel(name="dropdown", description="Dropdown select"),
    CustomLabel(name="required_field", description="Field marked as required (*)"),
]

result = detector.extract(image, custom_labels=form_labels)

# Count field types
field_counts = {}
for box in result.bboxes:
    label = str(box.label)
    field_counts[label] = field_counts.get(label, 0) + 1

for field_type, count in field_counts.items():
    print(f"  {field_type}: {count}")
```

---

## Performance Characteristics

### Speed Comparison with DocLayout-YOLO

| Model | Speed | Trade-offs |
|-------|-------|-----------|
| **DocLayout-YOLO** | 0.1-0.2s/page | Fast but fixed labels |
| **Qwen Layout (PyTorch)** | 2-5s/page | Slower but flexible |
| **Qwen Layout (VLLM)** | 0.5-1.5s/page | Better speed with batching |

### Memory Requirements

| Backend | Min VRAM | Typical | Batch |
|---------|----------|---------|-------|
| PyTorch | 8 GB | 16 GB | 1 |
| VLLM | 12 GB | 24 GB | 2-4 |
| MLX | 8 GB | 16 GB | 1 |

---

## Troubleshooting

### Custom Labels Not Detected

**Symptom**: Custom labels return 0 detections

**Solutions**:

```python
# 1. Provide more detailed descriptions
custom_labels = [
    CustomLabel(
        name="code_block",
        description="Monospaced font code/programming examples in gray background",
        detection_prompt="Look for gray-boxed code sections with monospaced text",
    ),
]

# 2. Reduce temperature for more confident predictions
config = QwenLayoutPyTorchConfig(
    temperature=0.0,  # Most deterministic
)

# 3. Use larger model variant
config = QwenLayoutPyTorchConfig(
    model="Qwen/Qwen3-VL-32B-Instruct",
)

# 4. Check with standard labels first (confidence building)
result = detector.extract(image)  # No custom labels
print(f"Found {result.element_count} standard elements")
# Then try with custom
```

### Memory Issues

**Symptom**: CUDA out of memory

**Solutions**:

```python
# Use smaller model
config = QwenLayoutPyTorchConfig(
    model="Qwen/Qwen3-VL-4B-Instruct",
)

# Reduce max_new_tokens
config = QwenLayoutPyTorchConfig(
    max_new_tokens=2048,
)

# Enable quantization
config = QwenLayoutPyTorchConfig(
    load_in_4bit=True,
)

# Use CPU (slow but works)
config = QwenLayoutPyTorchConfig(
    device="cpu",
)
```

---

## Qwen Layout vs DocLayout-YOLO

| Aspect | Qwen Layout | DocLayout-YOLO |
|--------|------------|-----------------|
| **Custom Labels** | Yes (unlimited) | No (10 fixed) |
| **Speed** | Slower | Very fast |
| **Accuracy** | Higher | Good |
| **Memory** | 8-16 GB | 2-4 GB |
| **Backends** | 4 (PyTorch, VLLM, MLX, API) | 1 (PyTorch) |
| **Best For** | Flexibility, custom layouts | Speed, batch processing |

**Choose Qwen if**: You need custom layout categories or better accuracy
**Choose DocLayout-YOLO if**: You need speed and can use fixed categories

---

## API Reference

### QwenLayoutDetector.extract()

```python
def extract(
    image: Union[Image.Image, np.ndarray, str, Path],
    custom_labels: Optional[List[CustomLabel]] = None,
) -> LayoutOutput:
    """
    Extract layout from image with optional custom labels.

    Args:
        image: Input image
        custom_labels: List of CustomLabel objects for flexible detection

    Returns:
        LayoutOutput with detected layout boxes
    """
```

### LayoutOutput Properties (Standard Labels)

```python
result = detector.extract(image)

# Basic info
result.bboxes              # List[LayoutBox]
result.element_count       # Total detections
result.labels_found        # List of detected labels

# Filter by label
result.filter_by_label(LayoutLabel.TEXT)

# Convert coordinates
result.get_normalized_bboxes()  # 0-1024 scale
result.sort_by_position()       # Reading order

# Visualization
result.visualize(image, output_path="viz.png")

# Save/load
result.save_json("layout.json")
LayoutOutput.load_json("layout.json")
```

### CustomLabel Properties

```python
label = CustomLabel(
    name="code_block",
    description="Code examples",
    color="#E0E0E0",
    detection_prompt="Look for monospaced code",
)

print(label.name)               # "code_block"
print(label.description)        # Description text
print(label.color)              # "#E0E0E0"
print(label.detection_prompt)   # Custom hint
```

---

## Advanced Usage

### Hierarchical Layout Detection

```python
# Detect first pass: standard labels
result_std = detector.extract(image)

# Second pass: custom labels on specific regions
text_regions = result_std.filter_by_label(LayoutLabel.TEXT)

custom_labels = [
    CustomLabel(name="list_item"),
    CustomLabel(name="definition"),
    CustomLabel(name="example"),
]

# Could refine by cropping and re-detecting each region
for text_box in text_regions[:1]:  # First text block
    x1, y1, x2, y2 = text_box.bbox.to_xyxy()
    region = image.crop((x1, y1, x2, y2))

    result_detail = detector.extract(
        region,
        custom_labels=custom_labels,
    )
    print(f"Found {result_detail.element_count} fine-grained elements")
```

### Label Color Mapping for Visualization

```python
from PIL import Image, ImageDraw

# Colors for different label types
label_colors = {
    "title": "#E74C3C",        # Red
    "text": "#3498DB",         # Blue
    "abstract": "#2ECC71",     # Green
    "code_block": "#95A5A6",   # Gray
    "figure": "#9B59B6",       # Purple
    "table": "#F39C12",        # Orange
}

image = Image.open("document.png")
viz = image.copy()
draw = ImageDraw.Draw(viz)

# Draw with color mapping
for box in result.bboxes:
    color = label_colors.get(str(box.label), "#CCCCCC")
    coords = box.bbox.to_xyxy()
    draw.rectangle(coords, outline=color, width=2)

viz.save("layout_colored.png")
```

---

## See Also

- [DocLayout-YOLO](./doclayout-yolo.md) - Fixed label, fast detector
- [Qwen Text Extraction](../text-extraction/qwen.md) - Text extraction
- [Comparison Guide](./comparison.md) - Model selection matrix
