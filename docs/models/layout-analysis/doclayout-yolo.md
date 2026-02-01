# DocLayout-YOLO Layout Detection

## Model Overview

DocLayout-YOLO is a YOLO-based (You Only Look Once) object detector specifically optimized for document layout analysis. It's the fastest layout detection model in OmniDocs, making it ideal for processing large document collections.

**Model ID**: juliozhao/DocLayout-YOLO-DocStructBench
**Architecture**: YOLOv10 (object detection)
**Training Focus**: Academic papers, technical documents, arXiv papers
**Framework**: PyTorch only (no other backends)

### Key Capabilities

- **Fast Inference**: 0.1-0.3s per page (fastest in OmniDocs)
- **10 Layout Categories**: Title, text, figures, tables, formulas, captions, etc.
- **Fixed Labels**: Standardized output across all documents
- **Document-Optimized**: Trained on 100K+ academic papers
- **Confidence Scores**: Per-detection confidence for filtering

### Limitations

- **PyTorch only**: No VLLM, MLX, or API backends
- **GPU required**: No CPU inference (YOLO needs GPU)
- **Fixed categories**: Cannot customize labels
- **English-focused**: Optimized for English documents
- **Specialized**: Best for academic/technical documents
- **Layout only**: Does not extract text content (use with OCR/VLM)

---

## Installation & Configuration

### Basic Installation

```bash
# Install with layout analysis support
pip install omnidocs[pytorch]

# Specifically install doclayout-yolo
pip install doclayout-yolo
```

### Configuration

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

config = DocLayoutYOLOConfig(
    device="cuda",           # GPU required
    model_path=None,         # Auto-download from HuggingFace
    img_size=1024,           # Input image size
    confidence=0.25,         # Detection confidence threshold
)

extractor = DocLayoutYOLO(config=config)
```

**Config Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | "cuda" | Device: "cuda", "mps", "cpu" (GPU required) |
| `model_path` | str | None | Path to model weights, or None for auto-download |
| `img_size` | int | 1024 | Input size for inference (320-1920) |
| `confidence` | float | 0.25 | Confidence threshold (0-1, higher = stricter) |

---

## Layout Categories (10 Fixed)

DocLayout-YOLO detects exactly 10 layout element types:

| Category | Description | Common Content |
|----------|-------------|-----------------|
| **Title** | Document/section title | "Introduction", "Methods" |
| **Plain text** | Body paragraph | Main content paragraphs |
| **Figure** | Image/diagram (content region) | Graphs, plots, photos |
| **Figure caption** | Caption for figures | "Fig. 1: System Overview" |
| **Table** | Tabular data (content region) | Data tables, matrices |
| **Table caption** | Caption for tables | "Table 2: Performance Results" |
| **Table footnote** | Notes under tables | Footnotes, explanations |
| **Formula** | Isolated equation | Display math: $E=mc^2$ |
| **Formula caption** | Caption for formulas | "Equation 3.1: Distance metric" |
| **Abandon** | Elements to ignore | Watermarks, page numbers, artifacts |

---

## Usage Examples

### Basic Layout Detection

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

# Initialize
config = DocLayoutYOLOConfig(device="cuda", confidence=0.3)
extractor = DocLayoutYOLO(config=config)

# Load image
image = Image.open("document.png")

# Extract layout
result = extractor.extract(image)

# Access results
print(f"Found {result.element_count} elements")
print(f"Labels: {result.labels_found}")

for box in result.bboxes:
    print(f"  {box.label.value}: confidence={box.confidence:.2f}")
    print(f"    bbox={box.bbox.to_list()}")
```

### Filter by Confidence

```python
# Keep only high-confidence detections
high_conf = result.filter_by_confidence(min_confidence=0.5)

for box in high_conf:
    print(f"{box.label.value} ({box.confidence:.2%})")
```

### Filter by Label

```python
from omnidocs.tasks.layout_extraction import LayoutLabel

# Extract only text regions
text_boxes = result.filter_by_label(LayoutLabel.TEXT)
print(f"Found {len(text_boxes)} text blocks")

# Extract only figures
figures = result.filter_by_label(LayoutLabel.FIGURE)
for fig in figures:
    x1, y1, x2, y2 = fig.bbox.to_xyxy()
    width = x2 - x1
    height = y2 - y1
    print(f"Figure: {width}x{height} at ({x1}, {y1})")
```

### Normalized Coordinates

```python
# Get bounding boxes normalized to 0-1024 scale
normalized = result.get_normalized_bboxes()

for box_dict in normalized:
    print(f"{box_dict['label']}:")
    print(f"  bbox (0-1024): {box_dict['bbox']}")
    print(f"  confidence: {box_dict['confidence']:.2f}")
```

### Visualization

```python
from PIL import Image

# Load original image
image = Image.open("document.png")

# Create visualization with bounding boxes
viz = result.visualize(
    image,
    output_path="layout_visualization.png",
    show_labels=True,
    show_confidence=True,
    line_width=2,
)

# Display
viz.show()
```

### Batch Processing

```python
from pathlib import Path
import json

# Process multiple documents
doc_dir = Path("documents/")
results = {}

for img_path in sorted(doc_dir.glob("*.png")):
    print(f"Processing {img_path.name}...")
    image = Image.open(img_path)
    layout = extractor.extract(image)

    results[img_path.name] = layout.to_dict()

# Save results
with open("layout_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary statistics
total_elements = sum(r["element_count"] for r in results.values())
avg_elements = total_elements / len(results)
print(f"Total elements: {total_elements}")
print(f"Average per document: {avg_elements:.1f}")
```

---

## Performance Characteristics

### Speed Comparison

| Device | Image Size | Time |
|--------|-----------|------|
| A10G GPU | 1024x1024 | 0.1-0.2s |
| A10G GPU | 2048x2048 | 0.3-0.5s |
| CPU | 1024x1024 | 5-10s |
| CPU | 2048x2048 | 15-30s |

### Memory Requirements

| Batch Size | VRAM | Device |
|-----------|------|--------|
| 1 (single) | 2-4 GB | A10G |
| 2-4 | 4-8 GB | A10G |
| 1 | 1-2 GB | A100 |

### Typical Detection Counts

| Document Type | Elements | Speed |
|---------------|----------|-------|
| Single page paper | 10-30 | 0.1s |
| Research paper (5pp) | 50-150 | 0.5s |
| Scanned book page | 20-40 | 0.15s |

---

## Troubleshooting

### Model Download Issues

**Symptom**: Model fails to download on first run

**Solution**:

```python
# Set cache directory
import os
os.environ["HF_HOME"] = "/path/to/cache"

# Pre-download the model
from huggingface_hub import snapshot_download
snapshot_download("juliozhao/DocLayout-YOLO-DocStructBench")

# Now use extractor (will use cached model)
extractor = DocLayoutYOLO(config=config)
```

### Confidence Threshold Tuning

**Symptom**: Too many false positives OR missing real elements

**Solutions**:

```python
# Too many false positives → increase confidence
config = DocLayoutYOLOConfig(confidence=0.5)  # Stricter

# Missing elements → decrease confidence
config = DocLayoutYOLOConfig(confidence=0.1)  # More lenient

# Find optimal threshold
from PIL import Image
image = Image.open("test.png")

for conf in [0.1, 0.25, 0.5, 0.75]:
    config = DocLayoutYOLOConfig(confidence=conf)
    extractor = DocLayoutYOLO(config=config)
    result = extractor.extract(image)
    print(f"Confidence {conf}: {result.element_count} elements")
```

### Image Size Issues

**Symptom**: Poor detection on very large or small images

**Solutions**:

```python
from PIL import Image

image = Image.open("document.png")
print(f"Original size: {image.size}")

# Resize to standard size for better detection
target_size = 1024
image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

result = extractor.extract(image)
print(f"Found {result.element_count} elements")
```

---

## Model Selection Guide

### When to Use DocLayout-YOLO

**Best for**:
- Fast batch processing of large document collections
- Academic papers and technical documents
- When speed is critical (real-time requirements)
- You need layout detection only (will add OCR separately)

**Not ideal for**:
- Extracting actual text (use Qwen or DotsOCR)
- Complex/unusual layouts (use Qwen layout detector)
- Handwritten documents (use Surya)
- When you need custom layout categories (use Qwen)

### DocLayout-YOLO vs Other Layout Models

| Feature | DocLayout-YOLO | RT-DETR | Qwen Layout |
|---------|----------------|---------|-------------|
| **Speed** | Very Fast | Fast | Medium |
| **Categories** | 10 (fixed) | 12+ (fixed) | Unlimited (custom) |
| **Backend** | PyTorch only | PyTorch | Multi-backend |
| **Memory** | 2-4 GB | 4-8 GB | 8-16 GB |
| **Quality** | Good | Excellent | Excellent |
| **Use Case** | Fast detection | Precision | Flexibility |

**Choose DocLayout-YOLO if**: You need fast detection for batch processing
**Choose Qwen Layout if**: You need flexible categories or better quality

---

## API Reference

### DocLayoutYOLO.extract()

```python
def extract(image: Union[Image.Image, np.ndarray, str, Path]) -> LayoutOutput:
    """
    Run layout extraction on an image.

    Args:
        image: Input image (PIL Image, numpy array, or path)

    Returns:
        LayoutOutput with detected layout boxes
    """
```

### LayoutOutput Properties

```python
result = extractor.extract(image)

# Basic properties
result.bboxes              # List[LayoutBox] - all detections
result.element_count       # Number of elements
result.labels_found        # Set of unique labels
result.image_width         # Source image width
result.image_height        # Source image height
result.model_name          # "DocLayout-YOLO"

# Filter methods
result.filter_by_label(label)        # Filter by LayoutLabel
result.filter_by_confidence(min_conf) # Filter by confidence

# Coordinate conversion
result.get_normalized_bboxes()  # Convert to 0-1024 scale
result.sort_by_position()       # Sort by reading order

# Export
result.to_dict()           # Convert to dictionary
result.visualize(image)    # Create visualization
result.save_json(path)     # Save to JSON file
result.load_json(path)     # Load from JSON file
```

### LayoutBox Properties

```python
for box in result.bboxes:
    box.label             # LayoutLabel enum
    box.bbox              # BoundingBox object
    box.confidence        # float (0-1)
    box.class_id          # int - YOLO class ID
    box.original_label    # str - original YOLO label
```

### BoundingBox Methods

```python
bbox = box.bbox

# Access coordinates
bbox.x1, bbox.y1          # Top-left corner
bbox.x2, bbox.y2          # Bottom-right corner
bbox.width                # Width in pixels
bbox.height               # Height in pixels
bbox.area                 # Area in pixels²
bbox.center               # (center_x, center_y) tuple

# Convert formats
bbox.to_list()            # [x1, y1, x2, y2]
bbox.to_xyxy()            # (x1, y1, x2, y2)
bbox.to_xywh()            # (x, y, width, height)

# Normalize to 0-1024 range
normalized = bbox.to_normalized(image_width, image_height)

# Convert back to absolute
absolute = normalized.to_absolute(image_width, image_height)
```

---

## Advanced Usage

### Reading Order Detection

```python
# DocLayout-YOLO automatically sorts by position (top to bottom, left to right)
sorted_result = result.sort_by_position(top_to_bottom=True)

for i, box in enumerate(sorted_result.bboxes):
    print(f"{i+1}. {box.label.value} at ({box.bbox.y1:.0f}, {box.bbox.x1:.0f})")
```

### Region-Based Processing

```python
# Get all elements in upper half of page
upper_half = [
    box for box in result.bboxes
    if box.bbox.y1 < result.image_height // 2
]

# Get all large elements (> 1/4 page width)
page_width = result.image_width
large_elements = [
    box for box in result.bboxes
    if box.bbox.width > page_width // 4
]

print(f"Upper half: {len(upper_half)} elements")
print(f"Large: {len(large_elements)} elements")
```

### Export to Different Formats

```python
# Save as JSON for downstream processing
result.save_json("layout.json")

# Convert to dict for custom serialization
layout_dict = result.to_dict()

# Export to COCO format (for computer vision tools)
coco_format = {
    "images": [{
        "id": 0,
        "width": result.image_width,
        "height": result.image_height,
        "file_name": "document.png"
    }],
    "annotations": [
        {
            "id": i,
            "image_id": 0,
            "category_id": box.class_id,
            "bbox": list(box.bbox.to_xywh()),  # COCO format: [x, y, w, h]
            "area": box.bbox.area,
            "iscrowd": 0,
        }
        for i, box in enumerate(result.bboxes)
    ],
}
```

---

## Integration with Text Extraction

### Pipeline: Layout + OCR

```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig
from PIL import Image

# Step 1: Detect layout
layout_result = extractor.extract(image)

# Step 2: Extract text from regions
ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))

for box in layout_result.bboxes:
    if box.label.value in ["text", "title"]:
        # Crop region
        x1, y1, x2, y2 = box.bbox.to_xyxy()
        region = image.crop((x1, y1, x2, y2))

        # OCR the region
        ocr_result = ocr.extract(region)

        print(f"{box.label.value}: {ocr_result.full_text}")
```

### Pipeline: Layout + VLM

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Step 1: Detect layout
layout_result = extractor.extract(image)

# Step 2: Extract text per element
extractor_qwen = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

for i, box in enumerate(layout_result.bboxes):
    # Crop region
    x1, y1, x2, y2 = box.bbox.to_xyxy()
    region = image.crop((x1, y1, x2, y2))

    # Extract with Qwen
    result = extractor_qwen.extract(region)

    print(f"Element {i} ({box.label.value}):")
    print(result.content)
    print()
```

---

## See Also

- [RT-DETR Layout Detection](./rtdetr.md) - Alternative DETR-based model
- [Qwen Layout Detection](./qwen-layout.md) - For custom categories
- [Comparison Guide](./comparison.md) - Model selection matrix
- [YOLOv10 Paper](https://arxiv.org/abs/2405.14458)
