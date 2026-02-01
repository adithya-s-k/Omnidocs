# Layout Analysis Guide

Detect document structure and element boundaries using layout detection models. This guide explains how to extract layout information, work with labels, and build advanced document processing pipelines.

## Table of Contents

- [What is Layout Detection](#what-is-layout-detection)
- [Available Models](#available-models)
- [Fixed vs Custom Labels](#fixed-vs-custom-labels)
- [Basic Usage](#basic-usage)
- [Filtering and Analyzing Results](#filtering-and-analyzing-results)
- [Visualization](#visualization)
- [Advanced: Custom Labels](#advanced-custom-labels)
- [Troubleshooting](#troubleshooting)

## What is Layout Detection

Layout detection identifies document regions (elements) and classifies them by type. Unlike OCR (which extracts text) or text extraction (which formats content), layout detection provides structural information.

**Output:** List of bounding boxes with labels and confidence scores.

**Use Cases:**
- Document structure analysis
- Segmentation for downstream processing
- Building multi-stage pipelines (layout → text → output)
- Understanding document semantics
- Filtering unwanted elements (headers, footers, artifacts)

## Available Models

### 1. DocLayoutYOLO (Fast, fixed labels)

YOLO-based detector optimized for speed and accuracy on document layouts.

**Strengths:**
- Extremely fast (~0.3-0.5 sec per page)
- High accuracy on standard document elements
- Single-backend (PyTorch only)
- Low memory requirements

**Weaknesses:**
- Fixed labels only (no custom categories)
- Less accurate on irregular documents
- May struggle with non-English text

**Fixed Labels:**
- `title` - Document/section titles
- `text` - Body paragraphs
- `list` - Bullet or numbered lists
- `table` - Data tables
- `figure` - Images, diagrams, charts
- `caption` - Figure/table captions
- `formula` - Mathematical equations
- `footnote` - Footnotes
- `page_header` - Page headers
- `page_footer` - Page footers
- `unknown` - Unclassifiable elements

### 2. RT-DETR (Accuracy-focused, fixed labels)

High-accuracy detector with stronger backbone, but slower than YOLO.

**Strengths:**
- Higher accuracy than YOLO
- Good on challenging document types
- Handles small elements better

**Weaknesses:**
- Slower (~1-2 sec per page)
- Higher memory requirements
- Fixed labels only

**Same fixed labels as DocLayoutYOLO.**

### 3. QwenLayoutDetector (Flexible, custom labels)

Vision-language model supporting custom layout labels.

**Strengths:**
- Flexible custom labels (define your own)
- Strong on diverse document types
- Multi-backend support (PyTorch, VLLM, MLX, API)
- Better on irregular layouts

**Weaknesses:**
- Slower than YOLO (~2-3 sec per page)
- Higher VRAM requirements
- Requires more GPU memory

**Supports:**
- Standard LayoutLabel enum
- Custom labels (unlimited)
- Mixed standard + custom labels

## Fixed vs Custom Labels

### Fixed Labels (DocLayoutYOLO, RT-DETR)

Predefined categories that the model recognizes during training.

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

# Fixed labels are built-in and cannot be changed
config = DocLayoutYOLOConfig(device="cuda")
detector = DocLayoutYOLO(config=config)

result = detector.extract(image)

# Result contains elements with fixed labels
for element in result.elements:
    print(f"{element.label}: {element.bbox}")
    # Labels will always be from: title, text, list, table, figure, caption, formula, etc.
```

**Available Fixed Labels:**

| Label | Use Case | Typical Content |
|-------|----------|-----------------|
| `title` | Document/section heading | "Chapter 1", "Introduction" |
| `text` | Body paragraphs | Main content text |
| `list` | Bullet or numbered lists | "- Item 1", "1. Point A" |
| `table` | Data tables and grids | Tabular data |
| `figure` | Images, diagrams, charts | Photos, graphics, plots |
| `caption` | Figure/table descriptions | "Figure 1: Title" |
| `formula` | Mathematical equations | LaTeX, equations |
| `footnote` | Bottom-of-page notes | Footnotes, endnotes |
| `page_header` | Page header regions | Running headers |
| `page_footer` | Page footer regions | Running footers |
| `unknown` | Unclassifiable elements | Artifacts, noise |

### Custom Labels (QwenLayoutDetector only)

Define your own label categories for domain-specific documents.

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
from omnidocs.tasks.layout_extraction import CustomLabel

# Define custom labels
custom_labels = [
    CustomLabel(name="code_block", description="Code snippets"),
    CustomLabel(name="sidebar", description="Sidebar content"),
    CustomLabel(name="pull_quote", description="Quoted text"),
]

config = QwenLayoutPyTorchConfig(device="cuda")
detector = QwenLayoutDetector(backend=config)

result = detector.extract(image, custom_labels=custom_labels)

# Result contains elements with your custom labels
for element in result.elements:
    print(f"{element.label}: {element.bbox}")
    # Labels will be: code_block, sidebar, pull_quote, or standard labels
```

## Basic Usage

### Example 1: Fast Layout Detection (DocLayoutYOLO)

Detect layout using the fast YOLO-based model.

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs import Document
from PIL import Image

# Load image
image = Image.open("document_page.png")

# Initialize detector (fastest option)
config = DocLayoutYOLOConfig(
    device="cuda",  # or "cpu"
    img_size=1024,
    confidence=0.25,  # Detection confidence threshold
)
detector = DocLayoutYOLO(config=config)

# Extract layout
result = detector.extract(image)

print(f"Detected {len(result.elements)} layout elements")

# Display results
for element in result.elements:
    print(f"  {element.label:12} @ {element.bbox} (confidence: {element.confidence:.2f})")

# Count by label
from collections import Counter
label_counts = Counter(e.label for e in result.elements)
print(f"\nSummary: {dict(label_counts)}")
```

**Output Example:**
```
Detected 12 layout elements
  title         @ [50, 20, 500, 60] (confidence: 0.98)
  text          @ [50, 80, 950, 300] (confidence: 0.95)
  figure        @ [50, 320, 450, 650] (confidence: 0.92)
  caption       @ [50, 660, 450, 700] (confidence: 0.88)
  text          @ [480, 320, 950, 600] (confidence: 0.94)
  table         @ [50, 720, 950, 1000] (confidence: 0.91)

Summary: {'title': 1, 'text': 2, 'figure': 1, 'caption': 1, 'table': 1, ...}
```

### Example 2: Extract from PDF with Multiple Pages

Process all pages of a PDF document.

```python
from omnidocs import Document
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from pathlib import Path

# Load PDF
doc = Document.from_pdf("document.pdf")
print(f"Loaded {doc.page_count} pages")

# Initialize detector
config = DocLayoutYOLOConfig(device="cuda")
detector = DocLayoutYOLO(config=config)

# Process all pages
page_layouts = []
for page_idx in range(doc.page_count):
    page_image = doc.get_page(page_idx)
    result = detector.extract(page_image)
    page_layouts.append({
        "page": page_idx + 1,
        "num_elements": len(result.elements),
        "elements": result.elements,
    })
    print(f"Page {page_idx + 1}: {len(result.elements)} elements")

# Summary statistics
total_elements = sum(p["num_elements"] for p in page_layouts)
print(f"\nTotal: {total_elements} elements across {doc.page_count} pages")
```

### Example 3: High-Accuracy Detection (RT-DETR)

Use the more accurate RT-DETR model for challenging documents.

```python
from omnidocs.tasks.layout_extraction import RTDETRLayoutDetector, RTDETRLayoutConfig
from PIL import Image

image = Image.open("complex_document.png")

# Use RT-DETR for better accuracy (slower)
config = RTDETRLayoutConfig(
    device="cuda",
    confidence=0.3,  # Lower confidence threshold for more detections
)
detector = RTDETRLayoutDetector(config=config)

result = detector.extract(image)

# Filter by confidence
high_confidence = [e for e in result.elements if e.confidence >= 0.9]
print(f"High confidence detections: {len(high_confidence)}/{len(result.elements)}")

for element in high_confidence:
    print(f"  {element.label:12} {element.bbox} (conf: {element.confidence:.3f})")
```

### Example 4: Custom Labels with Qwen

Use Qwen to detect custom document elements.

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
from omnidocs.tasks.layout_extraction import CustomLabel
from PIL import Image

image = Image.open("technical_document.png")

# Define domain-specific labels
custom_labels = [
    CustomLabel(name="code_block", description="Syntax-highlighted code"),
    CustomLabel(name="api_doc", description="API documentation/reference"),
    CustomLabel(name="note_box", description="Important note or warning"),
    CustomLabel(name="example", description="Code example or usage"),
]

# Initialize with PyTorch backend
config = QwenLayoutPyTorchConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda",
)
detector = QwenLayoutDetector(backend=config)

# Extract with custom labels
result = detector.extract(image, custom_labels=custom_labels)

# Analyze custom labels
custom_elements = [e for e in result.elements if e.label in [l.name for l in custom_labels]]
print(f"Found {len(custom_elements)} domain-specific elements")

for element in custom_elements:
    print(f"  {element.label:12} {element.bbox}")
```

## Filtering and Analyzing Results

### Filter by Label

Extract only specific types of elements.

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

image = Image.open("document.png")
detector = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = detector.extract(image)

# Get only text elements
text_elements = [e for e in result.elements if e.label == "text"]
print(f"Text blocks: {len(text_elements)}")

# Get tables and figures
visual_elements = [e for e in result.elements if e.label in ["table", "figure"]]
print(f"Visual elements: {len(visual_elements)}")

# Exclude page artifacts (headers, footers)
content_elements = [
    e for e in result.elements
    if e.label not in ["page_header", "page_footer", "unknown"]
]
print(f"Main content elements: {len(content_elements)}")
```

### Filter by Confidence

Exclude low-confidence detections.

```python
# Keep only high-confidence detections
min_confidence = 0.8
filtered = [e for e in result.elements if e.confidence >= min_confidence]

print(f"Original: {len(result.elements)} elements")
print(f"Filtered (confidence >= {min_confidence}): {len(filtered)} elements")

# Analyze confidence distribution
confidences = [e.confidence for e in result.elements]
print(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")
print(f"Average: {sum(confidences)/len(confidences):.2f}")
```

### Filter by Bounding Box

Find elements in specific image regions.

```python
# Find elements in top half of page
image_height = image.height
top_half_elements = [
    e for e in result.elements
    if e.bbox[1] < image_height / 2  # y1 < height/2
]
print(f"Elements in top half: {len(top_half_elements)}")

# Find elements in specific region (e.g., sidebar)
def in_region(bbox, region):
    """Check if element overlaps with region."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = region
    # Check overlap
    return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

sidebar_region = (0, 0, 200, 1024)  # Left 200px
sidebar_elements = [e for e in result.elements if in_region(e.bbox, sidebar_region)]
print(f"Elements in sidebar region: {len(sidebar_elements)}")
```

### Analyze Element Sizes

Check element dimensions for quality control.

```python
# Calculate element sizes
element_sizes = []
for element in result.elements:
    bbox = element.bbox  # [x1, y1, x2, y2]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    element_sizes.append({
        "label": element.label,
        "width": width,
        "height": height,
        "area": area,
    })

# Find largest elements
largest = sorted(element_sizes, key=lambda e: e["area"], reverse=True)[:5]
print("Largest elements:")
for elem in largest:
    print(f"  {elem['label']}: {elem['width']:.0f}x{elem['height']:.0f} ({elem['area']:.0f} px²)")

# Find anomalies (very small or very large)
avg_area = sum(e["area"] for e in element_sizes) / len(element_sizes)
anomalies = [e for e in element_sizes if e["area"] < avg_area * 0.1 or e["area"] > avg_area * 10]
print(f"Anomalous elements: {len(anomalies)}")
```

## Visualization

### Visualize Detections

Draw bounding boxes on the image.

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image, ImageDraw
import random

image = Image.open("document.png")
detector = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = detector.extract(image)

# Create visualization
vis_image = image.copy()
draw = ImageDraw.Draw(vis_image)

# Color map for labels
colors = {
    "title": "#FF0000",
    "text": "#00FF00",
    "table": "#0000FF",
    "figure": "#FFFF00",
    "list": "#FF00FF",
    "caption": "#00FFFF",
}

# Draw bounding boxes
for element in result.elements:
    bbox = element.bbox
    label = element.label
    color = colors.get(label, "#FFFFFF")

    # Draw rectangle
    draw.rectangle(bbox, outline=color, width=2)

    # Draw label
    draw.text((bbox[0], bbox[1]-10), label, fill=color)

# Save visualization
vis_image.save("layout_visualization.png")
print("Saved visualization to layout_visualization.png")
```

### Use Built-in Visualization (if available)

Some models provide built-in visualization.

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

image = Image.open("document.png")
detector = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = detector.extract(image)

# If the model supports visualization
if hasattr(result, 'visualize'):
    vis_image = result.visualize(image)
    vis_image.save("layout_visualization.png")
else:
    print("Visualization not available for this model")
```

### Create Mask Images

Generate segmentation masks for each label.

```python
from PIL import Image, ImageDraw
import numpy as np

# Label to color mapping
label_colors = {
    "title": (255, 0, 0),
    "text": (0, 255, 0),
    "table": (0, 0, 255),
    "figure": (255, 255, 0),
    "list": (255, 0, 255),
    "caption": (0, 255, 255),
}

# Create mask image
mask = Image.new("RGB", image.size, color=(255, 255, 255))
draw = ImageDraw.Draw(mask)

# Draw filled rectangles
for element in result.elements:
    color = label_colors.get(element.label, (128, 128, 128))
    draw.rectangle(element.bbox, fill=color)

# Save and display
mask.save("layout_mask.png")

# Create overlay
overlay = Image.blend(image, mask, alpha=0.5)
overlay.save("layout_overlay.png")
print("Saved mask and overlay")
```

## Advanced: Custom Labels

### Multi-Stage Pipeline with Custom Labels

Detect custom elements, then extract text from specific types.

```python
from omnidocs.tasks.layout_extraction import QwenLayoutDetector
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig
from omnidocs.tasks.layout_extraction import CustomLabel
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig
from PIL import Image, ImageDraw
import numpy as np

image = Image.open("document.png")

# Stage 1: Layout detection with custom labels
custom_labels = [
    CustomLabel(name="code_snippet"),
    CustomLabel(name="data_table"),
    CustomLabel(name="api_reference"),
]

layout_config = QwenLayoutPyTorchConfig(device="cuda")
layout_detector = QwenLayoutDetector(backend=layout_config)
layout_result = layout_detector.extract(image, custom_labels=custom_labels)

# Group elements by type
code_snippets = [e for e in layout_result.elements if e.label == "code_snippet"]
data_tables = [e for e in layout_result.elements if e.label == "data_table"]
api_docs = [e for e in layout_result.elements if e.label == "api_reference"]

print(f"Found: {len(code_snippets)} code snippets, "
      f"{len(data_tables)} tables, "
      f"{len(api_docs)} API references")

# Stage 2: Extract text from specific regions
text_config = QwenTextPyTorchConfig(device="cuda")
text_extractor = QwenTextExtractor(backend=text_config)

# Process code snippets with special handling
for snippet in code_snippets:
    x1, y1, x2, y2 = snippet.bbox
    snippet_img = image.crop((x1, y1, x2, y2))

    # Use specialized prompt for code
    code_prompt = "Extract the code from this code snippet. Format as a code block with language identifier."
    result = text_extractor.extract(
        snippet_img,
        output_format="markdown",
        custom_prompt=code_prompt,
    )
    print(f"Code snippet:\n{result.content}\n")
```

### Custom Labels with Constraints

Add validation and constraints to custom labels.

```python
from omnidocs.tasks.layout_extraction import CustomLabel

# Create labels with metadata
labels = [
    CustomLabel(
        name="warning_box",
        description="Important warning or alert",
        color="#FF0000",  # Custom metadata
    ),
    CustomLabel(
        name="tip_box",
        description="Helpful tip or best practice",
        color="#00FF00",
    ),
    CustomLabel(
        name="example_code",
        description="Code example or snippet",
        color="#0000FF",
    ),
]

# Use in extraction
detector = QwenLayoutDetector(backend=config)
result = detector.extract(image, custom_labels=labels)

# Group by custom category
warnings = [e for e in result.elements if e.label == "warning_box"]
tips = [e for e in result.elements if e.label == "tip_box"]
examples = [e for e in result.elements if e.label == "example_code"]

print(f"Warnings: {len(warnings)}, Tips: {len(tips)}, Examples: {len(examples)}")
```

## Troubleshooting

### Missing Elements

**Problem:** Some elements not detected.

**Solutions:**
1. Lower `confidence` threshold
2. Try different model (RT-DETR instead of YOLO)
3. Resize image if very small
4. Check image quality

```python
# Solution 1: Lower confidence threshold
config = DocLayoutYOLOConfig(device="cuda", confidence=0.15)  # Default: 0.25

# Solution 2: Try RT-DETR
from omnidocs.tasks.layout_extraction import RTDETRLayoutDetector

detector = RTDETRLayoutDetector(config=RTDETRLayoutConfig(device="cuda"))
result = detector.extract(image)

# Solution 3: Resize image
if image.width < 512:
    image = image.resize((image.width * 2, image.height * 2))

# Solution 4: Check image quality
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")
if image.mode == "RGBA":
    # Convert RGBA to RGB
    image = image.convert("RGB")
```

### False Positives

**Problem:** Detecting too many incorrect elements.

**Solutions:**
1. Increase `confidence` threshold
2. Post-filter by size or region
3. Use different model

```python
# Solution 1: Increase confidence threshold
config = DocLayoutYOLOConfig(device="cuda", confidence=0.5)  # Higher threshold

result = detector.extract(image)

# Solution 2: Filter by size
filtered = [
    e for e in result.elements
    if (e.bbox[2] - e.bbox[0]) > 50  # Width > 50px
    and (e.bbox[3] - e.bbox[1]) > 20  # Height > 20px
]

# Solution 3: Try different model
# If YOLO has too many false positives, try RT-DETR or Qwen
```

### Overlapping Detections

**Problem:** Elements overlap incorrectly.

**Solutions:**
1. Post-process to remove overlaps
2. Use different confidence threshold
3. Try different model

```python
def remove_overlaps(elements, overlap_threshold=0.5):
    """Remove overlapping detections."""
    if not elements:
        return []

    # Sort by confidence (descending)
    sorted_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)

    # Keep non-overlapping elements
    kept = []
    for elem in sorted_elements:
        overlaps = False
        for kept_elem in kept:
            # Calculate intersection over union
            # ... (implementation details)
            pass
        if not overlaps:
            kept.append(elem)

    return kept

filtered = remove_overlaps(result.elements)
print(f"Removed {len(result.elements) - len(filtered)} overlapping detections")
```

### Slow Inference

**Problem:** Layout detection too slow.

**Solutions:**
1. Use faster model (YOLO instead of RT-DETR)
2. Reduce image resolution
3. Use VLLM for Qwen

```python
# Solution 1: Use YOLO (fastest)
from omnidocs.tasks.layout_extraction import DocLayoutYOLO
detector = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

# Solution 2: Resize image
original_size = image.size
image = image.resize((image.width // 2, image.height // 2))
result = detector.extract(image)
# Scale bboxes back
for elem in result.elements:
    elem.bbox = [x * 2 for x in elem.bbox]

# Solution 3: Use VLLM for Qwen
from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig
config = QwenLayoutVLLMConfig(tensor_parallel_size=1)
```

---

**Next Steps:**
- See [Text Extraction Guide](text-extraction.md) for extracting content from detected regions
- See [OCR Extraction Guide](ocr-extraction.md) for character-level extraction
- See [Batch Processing Guide](batch-processing.md) for processing many documents
