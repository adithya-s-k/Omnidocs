# Layout Extraction Tutorial

This tutorial will guide you through using OmniDocs' layout extraction module to detect and classify regions in documents and images.

## What is Layout Extraction?

Layout extraction (or layout analysis) is the process of detecting structural elements in a document image—such as text blocks, titles, tables, figures, and formulas—and identifying their locations with bounding boxes.

This is typically the first step in a document processing pipeline:

1. **Layout Extraction** - Find where elements are located
2. **OCR** - Extract text from text regions
3. **Table Extraction** - Parse table structures
4. **Further Processing** - Combine results for your use case

## Quick Start

Here's a minimal working example to get you started:

```python
from omnidocs.tasks.layout_analysis import YOLOLayoutDetector

# Initialize the detector
detector = YOLOLayoutDetector()

# Run detection on an image
annotated_image, layout_output = detector.detect("document.png")

# Print detected elements
for box in layout_output.bboxes:
    print(f"Found {box.label} at {box.bbox} (confidence: {box.confidence:.2f})")

# Save the annotated image
annotated_image.save("output.png")
```

## Available Detectors

OmniDocs provides two main layout detectors:

| Detector | Model | Best For |
|----------|-------|----------|
| `YOLOLayoutDetector` | DocLayout-YOLO | Fast inference, general documents |
| `RTDETRLayoutDetector` | RT-DETR (Transformer) | Higher accuracy, more label types |

Both detectors share the same API, so you can easily switch between them.

## Using DocLayoutYOLO

DocLayout-YOLO is a YOLO-based model optimized for document layout detection. It's fast and works well for most document types.

### Basic Usage

```python
from omnidocs.tasks.layout_analysis import YOLOLayoutDetector

# Initialize with default settings
detector = YOLOLayoutDetector()

# Or customize settings
detector = YOLOLayoutDetector(
    device="cuda",      # Use GPU (default: auto-detect)
    show_log=True       # Enable logging for debugging
)

# Detect layout
annotated_image, layout_output = detector.detect(
    "document.png",
    conf_threshold=0.3,  # Confidence threshold (default: 0.2)
    img_size=1024        # Input image size (default: 1024)
)
```

### Detected Labels

DocLayout-YOLO detects these element types:

- `text` - Regular text paragraphs
- `title` - Document titles and headings
- `image` - Figures and images
- `formula` - Mathematical formulas
- `caption` - Figure/table captions
- `table` - Tables

## Using RT-DETR

RT-DETR is a transformer-based real-time detection model. It typically provides higher accuracy and detects more element types.

### Basic Usage

```python
from omnidocs.tasks.layout_analysis import RTDETRLayoutDetector

# Initialize with default settings (CPU-only by default)
detector = RTDETRLayoutDetector()

# Or customize settings
detector = RTDETRLayoutDetector(
    device="cuda",       # Use GPU
    use_cpu_only=False,  # Allow GPU usage
    show_log=True,       # Enable logging
    num_threads=4        # CPU threads (when using CPU)
)

# Detect layout
annotated_image, layout_output = detector.detect(
    "document.png",
    confidence_threshold=0.5  # Default: 0.6
)
```

### Detected Labels

RT-DETR detects more element types, which are mapped to standard labels:

| Original Label | Standard Label |
|---------------|----------------|
| text | `text` |
| title | `title` |
| section-header | `title` |
| picture | `image` |
| table | `table` |
| formula | `formula` |
| caption | `caption` |
| list-item | `list` |
| footnote | `text` |
| page-header | `text` |
| page-footer | `text` |

## DocLayoutYOLO vs RT-DETR: When to Use Which?

| Criteria | DocLayoutYOLO | RT-DETR |
|----------|---------------|---------|
| **Speed** | Faster | Slower |
| **Accuracy** | Good | Better |
| **GPU Memory** | Lower | Higher |
| **Label Variety** | 6 types | 11 types |
| **Best For** | Quick processing, batch jobs | Detailed analysis, complex layouts |

**Choose DocLayoutYOLO when:**
- You need fast inference
- Processing large batches of documents
- Running on limited hardware

**Choose RT-DETR when:**
- Accuracy is more important than speed
- Documents have complex layouts
- You need to detect headers, footers, or list items

## Working with Results

### The LayoutOutput Object

Both detectors return a `LayoutOutput` object containing:

```python
annotated_image, layout_output = detector.detect("document.png")

# Access detected boxes
for box in layout_output.bboxes:
    print(f"Label: {box.label}")
    print(f"Bbox: {box.bbox}")  # [x1, y1, x2, y2]
    print(f"Confidence: {box.confidence}")

# Optional metadata
print(f"Page number: {layout_output.page_number}")
print(f"Image size: {layout_output.image_size}")
```

### Filtering Results

Filter detections by label or confidence:

```python
# Get only tables
tables = [box for box in layout_output.bboxes if box.label == "table"]

# Get high-confidence detections only
confident = [box for box in layout_output.bboxes if box.confidence > 0.8]

# Get text and titles
text_elements = [box for box in layout_output.bboxes if box.label in ("text", "title")]
```

### Sorting Results

Sort boxes by position (useful for reading order):

```python
# Sort top-to-bottom, then left-to-right
sorted_boxes = sorted(
    layout_output.bboxes,
    key=lambda b: (b.bbox[1], b.bbox[0])  # Sort by y1, then x1
)

# Sort by confidence (highest first)
by_confidence = sorted(
    layout_output.bboxes,
    key=lambda b: b.confidence,
    reverse=True
)

# Sort by area (largest first)
def box_area(box):
    x1, y1, x2, y2 = box.bbox
    return (x2 - x1) * (y2 - y1)

by_area = sorted(layout_output.bboxes, key=box_area, reverse=True)
```

### Saving Results to JSON

```python
# Save layout data to JSON
layout_output.save_json("layout_results.json")
```

## Visualization

### Built-in Visualization

Save annotated images with detected boxes:

```python
# Single image
detector.visualize(
    (annotated_image, layout_output),
    "output.png"
)

# This also saves a JSON file alongside: output.json
```

### Custom Visualization

Create your own visualizations using PIL:

```python
from PIL import Image, ImageDraw, ImageFont

# Load original image
image = Image.open("document.png")
draw = ImageDraw.Draw(image)

# Define colors for each label
colors = {
    "text": "blue",
    "title": "red",
    "table": "orange",
    "image": "purple",
    "formula": "yellow",
    "caption": "cyan",
    "list": "green"
}

# Draw boxes
for box in layout_output.bboxes:
    color = colors.get(box.label, "gray")
    x1, y1, x2, y2 = box.bbox

    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Draw label
    label_text = f"{box.label} ({box.confidence:.2f})"
    draw.text((x1, y1 - 15), label_text, fill=color)

image.save("custom_visualization.png")
```

### Processing Multiple Pages (PDFs)

```python
# Detect layout on all pages of a PDF
results = detector.detect_all("document.pdf")

for annotated_image, layout_output in results:
    print(f"Page {layout_output.page_number}: {len(layout_output.bboxes)} elements found")

# Save all pages
detector.visualize_all(results, output_dir="output/", prefix="page")
# Creates: output/page_1.png, output/page_1.json, output/page_2.png, ...
```

## Understanding Coordinates

### Absolute Coordinates

OmniDocs uses **absolute pixel coordinates** for bounding boxes:

```python
bbox = [x1, y1, x2, y2]
# x1, y1 = top-left corner
# x2, y2 = bottom-right corner
# All values are in pixels
```

Example:
```python
box.bbox = [100, 200, 500, 400]
# Top-left: (100, 200)
# Bottom-right: (500, 400)
# Width: 500 - 100 = 400 pixels
# Height: 400 - 200 = 200 pixels
```

### Converting to Normalized Coordinates

Some applications require normalized coordinates (0.0 to 1.0). Here's how to convert:

```python
def normalize_bbox(bbox, image_width, image_height):
    """Convert absolute coordinates to normalized (0-1) coordinates."""
    x1, y1, x2, y2 = bbox
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    ]

def denormalize_bbox(bbox, image_width, image_height):
    """Convert normalized coordinates back to absolute pixels."""
    x1, y1, x2, y2 = bbox
    return [
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height
    ]

# Example usage
image_width, image_height = layout_output.image_size

for box in layout_output.bboxes:
    normalized = normalize_bbox(box.bbox, image_width, image_height)
    print(f"Absolute: {box.bbox}")
    print(f"Normalized: {normalized}")
```

### Why Absolute vs Normalized?

| Coordinate Type | When to Use |
|-----------------|-------------|
| **Absolute** | Drawing on images, cropping regions, pixel-level operations |
| **Normalized** | Storing results for different resolutions, ML training data, cross-document comparison |

## Complete Example

Here's a complete workflow that ties everything together:

```python
from omnidocs.tasks.layout_analysis import YOLOLayoutDetector, RTDETRLayoutDetector
from PIL import Image

def analyze_document(image_path, use_rtdetr=False):
    """Analyze a document and extract structured layout information."""

    # Choose detector
    if use_rtdetr:
        detector = RTDETRLayoutDetector(show_log=True)
    else:
        detector = YOLOLayoutDetector(show_log=True)

    # Run detection
    annotated_image, layout_output = detector.detect(image_path)

    # Organize results by type
    results = {
        "text_blocks": [],
        "titles": [],
        "tables": [],
        "figures": [],
        "formulas": []
    }

    for box in layout_output.bboxes:
        entry = {
            "bbox": box.bbox,
            "confidence": box.confidence
        }

        if box.label == "text":
            results["text_blocks"].append(entry)
        elif box.label == "title":
            results["titles"].append(entry)
        elif box.label == "table":
            results["tables"].append(entry)
        elif box.label == "image":
            results["figures"].append(entry)
        elif box.label == "formula":
            results["formulas"].append(entry)

    # Print summary
    print(f"\nDocument Analysis Summary:")
    print(f"  Text blocks: {len(results['text_blocks'])}")
    print(f"  Titles: {len(results['titles'])}")
    print(f"  Tables: {len(results['tables'])}")
    print(f"  Figures: {len(results['figures'])}")
    print(f"  Formulas: {len(results['formulas'])}")

    # Save visualization
    detector.visualize((annotated_image, layout_output), "analysis_result.png")

    return results

# Run analysis
results = analyze_document("my_document.png", use_rtdetr=False)
```

## Troubleshooting

### Common Issues

**Model download fails:**
```python
# Models are downloaded from HuggingFace Hub on first use
# Ensure you have internet access and sufficient disk space
# Models are cached in ~/.cache/omnidocs/ or the configured model directory
```

**Out of memory (GPU):**
```python
# Use CPU instead
detector = YOLOLayoutDetector(device="cpu")

# Or for RT-DETR
detector = RTDETRLayoutDetector(use_cpu_only=True)
```

**Low detection quality:**
```python
# Try adjusting the confidence threshold
# Lower threshold = more detections (but more false positives)
annotated_image, layout_output = detector.detect(
    "document.png",
    conf_threshold=0.1  # YOLO
    # or confidence_threshold=0.3  # RT-DETR
)
```

**Slow inference:**
```python
# Use GPU if available
detector = YOLOLayoutDetector(device="cuda")

# Or reduce image size for YOLO
annotated_image, layout_output = detector.detect(
    "document.png",
    img_size=640  # Default is 1024
)
```

## Next Steps

- Explore the [API Reference](../api_reference/tasks/layout_analysis.md) for detailed documentation
- Try the [Jupyter notebooks](../tasks/layout_analysis/tutorials/) for interactive examples
- Combine layout extraction with [OCR](../tasks/ocr/overview.md) and [Table Extraction](../tasks/table_extraction/overview.md)
