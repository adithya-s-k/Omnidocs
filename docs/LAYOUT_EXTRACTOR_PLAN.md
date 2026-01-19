# Layout Extractor Implementation Plan

> **Status**: Complete
> **Created**: January 20, 2026
> **Issue**: #TBD - Add Layout Extraction Models (DocLayout-YOLO & RT-DETR)

---

## Overview

This document outlines the implementation plan for integrating DocLayout-YOLO and RT-DETR layout detection models into OmniDocs v2.0.

### Goals

1. Implement two fixed-label layout detectors following the new architecture
2. Use Pydantic configs with environment variable support for model paths
3. Unified `.extract()` API returning `LayoutOutput`
4. Coordinate normalization (0-1024 range) for consistent results
5. Built-in visualization support
6. Comprehensive test coverage

---

## Architecture Reference

Based on `omnidocs_ideation/FINAL_DEVEX.md` and `omnidocs_ideation/BACKEND_ARCHITECTURE.md`:

### Single-Backend Pattern

Both DocLayout-YOLO and RT-DETR are **single-backend models** (PyTorch only):

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)
```

### Directory Structure

```
omnidocs/tasks/layout_extraction/
├── __init__.py           # Export models + configs
├── base.py               # BaseLayoutExtractor abstract class
├── models.py             # LayoutLabel, LayoutBox, LayoutOutput, BoundingBox (Pydantic)
├── doc_layout_yolo.py    # DocLayoutYOLO + DocLayoutYOLOConfig
└── rtdetr.py             # RTDETRLayoutExtractor + RTDETRConfig
```

---

## Coordinate Systems

### Absolute Coordinates (Default)

By default, all bounding boxes are returned in **absolute pixel coordinates** relative to the original image size.

```python
result = layout.extract(image)
for box in result.bboxes:
    print(f"Pixel coords: {box.bbox.to_list()}")
    # Output: [150.5, 200.0, 450.5, 300.0]
```

### Normalized Coordinates (0-1024)

For consistent coordinates across different image sizes, use normalized coordinates. All coordinates are scaled to a **virtual 1024×1024 canvas**.

```python
# Get normalized bboxes
normalized = result.get_normalized_bboxes()
for box in normalized:
    print(f"Normalized coords: {box['bbox']}")
    # Output: [102.4, 256.0, 460.8, 384.0]

# Convert individual bbox
norm_bbox = box.bbox.to_normalized(image_width, image_height)

# Convert back to absolute
abs_bbox = norm_bbox.to_absolute(new_width, new_height)
```

**Why 0-1024?** This range provides:
- Integer precision for most use cases
- Easy integration with models expecting normalized inputs
- Consistent scale regardless of source image dimensions

---

## Visualization

The `LayoutOutput` class includes a built-in `visualize()` method for debugging and inspection:

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

# Load image and extract layout
image = Image.open("document.png")
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)

# Visualize with default settings
viz = result.visualize(image)
viz.show()

# Save to file
result.visualize(image, output_path="layout_visualization.png")

# Customize visualization
viz = result.visualize(
    image,
    show_labels=True,       # Show label text
    show_confidence=True,   # Show confidence scores
    line_width=3,           # Bounding box line width
    font_size=12,           # Label font size
)
```

### Label Colors

Each layout category has a distinct color for easy identification:

| Label | Color |
|-------|-------|
| TITLE | Red (#E74C3C) |
| TEXT | Blue (#3498DB) |
| LIST | Green (#2ECC71) |
| FIGURE | Purple (#9B59B6) |
| TABLE | Orange (#F39C12) |
| CAPTION | Teal (#1ABC9C) |
| FORMULA | Pink (#E91E63) |
| FOOTNOTE | Gray (#607D8B) |
| PAGE_HEADER/FOOTER | Brown (#795548) |
| ABANDON | Light Gray (#BDC3C7) |

---

## Implementation Details

### 1. Standardized Layout Labels (`LayoutLabel` Enum)

```python
from enum import Enum

class LayoutLabel(str, Enum):
    """Standardized layout labels used across all extractors."""
    TITLE = "title"
    TEXT = "text"
    LIST = "list"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    FORMULA = "formula"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    ABANDON = "abandon"
    UNKNOWN = "unknown"
```

### 2. Label Mapping

Each model maps its native labels to standardized `LayoutLabel` values:

```python
# DocLayout-YOLO mapping
DOCLAYOUT_YOLO_MAPPING = LabelMapping({
    "title": LayoutLabel.TITLE,
    "plain_text": LayoutLabel.TEXT,
    "figure": LayoutLabel.FIGURE,
    "figure_caption": LayoutLabel.CAPTION,
    "table": LayoutLabel.TABLE,
    "table_caption": LayoutLabel.CAPTION,
    "isolate_formula": LayoutLabel.FORMULA,
    # ...
})

# RT-DETR mapping
RTDETR_MAPPING = LabelMapping({
    "title": LayoutLabel.TITLE,
    "section-header": LayoutLabel.TITLE,
    "text": LayoutLabel.TEXT,
    "picture": LayoutLabel.FIGURE,
    "table": LayoutLabel.TABLE,
    "list-item": LayoutLabel.LIST,
    # ...
})
```

### 3. Pydantic Output Models

```python
class BoundingBox(BaseModel):
    """Bounding box with normalization support."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_normalized(self, image_width: int, image_height: int) -> "BoundingBox":
        """Convert to 0-1024 normalized coordinates."""
        ...

    def to_absolute(self, image_width: int, image_height: int) -> "BoundingBox":
        """Convert from normalized to absolute pixel coordinates."""
        ...


class LayoutBox(BaseModel):
    """Single detected layout element."""
    label: LayoutLabel
    bbox: BoundingBox
    confidence: float
    class_id: Optional[int] = None
    original_label: Optional[str] = None


class LayoutOutput(BaseModel):
    """Layout detection results."""
    bboxes: List[LayoutBox]
    image_width: int
    image_height: int
    model_name: Optional[str] = None

    def get_normalized_bboxes(self) -> List[Dict]:
        """Get all bboxes in 0-1024 normalized coordinates."""
        ...

    def visualize(self, image, output_path=None, ...) -> Image:
        """Visualize layout detection results."""
        ...
```

---

## Config Classes

### DocLayoutYOLOConfig

```python
class DocLayoutYOLOConfig(BaseModel):
    device: str = "cuda"
    model_path: Optional[str] = None
    img_size: int = 1024          # Range: 320-1920
    confidence: float = 0.25      # Range: 0.0-1.0
```

### RTDETRConfig

```python
class RTDETRConfig(BaseModel):
    device: str = "cuda"
    model_path: Optional[str] = None
    model_name: str = "HuggingPanda/docling-layout"
    image_size: int = 640         # Range: 320-1280
    confidence: float = 0.4       # Range: 0.0-1.0
```

---

## Model Path Configuration

### Environment Variable Priority

1. **Config `model_path`** - If explicitly provided in config, use it
2. **`OMNIDOCS_MODELS_DIR`** - Environment variable for models directory
3. **Default** - `~/.omnidocs/models/`

### Download Locations

- DocLayout-YOLO: `~/.omnidocs/models/doclayout_yolo/`
- RT-DETR: `~/.omnidocs/models/rtdetr_layout/`

---

## Test Coverage

### Test Files

```
tests/tasks/layout_extraction/
├── conftest.py              # Shared fixtures
├── test_models.py           # 41 tests for Pydantic models
├── test_doc_layout_yolo.py  # Config & extractor tests
└── test_rtdetr.py           # Config & extractor tests
```

### Total: 71 tests

- Pydantic model validation
- Config validation (bounds, extra fields forbidden)
- Model path resolution
- Normalization roundtrip
- Visualization output
- Integration tests (marked `@pytest.mark.slow`)

---

## Usage Examples

### Basic Usage

```python
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

# Load image
image = Image.open("document.png")

# Initialize layout detector
layout = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(device="cuda", confidence=0.3)
)

# Extract layout
result = layout.extract(image)

# Access results
for box in result.bboxes:
    print(f"{box.label.value}: {box.bbox.to_list()} (conf: {box.confidence:.2f})")
```

### With Normalized Coordinates

```python
# Get normalized bounding boxes (0-1024 range)
normalized = result.get_normalized_bboxes()

for box in normalized:
    print(f"{box['label']}: {box['bbox']} (normalized)")
```

### Visualization

```python
# Quick visualization
viz = result.visualize(image)
viz.show()

# Save with custom settings
result.visualize(
    image,
    output_path="layout_debug.png",
    show_labels=True,
    show_confidence=True,
    line_width=2,
)
```

### With RT-DETR

```python
from omnidocs.tasks.layout_extraction import RTDETRLayoutExtractor, RTDETRConfig

layout = RTDETRLayoutExtractor(
    config=RTDETRConfig(device="cuda", confidence=0.4)
)

result = layout.extract(image)

# Filter by label
tables = result.filter_by_label(LayoutLabel.TABLE)
figures = result.filter_by_label(LayoutLabel.FIGURE)

# Filter by confidence
high_conf = result.filter_by_confidence(0.8)

# Sort by reading order
sorted_result = result.sort_by_position()
```

---

## Implementation Checklist

- [x] Create `omnidocs/tasks/layout_extraction/` directory structure
- [x] Implement `models.py` (Pydantic models with normalization)
- [x] Implement `base.py` (abstract base)
- [x] Implement `doc_layout_yolo.py`
- [x] Implement `rtdetr.py`
- [x] Add standardized `LayoutLabel` enum with mappings
- [x] Add coordinate normalization (0-1024)
- [x] Add `visualize()` method with label colors
- [x] Update `__init__.py` with exports
- [x] Add dependencies to pyproject.toml
- [x] Create test fixtures
- [x] Write unit tests (71 tests, all passing)
- [ ] Create GitHub issue
- [ ] Create pull request

---

## References

- Working implementation: `document_parsing_workflow/02_layout_doclayout_yolo.py`
- Working implementation: `document_parsing_workflow/03_layout_rtdetr.py`
- Old implementation: `omnidocs_old/omnidocs/tasks/layout_analysis/`
- Design docs: `omnidocs_ideation/FINAL_DEVEX.md`
- Architecture: `omnidocs_ideation/BACKEND_ARCHITECTURE.md`

---

**Last Updated**: January 20, 2026
