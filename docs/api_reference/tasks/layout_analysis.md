# 📐 Layout Analysis

This section documents the API for layout analysis tasks, including various extractors for detecting and analyzing document structure.

## Overview

Layout analysis in OmniDocs focuses on identifying and categorizing different regions within a document, such as text blocks, images, tables, and figures. This is crucial for understanding the document's overall structure and reading order.

## Available Extractors

### DocLayoutYOLOExtractor

A layout detection model based on YOLO-v10, designed for diverse document types.

::: omnidocs.tasks.layout_analysis.extractors.doc_layout_yolo.YOLOLayoutDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.layout_analysis.extractors.doc_layout_yolo import YOLOLayoutDetector

extractor = YOLOLayoutDetector()
result = extractor.extract("document.pdf")
print(f"Detected {len(result.layouts)} layout elements.")
```

### FlorenceLayoutExtractor

A fine-tuned model for document layout analysis, improving bounding box accuracy in document images.

::: omnidocs.tasks.layout_analysis.extractors.florence.FlorenceLayoutDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.layout_analysis.extractors.florence import FlorenceLayoutDetector

extractor = FlorenceLayoutDetector()
result = extractor.extract("image.png")
print(f"Detected {len(result.layouts)} layout elements.")
```

### PaddleLayoutExtractor

An OCR tool that supports multiple languages and provides layout detection capabilities.

::: omnidocs.tasks.layout_analysis.extractors.paddle.PaddleLayoutDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.layout_analysis.extractors.paddle import PaddleLayoutDetector

extractor = PaddleLayoutDetector()
result = extractor.extract("image.png")
print(f"Detected {len(result.layouts)} layout elements.")
```

### RTDETRLayoutExtractor

Implementation of RT-DETR, a real-time detection transformer focusing on object detection tasks.

::: omnidocs.tasks.layout_analysis.extractors.rtdetr.RTDETRLayoutDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.layout_analysis.extractors.rtdetr import RTDETRLayoutDetector

extractor = RTDETRLayoutDetector()
result = extractor.extract("image.png")
print(f"Detected {len(result.layouts)} layout elements.")
```

### SuryaLayoutExtractor

OCR and layout analysis tool supporting 90+ languages, including reading order and table recognition.

::: omnidocs.tasks.layout_analysis.extractors.surya.SuryaLayoutDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.layout_analysis.extractors.surya import SuryaLayoutDetector

extractor = SuryaLayoutDetector()
result = extractor.extract("document.pdf")
print(f"Detected {len(result.layouts)} layout elements.")
```

## LayoutOutput

The standardized output format for layout analysis results.

::: omnidocs.tasks.layout_analysis.base.LayoutOutput
    options:
      show_root_heading: true
      show_source: false

#### Key Properties

- `layouts` (List[LayoutElement]): List of detected layout elements.
- `source_file` (str): Path to the processed file.
- `source_img_size` (Tuple[int, int]): Dimensions of the source image.

#### Key Methods

- `save_json(output_path)`: Save results to a JSON file.
- `visualize(image_path, output_path)`: Visualize layout elements on the source image.

## LayoutElement

Represents a single detected layout element.

::: omnidocs.tasks.layout_analysis.base.BaseLayoutMapper
    options:
      show_root_heading: true
      show_source: false

#### Attributes

- `type` (str): Type of the element (e.g., 'text', 'title', 'table', 'figure').
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2].
- `text_content` (Optional[str]): Text content if applicable.
- `confidence` (Optional[float]): Confidence score of the detection.

## BaseLayoutExtractor

The abstract base class for all layout analysis extractors.

::: omnidocs.tasks.layout_analysis.base.BaseLayoutDetector
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract
        - preprocess_input
        - postprocess_output
        - visualize

## LayoutMapper

Handles mapping of layout labels and normalization of bounding boxes.

::: omnidocs.tasks.layout_analysis.base.BaseLayoutMapper
    options:
      show_root_heading: true
      show_source: false

## Related Resources

- [Layout Analysis Overview](../tasks/layout_analysis/overview.md)
- [Core Classes](../core.md)
- [Utilities](../utils.md)
