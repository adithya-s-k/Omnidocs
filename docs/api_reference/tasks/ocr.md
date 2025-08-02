# 🖹 OCR (Optical Character Recognition)

This section documents the API for OCR tasks, providing various extractors to recognize and extract text from images and scanned documents.

## Overview

OCR in OmniDocs enables the conversion of images (e.g., scanned documents, photos) into machine-readable text. It supports multiple engines, allowing you to choose the best balance of speed, accuracy, and language support for your needs.

## Available Extractors

### EasyOCRExtractor

A simple and easy-to-use OCR library that supports multiple languages and is built on PyTorch.

::: omnidocs.tasks.ocr_extraction.extractors.easy_ocr.EasyOCRExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

extractor = EasyOCRExtractor(languages=['en'])
result = extractor.extract("scanned_document.png")
print(f"Extracted text: {result.full_text[:200]}...")
```

### PaddleOCRExtractor

An OCR tool that supports multiple languages and provides layout detection capabilities.

::: omnidocs.tasks.ocr_extraction.extractors.paddle.PaddleOCRExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.ocr_extraction.extractors.paddle import PaddleOCRExtractor

extractor = PaddleOCRExtractor(languages=['en'])
result = extractor.extract("scanned_document.png")
print(f"Extracted text: {result.full_text[:200]}...")
```

### SuryaOCRExtractor

A modern, high-accuracy OCR engine, part of the Surya library, with strong support for Indian languages.

::: omnidocs.tasks.ocr_extraction.extractors.surya_ocr.SuryaOCRExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.ocr_extraction.extractors.surya_ocr import SuryaOCRExtractor

extractor = SuryaOCRExtractor(languages=['en'])
result = extractor.extract("scanned_document.png")
print(f"Extracted text: {result.full_text[:200]}...")
```

### TesseractOCRExtractor

An open-source OCR engine that supports multiple languages and is widely used for text extraction from images.

::: omnidocs.tasks.ocr_extraction.extractors.tesseract_ocr.TesseractOCRExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.ocr_extraction.extractors.tesseract_ocr import TesseractOCRExtractor

extractor = TesseractOCRExtractor(languages=['eng']) # Tesseract uses 'eng' for English
result = extractor.extract("scanned_document.png")
print(f"Extracted text: {result.full_text[:200]}...")
```

## OCROutput

The standardized output format for OCR results.

::: omnidocs.tasks.ocr_extraction.base.OCROutput
    options:
      show_root_heading: true
      show_source: false

#### Key Properties

- `texts` (List[OCRText]): List of individual text regions detected.
- `full_text` (str): The combined text from all detected regions.
- `source_img_size` (Tuple[int, int]): Dimensions of the source image.

#### Key Methods

- `save_json(output_path)`: Save results to a JSON file.
- `visualize(image_path, output_path)`: Visualize OCR results with bounding boxes on the source image.
- `get_text_by_confidence(min_confidence)`: Filter text regions by confidence score.
- `get_sorted_by_reading_order()`: Sort text regions by reading order.

## OCRText

Represents a single text region detected by OCR.

::: omnidocs.tasks.ocr_extraction.base.OCRText
    options:
      show_root_heading: true
      show_source: false

#### Attributes

- `text` (str): The recognized text content.
- `confidence` (float): Confidence score of the recognition (0.0-1.0).
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2].
- `polygon` (List[List[float]]): Precise polygon coordinates of the text region.
- `language` (Optional[str]): Detected language code.
- `reading_order` (int): Reading order index of the text region.

## BaseOCRExtractor

The abstract base class for all OCR extractors.

::: omnidocs.tasks.ocr_extraction.base.BaseOCRExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract
        - extract_all
        - extract_with_layout
        - preprocess_input
        - postprocess_output
        - visualize
        - get_supported_languages
        - set_languages

## BaseOCRMapper

Handles language code mapping and normalization for OCR engines.

::: omnidocs.tasks.ocr_extraction.base.BaseOCRMapper
    options:
      show_root_heading: true
      show_source: false

## Related Resources

- [OCR Overview](../tasks/ocr/overview.md)
- [EasyOCR Tutorial](../../tasks/ocr/tutorials/easyocr.ipynb)
- [PaddleOCR Tutorial](../../tasks/ocr/tutorials/paddle.ipynb)
- [Surya OCR Tutorial](../../tasks/ocr/tutorials/suryaocr.ipynb)
- [Tesseract Tutorial](../../tasks/ocr/tutorials/tesseract.ipynb)
- [Core Classes](../core.md)
- [Utilities](../utils.md)
