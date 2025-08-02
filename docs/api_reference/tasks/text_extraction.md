# 📝 Text Extraction

This section documents the API for text extraction tasks, providing various extractors to retrieve textual content from documents.

## Overview

Text extraction in OmniDocs focuses on accurately pulling out text from different document formats (PDFs, images, etc.), often preserving layout and structural information. This is a fundamental step for many document understanding applications.

## Available Extractors

### DoclingParseExtractor

A unified parsing library for PDF, DOCX, PPTX, HTML, and MD, with OCR and structure capabilities.

::: omnidocs.tasks.text_extraction.extractors.docling_parse.DoclingTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.docling_parse import DoclingTextExtractor

extractor = DoclingTextExtractor()
result = extractor.extract("document.pdf")
print(f"Extracted text: {result.full_text[:200]}...")
```

### PDFPlumberTextExtractor

A library for extracting text and tables from PDFs with layout details.

::: omnidocs.tasks.text_extraction.extractors.pdfplumber.PdfplumberTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.pdfplumber import PdfplumberTextExtractor

extractor = PdfplumberTextExtractor()
result = extractor.extract("document.pdf")
print(f"Extracted text: {result.full_text[:200]}...")
```

### PDFTextExtractor

A simple, fast PDF text extraction with layout options.

::: omnidocs.tasks.text_extraction.extractors.pdftext.PdftextTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.pdftext import PdftextTextExtractor

extractor = PdftextTextExtractor()
result = extractor.extract("document.pdf")
print(f"Extracted text: {result.full_text[:200]}...")
```

### PyMuPDFTextExtractor

A fast, multi-format text extraction library with layout and font information.

::: omnidocs.tasks.text_extraction.extractors.pymupdf.PyMuPDFTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.pymupdf import PyMuPDFTextExtractor

extractor = PyMuPDFTextExtractor()
result = extractor.extract("document.pdf")
print(f"Extracted text: {result.full_text[:200]}...")
```

### PyPDF2TextExtractor

A pure Python library for extracting text from PDFs, supporting encrypted PDFs and form fields.

::: omnidocs.tasks.text_extraction.extractors.pypdf2.PyPDF2TextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.pypdf2 import PyPDF2TextExtractor

extractor = PyPDF2TextExtractor()
result = extractor.extract("document.pdf")
print(f"Extracted text: {result.full_text[:200]}...")
```

### SuryaTextExtractor

Surya-based text extraction for images and documents.

::: omnidocs.tasks.text_extraction.extractors.surya_text.SuryaTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.surya_text import SuryaTextExtractor

extractor = SuryaTextExtractor()
result = extractor.extract("image.png")
print(f"Extracted text: {result.full_text[:200]}...")
```

## TextOutput

The standardized output format for text extraction results.

::: omnidocs.tasks.text_extraction.base.TextOutput
    options:
      show_root_heading: true
      show_source: false

### Key Properties

- `text_blocks` (List[TextBlock]): List of extracted text blocks with positions.
- `full_text` (str): The complete extracted text content.
- `source_file` (str): Path to the processed file.

### Key Methods

- `save_json(output_path)`: Save results to a JSON file.

## TextBlock

Represents a single block of text with its bounding box.

::: omnidocs.tasks.text_extraction.base.TextBlock
    options:
      show_root_heading: true
      show_source: false

#### Attributes

- `text` (str): The text content of the block.
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2].
- `page_number` (int): The page number where the text block is found.

## BaseTextExtractor

The abstract base class for all text extraction extractors.

::: omnidocs.tasks.text_extraction.base.BaseTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract
        - preprocess_input
        - postprocess_output

## Related Resources

- [Text Extraction Overview](../tasks/text_extraction/overview.md)
- [Text Extractors Tutorial](../../tasks/text_extraction/tutorials/text_extractors.ipynb)
- [Core Classes](../core.md)
- [Utilities](../utils.md)
