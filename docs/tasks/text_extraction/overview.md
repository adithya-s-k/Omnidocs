# Text Extraction

OmniDocs provides several extractors for extracting raw text from PDFs and images. Each extractor is implemented as a class with a consistent interface, making it easy to switch between methods for different document types.

## Available Text Extractors

- **PyMuPDFTextExtractor**: Fast, reliable text extraction from PDFs using PyMuPDF.
- **PdfplumberTextExtractor**: Lightweight PDF text extraction with layout awareness.
- **PyPDF2TextExtractor**: Simple PDF text extraction using PyPDF2.
- **PdftextTextExtractor**: General PDF text extraction backend.
- **SuryaTextExtractor**: Deep learning-based text extraction for scanned or complex documents.
- **DoclingTextExtractor**: Advanced parsing and text extraction for structured documents.

## Tutorials

- [Text Extractors](tutorials/text_extractors.ipynb)

## Basic Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.pymupdf import PyMuPDFExtractor

# Initialize extractor
extractor = PyMuPDFExtractor()

# Extract text from PDF
result = extractor.extract("path/to/file.pdf")

# Access text
print(result.text)
```

## Batch Processing Multiple PDFs

```python
import os
from omnidocs.tasks.text_extraction.extractors.pdfplumber import PDFPlumberExtractor

extractor = PDFPlumberExtractor()
pdf_folder = "path/to/pdf_folder/"

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        try:
            result = extractor.extract(pdf_path)
            print(f"{filename}: {len(result.text)} characters extracted")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
```

## Working with Different Extractors

```python
from omnidocs.tasks.text_extraction.extractors import (
    PyMuPDFTextExtractor, PdfplumberTextExtractor, PyPDF2TextExtractor,
    PdftextTextExtractor, SuryaTextExtractor, DoclingTextExtractor
)

extractors = [
    PyMuPDFTextExtractor(),
    PdfplumberTextExtractor(),
    PyPDF2TextExtractor(),
    PdftextTextExtractor(),
    SuryaTextExtractor(),
    DoclingTextExtractor()
]
pdf_path = "sample.pdf"

for extractor in extractors:
    try:
        result = extractor.extract(pdf_path)
        print(f"{extractor.__class__.__name__}: {len(result.text)} characters")
    except Exception as e:
        print(f"{extractor.__class__.__name__} error: {e}")
```

## Output Format

All extractors return a `TextOutput` object with:

```python
class TextOutput:
    text: str           # Extracted text content
    source_file: str    # Path to the processed file
```

## Installation Requirements

Each extractor may require specific dependencies:

### PyMuPDFTextExtractor
```bash
pip install pymupdf
```

### PdfplumberTextExtractor
```bash
pip install pdfplumber
```

### PyPDF2TextExtractor
```bash
pip install pypdf2
```

### PdftextTextExtractor
```bash
pip install pdfminer.six
```

### SuryaTextExtractor
```bash
pip install surya-ocr torch
```

### DoclingTextExtractor
```bash
pip install docling-tools
```

## Troubleshooting

**1. No Text Detected:**
  - Try a different extractor.
  - Check if the PDF is scanned (use OCR extractors if so).

**2. Import Errors:**
  - Install missing dependencies as shown above.

**3. Output Not as Expected:**
  - Inspect the extracted text and try another extractor for better results.

## Best Practices

1. **Choose the Right Extractor:**
   - Use PyMuPDF for fast, general-purpose extraction.
   - Use PDFPlumber for layout-aware extraction.
   - Use PyPDF2 for simple, lightweight extraction.

2. **Optimize Performance:**
   - Batch process files and initialize extractors once.

3. **Handle Errors Gracefully:**
   - Wrap extraction in try-except blocks.
   - Log or print errors for debugging.

4. **Validate Output:**
   - Always inspect the extracted text for completeness and accuracy.
