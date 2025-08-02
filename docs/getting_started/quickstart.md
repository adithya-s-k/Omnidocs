# Quick Start Guide

Get up and running with OmniDocs in minutes! This guide will walk you through installation and your first document processing tasks.

## 🚀 Installation

### Option 1: PyPI (Recommended)
```bash
pip install omnidocs
```

### Option 2: From Source
```bash
git clone https://github.com/adithya-s-k/OmniDocs.git
cd OmniDocs
pip install -e .
```

### Option 3: With uv (Fastest)
```bash
uv pip install omnidocs
```

## 🎯 Your First Document Processing

### 1. Extract Text with OCR

```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

# Initialize extractor
extractor = EasyOCRExtractor(languages=['en'])

# Extract text from image
result = extractor.extract("path/to/your/image.png")

# Print extracted text
print("Extracted Text:")
print(result.full_text)

# Access individual text regions
for text_region in result.texts:
    print(f"Text: {text_region.text}")
    print(f"Confidence: {text_region.confidence:.2f}")
    print(f"Bounding Box: {text_region.bbox}")
```

### 2. Extract Tables from PDF

```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

# Initialize table extractor
extractor = CamelotExtractor()

# Extract tables
result = extractor.extract("path/to/your/document.pdf")

# Print number of tables found
print(f"Found {len(result.tables)} tables")

# Access first table as DataFrame
if result.tables:
    first_table = result.tables[0]
    print("First table:")
    print(first_table.df.head())
    
    # Save as CSV
    first_table.df.to_csv("extracted_table.csv", index=False)
```

### 3. Extract Mathematical Expressions

```python
from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor

# Initialize math extractor
extractor = NougatExtractor()

# Extract math expressions
result = extractor.extract("path/to/academic/paper.pdf")

# Print extracted LaTeX
print("Extracted LaTeX:")
print(result.full_text)
```

### 4. Comprehensive Text Extraction

```python
from omnidocs.tasks.text_extraction.extractors.pymupdf import PyMuPDFExtractor

# Initialize text extractor
extractor = PyMuPDFExtractor()

# Extract text with layout information
result = extractor.extract("path/to/document.pdf")

# Print extracted text
print("Document Text:")
print(result.full_text)

# Access text blocks with positions
for text_block in result.text_blocks:
    print(f"Block: {text_block.text[:50]}...")
    print(f"Position: {text_block.bbox}")
```

## 🔄 Switching Between Extractors

One of OmniDocs' key features is the ability to easily switch between different extractors:

```python
# Try different OCR engines
from omnidocs.tasks.ocr_extraction.extractors import (
    EasyOCRExtractor, TesseractOCRExtractor, PaddleOCRExtractor
)

extractors = [
    EasyOCRExtractor(languages=['en']),
    TesseractOCRExtractor(languages=['eng']),
    PaddleOCRExtractor(languages=['en'])
]

image_path = "path/to/image.png"

for extractor in extractors:
    result = extractor.extract(image_path)
    print(f"{extractor.__class__.__name__}: {result.full_text[:100]}...")
```

## 📊 Batch Processing

Process multiple files efficiently:

```python
import os
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

extractor = CamelotExtractor()
pdf_folder = "path/to/pdf/folder"

results = {}
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith('.pdf'):
        file_path = os.path.join(pdf_folder, filename)
        try:
            result = extractor.extract(file_path)
            results[filename] = len(result.tables)
            print(f"✅ {filename}: {len(result.tables)} tables")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")

print(f"\nProcessed {len(results)} files successfully")
```

## 🎨 Visualization

Visualize extraction results:

```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

extractor = EasyOCRExtractor(languages=['en'])
result = extractor.extract("path/to/image.png")

# Visualize OCR results with bounding boxes
extractor.visualize(
    result=result,
    image_path="path/to/image.png",
    output_path="ocr_visualization.png",
    show_text=True,
    show_confidence=True
)
```

## 🔧 Configuration Examples

### Custom OCR Configuration
```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

extractor = EasyOCRExtractor(
    languages=['en', 'fr', 'de'],  # Multiple languages
    device='cuda',                 # Use GPU if available
    show_log=True                  # Enable logging
)
```

### Custom Table Extraction
```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

extractor = CamelotExtractor(
    flavor='lattice',              # Use lattice parsing
    pages='all',                   # Process all pages
    table_areas=None,              # Auto-detect table areas
    columns=None                   # Auto-detect columns
)
```

## 🚨 Common Issues & Solutions

### Issue: Import Errors
```bash
# Install missing dependencies
pip install torch torchvision  # For deep learning models
pip install camelot-py[cv]      # For Camelot
pip install easyocr            # For EasyOCR
```

### Issue: CUDA/GPU Problems
```python
# Force CPU usage if GPU issues
extractor = EasyOCRExtractor(device='cpu')
```

### Issue: Language Not Supported
```python
# Check supported languages
extractor = EasyOCRExtractor()
print(extractor.get_supported_languages())
```

## 📚 Next Steps

Now that you're up and running:

1. **Explore Tutorials**: Check out detailed [task-specific tutorials](../tasks/)
2. **Read API Reference**: Dive into the [complete API documentation](../api_reference/)
3. **Join Community**: Report issues or contribute on [GitHub](https://github.com/adithya-s-k/OmniDocs)

## 🎯 Real-World Examples

### Document Processing Pipeline
```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

def process_document(file_path):
    """Complete document processing pipeline"""
    results = {}
    
    # Extract text with OCR
    ocr_extractor = EasyOCRExtractor(languages=['en'])
    ocr_result = ocr_extractor.extract(file_path)
    results['text'] = ocr_result.full_text
    
    # Extract tables
    table_extractor = CamelotExtractor()
    table_result = table_extractor.extract(file_path)
    results['tables'] = [table.df for table in table_result.tables]
    
    return results

# Process a document
document_data = process_document("business_report.pdf")
print(f"Extracted {len(document_data['text'])} characters of text")
print(f"Found {len(document_data['tables'])} tables")
```

You’re now ready to build document-AI applications with OmniDocs.
