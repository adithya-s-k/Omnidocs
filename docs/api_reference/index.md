# 📚 API Reference

Welcome to the comprehensive OmniDocs API Reference! This documentation provides detailed information about all classes, functions, and modules in the OmniDocs ecosystem.

## 🚀 Quick Navigation

### 🧩 Core Components
- **[Core Classes](core.md)** - Base classes and fundamental components
- **[Utils](utils.md)** - Utility functions and helpers

### 📋 Tasks & Extractors
- **[Layout Analysis](tasks/layout_analysis.md)** - Document structure detection
- **[Text Extraction](tasks/text_extraction.md)** - Text parsing from documents
- **[Table Extraction](tasks/table_extraction.md)** - Tabular data extraction
- **[OCR](tasks/ocr.md)** - Optical Character Recognition
- **[Math Expression](tasks/math_expression.md)** - Mathematical formula extraction

## 🎯 Getting Started with the API

### Basic Usage Pattern

All OmniDocs extractors follow a consistent interface:

```python
# 1. Import the extractor
from omnidocs.tasks.{task}.extractors.{extractor} import {ExtractorClass}

# 2. Initialize with configuration
extractor = ExtractorClass(
    # Common parameters
    device='cpu',           # or 'cuda' for GPU
    show_log=True,         # Enable logging
    languages=['en'],      # Supported languages
    # Extractor-specific parameters...
)

# 3. Extract from document
result = extractor.extract("path/to/document.pdf")

# 4. Access results
print(result.full_text)    # For text-based results
print(result.tables)       # For table results
print(result.texts)        # For OCR results
```

### Common Parameters

Most extractors support these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `'cpu'` | Device to run on ('cpu' or 'cuda') |
| `show_log` | `bool` | `False` | Enable detailed logging |
| `languages` | `List[str]` | `['en']` | Languages to support |

### Result Objects

All extractors return structured result objects:

#### OCROutput
```python
class OCROutput:
    texts: List[OCRText]           # Individual text regions
    full_text: str                 # Combined text
    source_img_size: Tuple[int, int]  # Original image dimensions
    processing_time: Optional[float]   # Extraction time
    metadata: Dict[str, Any]       # Additional information
```

#### TableOutput
```python
class TableOutput:
    tables: List[Table]            # Extracted tables
    source_file: str              # Source document path
    processing_time: Optional[float]  # Extraction time
    metadata: Dict[str, Any]      # Additional information
```

#### TextOutput
```python
class TextOutput:
    text_blocks: List[TextBlock]   # Text blocks with positions
    full_text: str                # Combined text
    source_file: str              # Source document path
    processing_time: Optional[float]  # Extraction time
    metadata: Dict[str, Any]      # Additional information
```

## 🔧 Advanced Usage

### Batch Processing

Process multiple documents efficiently:

```python
from pathlib import Path
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

extractor = EasyOCRExtractor()
documents = Path("documents/").glob("*.pdf")

results = []
for doc in documents:
    try:
        result = extractor.extract(str(doc))
        results.append({
            'file': doc.name,
            'text': result.full_text,
            'confidence': sum(t.confidence for t in result.texts) / len(result.texts)
        })
    except Exception as e:
        print(f"Error processing {doc}: {e}")
```

### Custom Configuration

Configure extractors for specific use cases:

```python
# High-accuracy OCR setup
ocr_extractor = EasyOCRExtractor(
    languages=['en', 'fr', 'de'],
    device='cuda',
    show_log=True
)

# Fast table extraction setup
table_extractor = CamelotExtractor(
    flavor='stream',        # Faster than 'lattice'
    edge_tol=500,          # Edge tolerance
    row_tol=2              # Row tolerance
)
```

### Error Handling

Robust error handling patterns:

```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

extractor = CamelotExtractor()

try:
    result = extractor.extract("document.pdf")
    if result.tables:
        print(f"Successfully extracted {len(result.tables)} tables")
    else:
        print("No tables found in document")
except FileNotFoundError:
    print("Document file not found")
except Exception as e:
    print(f"Extraction failed: {e}")
```

## 🎨 Visualization

Most extractors support result visualization:

```python
# Visualize OCR results
ocr_result = ocr_extractor.extract("image.png")
ocr_extractor.visualize(
    result=ocr_result,
    image_path="image.png",
    output_path="ocr_visualization.png",
    show_text=True,
    show_confidence=True
)

# Visualize table extraction
table_result = table_extractor.extract("document.pdf")
table_extractor.visualize(
    result=table_result,
    image_path="document.pdf",
    output_path="table_visualization.png"
)
```

## 📊 Performance Optimization

### GPU Acceleration

Enable GPU support for faster processing:

```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = 'cpu'
    print("Using CPU")

# Initialize with GPU
extractor = EasyOCRExtractor(device=device)
```

### Memory Management

For large-scale processing:

```python
import gc
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

extractor = EasyOCRExtractor()

for i, document in enumerate(large_document_list):
    result = extractor.extract(document)
    # Process result...
    
    # Clean up memory every 100 documents
    if i % 100 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

## 🔍 Debugging

### Enable Detailed Logging

```python
import logging
from omnidocs.utils.logging import get_logger

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = get_logger(__name__)

# Initialize extractor with logging
extractor = EasyOCRExtractor(show_log=True)
```

### Inspect Results

```python
result = extractor.extract("document.pdf")

# Inspect result structure
print(f"Result type: {type(result)}")
print(f"Available attributes: {dir(result)}")

# For OCR results
if hasattr(result, 'texts'):
    print(f"Number of text regions: {len(result.texts)}")
    for i, text in enumerate(result.texts[:3]):  # First 3
        print(f"Text {i}: {text.text[:50]}...")
        print(f"Confidence: {text.confidence:.3f}")
        print(f"Bbox: {text.bbox}")

# For table results
if hasattr(result, 'tables'):
    print(f"Number of tables: {len(result.tables)}")
    for i, table in enumerate(result.tables):
        print(f"Table {i} shape: {table.df.shape}")
```

## 📚 Examples by Use Case

### Document Digitization
```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

extractor = EasyOCRExtractor(languages=['en'])
result = extractor.extract("scanned_document.png")
with open("digitized.txt", "w") as f:
    f.write(result.full_text)
```

### Financial Report Processing
```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

extractor = CamelotExtractor()
result = extractor.extract("financial_report.pdf")
for i, table in enumerate(result.tables):
    table.df.to_csv(f"financial_table_{i}.csv", index=False)
```

### Academic Paper Analysis
```python
from omnidocs.tasks.math_expression_extraction.extractors.nougat import NougatExtractor

extractor = NougatExtractor()
result = extractor.extract("research_paper.pdf")
print("Extracted LaTeX formulas:")
print(result.full_text)
```

## 🚨 Common Issues & Solutions

### Import Errors
```python
# Check if dependencies are installed
try:
    from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor
    print("✅ EasyOCR available")
except ImportError as e:
    print(f"❌ EasyOCR not available: {e}")
    print("Install with: pip install easyocr")
```

### Memory Issues
```python
# For large documents, process page by page
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

extractor = CamelotExtractor()
# Process specific pages instead of all pages
result = extractor.extract("large_document.pdf", pages="1-5")
```

### Language Support
```python
# Check supported languages
extractor = EasyOCRExtractor()
supported = extractor.get_supported_languages()
print(f"Supported languages: {supported}")
```

## 🔗 Related Resources

- **[Getting Started Guide](../getting_started/quickstart.md)** - Quick introduction
- **[Task Tutorials](../tasks/)** - Detailed task-specific guides
- **[GitHub Repository](https://github.com/adithya-s-k/OmniDocs)** - Source code and issues
- **[Contributing Guide](../../CONTRIBUTING.md)** - How to contribute

---

*This API reference is automatically generated from the source code. For the most up-to-date information, please refer to the docstrings in the source code.*
