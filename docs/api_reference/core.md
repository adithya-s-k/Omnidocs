# 🧩 Core Classes

This section documents the core base classes and fundamental components that power all OmniDocs extractors.

## Base Extractor Classes

### BaseOCRExtractor

The foundation for all OCR (Optical Character Recognition) extractors.

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

#### Key Features

- **Unified Interface**: Consistent API across all OCR engines
- **Language Support**: Multi-language text recognition
- **Batch Processing**: Process multiple documents efficiently
- **Visualization**: Built-in result visualization
- **Device Management**: CPU/GPU support

#### Usage Example

```python
from omnidocs.tasks.ocr_extraction.extractors.easy_ocr import EasyOCRExtractor

# Initialize extractor
extractor = EasyOCRExtractor(
    languages=['en', 'fr'],
    device='cuda',
    show_log=True
)

# Extract text
result = extractor.extract("document.png")
print(f"Extracted: {result.full_text}")

# Visualize results
extractor.visualize(
    result=result,
    image_path="document.png",
    output_path="visualization.png"
)
```

### BaseTableExtractor

The foundation for all table extraction implementations.

::: omnidocs.tasks.table_extraction.base.BaseTableExtractor
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

#### Key Features

- **Multiple Formats**: Support for PDF and image inputs
- **Structured Output**: Returns pandas DataFrames
- **Coordinate Transformation**: Handles PDF to image coordinate mapping
- **Batch Processing**: Process multiple documents
- **Visualization**: Table detection visualization

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

# Initialize extractor
extractor = CamelotExtractor(
    flavor='lattice',
    pages='all'
)

# Extract tables
result = extractor.extract("report.pdf")

# Access tables as DataFrames
for i, table in enumerate(result.tables):
    print(f"Table {i} shape: {table.df.shape}")
    table.df.to_csv(f"table_{i}.csv", index=False)
```

### BaseTextExtractor

The foundation for text extraction from documents.

::: omnidocs.tasks.text_extraction.base.BaseTextExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract
        - extract_all
        - preprocess_input
        - postprocess_output

#### Key Features

- **Multi-format Support**: PDF, DOCX, HTML, and more
- **Layout Preservation**: Maintains document structure
- **Metadata Extraction**: Document properties and formatting
- **Batch Processing**: Handle multiple documents

#### Usage Example

```python
from omnidocs.tasks.text_extraction.extractors.pymupdf import PyMuPDFExtractor

# Initialize extractor
extractor = PyMuPDFExtractor()

# Extract text with layout
result = extractor.extract("document.pdf")

# Access structured text
print(f"Full text: {result.full_text}")
for block in result.text_blocks:
    print(f"Block: {block.text[:50]}...")
    print(f"Position: {block.bbox}")
```

## Data Models

### OCRText

Represents a single text region detected by OCR.

::: omnidocs.tasks.ocr_extraction.base.OCRText
    options:
      show_root_heading: true
      show_source: false

#### Attributes

- `text` (str): The recognized text content
- `confidence` (float): Recognition confidence score (0.0-1.0)
- `bbox` (List[float]): Bounding box coordinates [x1, y1, x2, y2]
- `polygon` (List[List[float]]): Precise polygon coordinates
- `language` (Optional[str]): Detected language code
- `reading_order` (int): Reading order index

#### Example

```python
# Access OCR text regions
for text_region in ocr_result.texts:
    print(f"Text: {text_region.text}")
    print(f"Confidence: {text_region.confidence:.3f}")
    print(f"Bbox: {text_region.bbox}")
    print(f"Language: {text_region.language}")
```

### OCROutput

Complete OCR extraction result.

::: omnidocs.tasks.ocr_extraction.base.OCROutput
    options:
      show_root_heading: true
      show_source: false

#### Key Methods

- `get_text_by_confidence(min_confidence)`: Filter by confidence threshold
- `get_sorted_by_reading_order()`: Sort by reading order
- `save_json(output_path)`: Save results to JSON
- `to_dict()`: Convert to dictionary

#### Example

```python
result = extractor.extract("image.png")

# Filter high-confidence text
high_conf_texts = result.get_text_by_confidence(0.8)
print(f"High confidence regions: {len(high_conf_texts)}")

# Save results
result.save_json("ocr_results.json")
```

### Table

Represents an extracted table with structure and data.

::: omnidocs.tasks.table_extraction.base.Table
    options:
      show_root_heading: true
      show_source: false

#### Key Properties

- `df` (pandas.DataFrame): Table data as DataFrame
- `bbox` (List[float]): Table bounding box
- `confidence` (float): Extraction confidence
- `page_number` (int): Source page number

#### Key Methods

- `to_csv()`: Export as CSV string
- `to_html()`: Export as HTML string
- `to_dict()`: Convert to dictionary

#### Example

```python
for table in table_result.tables:
    # Access as DataFrame
    df = table.df
    print(f"Table shape: {df.shape}")
    
    # Export formats
    csv_content = table.to_csv()
    html_content = table.to_html()
    
    # Save to file
    df.to_excel(f"table_page_{table.page_number}.xlsx")
```

### TableOutput

Complete table extraction result.

::: omnidocs.tasks.table_extraction.base.TableOutput
    options:
      show_root_heading: true
      show_source: false

#### Key Methods

- `get_tables_by_confidence(min_confidence)`: Filter by confidence
- `save_tables_as_csv(output_dir)`: Save all tables as CSV files
- `save_json(output_path)`: Save metadata to JSON

#### Example

```python
result = extractor.extract("document.pdf")

# Filter high-confidence tables
good_tables = result.get_tables_by_confidence(0.7)

# Save all tables
csv_files = result.save_tables_as_csv("output_tables/")
print(f"Saved {len(csv_files)} CSV files")
```

## Mapper Classes

### BaseOCRMapper

Handles language code mapping and normalization for OCR engines.

::: omnidocs.tasks.ocr_extraction.base.BaseOCRMapper
    options:
      show_root_heading: true
      show_source: false

#### Key Methods

- `to_standard_language(engine_language)`: Convert to standard language code
- `from_standard_language(standard_language)`: Convert from standard language code
- `get_supported_languages()`: List supported languages
- `normalize_bbox(bbox, img_width, img_height)`: Normalize bounding box coordinates

### BaseTableMapper

Handles coordinate transformation and table structure mapping.

::: omnidocs.tasks.table_extraction.base.BaseTableMapper
    options:
      show_root_heading: true
      show_source: false

#### Key Methods

- `normalize_bbox(bbox, img_width, img_height)`: Normalize coordinates
- `detect_header_rows(cells)`: Identify header rows

## Abstract Base Classes

All extractors inherit from these abstract base classes, ensuring consistent interfaces:

```python
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """Abstract base class for all extractors."""
    
    @abstractmethod
    def extract(self, input_path: Union[str, Path]) -> Any:
        """Extract data from input document."""
        pass
    
    @abstractmethod
    def preprocess_input(self, input_path: Union[str, Path]) -> Any:
        """Preprocess input for extraction."""
        pass
    
    @abstractmethod
    def postprocess_output(self, raw_output: Any) -> Any:
        """Convert raw output to standardized format."""
        pass
```

## Common Patterns

### Initialization Pattern

All extractors follow this initialization pattern:

```python
class SomeExtractor(BaseExtractor):
    def __init__(
        self,
        device: Optional[str] = None,
        show_log: bool = False,
        languages: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(device, show_log, languages)
        # Extractor-specific initialization
        self._load_model()
```

### Processing Pipeline

Standard processing flow:

1. **Input Validation**: Check file existence and format
2. **Preprocessing**: Convert to required format (PIL Image, etc.)
3. **Model Inference**: Run the actual extraction
4. **Postprocessing**: Convert to standardized output format
5. **Result Packaging**: Create result object with metadata

### Error Handling

Consistent error handling across extractors:

```python
try:
    result = extractor.extract("document.pdf")
except FileNotFoundError:
    print("Document not found")
except ImportError:
    print("Required dependencies not installed")
except Exception as e:
    print(f"Extraction failed: {e}")
```

## Performance Considerations

### Memory Management

- Use generators for batch processing large datasets
- Clear GPU memory between large operations
- Implement proper cleanup in `__del__` methods

### GPU Utilization

- Check GPU availability before initialization
- Batch operations when possible
- Use appropriate tensor data types

### Caching

- Cache model loading where appropriate
- Implement result caching for repeated operations
- Use memory-mapped files for large datasets

## Extension Points

### Custom Extractors

Create custom extractors by inheriting from base classes:

```python
from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor

class CustomOCRExtractor(BaseOCRExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
    
    def _load_model(self):
        # Load your custom model
        pass
    
    def postprocess_output(self, raw_output, img_size):
        # Convert to OCROutput format
        pass
```

### Custom Mappers

Implement custom language or coordinate mappers:

```python
from omnidocs.tasks.ocr_extraction.base import BaseOCRMapper

class CustomMapper(BaseOCRMapper):
    def __init__(self):
        super().__init__('custom_engine')
        self._setup_custom_mapping()
    
    def _setup_custom_mapping(self):
        # Define your language mappings
        pass
```

This core architecture ensures consistency, extensibility, and maintainability across all OmniDocs extractors.
