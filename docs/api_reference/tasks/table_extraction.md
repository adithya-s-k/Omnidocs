# 📊 Table Extraction

This section documents the API for table extraction tasks, providing various extractors to retrieve tabular data from documents.

## Overview

Table extraction in OmniDocs focuses on accurately identifying and extracting structured data from tables within PDFs and images. This is crucial for converting unstructured document data into usable formats like DataFrames.

## Available Extractors

### CamelotExtractor

Accurate table extraction from PDFs, supporting both lattice (for tables with lines) and stream (for tables without lines) modes.

::: omnidocs.tasks.table_extraction.extractors.camelot.CamelotExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

extractor = CamelotExtractor(flavor='lattice') # or 'stream'
result = extractor.extract("document.pdf")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
    print(table.df.head())
```

### PDFPlumberTableExtractor

A lightweight and fast PDF table extraction library.

::: omnidocs.tasks.table_extraction.extractors.pdfplumber.PDFPlumberExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.pdfplumber import PDFPlumberExtractor

extractor = PDFPlumberExtractor()
result = extractor.extract("document.pdf")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
```

### PPStructureTableExtractor

An OCR tool that supports multiple languages and provides table recognition capabilities.

::: omnidocs.tasks.table_extraction.extractors.ppstructure.PPStructureExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.ppstructure import PPStructureExtractor

extractor = PPStructureExtractor()
result = extractor.extract("image.png")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
```

### SuryaTableExtractor

Deep learning-based table structure recognition, part of the Surya library.

::: omnidocs.tasks.table_extraction.extractors.surya_table.SuryaTableExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.surya_table import SuryaTableExtractor

extractor = SuryaTableExtractor()
result = extractor.extract("document.pdf")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
```

### TableTransformerExtractor

A transformer-based model for table detection and extraction.

::: omnidocs.tasks.table_extraction.extractors.table_transformer.TableTransformerExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.table_transformer import TableTransformerExtractor

extractor = TableTransformerExtractor()
result = extractor.extract("image.png")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
```

### TableFormerExtractor

An advanced deep learning model for table structure parsing.

::: omnidocs.tasks.table_extraction.extractors.tableformer.TableFormerExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.tableformer import TableFormerExtractor

extractor = TableFormerExtractor()
result = extractor.extract("document.pdf")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
```

### TabulaExtractor

A Java-based tool for extracting tables from PDFs. Requires Java runtime installed.

::: omnidocs.tasks.table_extraction.extractors.tabula.TabulaExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract

#### Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.tabula import TabulaExtractor

extractor = TabulaExtractor()
result = extractor.extract("document.pdf")
for i, table in enumerate(result.tables):
    print(f"Table {i+1} shape: {table.df.shape}")
```

## TableOutput

The standardized output format for table extraction results.

::: omnidocs.tasks.table_extraction.base.TableOutput
    options:
      show_root_heading: true
      show_source: false

### Key Properties

- `tables` (List[Table]): List of extracted tables.
- `source_file` (str): Path to the processed file.
- `processing_time` (Optional[float]): Time taken for extraction.

#### Key Methods

- `save_json(output_path)`: Save results metadata to a JSON file.
- `save_tables_as_csv(output_dir)`: Save all extracted tables as individual CSV files.
- `get_tables_by_confidence(min_confidence)`: Filter tables by confidence score.

## Table

Represents a single extracted table.

::: omnidocs.tasks.table_extraction.base.Table
    options:
      show_root_heading: true
      show_source: false

### Attributes

- `df` (pandas.DataFrame): The extracted table data as a DataFrame.
- `bbox` (List[float]): Bounding box coordinates of the table.
- `page_number` (int): The page number where the table is found.
- `confidence` (Optional[float]): Confidence score of the table extraction.

#### Key Methods

- `to_csv()`: Convert the table DataFrame to a CSV string.
- `to_html()`: Convert the table DataFrame to an HTML string.

## BaseTableExtractor

The abstract base class for all table extraction extractors.

::: omnidocs.tasks.table_extraction.base.BaseTableExtractor
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - extract
        - preprocess_input
        - postprocess_output
        - visualize

## TableMapper

Handles mapping of table-related labels and normalization of bounding boxes.

::: omnidocs.tasks.table_extraction.base.BaseTableMapper
    options:
      show_root_heading: true
      show_source: false

## Related Resources

- [Table Extraction Overview](../tasks/table_extraction/overview.md)
- [Camelot Tutorial](../../tasks/table_extraction/tutorials/camelot.ipynb)
- [PDFPlumber Tutorial](../../tasks/table_extraction/tutorials/pdfplumber.ipynb)
- [Surya Table Tutorial](../../tasks/table_extraction/tutorials/surya_table.ipynb)
- [Tabula Tutorial](../../tasks/table_extraction/tutorials/tabula.ipynb)
- [TableTransformer Tutorial](../../tasks/table_extraction/tutorials/tabletransformer.ipynb)
- [TableFormer Tutorial](../../tasks/table_extraction/tutorials/tableformer.ipynb)
- [Core Classes](../core.md)
- [Utilities](../utils.md)
