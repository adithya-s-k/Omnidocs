# Table Extraction

OmniDocs provides multiple specialized extractors for table recognition in PDFs and images. Each extractor is implemented as a standalone class with a consistent interface, making it easy to swap between methods for different use cases.

## Available Table Extractors

- **CamelotExtractor**: Accurate table extraction from PDFs (lattice and stream modes).
- **PDFPlumberExtractor**: Lightweight, fast PDF table extraction.
- **SuryaTableExtractor**: Deep learning-based table structure recognition.
- **TabulaExtractor**: Java-based PDF table extraction (requires Java runtime).
- **TableTransformerExtractor**: Transformer-based table detection and extraction.
- **TableFormerExtractor**: Advanced table structure parsing with deep learning.

## Tutorials

- [Camelot](tutorials/camelot.ipynb)
- [PDFPlumber](tutorials/pdfplumber.ipynb)
- [Surya Table](tutorials/surya_table.ipynb)
- [Tabula](tutorials/tabula.ipynb)
- [TableTransformer](tutorials/tabletransformer.ipynb)
- [TableFormer](tutorials/tableformer.ipynb)

## Basic Usage Example

```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor

# Initialize extractor
extractor = CamelotExtractor()

# Extract tables from PDF
results = extractor.extract("path/to/file.pdf")

# Access tables
for i, table in enumerate(results.tables):
    print(f"Table {i+1} as DataFrame:\n", table.df)
```

## Batch Processing Multiple PDFs

```python
import os
from omnidocs.tasks.table_extraction.extractors.pdfplumber import PDFPlumberExtractor

extractor = PDFPlumberExtractor()
pdf_folder = "path/to/pdf_folder/"

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        try:
            results = extractor.extract(pdf_path)
            print(f"{filename}: {len(results.tables)} tables found")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
```

## Working with Different Extractors

```python
from omnidocs.tasks.table_extraction.extractors import (
    CamelotExtractor, PDFPlumberExtractor, SuryaTableExtractor, TabulaExtractor
)

extractors = [CamelotExtractor(), PDFPlumberExtractor(), SuryaTableExtractor(), TabulaExtractor()]
pdf_path = "sample.pdf"

for extractor in extractors:
    try:
        results = extractor.extract(pdf_path)
        print(f"{extractor.__class__.__name__}: {len(results.tables)} tables")
    except Exception as e:
        print(f"{extractor.__class__.__name__} error: {e}")
```

## Output Format

All extractors return a `TableOutput` object with:

```python
class TableOutput:
    tables: List[TableResult]  # Each table as a DataFrame and metadata
    source_file: str           # Path to the processed file
```

## Installation Requirements

Each extractor may require specific dependencies:

### CamelotExtractor
```bash
pip install camelot-py[cv] pandas
```

### PDFPlumberExtractor
```bash
pip install pdfplumber pandas
```

### SuryaTableExtractor
```bash
pip install surya-table torch pandas
```

### TabulaExtractor
```bash
pip install tabula-py pandas
# Requires Java installed and in PATH
```

### TableTransformerExtractor
```bash
pip install table-transformer torch pandas
```

### TableFormerExtractor
```bash
pip install tableformer torch pandas
```

## Troubleshooting

**1. Java Not Found (TabulaExtractor):**
  - Ensure Java is installed and added to your system PATH.

**2. No Tables Detected:**
  - Try a different extractor or adjust parameters (e.g., lattice/stream mode for Camelot).
  - Check PDF quality and ensure tables are not scanned images (use OCR if needed).

**3. Import Errors:**
  - Install missing dependencies as shown above.

**4. Output Not as Expected:**
  - Inspect the DataFrame output and adjust extraction settings.

## Best Practices

1. **Choose the Right Extractor:**
   - Use Camelot for vector PDFs with clear table lines.
   - Use PDFPlumber for lightweight, fast extraction.
   - Use SuryaTable, TableTransformer, or TableFormer for complex or scanned tables.
   - Use Tabula if you need Java-based extraction.

2. **Optimize Performance:**
   - Batch process files and initialize extractors once.
   - Use GPU-enabled extractors for large-scale jobs.

3. **Handle Errors Gracefully:**
   - Wrap extraction in try-except blocks.
   - Log or print errors for debugging.

4. **Validate Output:**
   - Always inspect the DataFrame output for correctness.
   - Post-process tables as needed for your workflow.