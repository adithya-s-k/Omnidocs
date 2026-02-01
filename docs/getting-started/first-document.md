# First Document: Loading and Processing

Learn how to load documents, access pages, and work with document metadata.

## Understanding the Document Class

The `Document` class is OmniDocs' central abstraction for working with PDF and image data. Key design principles:

- **Stateless**: Documents contain only source data, not analysis results
- **Lazy Loading**: Pages are rendered only when accessed (memory efficient)
- **Cached**: Once rendered, pages are cached to avoid re-rendering
- **Memory Safe**: Clear cache to free GPU/CPU memory as needed

```python
from omnidocs import Document

# This is fast - does NOT render pages yet
doc = Document.from_pdf("large_document.pdf")
# Takes < 1 second even for 1000-page PDFs

# Pages only render when you access them
page = doc.get_page(0)  # Renders on demand
page = doc.get_page(1)  # Cached, returns instantly
```

## Loading Documents

OmniDocs supports multiple input formats:

### From PDF File

```python
from omnidocs import Document

# Simple case
doc = Document.from_pdf("document.pdf")
print(f"Loaded: {doc.page_count} pages")

# With custom DPI (resolution)
doc = Document.from_pdf("document.pdf", dpi=300)
# Higher DPI = higher quality but slower rendering
# Default: 150 DPI (good balance)

# Load only specific pages
doc = Document.from_pdf("document.pdf", page_range=(0, 10))
# Loads pages 0-10 (inclusive, 0-indexed)
# Useful for working with partial PDFs

# Context manager (auto-closes document)
with Document.from_pdf("document.pdf") as doc:
    page = doc.get_page(0)
    # Document automatically closed after use
```

**Error Handling:**
```python
from omnidocs import Document
from omnidocs.document import DocumentLoadError, UnsupportedFormatError

try:
    doc = Document.from_pdf("missing.pdf")
except DocumentLoadError as e:
    print(f"File not found: {e}")
except UnsupportedFormatError as e:
    print(f"Wrong format: {e}")
```

### From URL

```python
from omnidocs import Document

# Download and load from URL
doc = Document.from_url("https://example.com/document.pdf")

# With timeout (default: 30 seconds)
doc = Document.from_url(
    "https://example.com/large_file.pdf",
    timeout=60  # Wait up to 60 seconds
)

# Error handling for downloads
from omnidocs.document import URLDownloadError

try:
    doc = Document.from_url("https://bad.url/doc.pdf")
except URLDownloadError as e:
    print(f"Download failed: {e}")
```

### From Raw Bytes

```python
from omnidocs import Document

# Useful for PDFs from databases or APIs
pdf_bytes = b"%PDF-1.4..."  # Raw PDF bytes

doc = Document.from_bytes(
    pdf_bytes,
    filename="document.pdf"  # Optional, for metadata
)

# Real-world example: Download with requests
import requests

response = requests.get("https://example.com/doc.pdf")
doc = Document.from_bytes(
    response.content,
    filename="downloaded.pdf"
)
```

### From Images

OmniDocs can treat images as single or multi-page documents:

```python
from omnidocs import Document

# Single image
doc = Document.from_image("page.png")
print(doc.page_count)  # Always 1

# Multiple images (treated as multi-page document)
doc = Document.from_images([
    "page1.png",
    "page2.png",
    "page3.png"
])
print(doc.page_count)  # 3
```

**Supported Image Formats:**
- PNG, JPG, JPEG, GIF, BMP, TIFF, WebP

## Accessing Pages

### Get Single Page

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# 0-indexed (first page is 0)
first_page = doc.get_page(0)
last_page = doc.get_page(doc.page_count - 1)

# Pages are PIL Images - compatible with all image tools
print(f"Image size: {first_page.size}")  # (width, height)

# Save page as image
first_page.save("page_1.png")
```

### Iterate Over All Pages

For large documents, iterate instead of loading all at once:

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Memory efficient - one page at a time
for i, page in enumerate(doc.iter_pages()):
    print(f"Page {i + 1}")
    # Process page
    # Page is cleared from memory before next iteration
```

**Why Iterate?**
- **Memory Safe**: Only one page in memory at a time
- **Progress Tracking**: Easy to show progress for large docs
- **Early Exit**: Can stop processing early
- **Cache Control**: Can clear cache between pages

### Load All Pages at Once

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Load all pages into memory
all_pages = doc.pages  # List of PIL Images

# Only use this for small documents (< 50 pages)
# For larger documents, use iter_pages() instead
```

### Get Page Properties

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Get page dimensions WITHOUT rendering
width, height = doc.get_page_size(0)
# Fast - doesn't need to render the full page

# Get page as PIL Image (renders on demand)
page = doc.get_page(0)
print(f"Page: {page.size}")  # (width, height)
print(f"Mode: {page.mode}")  # Color mode (RGB, RGBA, etc.)
```

## Accessing Text Content

### Get Page Text

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Get text from specific page (1-indexed, like PDF viewers)
text = doc.get_page_text(1)  # First page
print(text[:100])  # First 100 characters

# Note: get_page(0) uses 0-indexing, get_page_text(1) uses 1-indexing
# get_page_text is slower (extracts text using pypdfium2/pdfplumber)
```

### Get Full Document Text

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Get all text (lazy, cached)
full_text = doc.text
print(f"Total: {len(full_text)} characters")

# Uses pypdfium2 (fast), falls back to pdfplumber if needed
# Cached after first access
```

**Warning:** PDF text extraction can fail for:
- Scanned PDFs (images, no selectable text)
- Encrypted PDFs
- PDFs with unusual encodings

For reliable text extraction, always use a Vision-Language Model (Qwen, DotsOCR, etc.), not PDF text extraction.

## Understanding Document Metadata

### Access Metadata

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Get metadata object
metadata = doc.metadata
print(metadata.page_count)      # Number of pages
print(metadata.source_type)     # "file", "url", "bytes", "image"
print(metadata.source_path)     # File path or URL
print(metadata.file_name)       # Filename
print(metadata.file_size)       # Size in bytes
print(metadata.format)          # "pdf", "png", "jpg", etc.
print(metadata.image_dpi)       # DPI for rendering (150, 300, etc.)
print(metadata.loaded_at)       # ISO timestamp

# For PDFs, also includes PDF metadata if available
if metadata.pdf_metadata:
    print(metadata.pdf_metadata)
    # {'Title': '...', 'Author': '...', 'Subject': '...'}
```

### Convert to Dictionary

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Convert metadata to dict for serialization
data = doc.to_dict()
print(data)

# Output:
# {
#     'source_type': 'file',
#     'source_path': '/path/to/document.pdf',
#     'file_name': 'document.pdf',
#     'file_size': 12345,
#     'page_count': 50,
#     'format': 'pdf',
#     'image_dpi': 150,
#     'loaded_at': '2024-02-01T12:34:56.123456'
# }

# Save metadata as JSON
import json
with open("metadata.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Memory Management

### Cache Behavior

OmniDocs caches rendered pages to avoid re-rendering:

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# First access: renders page (slow)
page1 = doc.get_page(0)  # ~200ms

# Second access: returns cached copy (instant)
page1_again = doc.get_page(0)  # <1ms
```

### Clear Cache

For large batch processing, clear cache to free memory:

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Clear specific page
doc.clear_cache(0)

# Clear all pages
doc.clear_cache()

# Good practice when processing many documents
for pdf_file in pdf_files:
    doc = Document.from_pdf(pdf_file)
    for page in doc.iter_pages():
        # Process page
        pass
    doc.clear_cache()  # Free memory before next document
```

### Close Document

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Explicitly close and free resources
doc.close()

# Or use context manager (recommended)
with Document.from_pdf("document.pdf") as doc:
    # Work with document
    pass
# Automatically closed
```

## Performance Tips

### Optimize DPI for Speed

```python
from omnidocs import Document

# Default: 150 DPI (good balance)
doc_balanced = Document.from_pdf("doc.pdf", dpi=150)

# Faster but lower quality (50 DPI)
doc_fast = Document.from_pdf("doc.pdf", dpi=100)

# Higher quality but slower (300 DPI)
doc_hq = Document.from_pdf("doc.pdf", dpi=300)
```

| DPI | Speed | Quality | Use Case |
|-----|-------|---------|----------|
| 72 | 10x | Poor | Text only, OCR |
| 100 | 5x | Fair | Quick scan |
| 150 | 1x | Good | Default |
| 200 | 0.5x | Very Good | Important docs |
| 300 | 0.2x | Excellent | Archives, legal |

### Process Large Documents Efficiently

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

doc = Document.from_pdf("1000_pages.pdf", dpi=150)
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# Pattern: Process one page at a time, save immediately
results = []
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page, output_format="markdown")

    # Save page result immediately (don't accumulate in memory)
    with open(f"page_{i+1:04d}.md", "w") as f:
        f.write(result.content)

    # Clear cache every N pages to free memory
    if (i + 1) % 10 == 0:
        doc.clear_cache()
        print(f"Processed {i + 1}/{doc.page_count} pages")

print("Done!")
```

### Working with Page Range

```python
from omnidocs import Document

# Load only pages 50-100 (saves memory)
doc = Document.from_pdf(
    "1000_pages.pdf",
    page_range=(49, 99)  # 0-indexed, inclusive
)

print(doc.page_count)  # 51 (pages 50-100 inclusive)

# Now access as normal
for page in doc.iter_pages():
    # Only processes the 51-page range
    pass
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Initialize extractor once (expensive, don't repeat)
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# Find all PDFs
pdf_files = sorted(Path("documents/").glob("*.pdf"))

# Process each
for pdf_path in pdf_files:
    print(f"Processing {pdf_path.name}...")

    with Document.from_pdf(str(pdf_path)) as doc:
        for i, page in enumerate(doc.iter_pages()):
            result = extractor.extract(page)
            # Save result
            output_file = pdf_path.stem / f"page_{i+1}.md"
            with open(output_file, "w") as f:
                f.write(result.content)

print(f"Processed {len(pdf_files)} documents")
```

## Real-World Examples

### Example 1: Extract and Save First Page

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Load document
doc = Document.from_pdf("report.pdf")

# Extract first page
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)
result = extractor.extract(doc.get_page(0), output_format="markdown")

# Save
with open("first_page.md", "w") as f:
    f.write(result.content)
```

### Example 2: Extract Table of Contents

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

doc = Document.from_pdf("book.pdf")
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# Extract first few pages (usually TOC is at start)
toc_pages = []
for i in range(min(5, doc.page_count)):
    page = doc.get_page(i)
    result = extractor.extract(page)
    toc_pages.append(result.content)

# Combine and save TOC
with open("table_of_contents.md", "w") as f:
    f.write("\n\n".join(toc_pages))
```

### Example 3: Document Summary

```python
from omnidocs import Document

doc = Document.from_pdf("document.pdf")

# Print document info
print(f"File: {doc.metadata.file_name}")
print(f"Size: {doc.metadata.file_size / (1024**2):.2f} MB")
print(f"Pages: {doc.page_count}")
print(f"Loaded: {doc.metadata.loaded_at}")

if doc.metadata.pdf_metadata:
    print(f"Title: {doc.metadata.pdf_metadata.get('Title', 'N/A')}")
    print(f"Author: {doc.metadata.pdf_metadata.get('Author', 'N/A')}")
```

## Troubleshooting

### "Page out of range"

```python
# Always check bounds
if page_num < 0 or page_num >= doc.page_count:
    print("Invalid page number")
else:
    page = doc.get_page(page_num)
```

### Document takes too long to load

```python
# Increase DPI if rendering is slow
doc = Document.from_pdf("doc.pdf", dpi=100)  # Lower quality, faster

# Or use page range
doc = Document.from_pdf("doc.pdf", page_range=(0, 10))  # First 10 pages
```

### "Out of memory" during iteration

```python
# Clear cache more frequently
for i, page in enumerate(doc.iter_pages()):
    # Process page
    if i % 5 == 0:  # Every 5 pages
        doc.clear_cache()
```

## Next Steps

- **[Quickstart](quickstart.md)** - Jump to extracting text
- **[Choosing Backends](choosing-backends.md)** - Select the right inference backend
- **[API Reference](../api/document.md)** - Complete API documentation

## Summary

```python
# Load document (4 ways)
doc = Document.from_pdf("file.pdf")
doc = Document.from_url("https://example.com/doc.pdf")
doc = Document.from_bytes(pdf_bytes)
doc = Document.from_image("page.png")

# Access pages
page = doc.get_page(0)           # Single page (0-indexed)
for page in doc.iter_pages():    # Iterate efficiently
    pass
width, height = doc.get_page_size(0)  # Get size without rendering

# Access metadata
print(doc.page_count)
print(doc.metadata.file_name)
print(doc.to_dict())

# Memory management
doc.clear_cache()                # Free memory
doc.clear_cache(0)               # Clear specific page
doc.close()                      # Close document
```

Happy processing!
