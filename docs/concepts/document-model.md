# Document Model

> **Core Principle**: The Document class represents source data only. It does NOT store task results, does NOT perform analysis, and does NOT modify the original content.

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Lazy Page Loading](#lazy-page-loading)
3. [Document Methods](#document-methods)
4. [DocumentMetadata](#documentmetadata)
5. [Memory Management](#memory-management)
6. [Common Patterns](#common-patterns)
7. [When to Use Document](#when-to-use-document)

---

## Design Philosophy

### Why Stateless?

The Document class is **intentionally stateless** - it doesn't store analysis results. This is a deliberate design choice with important consequences.

### Problem with Stateful Design

```python
# ❌ BAD: Document stores extraction results

class BadDocument:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages = []  # ← Stores rendered pages
        self.layout_results = []  # ← Stores layout analysis
        self.ocr_results = []  # ← Stores OCR results
        self.text_results = []  # ← Stores extracted text

    def extract_layout(self):
        # Stores result internally
        self.layout_results.append(...)

    def extract_text(self):
        # Stores result internally
        self.text_results.append(...)

    # Now what is doc.get_page(0)?
    # Is it the image, or metadata about extractions?
    # Which extractor's results take priority?
    # How do I re-run an extractor with different params?
```

**Problems**:
- Confusion: What is source data vs. analysis?
- Inflexibility: Can't re-extract with different parameters
- Memory: Multiple copies of results in document
- Coupling: Document depends on all extractors

### Solution: Stateless Design

```python
# ✅ GOOD: Document is just source data

class Document:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self._lazy_pages = []  # ← Only lazy wrappers, NOT rendered

    def get_page(self, idx):
        # Returns PIL Image - original pixel data
        return self._lazy_pages[idx].image

    @property
    def text(self):
        # Returns extracted text from PDF metadata
        # NOT analysis results
        return self._extracted_text_from_pdf

# Extractors are independent
layout_result = layout_extractor.extract(doc.get_page(0))
ocr_result = ocr_extractor.extract(doc.get_page(0))
text_result = text_extractor.extract(doc.get_page(0))

# User composes results
combined = {
    "layout": layout_result,
    "ocr": ocr_result,
    "text": text_result,
}
```

**Benefits**:
- **Clarity**: Document = source, Extractors = analysis
- **Flexibility**: Run any extractor, any number of times
- **Reusability**: Same document object, different analyses
- **Composability**: Easy to build pipelines
- **Caching**: User controls what to keep in memory

---

## Lazy Page Loading

The Document class uses **lazy evaluation** for pages: they're not rendered until accessed.

### How It Works

```python
doc = Document.from_pdf("paper.pdf")  # Fast - reads file, creates metadata
# No page rendering yet!

page = doc.get_page(0)  # Slow - renders page 0
page = doc.get_page(1)  # Renders page 1
page = doc.get_page(0)  # Fast - returns cached page 0
```

### The LazyPage Class

Behind the scenes, each page is wrapped in a `LazyPage` object:

```python
class LazyPage:
    def __init__(self, pdf_doc, page_index, dpi=150):
        self._pdf_doc = pdf_doc
        self._page_index = page_index
        self._dpi = dpi
        self._cached_image = None  # ← Not rendered yet
        self._cached_text = None

    @property
    def image(self) -> Image.Image:
        """Render page to PIL Image (cached after first access)."""
        if self._cached_image is None:
            # Render on first access
            scale = self._dpi / 72
            page = self._pdf_doc[self._page_index]
            bitmap = page.render(scale=scale)
            self._cached_image = bitmap.to_pil().convert("RGB")
        return self._cached_image  # ← Return cached copy

    @property
    def text(self) -> str:
        """Extract text (cached)."""
        if self._cached_text is None:
            # Extract on first access
            page = self._pdf_doc[self._page_index]
            textpage = page.get_textpage()
            self._cached_text = textpage.get_text_range()
        return self._cached_text

    def clear_cache(self):
        """Free memory."""
        self._cached_image = None
```

### Memory Efficiency

```python
# Loading 1000-page document
doc = Document.from_pdf("big_book.pdf")
# Memory used: Just metadata + LazyPage wrapper objects (~1MB)

# Processing pages one at a time
for page in doc.iter_pages():
    result = extractor.extract(page)
    # Memory used: 1 rendered page at a time (~5-10MB)
    # Each page only stays in memory while processing

    doc.clear_cache()  # Explicitly free page from memory
    # Page rendered again if accessed later (but usually not needed)
```

### With vs Without Lazy Loading

| Operation | Without Lazy Loading | With Lazy Loading |
|-----------|----------------------|-------------------|
| Load 100-page PDF | 5 seconds | 0.1 seconds |
| Access page 0 | Instant | 0.5 seconds |
| Access page 50 | Instant | 0.5 seconds |
| Memory for 100 pages | 500 MB | 1 MB |
| Process all pages sequentially | 500 MB RAM | 10 MB RAM |

---

## Document Methods

### Loading Documents

#### from_pdf()

Load a PDF file from disk.

```python
from omnidocs import Document

# Basic usage
doc = Document.from_pdf("paper.pdf")

# With options
doc = Document.from_pdf(
    "paper.pdf",
    page_range=(0, 4),  # Only pages 0-4 (0-indexed, inclusive)
    dpi=300,  # Higher resolution (default: 150)
)

# Properties
print(doc.page_count)  # 5 (with page_range specified)
print(doc.metadata.file_name)  # "paper.pdf"
print(doc.metadata.file_size)  # 2048000 bytes
```

**When to use**: Reading PDF files from disk

**Raises**:
- `DocumentLoadError` - File not found
- `UnsupportedFormatError` - Not a PDF file
- `PageRangeError` - Invalid page range

#### from_url()

Download and load a PDF from a URL.

```python
# Basic usage
doc = Document.from_url("https://example.com/paper.pdf")

# With options
doc = Document.from_url(
    "https://arxiv.org/pdf/2105.00001.pdf",
    timeout=60,  # Download timeout in seconds
    page_range=(0, 9),  # Only first 10 pages
)

print(doc.metadata.source_path)  # Full URL
print(doc.metadata.file_name)  # "paper.pdf"
```

**When to use**: Loading PDFs from the internet

**Raises**:
- `URLDownloadError` - Download failed (network error, 404, etc.)
- `PageRangeError` - Invalid page range

#### from_bytes()

Load PDF from bytes in memory.

```python
# From reading a file
with open("paper.pdf", "rb") as f:
    pdf_bytes = f.read()
doc = Document.from_bytes(pdf_bytes, filename="paper.pdf")

# From downloaded content
response = requests.get("https://example.com/doc.pdf")
doc = Document.from_bytes(response.content, filename="doc.pdf")

# With page range
doc = Document.from_bytes(
    pdf_bytes,
    filename="paper.pdf",
    page_range=(0, 19),  # First 20 pages
)
```

**When to use**: PDF already in memory, or from API responses

**Raises**:
- `PageRangeError` - Invalid page range

#### from_image()

Load a single image as a single-page document.

```python
doc = Document.from_image("page.png")

print(doc.page_count)  # 1
print(doc.metadata.format)  # "png"

page = doc.get_page(0)  # PIL Image of the page
```

**When to use**: Processing single images, screenshots, or scans

**Raises**:
- `DocumentLoadError` - File not found

#### from_images()

Load multiple images as a multi-page document.

```python
doc = Document.from_images([
    "page1.png",
    "page2.png",
    "page3.jpg",
])

print(doc.page_count)  # 3
print(doc.metadata.format)  # "images"

for page in doc.iter_pages():
    result = extractor.extract(page)
```

**When to use**: Processing scanned documents or image sequences

**Raises**:
- `DocumentLoadError` - Any file not found

---

### Accessing Document Data

#### get_page(idx)

Get a single page as a PIL Image.

```python
doc = Document.from_pdf("paper.pdf")

# 0-indexed
page = doc.get_page(0)  # First page (PIL Image, RGB)
page = doc.get_page(doc.page_count - 1)  # Last page

# Memory efficient for large documents
for i in range(doc.page_count):
    page = doc.get_page(i)
    result = extractor.extract(page)
    doc.clear_cache(i)  # Free memory
```

**Returns**: PIL Image in RGB mode

**Raises**: `PageRangeError` if index out of range

#### get_page_text(page_num)

Get text extracted from a PDF page.

```python
doc = Document.from_pdf("paper.pdf")

# NOTE: 1-indexed (like PDF viewers)
text = doc.get_page_text(1)  # First page (1-indexed)
text = doc.get_page_text(2)  # Second page

print(len(text))  # String of all text on page
```

**Note**: 1-indexed (PDF convention), not 0-indexed like `get_page()`

**Returns**: String of text extracted from PDF

**Raises**: `PageRangeError` if index out of range

#### get_page_size(idx)

Get page dimensions without rendering.

```python
doc = Document.from_pdf("paper.pdf")

width, height = doc.get_page_size(0)  # Fast, no rendering
print(f"Page 0: {width}x{height} pixels")

# Useful for pre-processing
if width > 2000 or height > 2000:
    print("High resolution page")
```

**Returns**: Tuple of (width, height) in pixels

**Raises**: `PageRangeError` if index out of range

**Advantages**: No rendering needed, very fast

#### iter_pages()

Iterate over pages one at a time (memory efficient).

```python
doc = Document.from_pdf("long_document.pdf")

for page in doc.iter_pages():
    # Each iteration:
    # - Loads next page
    # - Renders to image
    # - Provides PIL Image
    result = extractor.extract(page)

    # Clear memory after use
    doc.clear_cache()

# Equivalent to:
for i in range(doc.page_count):
    page = doc.get_page(i)
    result = extractor.extract(page)
```

**Yields**: PIL Images (one at a time)

**Memory**: Only 1 page rendered at a time

---

### Document Properties

#### page_count

Number of pages in the document.

```python
doc = Document.from_pdf("paper.pdf")
print(doc.page_count)  # 12

doc = Document.from_pdf("paper.pdf", page_range=(0, 4))
print(doc.page_count)  # 5 (only loaded pages)
```

#### metadata

DocumentMetadata object with source information.

```python
doc = Document.from_pdf("paper.pdf")

meta = doc.metadata
print(meta.source_type)  # "file"
print(meta.source_path)  # "/absolute/path/to/paper.pdf"
print(meta.file_name)  # "paper.pdf"
print(meta.file_size)  # Bytes
print(meta.page_count)  # Number of pages
print(meta.format)  # "pdf"
print(meta.image_dpi)  # 150
print(meta.loaded_at)  # ISO timestamp
```

#### text

Full document text (lazy, cached).

```python
doc = Document.from_pdf("paper.pdf")

# First access: extracts from PDF
full_text = doc.text
# Uses pypdfium2 first (fast), falls back to pdfplumber

# Subsequent accesses: return cached value
text_again = doc.text  # Instant
```

**Performance Note**: Caches after first access. Large documents may take a few seconds first time.

#### pages

List of all page images.

```python
doc = Document.from_pdf("paper.pdf")

# WARNING: Renders ALL pages into memory
all_pages = doc.pages  # [PIL Image, PIL Image, ...]

# Don't do this for large documents!
# Better: use iter_pages() or get_page() individually
```

**Warning**: Renders all pages at once. Use `iter_pages()` or `get_page()` for better memory efficiency.

---

## DocumentMetadata

Metadata about the document source.

```python
class DocumentMetadata(BaseModel):
    # Source information
    source_type: str  # "file", "url", "bytes", "image"
    source_path: Optional[str]  # Path or URL
    file_name: Optional[str]  # Just the filename
    file_size: Optional[int]  # Bytes

    # PDF metadata
    pdf_metadata: Optional[Dict[str, Any]]  # From PDF metadata

    # Document properties
    page_count: int  # Number of pages
    format: str  # "pdf", "png", "jpg", "images"

    # Image rendering
    image_dpi: int  # DPI for page rendering (50-600)
    image_format: str  # Color format (usually "RGB")

    # Text extraction
    text_extraction_engine: Optional[str]  # "pypdfium2", "pdfplumber"

    # Timestamps
    loaded_at: str  # ISO 8601 timestamp
```

### Example Usage

```python
doc = Document.from_pdf("paper.pdf")

meta = doc.metadata
print(f"Loaded: {meta.file_name}")
print(f"Pages: {meta.page_count}")
print(f"Size: {meta.file_size} bytes")
print(f"DPI: {meta.image_dpi}")
print(f"Source: {meta.source_type}")

# PDF-specific metadata
if meta.pdf_metadata:
    print(f"Title: {meta.pdf_metadata.get('Title')}")
    print(f"Author: {meta.pdf_metadata.get('Author')}")

# Convert to dict
meta_dict = meta.model_dump()
```

---

## Memory Management

### Caching Strategy

The Document class uses two levels of caching:

1. **Page Cache**: Individual pages cached in LazyPage
2. **Full Text Cache**: Entire document text cached

```python
doc = Document.from_pdf("paper.pdf")

# Page caching (automatic)
page = doc.get_page(0)  # Rendered and cached
page = doc.get_page(0)  # Returns cached copy (fast)

# Text caching (automatic)
text = doc.text  # Extracted and cached
text = doc.text  # Returns cached copy (instant)
```

### Explicit Cache Control

#### clear_cache()

Free cached pages from memory.

```python
doc = Document.from_pdf("big_book.pdf")

# Process with memory control
for i in range(doc.page_count):
    page = doc.get_page(i)
    result = extractor.extract(page)

    # Free previous page from memory
    if i > 0:
        doc.clear_cache(i - 1)

# Or clear all at once
doc.clear_cache()

# Check memory usage
import psutil
process = psutil.Process()
print(process.memory_info().rss)  # Resident memory
```

### When Processing Large Documents

```python
# ❌ BAD: Renders all pages at once
doc = Document.from_pdf("huge_book.pdf")  # 1000 pages
all_pages = doc.pages  # Tries to render all 1000 pages at once
# Memory error!

# ✅ GOOD: Process one page at a time
doc = Document.from_pdf("huge_book.pdf")
for page in doc.iter_pages():
    result = extractor.extract(page)
    # Only 1 page in memory at a time
    # Doc doesn't cache unless you keep page variables
```

### save_images()

Save all pages as individual image files.

```python
doc = Document.from_pdf("paper.pdf")

# Save all pages
paths = doc.save_images(
    output_dir="output/",
    prefix="page",  # Files: page_001.png, page_002.png, ...
    format="PNG",  # PNG or JPEG
)

print(f"Saved {len(paths)} images")
```

---

## Common Patterns

### Pattern 1: Process One Page

```python
from omnidocs import Document
from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

doc = Document.from_pdf("paper.pdf")
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig())

page = doc.get_page(0)
result = layout.extract(page)

print(f"Found {len(result.bboxes)} layout elements")
```

### Pattern 2: Process All Pages (Memory Efficient)

```python
doc = Document.from_pdf("paper.pdf")
extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig())

results = []
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page)
    results.append({
        "page": i,
        "elements": len(result.bboxes),
    })

    # Free memory after processing
    if i > 0:
        doc.clear_cache(i - 1)

print(f"Processed {len(results)} pages")
```

### Pattern 3: Extract Text from Entire Document

```python
doc = Document.from_pdf("paper.pdf")

# Option 1: Use PDF text extraction (fast, no model)
pdf_text = doc.text
print(len(pdf_text), "characters extracted from PDF")

# Option 2: Use VLM extraction (slower, higher quality)
extractor = QwenTextExtractor(backend=QwenTextPyTorchConfig())

full_content = ""
for page in doc.iter_pages():
    result = extractor.extract(page, output_format="markdown")
    full_content += result.content + "\n\n"

print(full_content[:500])
```

### Pattern 4: Process with Page Range

```python
# Only process pages 0-9
doc = Document.from_pdf(
    "large_document.pdf",
    page_range=(0, 9),  # First 10 pages only
)

# Process as usual
for page in doc.iter_pages():
    result = extractor.extract(page)
```

### Pattern 5: Load from Different Sources

```python
# From file
doc = Document.from_pdf("local_file.pdf")

# From URL
doc = Document.from_url("https://example.com/doc.pdf")

# From downloaded bytes
import requests
response = requests.get("https://example.com/file.pdf")
doc = Document.from_bytes(response.content, filename="doc.pdf")

# From images
doc = Document.from_images(["page1.png", "page2.png"])

# Same API for all sources!
for page in doc.iter_pages():
    result = extractor.extract(page)
```

### Pattern 6: Use as Context Manager

```python
# Automatically close and free resources
with Document.from_pdf("paper.pdf") as doc:
    for page in doc.iter_pages():
        result = extractor.extract(page)
# Resources freed automatically
```

---

## When to Use Document

### Use Document When

- **Loading PDFs or images** - Natural entry point
- **Need multiple pages** - Built-in pagination
- **Memory is constrained** - Lazy loading efficiency
- **Want metadata** - Source info, page count, etc.
- **Processing in pipelines** - Integrates with extractors

```python
doc = Document.from_pdf("paper.pdf")  # ✓ Use Document
for page in doc.iter_pages():
    result = extractor.extract(page)
```

### Use Direct Image When

- **Already have PIL Image** - Skip Document
- **Single image analysis** - Direct extractor call
- **Image from different source** - Camera, API, etc.

```python
from PIL import Image
image = Image.open("screenshot.png")  # ✓ Direct image
result = layout.extract(image)  # No Document needed
```

### Pattern: Conditional Use

```python
def process_document(source: str):
    # If source is a path or URL, use Document
    if source.endswith(".pdf"):
        doc = Document.from_pdf(source)
        for page in doc.iter_pages():
            result = extractor.extract(page)

    # If source is a URL
    elif source.startswith("http"):
        doc = Document.from_url(source)
        # ... same pattern

    # If source is an image
    elif source.endswith((".png", ".jpg")):
        image = Image.open(source)
        result = extractor.extract(image)
```

---

## Summary

The Document class provides:

| Feature | Benefit |
|---------|---------|
| Lazy page loading | Fast document loading, efficient memory |
| Automatic caching | Repeated access is instant |
| Multiple source formats | Unified API for PDF, URL, bytes, images |
| Explicit cache control | Fine-grained memory management |
| Metadata | Source information and document properties |
| Iterator interface | Processing large documents efficiently |
| Context manager | Automatic resource cleanup |

**Key Principle**: Document is stateless. It loads and provides source data, nothing more. Extractors handle analysis independently.

---

## Next Steps

- See [Architecture Overview](./architecture-overview.md) for how Document fits in the system
- See [Config Pattern](./config-pattern.md) for how to configure extractors
- See [Backend System](./backend-system.md) for understanding inference backends

