# Quickstart: 5 Minutes to Your First Extraction

Get OmniDocs running in 5 minutes with this minimal working example.

## 1. Install (1 minute)

```bash
pip install omnidocs[pytorch]
```

If you don't have a GPU, use the API backend instead:
```bash
pip install omnidocs[api]
export OPENAI_API_KEY="sk-..."
```

## 2. Load a Document (30 seconds)

OmniDocs can load PDFs, images, or URLs:

```python
from omnidocs import Document

# Load from PDF file
doc = Document.from_pdf("example.pdf")
print(f"Loaded {doc.page_count} pages")

# Or load from image
doc = Document.from_image("page.png")

# Or load from URL
doc = Document.from_url("https://example.com/document.pdf")
```

## 3. Extract Text (2 minutes)

Extract all text from the document in Markdown format:

```python
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Initialize the extractor (loads model on first use)
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",  # Fast, small model
        device="cuda",  # Use "cpu" if no GPU
    )
)

# Get the first page
first_page = doc.get_page(0)

# Extract text as Markdown
result = extractor.extract(
    first_page,
    output_format="markdown"
)

# Access the extracted content
print(result.content)
```

## 4. Get the Output (30 seconds)

```python
# The result object has these properties:
print(result.content)           # The extracted text (markdown)
print(result.format)            # Output format (markdown, html)
print(result.content_length)    # Number of characters
print(result.has_layout)        # Whether layout was included
```

**Example output:**
```markdown
# Document Title

This is the first paragraph extracted from your document.

## Section Heading

- Bullet point 1
- Bullet point 2

**Bold text** and *italic text* are preserved.
```

## Complete Example: Process an Entire Document

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Load document
doc = Document.from_pdf("research_paper.pdf")
print(f"Processing {doc.page_count} pages...")

# Initialize extractor
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        device="cuda",
    )
)

# Process all pages and collect results
all_content = []
for page_num in range(doc.page_count):
    page = doc.get_page(page_num)
    result = extractor.extract(page, output_format="markdown")
    all_content.append(result.content)
    print(f"Processed page {page_num + 1}/{doc.page_count}")

# Save combined output
full_text = "\n\n---\n\n".join(all_content)
with open("output.md", "w") as f:
    f.write(full_text)

print("Saved extracted text to output.md")
```

## Faster Processing with Efficient Iteration

For large documents, process pages efficiently:

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Load document
doc = Document.from_pdf("large_document.pdf", dpi=150)

# Initialize extractor once
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        device="cuda",
    )
)

# Process pages one at a time (memory efficient)
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page, output_format="markdown")

    # Save each page immediately to avoid memory buildup
    with open(f"page_{i+1:03d}.md", "w") as f:
        f.write(result.content)

    # Clear cache to free GPU memory
    if i % 5 == 0:
        doc.clear_cache()

print(f"Saved all {doc.page_count} pages to output files")
```

## Choose Your Backend

Different backends for different needs:

### PyTorch (Default, Recommended)
Best for local GPU inference on NVIDIA/AMD GPUs.

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-2B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",  # Faster with less precision
)
```

### VLLM (High Throughput)
Best for processing 100+ documents quickly.

```bash
pip install omnidocs[vllm]
```

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

config = QwenTextVLLMConfig(
    model="Qwen/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=2,  # Multi-GPU
    gpu_memory_utilization=0.9,
)
```

### MLX (Apple Silicon)
Best for Mac development.

```bash
pip install omnidocs[mlx]
```

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

config = QwenTextMLXConfig(
    model="Qwen/Qwen3-VL-2B-Instruct",
)
```

### API (Cloud-Based)
Best for no GPU setup.

```bash
pip install omnidocs[api]
export OPENAI_API_KEY="sk-..."
```

```python
from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

config = QwenTextAPIConfig(
    model="qwen3-vl-8b",
    api_key="sk-...",
)
```

## Common Tasks

### Extract from PDF and Save to File

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

doc = Document.from_pdf("input.pdf")
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# Extract first page
result = extractor.extract(doc.get_page(0))

# Save to file
with open("output.md", "w") as f:
    f.write(result.content)
```

### Extract with Layout Information

```python
# Include structure and layout in output
result = extractor.extract(
    page,
    output_format="markdown",
    include_layout=True  # Adds structure information
)

# Access layout information
if result.layout:
    print(f"Found {len(result.layout)} layout elements")
    for elem in result.layout:
        print(f"- {elem.category}: {elem.content[:50]}...")
```

### Batch Process Multiple PDFs

```python
from pathlib import Path
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

# Find all PDFs in a directory
pdf_dir = Path("documents/")
pdf_files = list(pdf_dir.glob("*.pdf"))

# Initialize extractor once
extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

# Process each PDF
for pdf_path in pdf_files:
    print(f"Processing {pdf_path.name}...")
    doc = Document.from_pdf(str(pdf_path))

    for i, page in enumerate(doc.iter_pages()):
        result = extractor.extract(page)

        # Save output
        output_path = pdf_path.stem / f"page_{i+1}.md"
        with open(output_path, "w") as f:
            f.write(result.content)
```

## Output Formats

OmniDocs can extract to different formats:

### Markdown (Default)
Preserves formatting with Markdown syntax:

```python
result = extractor.extract(page, output_format="markdown")
# Output: # Heading\n\nParagraph text\n\n- List items
```

### HTML
Preserves formatting with HTML tags:

```python
result = extractor.extract(page, output_format="html")
# Output: <h1>Heading</h1><p>Paragraph text</p><ul><li>List items</li></ul>
```

## What's Next?

- **[First Document Guide](first-document.md)** - Deep dive into loading and accessing documents
- **[Backend Selection](choosing-backends.md)** - Understand PyTorch vs VLLM vs MLX vs API
- **[Full API Reference](../api/document.md)** - Complete API documentation

## Quick Reference

```python
# Load document
doc = Document.from_pdf("file.pdf")
doc = Document.from_image("file.png")
doc = Document.from_url("https://example.com/doc.pdf")

# Access pages
page = doc.get_page(0)           # Single page (0-indexed)
for page in doc.iter_pages():    # Iterate efficiently
    pass

# Extract text
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

extractor = QwenTextExtractor(
    backend=QwenTextPyTorchConfig(device="cuda")
)

result = extractor.extract(page, output_format="markdown")

# Access results
print(result.content)            # Extracted text
print(result.format)             # Output format
print(result.has_layout)         # Has layout info
```

## Troubleshooting Quick Fixes

**"CUDA out of memory":**
```python
# Use a smaller model
config = QwenTextPyTorchConfig(
    model="Qwen/Qwen3-VL-2B-Instruct"  # Instead of 8B
)
```

**"Model download is slow":**
```python
# Models download to HuggingFace cache directory
# First run takes longer (~10 min), subsequent runs are instant
# Check progress in terminal
```

**"I don't have a GPU":**
```python
# Use API backend instead
pip install omnidocs[api]
```

Happy extracting! For more advanced usage, see the [complete documentation](../index.md).
