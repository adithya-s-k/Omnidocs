# Text Extraction

Convert document images to readable, formatted text.

---

## Input / Output

**Input:** Document image (PNG, JPG) or PDF page

**Output:** Formatted text (Markdown or HTML)

```python
result = extractor.extract(image, output_format="markdown")
print(result.content)
```

```markdown
# Document Title

This is the first paragraph with **bold** and *italic* text.

## Section 1

- Bullet point 1
- Bullet point 2

| Column A | Column B |
|----------|----------|
| Data 1   | Data 2   |
```

---

## Quick Start

```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Load document
doc = Document.from_pdf("document.pdf")

# Initialize extractor
extractor = QwenTextExtractor(
    backend=QwenPyTorchConfig(device="cuda")
)

# Extract
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

---

## Output Formats

```python
# Markdown (default)
result = extractor.extract(image, output_format="markdown")

# HTML
result = extractor.extract(image, output_format="html")
```

---

## Available Models

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| [Qwen](../models/qwen.md) | 2-3s/page | Excellent | General purpose |
| [DotsOCR](../models/dotsocr.md) | 3-5s/page | Very Good | Technical docs, layout-aware |

---

## When to Use

✅ Converting PDFs to Markdown
✅ Extracting article content
✅ Document parsing for RAG pipelines
✅ Content migration

❌ Need word coordinates → Use [OCR](ocr.md)
❌ Need structure only → Use [Layout Analysis](layout-analysis.md)
