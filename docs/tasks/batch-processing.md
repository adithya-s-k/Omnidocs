# Batch Processing

Process multiple documents efficiently.

---

## Quick Start

```python
from pathlib import Path
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

# Initialize once
extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))

# Process all PDFs
for pdf_path in Path("documents/").glob("*.pdf"):
    doc = Document.from_pdf(pdf_path)

    for i, page in enumerate(doc.iter_pages()):
        result = extractor.extract(page, output_format="markdown")

        output_path = Path("output") / f"{pdf_path.stem}_page_{i+1}.md"
        output_path.write_text(result.content)
```

---

## With Progress Tracking

```python
from pathlib import Path
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig
import time

pdf_files = list(Path("documents/").glob("*.pdf"))
extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))

start = time.time()

for idx, pdf_path in enumerate(pdf_files, 1):
    doc = Document.from_pdf(pdf_path)

    for page in doc.iter_pages():
        result = extractor.extract(page, output_format="markdown")

    elapsed = time.time() - start
    remaining = (len(pdf_files) - idx) * (elapsed / idx)
    print(f"[{idx}/{len(pdf_files)}] {pdf_path.name} - ETA: {remaining/60:.1f}min")
```

---

## Memory Management

For large batches, clear cache periodically:

```python
for i, page in enumerate(doc.iter_pages()):
    result = extractor.extract(page)
    save_result(result)

    # Free memory every 10 pages
    if i % 10 == 0:
        doc.clear_cache()
```

---

## Error Handling

```python
results = []
errors = []

for pdf_path in pdf_files:
    try:
        doc = Document.from_pdf(pdf_path)
        result = extractor.extract(doc.get_page(0))
        results.append({"path": str(pdf_path), "success": True})
    except Exception as e:
        errors.append({"path": str(pdf_path), "error": str(e)})

print(f"Succeeded: {len(results)}, Failed: {len(errors)}")
```

---

## Stream Results to Disk

Don't accumulate results in memory:

```python
import json

with open("results.jsonl", "w") as f:
    for pdf_path in pdf_files:
        doc = Document.from_pdf(pdf_path)
        result = extractor.extract(doc.get_page(0))

        record = {
            "path": str(pdf_path),
            "word_count": result.word_count,
        }
        f.write(json.dumps(record) + "\n")
```

---

## Performance Tips

| Tip | Impact |
|-----|--------|
| Initialize extractor once | Saves 2-3s per batch |
| Use VLLM for large batches | 2-4x throughput |
| Clear cache periodically | Prevents OOM |
| Stream results to disk | Constant memory |

---

## For Cloud Scale

See [Deployment](deployment.md) for processing on Modal GPUs.
