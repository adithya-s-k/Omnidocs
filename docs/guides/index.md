# Guides

Task-oriented tutorials for common OmniDocs workflows.

---

## By Task

| Guide | What You'll Do | Time |
|-------|----------------|------|
| [Text Extraction](text-extraction.md) | Convert documents to Markdown/HTML | 5 min |
| [Layout Analysis](layout-analysis.md) | Detect titles, tables, figures | 5 min |
| [OCR Extraction](ocr-extraction.md) | Get text with coordinates | 5 min |
| [Cache Management](cache-management.md) | Configure model storage | 5 min |
| [Batch Processing](batch-processing.md) | Process 100+ documents | 10 min |
| [Model Cache](model-cache.md) | Share models across extractors | 5 min |
| [Modal Deployment](deployment-modal.md) | Deploy to cloud GPUs | 15 min |

---

## Quick Examples

### Text Extraction
```python
from omnidocs import Document
from omnidocs.tasks.text_extraction import QwenTextExtractor
from omnidocs.tasks.text_extraction.qwen import QwenPyTorchConfig

doc = Document.from_pdf("paper.pdf")
extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))
result = extractor.extract(doc.get_page(0), output_format="markdown")
print(result.content)
```

### Layout Analysis
```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
result = layout.extract(image)
for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
```

### OCR Extraction
```python
from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractConfig

ocr = TesseractOCR(config=TesseractConfig(languages=["eng"]))
result = ocr.extract(image)
for block in result.text_blocks:
    print(f"{block.text} @ {block.bbox}")
```

### Batch Processing
```python
from pathlib import Path

extractor = QwenTextExtractor(backend=QwenPyTorchConfig(device="cuda"))

for pdf in Path("docs/").glob("*.pdf"):
    doc = Document.from_pdf(str(pdf))
    for i, page in enumerate(doc.iter_pages()):
        result = extractor.extract(page)
        (pdf.stem / f"page_{i}.md").write_text(result.content)
```

---

## Common Workflows

### Academic Paper Processing
1. Layout: DocLayoutYOLO (0.1-0.2s/page)
2. Text: Qwen3-VL-8B (2-4s/page)
3. Result: Structure + content

### Batch Document Processing
1. Layout: DocLayoutYOLO
2. Text: DotsOCR + VLLM (multi-GPU)
3. Result: 5-10k docs/hour

### Form Field Extraction
1. Layout: Qwen Layout (custom labels)
2. OCR: Tesseract per field
3. Result: Structured form data

---

## Performance Reference

| Task | Model | Device | Speed |
|------|-------|--------|-------|
| Text Extraction | Qwen3-VL-8B | A10G | 2-3s/page |
| Text Extraction | Qwen3-VL-8B | CPU | 15-30s/page |
| Layout Detection | DocLayoutYOLO | A10G | 0.1-0.2s/page |
| OCR | Tesseract | CPU | 0.5-1s/page |

---

## Troubleshooting

**Out of memory** → Use smaller model (2B instead of 8B)

**Slow inference** → Use VLLM backend for batches

**Poor accuracy** → Try larger model or different model

**Missing elements** → Adjust confidence threshold

See individual guides for detailed troubleshooting.
