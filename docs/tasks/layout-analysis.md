# Layout Analysis

Detect document structure: titles, tables, figures, and more.

---

## Models

| Model | Speed | Labels | Backends |
|-------|-------|--------|----------|
| **DocLayoutYOLO** | 0.1-0.2s/page | Fixed (11 types) | PyTorch |
| **Qwen Layout** | 2-3s/page | Custom | PyTorch, VLLM, MLX, API |

**Recommendation:** Use DocLayoutYOLO for speed, Qwen Layout for custom labels.

---

## Basic Usage

```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

image = Image.open("document.png")

# Initialize detector
layout = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(
        device="cuda",
        confidence=0.25,
    )
)

# Detect layout
result = layout.extract(image)

# Print detected elements
for elem in result.elements:
    print(f"{elem.label}: {elem.bbox} (conf: {elem.confidence:.2f})")
```

---

## Fixed Labels (DocLayoutYOLO)

| Label | Description |
|-------|-------------|
| `title` | Document/section headings |
| `text` | Body paragraphs |
| `list` | Bullet or numbered lists |
| `table` | Data tables |
| `figure` | Images, diagrams, charts |
| `caption` | Figure/table captions |
| `formula` | Math equations |
| `footnote` | Footnotes |
| `page_header` | Page headers |
| `page_footer` | Page footers |

---

## Filter Results

```python
# Get only tables and figures
visual_elements = [e for e in result.elements if e.label in ["table", "figure"]]

# Get high-confidence detections
confident = [e for e in result.elements if e.confidence >= 0.8]

# Exclude headers/footers
content = [e for e in result.elements if e.label not in ["page_header", "page_footer"]]
```

---

## Custom Labels (Qwen Layout)

Use Qwen for domain-specific labels.

```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector
from omnidocs.tasks.layout_analysis.qwen import QwenLayoutPyTorchConfig
from omnidocs.tasks.layout_analysis import CustomLabel

# Define custom labels
custom_labels = [
    CustomLabel(name="code_block", description="Code snippets"),
    CustomLabel(name="sidebar", description="Sidebar content"),
]

config = QwenLayoutPyTorchConfig(device="cuda")
detector = QwenLayoutDetector(backend=config)

result = detector.extract(image, custom_labels=custom_labels)

for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
```

---

## Process PDF

```python
from omnidocs import Document
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig

doc = Document.from_pdf("document.pdf")
layout = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))

for page_idx in range(doc.page_count):
    page_image = doc.get_page(page_idx)
    result = layout.extract(page_image)
    print(f"Page {page_idx + 1}: {len(result.elements)} elements")
```

---

## Troubleshooting

**Missing elements**
```python
# Lower confidence threshold
config = DocLayoutYOLOConfig(device="cuda", confidence=0.15)
```

**Too many false detections**
```python
# Increase confidence threshold
config = DocLayoutYOLOConfig(device="cuda", confidence=0.5)
```

**Need custom labels**
```python
# Switch to Qwen Layout
from omnidocs.tasks.layout_analysis import QwenLayoutDetector
```
