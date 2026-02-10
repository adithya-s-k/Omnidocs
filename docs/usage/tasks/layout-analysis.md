# Layout Analysis

Detect document structure and element boundaries.

---

## Input / Output

**Input:** Document image

**Output:** List of bounding boxes with labels and confidence scores

```python
result = detector.extract(image)
for elem in result.elements:
    print(f"{elem.label}: {elem.bbox} ({elem.confidence:.2f})")
```

```
title: [50, 20, 500, 60] (0.98)
text: [50, 80, 900, 300] (0.95)
table: [50, 320, 900, 600] (0.92)
figure: [50, 620, 400, 900] (0.89)
```

---

## Quick Start

```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

image = Image.open("document.png")

detector = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(device="cuda")
)

result = detector.extract(image)

for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
```

---

## Available Models

| Model | Speed | Labels | Best For |
|-------|-------|--------|----------|
| [DocLayoutYOLO](../models/doclayout-yolo.md) | 0.1-0.2s/page | Fixed (11) | Speed |
| [RT-DETR](../models/rtdetr.md) | 0.3-0.5s/page | Fixed (11) | Accuracy |
| [Qwen Layout](../models/qwen.md#layout-analysis) | 2-3s/page | **Custom** | Flexibility |
| [VLM API](../models/vlm-api.md) | Varies | **Custom** | No GPU, any cloud provider |

---

## Fixed Labels

Models like DocLayoutYOLO and RT-DETR detect these predefined labels:

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
| `page_header` | Running headers |
| `page_footer` | Running footers |

---

## Custom Labels (Qwen Layout)

Qwen Layout can detect **any custom elements** you define.

### Simple String Labels

```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector
from omnidocs.tasks.layout_analysis.qwen import QwenLayoutPyTorchConfig

detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(device="cuda")
)

# Detect custom elements
result = detector.extract(
    image,
    custom_labels=["code_block", "sidebar", "pull_quote", "diagram"]
)

for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
```

### Structured Labels with Metadata

For advanced use cases, use `CustomLabel` with descriptions:

```python
from omnidocs.tasks.layout_analysis import QwenLayoutDetector, CustomLabel
from omnidocs.tasks.layout_analysis.qwen import QwenLayoutPyTorchConfig

detector = QwenLayoutDetector(
    backend=QwenLayoutPyTorchConfig(device="cuda")
)

# Structured labels with metadata
result = detector.extract(
    image,
    custom_labels=[
        CustomLabel(
            name="code_block",
            description="Programming source code areas",
            detection_prompt="Regions with monospace text and syntax highlighting",
            color="#2ecc71",
        ),
        CustomLabel(
            name="sidebar",
            description="Sidebar or callout content",
            detection_prompt="Boxed regions with supplementary information",
            color="#3498db",
        ),
        CustomLabel(
            name="warning_box",
            description="Warning or alert boxes",
            detection_prompt="Highlighted boxes with warning icons or red/yellow colors",
            color="#e74c3c",
        ),
    ]
)

for elem in result.elements:
    print(f"{elem.label}: {elem.bbox}")
```

### Reusable Label Sets

Create reusable label collections for your domain:

```python
from omnidocs.tasks.layout_analysis import CustomLabel

class TechnicalDocLabels:
    """Labels for technical documentation."""

    CODE_BLOCK = CustomLabel(
        name="code_block",
        description="Source code listings",
        color="#2ecc71"
    )

    API_REFERENCE = CustomLabel(
        name="api_reference",
        description="API documentation tables",
        color="#3498db"
    )

    DIAGRAM = CustomLabel(
        name="diagram",
        description="Architecture diagrams",
        color="#9b59b6"
    )

    @classmethod
    def all(cls):
        return [cls.CODE_BLOCK, cls.API_REFERENCE, cls.DIAGRAM]

# Use across projects
result = detector.extract(image, custom_labels=TechnicalDocLabels.all())
```

---

## Fixed vs Custom Labels

| Feature | Fixed (YOLO, RT-DETR) | Custom (Qwen) |
|---------|----------------------|---------------|
| **Speed** | 0.1-0.5s/page | 2-3s/page |
| **Labels** | 11 predefined | Unlimited custom |
| **Accuracy** | High on standard docs | Good on any doc |
| **Use case** | Standard documents | Domain-specific |

**Choose Fixed Labels when:**
- Processing standard documents
- Speed is critical
- Standard elements are sufficient

**Choose Custom Labels when:**
- Need domain-specific elements (code, sidebars, etc.)
- Processing non-standard documents
- Flexibility is more important than speed

---

## Filtering Results

```python
# By label
tables = [e for e in result.elements if e.label == "table"]
figures = [e for e in result.elements if e.label == "figure"]

# By confidence
confident = [e for e in result.elements if e.confidence >= 0.8]

# Exclude headers/footers
content = [e for e in result.elements
           if e.label not in ["page_header", "page_footer"]]
```

---

## When to Use

✅ Document structure analysis
✅ Finding tables and figures
✅ Building multi-stage pipelines
✅ Filtering unwanted elements
✅ Domain-specific element detection (custom labels)

❌ Need readable text → Use [Text Extraction](text-extraction.md)
❌ Need word positions → Use [OCR](ocr.md)

---

## Upcoming Models

| Model | Description | Status |
|-------|-------------|--------|
| **SuryaLayout** | Modern layout detection | 🔜 Soon |
