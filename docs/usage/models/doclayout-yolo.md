# DocLayoutYOLO

Fast document layout detection.

---

## Overview

| | |
|---|---|
| **Tasks** | Layout Analysis |
| **Backends** | PyTorch |
| **Speed** | 0.1-0.2s/page |
| **Quality** | Good |
| **VRAM** | 2-4GB |

---

## Why DocLayoutYOLO

- **Extremely fast** - 5-10x faster than VLM-based detection
- **Low memory** - Runs on modest GPUs or CPU
- **Reliable** - YOLO architecture, battle-tested
- **Fixed labels** - 11 pre-trained categories

---

## Basic Usage

```python
from omnidocs.tasks.layout_analysis import DocLayoutYOLO, DocLayoutYOLOConfig
from PIL import Image

image = Image.open("document.png")

detector = DocLayoutYOLO(
    config=DocLayoutYOLOConfig(device="cuda")
)

result = detector.extract(image)

for elem in result.elements:
    print(f"{elem.label}: {elem.bbox} ({elem.confidence:.2f})")
```

---

## Configuration

```python
config = DocLayoutYOLOConfig(
    device="cuda",        # "cuda", "cpu"
    confidence=0.25,      # Detection threshold (0.0-1.0)
    img_size=1024,        # Input image size
)
```

---

## Detected Labels

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
| `unknown` | Unclassified elements |

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

## When to Use DocLayoutYOLO vs Qwen Layout

| Use Case | Model |
|----------|-------|
| Speed-critical | DocLayoutYOLO |
| Custom labels needed | Qwen Layout |
| Limited GPU memory | DocLayoutYOLO |
| Higher accuracy | Qwen Layout |
| Batch processing | DocLayoutYOLO |

---

## Troubleshooting

**Missing elements**
```python
# Lower confidence threshold
config = DocLayoutYOLOConfig(confidence=0.15)
```

**Too many false detections**
```python
# Increase confidence threshold
config = DocLayoutYOLOConfig(confidence=0.5)
```

**Slow on CPU**
```python
# Expected: ~1-2s/page on CPU vs 0.1-0.2s on GPU
# Consider using GPU if available
```
