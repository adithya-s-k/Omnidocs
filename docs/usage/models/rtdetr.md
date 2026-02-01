# RT-DETR

High-accuracy document layout detection.

---

## Overview

| | |
|---|---|
| **Tasks** | Layout Analysis |
| **Backends** | PyTorch |
| **Speed** | 0.3-0.5s/page |
| **Quality** | Excellent |
| **VRAM** | 4-6GB |

---

## Why RT-DETR

- **Higher accuracy** than YOLO-based detectors
- **Better on small elements** - catches details YOLO misses
- **Transformer architecture** - modern, effective
- **Same labels** as DocLayoutYOLO - drop-in replacement

---

## Basic Usage

```python
from omnidocs.tasks.layout_analysis import RTDETRLayoutDetector, RTDETRConfig
from PIL import Image

image = Image.open("document.png")

detector = RTDETRLayoutDetector(
    config=RTDETRConfig(device="cuda")
)

result = detector.extract(image)

for elem in result.elements:
    print(f"{elem.label}: {elem.bbox} ({elem.confidence:.2f})")
```

---

## Configuration

```python
config = RTDETRConfig(
    device="cuda",        # "cuda" or "cpu"
    confidence=0.3,       # Detection threshold
)
```

---

## RT-DETR vs DocLayoutYOLO

| | RT-DETR | DocLayoutYOLO |
|---|---------|---------------|
| **Speed** | 0.3-0.5s/page | 0.1-0.2s/page |
| **Accuracy** | Higher | Good |
| **Small elements** | Better | May miss |
| **Memory** | 4-6GB | 2-4GB |
| **Use case** | Accuracy-critical | Speed-critical |

---

## When to Use

✅ Need highest accuracy
✅ Documents with small elements
✅ Quality over speed

❌ Speed-critical → Use [DocLayoutYOLO](doclayout-yolo.md)
❌ Custom labels → Use [Qwen Layout](qwen.md#layout-analysis)
