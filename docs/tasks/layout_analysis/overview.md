# Layout Analysis in OmniDocs

Layout analysis is the process of detecting and classifying regions (text, tables, images, etc.) in documents or images. OmniDocs provides a unified interface to several state-of-the-art layout detection backends, making it easy to experiment, compare, and integrate them into your workflows.

## Layout Analysis?

Layout analysis breaks a document page into its logical components—like paragraphs, tables, figures, and headers—by predicting bounding boxes and labels for each region. This is a crucial first step for downstream tasks like OCR, table extraction, and document understanding.

## 🧩 Supported Layout Detectors

OmniDocs supports multiple layout detection engines, each with its own strengths:

| Detector   | Backend/Model         | Highlights                                  |
|------------|----------------------|---------------------------------------------|
| **Paddle** | PaddleOCR Layout     | Fast, robust, easy to use, good for scanned docs |
| **RTDETR** | RT-DETR              | Real-time, transformer-based, accurate      |
| **Surya**  | Surya Layout         | Modern, high-accuracy, Indian docs friendly |
| **YOLO**   | YOLOv8/YOLOv5        | Fast, customizable, works on many layouts   |
| **Florence** | Florence Layout    | (If available) Large foundation model, generalizes well |

> **Tip:** You can easily switch between detectors by changing a single import/class name.

## 📝 How to Use

All layout detectors follow the same API pattern:

```python
from omnidocs.tasks.layout_analysis.extractors.paddle import PaddleLayoutDetector

detector = PaddleLayoutDetector(device='cpu')
image_path = "path/to/your/document.png"
annotated_image, layout_output = detector.detect(image_path)

# Visualize results
detector.visualize((annotated_image, layout_output), "output.png")
```

- **Change `PaddleLayoutDetector` to any other detector** (e.g., `RTDETRLayoutDetector`, `SuryaLayoutDetector`, `YOLOLayoutDetector`) to use a different backend.
- All detectors return both the annotated image and a structured output with bounding boxes and labels.

## 📒 Example Notebooks

See the [tutorial notebooks](./tutorials/) for hands-on examples:
- [Paddle Layout Analysis](./tutorials/paddle.ipynb)
- [YOLO Layout Analysis](./tutorials/yolo.ipynb)
- [RTDETR Layout Analysis](./tutorials/rtdetr.ipynb)
- [Surya Layout Analysis](./tutorials/surya.ipynb)

Each notebook demonstrates:
- How to initialize and use the detector
- How to visualize results
- How to interpret the output

## 🛠️ Advanced Tips

- **Device Selection:** Most detectors support `device='cpu'` or `device='cuda'` for GPU acceleration.
- **Custom Models:** For YOLO and Surya, you can plug in your own trained weights.
- **Batch Processing:** Use Python loops or scripts to process folders of images.
- **Output Structure:** All detectors return bounding boxes, class labels, and (optionally) confidence scores.

## 🔗 Next Steps

- Try out the [notebooks](./tutorials/) with your own documents.
- Read the [API Reference](../../api_reference/) for advanced usage and customization.
- Explore downstream tasks like OCR and table extraction using the detected layouts.

---

OmniDocs makes layout analysis accessible, reproducible, and extensible, no matter which backend you choose. 