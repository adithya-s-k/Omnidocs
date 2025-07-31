
# Optical Character Recognition (OCR) in OmniDocs

OmniDocs provides a unified, production-ready interface for extracting text from images and documents using multiple OCR engines. Whether you need fast, lightweight extraction or advanced multilingual support, OmniDocs makes it easy to switch between backends and integrate OCR into your document workflows.

## 🚀 Key Features

- **Multiple OCR Engines:** Seamlessly switch between PaddleOCR, Tesseract, EasyOCR, and Surya OCR.
- **Unified API:** Consistent input/output formats across all engines.
- **Multilingual Support:** Extract text in dozens of languages, with automatic language mapping.
- **Bounding Boxes & Layout:** Get word/line bounding boxes, reading order, and more.
- **Visualization:** Easily visualize OCR results on images.
- **Batch Processing:** Process single files or entire folders with the same interface.

## 🧩 Supported OCR Engines

| Engine      | Source & Docs | License | CPU | GPU | Highlights |
|-------------|--------------|---------|-----|-----|------------|
| **PaddleOCR** | [GitHub](https://github.com/PaddlePaddle/PaddleOCR) | Apache 2.0 | ✅ | ✅ | Fast, accurate, layout-aware, 90+ languages |
| **Tesseract** | [GitHub](https://github.com/tesseract-ocr/tesseract) | BSD-3-Clause | ✅ | ✅ | Classic, robust, many languages |
| **EasyOCR** | [GitHub](https://github.com/JaidedAI/EasyOCR) | MIT | ✅ | ✅ | PyTorch-based, easy to use, many languages |
| **Surya OCR** | [GitHub](https://github.com/VikParuchuri/surya) | GPL-3.0-or-later | ✅ | ✅ | Modern, high-accuracy, Indian languages |

## 📝 Quick Example

```python
from omnidocs.tasks.ocr_extraction import EasyOCRExtractor

extractor = EasyOCRExtractor(languages=["en"], device="cpu")
result = extractor.extract("path/to/image.png")
print(result.full_text)
```

You can swap `EasyOCRExtractor` for `TesseractOCRExtractor`, `PaddleOCRExtractor`, or `SuryaOCRExtractor` with no code changes.

## 🎨 Visualization

OmniDocs can visualize OCR results with bounding boxes and recognized text:

```python
extractor.visualize(result, "path/to/image.png", output_path="ocr_vis.png", show_text=True)
```

## 📚 Advanced Usage

- **Language Mapping:** Standardizes language codes across engines.
- **Batch Extraction:** Use `extract_all` for folders or lists of images.
- **Custom Preprocessing:** Override or extend input preprocessing as needed.

## 📖 Tutorials & Further Reading

- [EasyOCR Tutorial](../tutorials/easyocr.ipynb)
- [Tesseract Tutorial](../tutorials/tesseract.ipynb)
- [PaddleOCR Tutorial](../tutorials/paddle.ipynb)
- [Surya OCR Tutorial](../tutorials/surya.ipynb)
- [Visual Comparison OCR Test Notebook](../../getting_started/ocr_test.ipynb)
- [API Reference](../../api_reference/overview.md)

---
For more, see the [README](../../../README.md) and the main [OmniDocs documentation](../../index.md).
