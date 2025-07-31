# Getting Started with OmniDocs

Welcome to OmniDocs! This guide will help you get up and running with powerful document AI extraction in just a few steps.

---

## 🚀 What is OmniDocs?
OmniDocs is a unified Python library for extracting tables, text, math, and OCR data from PDFs and images using state-of-the-art models and classic tools—all with a simple, consistent API.

---

## 🛠️ Installation

Choose your preferred method:

- **PyPI (Recommended):**
  ```bash
  pip install omnidocs
  ```
- **uv pip (Fastest):**
  ```bash
  uv pip install omnidocs
  ```
- **From Source:**
  ```bash
  git clone https://github.com/adithya-s-k/OmniDocs.git
  cd OmniDocs
  pip install . 
  or 
  uv sync 
  ```
- **Conda (if available):**
  ```bash
  conda install -c conda-forge omnidocs
  ```

---

## 🏗️ Setting Up Your Environment

It's best to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## 📄 Quick Example

Extract tables from a PDF in just a few lines:
```python
from omnidocs.tasks.table_extraction.extractors.camelot import CamelotExtractor
extractor = CamelotExtractor()
results = extractor.extract("sample.pdf")
print(results.tables[0].df)  # Print first table as DataFrame
```

---

## 📚 Explore Tutorials
- [Table Extraction](../tasks/table_extraction/overview.md)
- [Text Extraction](../tasks/text_extraction/overview.md)
- [Math Extraction](../tasks/math_extraction/overview.md)
- [OCR Extraction](../tasks/ocr/overview.md)

---

## 🧑‍💻 Need Help?
- See the [API Reference](../api_reference/python_api.md)
- Open an issue on [GitHub](https://github.com/adithya-s-k/OmniDocs/issues)

---

Happy Document AI-ing! 🎉
