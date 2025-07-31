# OmniDocs: Your Complete Toolkit for Intelligent Document Understanding

![OmniDocs Banner](https://raw.githubusercontent.com/adithya-s-k/Omnidocs/refs/heads/main/assets/omnidocs_banner.png)

<p align="left">
  <a href="https://pypi.org/project/omnidocs/"><img src="https://img.shields.io/pypi/v/omnidocs.svg?color=blue" alt="PyPI version"></a>
  <a href="https://github.com/adithya-s-k/Omnidocs/actions"><img src="https://img.shields.io/github/actions/workflow/status/adithya-s-k/Omnidocs/ci.yml?branch=main" alt="Build Status"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/github/license/adithya-s-k/Omnidocs" alt="License"></a>
</p>

**OmniDocs** is a unified Python framework for intelligent document understanding. Extract text, tables, layout, and more from PDFs and images using state-of-the-art models—all with a single, powerful API.

---

## ✨ Key Features

- **Unified API:** One simple interface for every document AI task.
- **Layout Detection:** YOLO, Surya, PaddleOCR, and more.
- **OCR Extraction:** PaddleOCR, Tesseract, EasyOCR, Surya OCR.
- **Text Extraction:**
  PyPDF2, PyMuPDF, pdfplumber, docling_parse, pdftext, Surya
- **Math Expression Extraction:**
  - **Donut:** NAVER CLOVA Donut model for math/LaTeX extraction.
  - **Nougat:** Facebook's Nougat model for LaTeX from academic documents.
  - **Surya Math:** Surya-based mathematical expression extraction.
  - **UniMERNet:** Universal Mathematical Expression Recognition Network.
- **Table Extraction:** Camelot, Tabula, PDFPlumber, Table Transformer, TableFormer, Surya Table
- **Reading Order & Structure:** Advanced document parsing that just works.
- **Multilingual:** Supports 90+ languages out of the box.
- **Extensible:** Easily plug in your own models and build custom workflows.

---

## ⚡ Installation

Install OmniDocs from PyPI with a single command:

```bash
pip install omnidocs
````

For full setup (GPU, `conda`, `poetry`, etc.), check the [**Installation Guide**](./getting_started/installation.md).

---

## 🚀 Quick Start

See the [**Quick Start Guide**](./getting_started/quickstart.md) for a comprehensive introduction to using OmniDocs.

---


## 📚 The Arsenal: All Supported Backends

<details>
<summary><strong>Layout Analysis Models (Click to Expand)</strong></summary>

* DocLayout YOLO
* PPStructure (Paddle OCR)
* RT DETR (Docling)
* Florence-2-DocLayNet
* Surya Layout

</details>

<details>
<summary><strong>Text Extraction Libraries (Click to Expand)</strong></summary>

* **PyPDF2** — Pure Python PDF text extraction, supports encrypted PDFs and form fields.
* **PyMuPDF** — Fast, multi-format text extraction with layout and font info.
* **pdfplumber** — PDF text and table extraction with layout details.
* **docling_parse** — Unified parsing for PDF, DOCX, PPTX, HTML, MD, with OCR and structure.
* **pdftext** — Simple, fast PDF text extraction with layout options.
* **surya_text** — Surya-based text extraction for images and documents.

</details>

<details>
<summary><strong>OCR Models (Click to Expand)</strong></summary>

* Paddle OCR
* Tesseract
* EasyOCR
* Surya OCR

</details>


<details>
<summary><strong>Math Expression Extraction Models (Click to Expand)</strong></summary>

* **Donut** — NAVER CLOVA Donut model for math/LaTeX extraction.
* **Nougat** — Facebook's Nougat model for LaTeX from academic documents.
* **Surya Math** — Surya-based mathematical expression extraction.
* **UniMERNet** — Universal Mathematical Expression Recognition Network.

</details>

<details>
<summary><strong>Table Extraction Models (Click to Expand)</strong></summary>

* PPStructure (Paddle OCR)
* Camelot
* Tabula
* PDFPlumber
* Table Transformer
* TableFormer
* Surya Table

</details>

---


## 🗺️ Learn More

* **[Tutorials](./tasks/):** Hands-on notebooks and guides for every task.
* **[API Reference](./api_reference/):** The full dictionary of all public methods and classes.

## 🤝 Contributing

Contributions are welcome! If you want to help make OmniDocs even better, see our [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide.

---

## 🛡️ License

The **OmniDocs** framework is MIT licensed. The underlying models and libraries may have their own licenses—please verify before use in production.

---

## 🌟 Support the Project

If you find **OmniDocs** helpful, please ⭐ the repo on [GitHub](https://github.com/adithya-s-k/Omnidocs)!

---

## 🗨️ Join the Community

* **Issues:** Report bugs or suggest features [here](https://github.com/adithya-s-k/OmniDocs/issues)
* **Email:** [adithyaskolavi@gmail.com](mailto:adithyaskolavi@gmail.com) or [laxmansrivastacc@gmail.com](mailto:laxmansrivastacc@gmail.com)
