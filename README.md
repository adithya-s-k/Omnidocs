
# 🚀 OmniDocs

![OmniDocs Banner](./assets/omnidocs_banner.png)

<p align="center">
  <b>Unified, modern, and blazing-fast Document AI for Python</b><br>
  <a href="https://github.com/adithya-s-k/OmniDocs/actions"><img src="https://img.shields.io/github/actions/workflow/status/adithya-s-k/OmniDocs/ci.yml?branch=main" alt="CI Status"></a>
  <a href="https://pypi.org/project/omnidocs/"><img src="https://img.shields.io/pypi/v/omnidocs.svg" alt="PyPI version"></a>
  <a href="https://github.com/adithya-s-k/OmniDocs/blob/main/LICENSE.md"><img src="https://img.shields.io/github/license/adithya-s-k/OmniDocs" alt="License"></a>
</p>

**OmniDocs** is your all in one Python toolkit for extracting tables, text, math, and OCR from PDFs and image, powered by classic libraries and state of the art deep learning models. Build robust document workflows with a single, consistent API.

- 🧩 Unified, production-ready API for all tasks
- 🏎️ Fast, GPU-accelerated, and easy to extend

---

## ⚡ Quick Start

Get started quickly with practical examples for various document processing tasks in the [**Quick Start Guide**](docs/getting_started/quickstart.md).



## 🏁 Get Started

- See the [Getting Started Guide](docs/getting_started/quickstart.md)
- Dive into the [API Reference](docs/api_reference/index.md)

## 📖 Tutorials

- **Table Extraction:** [Overview](docs/tasks/table_extraction/overview.md)
  - [Camelot](docs/tasks/table_extraction/tutorials/camelot.ipynb)
  - [PDFPlumber](docs/tasks/table_extraction/tutorials/pdfplumber.ipynb)
  - [Surya Table](docs/tasks/table_extraction/tutorials/surya_table.ipynb)
  - [Tabula](docs/tasks/table_extraction/tutorials/tabula.ipynb)
  - [TableTransformer](docs/tasks/table_extraction/tutorials/tabletransformer.ipynb)
  - [TableFormer](docs/tasks/table_extraction/tutorials/tableformer.ipynb)
- **Text Extraction:** [Overview](docs/tasks/text_extraction/overview.md)
  - [PyMuPDF](docs/tasks/text_extraction/tutorials/text_extractors.ipynb)
  - [PDFPlumber](docs/tasks/text_extraction/tutorials/text_extractors.ipynb)
  - [PyPDF2](docs/tasks/text_extraction/tutorials/text_extractors.ipynb)
  - [PDFText](docs/tasks/text_extraction/tutorials/text_extractors.ipynb)
  - [Surya Text](docs/tasks/text_extraction/tutorials/text_extractors.ipynb)
  - [Docling Parse](docs/tasks/text_extraction/tutorials/text_extractors.ipynb)
- **Math Extraction:** [Overview](docs/tasks/math_extraction/overview.md)
  - [UniMERNet](docs/tasks/math_extraction/tutorials/unimernet.ipynb)
  - [SuryaMath](docs/tasks/math_extraction/tutorials/suryamath.ipynb)
  - [Nougat](docs/tasks/math_extraction/tutorials/nougat.ipynb)
  - [Donut](docs/tasks/math_extraction/tutorials/donut.ipynb)
- **OCR Extraction:** [Overview](docs/tasks/ocr/overview.md)
  - [PaddleOCR](docs/tasks/ocr/tutorials/paddle.ipynb)
  - [Tesseract](docs/tasks/ocr/tutorials/tesseract.ipynb)
  - [EasyOCR](docs/tasks/ocr/tutorials/easyocr.ipynb)
  - [SuryaOCR](docs/tasks/ocr/tutorials/suryaocr.ipynb)

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

## 🏗️ How It Works

**OmniDocs** organizes document processing tasks into modular components. Each component corresponds to a specific task and offers:

1. **A Unified Interface:** Consistent input and output formats.
2. **Model Independence:** Switch between libraries or models effortlessly.
3. **Pipeline Flexibility:** Combine components to create custom workflows.

## 📈 Roadmap

- Add support for semantic understanding tasks (e.g., entity extraction).
- Integrate pre-trained transformer models for context-aware document analysis.
- Expand pipelines for multilingual document processing.
- Add CLI support for batch processing.

## 🤝 Contributing

We welcome contributions to **OmniDocs**! Here's how you can help:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and open a pull request.

For more details, refer to our [CONTRIBUTING.md](CONTRIBUTING.md).

## 🛡️ License

This project is licensed under multiple licenses, depending on the models and libraries you use in your pipeline. Please refer to the individual licenses of each component for specific terms and conditions.

## 🌟 Support the Project

If you find **OmniDocs** helpful, please give us a ⭐ on GitHub and share it with others in the community.

## 🗨️ Join the Community

For discussions, questions, or feedback:

- **Issues:** Report bugs or suggest features [here](https://github.com/adithya-s-k/OmniDocs/issues).
- **Email:** Reach out at adithyaskolavi@gmail.com, laxmansrivastacc@gmail.com
