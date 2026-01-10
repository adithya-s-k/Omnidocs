# 🐍 Python API Reference

Welcome to the **OmniDocs Python API Reference**! This page provides live, auto-generated documentation for every major module, extractor, and utility in the OmniDocs ecosystem. Use this as your single source of truth for all classes, functions, and configuration options.

---

## 📦 Core Package

The main OmniDocs package provides the top-level API, configuration, and shared utilities.

::: omnidocs

---

## 🧩 Tasks & Extractors

OmniDocs organizes all document AI into modular tasks. Each task has its own extractors, which you can import and use directly. Click any section below to expand the full API for that task.

### 📐 Layout Analysis

**Detect and analyze document structure, regions, and reading order.**

::: omnidocs.tasks.layout_analysis
::: omnidocs.tasks.layout_analysis.extractors.doc_layout_yolo
::: omnidocs.tasks.layout_analysis.extractors.florence
::: omnidocs.tasks.layout_analysis.extractors.paddle
::: omnidocs.tasks.layout_analysis.extractors.rtdetr
::: omnidocs.tasks.layout_analysis.extractors.surya

---

### 📝 Text Extraction

**Extract raw and structured text from PDFs and images using classic and deep learning methods.**

::: omnidocs.tasks.text_extraction
::: omnidocs.tasks.text_extraction.extractors.pymupdf
::: omnidocs.tasks.text_extraction.extractors.pdfplumber
::: omnidocs.tasks.text_extraction.extractors.pypdf2
::: omnidocs.tasks.text_extraction.extractors.pdftext
::: omnidocs.tasks.text_extraction.extractors.surya_text
::: omnidocs.tasks.text_extraction.extractors.docling_parse

---

### 🔢 Math Expression Extraction

**Recognize and extract LaTeX math expressions from images and PDFs.**

::: omnidocs.tasks.math_expression_extraction
::: omnidocs.tasks.math_expression_extraction.extractors.donut
::: omnidocs.tasks.math_expression_extraction.extractors.nougat
::: omnidocs.tasks.math_expression_extraction.extractors.surya_math
::: omnidocs.tasks.math_expression_extraction.extractors.unimernet

---

### 🖹 OCR (Optical Character Recognition)

**Extract text from scanned documents and images using OCR models.**

<!-- ::: omnidocs.tasks.ocr -->
::: omnidocs.tasks.ocr_extraction.extractors.paddle
::: omnidocs.tasks.ocr_extraction.extractors.tesseract_ocr
::: omnidocs.tasks.ocr_extraction.extractors.easy_ocr
::: omnidocs.tasks.ocr_extraction.extractors.surya_ocr

---

### 📊 Table Extraction

**Extract tabular data from PDFs and images using classic and deep learning models.**

::: omnidocs.tasks.table_extraction
::: omnidocs.tasks.table_extraction.extractors.camelot
::: omnidocs.tasks.table_extraction.extractors.pdfplumber
::: omnidocs.tasks.table_extraction.extractors.surya_table
::: omnidocs.tasks.table_extraction.extractors.tabula
::: omnidocs.tasks.table_extraction.extractors.table_transformer
::: omnidocs.tasks.table_extraction.extractors.tableformer

---

## 🛠️ Utilities & Helpers

Common utility functions, data structures, and helpers used throughout OmniDocs.

::: omnidocs.utils

---

## 🧑‍💻 Usage Tips

- All extractors follow a consistent interface: `extractor = ...Extractor(); result = extractor.extract(input)`
- Results are returned as structured objects (e.g., `TableOutput`, `TextOutput`, etc.)
- See the [Getting Started guide](../getting_started/getting_started.md) for real-world examples.
- For advanced configuration, check each extractor’s docstring for parameters and options.

---

## 📚 More Resources

- [Project README](../../README.md)
- [Getting Started](../getting_started/getting_started.md)
- [Open an Issue](https://github.com/adithya-s-k/OmniDocs/issues)

