# API Reference Overview

Welcome to the OmniDocs API Reference! This section provides a high-level overview of the library's architecture and how its various components fit together. For detailed documentation on specific classes, functions, and modules, please refer to the dedicated sections.

## Core Concepts

OmniDocs is built around a modular and extensible design, centered on the following core concepts:

-   **Extractors**: These are the primary classes responsible for performing specific document processing tasks (e.g., OCR, table extraction, text extraction). All extractors inherit from a common `BaseExtractor` class, ensuring a consistent API.
-   **Data Models**: Standardized Pydantic models are used to represent the output of each extraction task (e.g., `OCROutput`, `TableOutput`, `TextOutput`). This ensures consistency and ease of use across different extractors.
-   **Mappers**: Helper classes that handle task-specific logic, such as language code mapping for OCR engines or coordinate transformations for layout analysis.
-   **Utilities**: A collection of helper functions for common tasks like logging, document handling (opening PDFs/images), image processing, and file validation.

## Architecture

The library is structured into the following main packages:

-   `omnidocs.tasks`: Contains sub-packages for each document AI task (e.g., `layout_analysis`, `ocr_extraction`, `table_extraction`, `text_extraction`, `math_expression_extraction`). Each task sub-package further contains its specific `extractors` and `base` classes.
-   `omnidocs.utils`: Provides general-purpose utility functions and helpers used across the library.
-   `omnidocs.models`: (If applicable) Contains definitions for deep learning models or model-related utilities.
-   `omnidocs.workflows`: (If applicable) Contains higher-level pipelines that combine multiple extractors to achieve complex document processing workflows.

## How to Use This Reference

This API reference is organized to help you quickly find the information you need:

-   **[Python API](index.md)**: Start here for a general introduction to the API, common usage patterns, and result object structures.
-   **[Core Classes](core.md)**: Dive into the foundational base classes and data models that define the common interfaces and outputs across OmniDocs.
-   **[Tasks](tasks/layout_analysis.md)**: Explore the specific extractors available for each document AI task. Each task section provides detailed documentation for its extractors, including initialization parameters, `extract` method signatures, and usage examples.
-   **[Utilities](utils.md)**: Find documentation for various helper functions related to logging, file handling, image processing, and more.

## Getting Started

If you're new to OmniDocs, we recommend starting with the [**Quick Start Guide**](../getting_started/quickstart.md) for hands-on examples and a smooth onboarding experience.

## Contributing

We welcome contributions to OmniDocs! Please refer to our [**Contributing Guide**](../../CONTRIBUTING.md) for details on how to get involved.
