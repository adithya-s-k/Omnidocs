"""
OmniDocs - Unified Python toolkit for visual document processing.

Core exports:
- Document: Stateless document container for loading PDFs and images
- DocumentBatch: Batch document loader for multi-document processing
- Batch processing utilities
"""

from omnidocs.batch import (
    DocumentBatch,
    process_directory,
    process_document,
)
from omnidocs.document import (
    Document,
    DocumentLoadError,
    DocumentMetadata,
    PageRangeError,
    UnsupportedFormatError,
    URLDownloadError,
)
from omnidocs.utils.aggregation import (
    BatchResult,
    DocumentResult,
    merge_text_results,
)

from ._version import __version__

__all__ = [
    "__version__",
    # Document loading
    "Document",
    "DocumentMetadata",
    "DocumentLoadError",
    "URLDownloadError",
    "PageRangeError",
    "UnsupportedFormatError",
    # Batch processing
    "DocumentBatch",
    "process_directory",
    "process_document",
    # Result aggregation
    "BatchResult",
    "DocumentResult",
    "merge_text_results",
]
