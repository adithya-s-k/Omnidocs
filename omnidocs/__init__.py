"""
OmniDocs - Unified Python toolkit for visual document processing.

Core exports:
- Document: Stateless document container for loading PDFs and images
"""

from omnidocs.document import (
    Document,
    DocumentLoadError,
    DocumentMetadata,
    PageRangeError,
    UnsupportedFormatError,
    URLDownloadError,
)

from ._version import __version__

__all__ = [
    "__version__",
    "Document",
    "DocumentMetadata",
    "DocumentLoadError",
    "URLDownloadError",
    "PageRangeError",
    "UnsupportedFormatError",
]
