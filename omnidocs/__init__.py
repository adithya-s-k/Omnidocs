"""
OmniDocs - Unified Python toolkit for visual document processing.

Core exports:
- Document: Stateless document container for loading PDFs and images
"""

from ._version import __version__
from omnidocs.document import (
    Document,
    DocumentMetadata,
    DocumentLoadError,
    URLDownloadError,
    PageRangeError,
    UnsupportedFormatError,
)

__all__ = [
    "__version__",
    "Document",
    "DocumentMetadata",
    "DocumentLoadError",
    "URLDownloadError",
    "PageRangeError",
    "UnsupportedFormatError",
]
