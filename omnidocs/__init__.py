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
from omnidocs.utils.cache import (
    configure_backend_cache,
    get_cache_info,
    get_model_cache_dir,
)

from ._version import __version__

# Configure backend cache directories on import
try:
    configure_backend_cache()
except (OSError, PermissionError) as e:
    import warnings
    warnings.warn(
        f"Failed to configure model cache directory: {e}. "
        "Set OMNIDOCS_MODEL_CACHE to a writable path.",
        stacklevel=1,
    )

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
    # Cache management
    "get_model_cache_dir",
    "configure_backend_cache",
    "get_cache_info",
]
