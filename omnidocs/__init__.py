"""
OmniDocs - Unified Python toolkit for visual document processing.

Core exports:
- Document: Stateless document container for loading PDFs and images
- DocumentBatch: Batch document loader for multi-document processing
- Batch processing utilities
- Model cache for sharing models across extractors
"""

from omnidocs.batch import (
    DocumentBatch,
    process_directory,
    process_document,
)
from omnidocs.cache import (
    add_reference,
    clear_cache,
    get_cache_config,
    get_cache_info,
    get_cache_key,
    get_cached,
    get_or_load,
    list_cached_keys,
    remove_cached,
    set_cache_config,
    set_cached,
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
    # Model cache
    "get_cache_key",
    "get_cached",
    "set_cached",
    "get_or_load",
    "add_reference",
    "remove_cached",
    "clear_cache",
    "get_cache_info",
    "list_cached_keys",
    "set_cache_config",
    "get_cache_config",
]
