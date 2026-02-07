"""
OmniDocs Utilities.

Provides utility functions for result aggregation, visualization, export, and cache management.
"""

from .aggregation import (
    BatchResult,
    DocumentResult,
    merge_text_results,
)
from .cache import (
    configure_backend_cache,
    get_model_cache_dir,
    get_storage_info,
)

__all__ = [
    "DocumentResult",
    "BatchResult",
    "merge_text_results",
    "get_model_cache_dir",
    "configure_backend_cache",
    "get_storage_info",
]
