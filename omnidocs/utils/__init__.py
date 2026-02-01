"""
OmniDocs Utilities.

Provides utility functions for result aggregation, visualization, and export.
"""

from .aggregation import (
    BatchResult,
    DocumentResult,
    merge_text_results,
)

__all__ = [
    "DocumentResult",
    "BatchResult",
    "merge_text_results",
]
