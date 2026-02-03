"""
TableFormer module for table structure extraction.

Provides the TableFormer-based table structure extractor.
"""

from .config import TableFormerConfig, TableFormerMode
from .pytorch import TableFormerExtractor

__all__ = [
    "TableFormerConfig",
    "TableFormerMode",
    "TableFormerExtractor",
]
