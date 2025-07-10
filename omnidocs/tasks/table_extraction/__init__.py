"""
Table extraction module for OmniDocs.

This module provides base classes and implementations for table detection and extraction
from images and documents.
"""

from .base import BaseTableExtractor, BaseTableMapper, TableOutput, Table, TableCell

__all__ = [
    'BaseTableExtractor',
    'BaseTableMapper',
    'TableOutput',
    'Table',
    'TableCell'
]
