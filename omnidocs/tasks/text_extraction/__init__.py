"""
Text extraction module for OmniDocs.

This module provides base classes and implementations for text extraction
from documents (PDFs, images, etc.).
"""

from .base import BaseTextExtractor, BaseTextMapper, TextOutput, TextBlock

__all__ = [
    'BaseTextExtractor',
    'BaseTextMapper',
    'TextOutput',
    'TextBlock'
]
