"""
OCR extraction module for OmniDocs.

This module provides base classes and implementations for optical character recognition (OCR)
text extraction from images and documents.
"""

from .base import BaseOCRExtractor, BaseOCRMapper, OCROutput, OCRText

__all__ = [
    'BaseOCRExtractor',
    'BaseOCRMapper', 
    'OCROutput',
    'OCRText'
]