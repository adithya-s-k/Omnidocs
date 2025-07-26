"""
Text Extractors module for OmniDocs.

This module provides various text extractor implementations.
"""

from .pymupdf import PyMuPDFTextExtractor
from .pypdf2 import PyPDF2TextExtractor
from .pdftext import PdftextTextExtractor
from .docling_parse import DoclingTextExtractor

from .pdfplumber import PdfplumberTextExtractor
PDFPLUMBER_AVAILABLE = True

# Create aliases for backward compatibility and easier naming
PyMuPDFExtractor = PyMuPDFTextExtractor
PyPDF2Extractor = PyPDF2TextExtractor
PDFTextExtractor = PdftextTextExtractor
DoclingExtractor = DoclingTextExtractor

# Aliases for all extractors
PDFPlumberTextExtractor = PdfplumberTextExtractor

# All extractors available
__all__ = [
    'PyMuPDFExtractor',
    'PyPDF2Extractor',
    'PDFPlumberTextExtractor',
    'PDFTextExtractor',
    'DoclingExtractor',
    'PyMuPDFTextExtractor',
    'PyPDF2TextExtractor',
    'PdfplumberTextExtractor',
    'PdftextTextExtractor',
    'DoclingTextExtractor'
]
