"""
Text Extractors module for OmniDocs.

This module provides various text extractor implementations.
"""

from .pymupdf import PyMuPDFTextExtractor
from .pypdf2 import PyPDF2TextExtractor
from .pdfplumber import PdfplumberTextExtractor
from .pdftext import PdftextTextExtractor
from .docling_parse import DoclingTextExtractor

# Create aliases for backward compatibility and easier naming
PyMuPDFExtractor = PyMuPDFTextExtractor
PyPDF2Extractor = PyPDF2TextExtractor
PDFPlumberTextExtractor = PdfplumberTextExtractor
PDFTextExtractor = PdftextTextExtractor
DoclingExtractor = DoclingTextExtractor

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
