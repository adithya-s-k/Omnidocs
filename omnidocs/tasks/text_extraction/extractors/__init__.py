"""
Text Extractors module for OmniDocs.

This module provides various text extractor implementations.
"""

from .pymupdf import PyMuPDFTextExtractor
from .pypdf2 import PyPDF2TextExtractor
from .pdftext import PdftextTextExtractor
from .docling_parse import DoclingTextExtractor
from .surya_text import SuryaTextExtractor

from .pdfplumber import PdfplumberTextExtractor
PDFPLUMBER_AVAILABLE = True

# All extractors available
__all__ = [
    'PyMuPDFTextExtractor',
    'PyPDF2TextExtractor',
    'PdfplumberTextExtractor',
    'PdftextTextExtractor',
    'DoclingTextExtractor',
    'SuryaTextExtractor'
]