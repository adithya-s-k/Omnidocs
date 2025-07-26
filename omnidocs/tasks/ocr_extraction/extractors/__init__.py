"""
OCR Extractors module for OmniDocs.

This module provides various OCR extractor implementations.
"""

from .paddle import PaddleOCRExtractor
from .tesseract_ocr import TesseractOCRExtractor
from .easy_ocr import EasyOCRExtractor
from .surya_ocr import SuryaOCRExtractor

__all__ = [
    'PaddleOCRExtractor',
    'TesseractOCRExtractor', 
    'EasyOCRExtractor',
    'SuryaOCRExtractor'
]
