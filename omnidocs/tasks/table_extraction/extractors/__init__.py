"""
Table Extractors module for OmniDocs.

This module provides various table extractor implementations.
"""

from .camelot import CamelotExtractor
from .pdfplumber import PDFPlumberExtractor
from .ppstructure import PPStructureExtractor
from .table_transformer import TableTransformerExtractor
from .tableformer import TableFormerExtractor
from .tabula import TabulaExtractor
from .surya_table import SuryaTableExtractor

__all__ = [
    'CamelotExtractor',
    'PDFPlumberExtractor',
    'PPStructureExtractor',
    'SuryaTableExtractor',
    'TableFormerExtractor',
    'TableTransformerExtractor',
    'TabulaExtractor',
]
