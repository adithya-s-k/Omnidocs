"""
Utilities module for OmniDocs.

This module provides common utilities used across different tasks and components.
"""

from .logging import get_logger
from .language import (
    LanguageCode,
    GlobalLanguageMapper,
    LanguageDetector,
    global_language_mapper,
    get_language_mapper,
    detect_language,
    is_supported_language,
    get_all_supported_languages
)

__all__ = [
    'get_logger',
    'LanguageCode',
    'GlobalLanguageMapper', 
    'LanguageDetector',
    'global_language_mapper',
    'get_language_mapper',
    'detect_language',
    'is_supported_language',
    'get_all_supported_languages'
]
