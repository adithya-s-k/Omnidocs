"""Test utilities for OmniDocs."""

from .evaluation import evaluate_text_extraction
from .synthetic_document import create_synthetic_document

__all__ = ["create_synthetic_document", "evaluate_text_extraction"]
