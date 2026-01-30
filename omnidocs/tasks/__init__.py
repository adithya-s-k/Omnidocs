"""
OmniDocs Task Modules.

Each task module provides extractors for specific document processing tasks.

Available task modules:
    - layout_extraction: Detect document structure (titles, tables, figures, etc.)
    - ocr_extraction: Extract text with bounding boxes from images
    - text_extraction: Convert document images to HTML/Markdown
"""

from omnidocs.tasks import layout_extraction, ocr_extraction, text_extraction

__all__ = [
    "layout_extraction",
    "ocr_extraction",
    "text_extraction",
]
