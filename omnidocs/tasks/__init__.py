"""
OmniDocs Task Modules.

Each task module provides extractors for specific document processing tasks.

Available task modules:
    - layout_extraction: Detect document structure (titles, tables, figures, etc.)
    - text_extraction: Convert document images to HTML/Markdown
"""

from omnidocs.tasks import layout_extraction, text_extraction

__all__ = [
    "layout_extraction",
    "text_extraction",
]
