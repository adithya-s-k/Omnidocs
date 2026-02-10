"""
Structured Extraction Module.

Provides extractors for extracting structured data from document images
using Pydantic schemas for type-safe output.

Example:
    ```python
    from pydantic import BaseModel
    from omnidocs.vlm import VLMAPIConfig
    from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

    class Invoice(BaseModel):
        vendor: str
        total: float
        items: list[str]

    config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
    extractor = VLMStructuredExtractor(config=config)
    result = extractor.extract(
        "invoice.png",
        schema=Invoice,
        prompt="Extract invoice details from this document.",
    )
    print(result.vendor, result.total)
    ```
"""

from .base import BaseStructuredExtractor
from .models import StructuredOutput
from .vlm import VLMStructuredExtractor

__all__ = [
    "BaseStructuredExtractor",
    "StructuredOutput",
    "VLMStructuredExtractor",
]
