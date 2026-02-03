"""
Granite Docling Text Extraction - API Backend

Standalone test script. Requires OPENROUTER_API_KEY environment variable.

Usage:
    export OPENROUTER_API_KEY=your_key
    python -m tests.standalone.text_extraction.granite_docling_api path/to/image.png
"""

import os
from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class GraniteDoclingTextAPITest(BaseOmnidocsTest):
    """Test Granite Docling text extraction with API backend."""

    @property
    def test_name(self) -> str:
        return "granite_docling_text_api"

    @property
    def backend_name(self) -> str:
        return "api"

    @property
    def task_name(self) -> str:
        return "text_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextAPIConfig,
        )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable required for API backend"
            )

        return GraniteDoclingTextExtractor(
            backend=GraniteDoclingTextAPIConfig(api_key=api_key)
        )

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image, output_format="markdown")

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "content_length": len(result.content),
            "model": result.model_name,
        }


if __name__ == "__main__":
    run_standalone_test(GraniteDoclingTextAPITest)
