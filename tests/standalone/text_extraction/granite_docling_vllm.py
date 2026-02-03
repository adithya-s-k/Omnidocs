"""
Granite Docling Text Extraction - VLLM Backend (GPU)

Standalone test script. Run on Modal with GPU.

IMPORTANT: VLLM requires revision="untied" for this model due to tied weights.
The OmniDocs config handles this automatically.

Usage:
    python -m tests.standalone.text_extraction.granite_docling_vllm path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class GraniteDoclingTextVLLMTest(BaseOmnidocsTest):
    """Test Granite Docling text extraction with VLLM backend."""

    @property
    def test_name(self) -> str:
        return "granite_docling_text_vllm"

    @property
    def backend_name(self) -> str:
        return "vllm"

    @property
    def task_name(self) -> str:
        return "text_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextVLLMConfig,
        )

        return GraniteDoclingTextExtractor(
            backend=GraniteDoclingTextVLLMConfig(
                fast_boot=True,
            )
        )

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image, output_format="markdown")

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "content_length": len(result.content),
            "model": result.model_name,
        }


if __name__ == "__main__":
    run_standalone_test(GraniteDoclingTextVLLMTest)
