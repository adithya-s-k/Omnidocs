"""
Tesseract OCR Extraction - CPU

Standalone test script. Run locally on CPU.

Usage:
    python -m tests.standalone.ocr_extraction.tesseract_cpu path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class TesseractCPUTest(BaseOmnidocsTest):
    """Test Tesseract OCR with CPU."""

    @property
    def test_name(self) -> str:
        return "tesseract_cpu"

    @property
    def backend_name(self) -> str:
        return "pytorch_cpu"

    @property
    def task_name(self) -> str:
        return "ocr_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig

        return TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image)

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "num_blocks": len(result.text_blocks),
            "total_text_length": sum(len(b.text) for b in result.text_blocks),
        }


if __name__ == "__main__":
    run_standalone_test(TesseractCPUTest)
