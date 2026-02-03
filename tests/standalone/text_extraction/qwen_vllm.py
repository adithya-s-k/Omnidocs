"""
Qwen Text Extraction - VLLM Backend

Standalone test script. Run locally with GPU or via Modal runner.

Usage:
    python -m tests.standalone.text_extraction.qwen_vllm path/to/image.png
"""

from typing import Any, Dict

from PIL import Image

from tests.standalone.base import BaseOmnidocsTest, run_standalone_test


class QwenTextVLLMTest(BaseOmnidocsTest):
    """Test Qwen text extraction with VLLM backend."""

    @property
    def test_name(self) -> str:
        return "qwen_text_vllm"

    @property
    def backend_name(self) -> str:
        return "vllm"

    @property
    def task_name(self) -> str:
        return "text_extraction"

    def create_extractor(self) -> Any:
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

        return QwenTextExtractor(backend=QwenTextVLLMConfig(gpu_memory_utilization=0.85))

    def run_extraction(self, extractor: Any, image: Image.Image) -> Any:
        return extractor.extract(image, output_format="markdown")

    def get_metadata(self, result: Any) -> Dict[str, Any]:
        return {
            "content_length": len(result.content),
            "model": result.model_name,
        }


if __name__ == "__main__":
    run_standalone_test(QwenTextVLLMTest)
