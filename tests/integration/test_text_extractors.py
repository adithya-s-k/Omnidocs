"""
Integration tests for text extraction.

These tests run actual model inference and require appropriate backends.
Use pytest markers to select which tests to run based on available hardware.

Usage:
    pytest tests/integration/test_text_extractors.py -m cpu
    pytest tests/integration/test_text_extractors.py -m "gpu and pytorch"
    pytest tests/integration/test_text_extractors.py -m mlx
"""

import pytest
from PIL import Image

from tests.utils.evaluation import evaluate_text_extraction


class TestQwenTextExtractor:
    """Tests for Qwen-based text extraction."""

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_qwen_pytorch_extraction(self, sample_document: Image.Image):
        """Test Qwen text extraction with PyTorch backend."""
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

        extractor = QwenTextExtractor(
            backend=QwenTextPyTorchConfig(model="Qwen/Qwen2-VL-7B-Instruct")
        )
        result = extractor.extract(sample_document, output_format="markdown")

        assert result is not None
        assert result.content
        assert len(result.content) > 0
        assert result.model_name

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.gpu
    @pytest.mark.vllm
    def test_qwen_vllm_extraction(self, sample_document: Image.Image):
        """Test Qwen text extraction with VLLM backend."""
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

        extractor = QwenTextExtractor(
            backend=QwenTextVLLMConfig(gpu_memory_utilization=0.85)
        )
        result = extractor.extract(sample_document, output_format="markdown")

        assert result is not None
        assert result.content
        assert len(result.content) > 0

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.mlx
    def test_qwen_mlx_extraction(self, sample_document: Image.Image):
        """Test Qwen text extraction with MLX backend."""
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

        extractor = QwenTextExtractor(backend=QwenTextMLXConfig())
        result = extractor.extract(sample_document, output_format="markdown")

        assert result is not None
        assert result.content
        assert len(result.content) > 0

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.api
    def test_qwen_api_extraction(self, sample_document: Image.Image):
        """Test Qwen text extraction with API backend."""
        from omnidocs.tasks.text_extraction import QwenTextExtractor
        from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

        extractor = QwenTextExtractor(backend=QwenTextAPIConfig())
        result = extractor.extract(sample_document, output_format="markdown")

        assert result is not None
        assert result.content
        assert len(result.content) > 0


class TestNanonetsTextExtractor:
    """Tests for Nanonets-based text extraction."""

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_nanonets_pytorch_extraction(self, sample_document: Image.Image):
        """Test Nanonets text extraction with PyTorch backend."""
        from omnidocs.tasks.text_extraction import NanonetsTextExtractor
        from omnidocs.tasks.text_extraction.nanonets import NanonetsTextPyTorchConfig

        extractor = NanonetsTextExtractor(backend=NanonetsTextPyTorchConfig())
        result = extractor.extract(sample_document)

        assert result is not None
        assert result.content
        assert len(result.content) > 0

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.gpu
    @pytest.mark.vllm
    def test_nanonets_vllm_extraction(self, sample_document: Image.Image):
        """Test Nanonets text extraction with VLLM backend."""
        from omnidocs.tasks.text_extraction import NanonetsTextExtractor
        from omnidocs.tasks.text_extraction.nanonets import NanonetsTextVLLMConfig

        extractor = NanonetsTextExtractor(
            backend=NanonetsTextVLLMConfig(gpu_memory_utilization=0.85)
        )
        result = extractor.extract(sample_document)

        assert result is not None
        assert result.content
        assert len(result.content) > 0

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.mlx
    def test_nanonets_mlx_extraction(self, sample_document: Image.Image):
        """Test Nanonets text extraction with MLX backend."""
        from omnidocs.tasks.text_extraction import NanonetsTextExtractor
        from omnidocs.tasks.text_extraction.nanonets import NanonetsTextMLXConfig

        extractor = NanonetsTextExtractor(backend=NanonetsTextMLXConfig())
        result = extractor.extract(sample_document)

        assert result is not None
        assert result.content
        assert len(result.content) > 0


class TestDotsOCRTextExtractor:
    """Tests for DotsOCR-based text extraction."""

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_dotsocr_pytorch_extraction(self, sample_document: Image.Image):
        """Test DotsOCR text extraction with PyTorch backend."""
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        extractor = DotsOCRTextExtractor(backend=DotsOCRPyTorchConfig())
        result = extractor.extract(sample_document)

        assert result is not None
        assert result.content
        assert len(result.content) > 0

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.gpu
    @pytest.mark.vllm
    def test_dotsocr_vllm_extraction(self, sample_document: Image.Image):
        """Test DotsOCR text extraction with VLLM backend."""
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        extractor = DotsOCRTextExtractor(
            backend=DotsOCRVLLMConfig(gpu_memory_utilization=0.85)
        )
        result = extractor.extract(sample_document)

        assert result is not None
        assert result.content
        assert len(result.content) > 0

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.api
    def test_dotsocr_api_extraction(self, sample_document: Image.Image):
        """Test DotsOCR text extraction with API backend."""
        from omnidocs.tasks.text_extraction import DotsOCRTextExtractor
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

        extractor = DotsOCRTextExtractor(backend=DotsOCRAPIConfig())
        result = extractor.extract(sample_document)

        assert result is not None
        assert result.content
        assert len(result.content) > 0


class TestTextExtractionQuality:
    """Tests for text extraction quality evaluation."""

    @pytest.mark.integration
    @pytest.mark.text_extraction
    @pytest.mark.cpu
    def test_text_extraction_evaluation(self, sample_document_with_ground_truth):
        """Test text extraction quality evaluation."""
        doc = sample_document_with_ground_truth
        ground_truth = doc.full_text

        # This test demonstrates how to evaluate extraction quality
        # In real tests, you would use an actual extractor
        eval_result = evaluate_text_extraction(
            extracted=ground_truth,  # Perfect match for demo
            ground_truth=ground_truth,
        )

        assert eval_result.character_accuracy == 1.0
        assert eval_result.word_accuracy == 1.0
        assert eval_result.is_passing
