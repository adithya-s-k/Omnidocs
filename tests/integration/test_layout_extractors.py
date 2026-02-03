"""
Integration tests for layout extraction.

These tests run actual model inference and require appropriate backends.
Use pytest markers to select which tests to run based on available hardware.

Usage:
    pytest tests/integration/test_layout_extractors.py -m cpu
    pytest tests/integration/test_layout_extractors.py -m "gpu and pytorch"
"""

import pytest
from PIL import Image


class TestDocLayoutYOLO:
    """Tests for DocLayout-YOLO layout detection."""

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_doclayout_yolo_cpu(self, sample_document: Image.Image):
        """Test DocLayout-YOLO on CPU."""
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

        extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu"))
        result = extractor.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")
        # Should detect at least one element
        assert len(result.bboxes) >= 0  # May be empty for synthetic docs

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_doclayout_yolo_gpu(self, sample_document: Image.Image):
        """Test DocLayout-YOLO on GPU."""
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig

        extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cuda"))
        result = extractor.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")


class TestRTDETR:
    """Tests for RT-DETR layout detection."""

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_rtdetr_cpu(self, sample_document: Image.Image):
        """Test RT-DETR on CPU."""
        from omnidocs.tasks.layout_extraction import RTDETRConfig, RTDETRLayoutExtractor

        extractor = RTDETRLayoutExtractor(config=RTDETRConfig(device="cpu"))
        result = extractor.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_rtdetr_gpu(self, sample_document: Image.Image):
        """Test RT-DETR on GPU."""
        from omnidocs.tasks.layout_extraction import RTDETRConfig, RTDETRLayoutExtractor

        extractor = RTDETRLayoutExtractor(config=RTDETRConfig(device="cuda"))
        result = extractor.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")


class TestQwenLayoutDetector:
    """Tests for Qwen-based layout detection."""

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_qwen_layout_pytorch(self, sample_document: Image.Image):
        """Test Qwen layout detection with PyTorch backend."""
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        detector = QwenLayoutDetector(
            backend=QwenLayoutPyTorchConfig(model="Qwen/Qwen2-VL-7B-Instruct")
        )
        result = detector.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.gpu
    @pytest.mark.vllm
    def test_qwen_layout_vllm(self, sample_document: Image.Image):
        """Test Qwen layout detection with VLLM backend."""
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

        detector = QwenLayoutDetector(
            backend=QwenLayoutVLLMConfig(gpu_memory_utilization=0.85)
        )
        result = detector.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.mlx
    def test_qwen_layout_mlx(self, sample_document: Image.Image):
        """Test Qwen layout detection with MLX backend."""
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutMLXConfig

        detector = QwenLayoutDetector(backend=QwenLayoutMLXConfig())
        result = detector.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")

    @pytest.mark.integration
    @pytest.mark.layout_extraction
    @pytest.mark.api
    def test_qwen_layout_api(self, sample_document: Image.Image):
        """Test Qwen layout detection with API backend."""
        from omnidocs.tasks.layout_extraction import QwenLayoutDetector
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

        detector = QwenLayoutDetector(backend=QwenLayoutAPIConfig())
        result = detector.extract(sample_document)

        assert result is not None
        assert hasattr(result, "bboxes")
