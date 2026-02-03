"""
Integration tests for reading order prediction.

These tests run actual model inference and require appropriate backends.
Use pytest markers to select which tests to run based on available hardware.

Usage:
    pytest tests/integration/test_reading_order.py -m cpu
"""

import shutil

import pytest
from PIL import Image

# Check for optional dependencies
easyocr_available = False
try:
    import easyocr  # noqa: F401

    easyocr_available = True
except ImportError:
    pass

tesseract_available = shutil.which("tesseract") is not None


@pytest.mark.skipif(not easyocr_available, reason="easyocr not installed")
class TestRuleBasedReadingOrder:
    """Tests for rule-based reading order prediction."""

    @pytest.mark.integration
    @pytest.mark.reading_order
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_rule_based_predictor(self, sample_document: Image.Image):
        """Test rule-based reading order predictor."""
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
        from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor

        # Initialize components
        layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu"))
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        predictor = RuleBasedReadingOrderPredictor()

        # Extract layout and OCR
        layout_result = layout_extractor.extract(sample_document)
        ocr_result = ocr.extract(sample_document)

        # Predict reading order
        result = predictor.predict(layout_result, ocr_result)

        assert result is not None
        assert hasattr(result, "ordered_elements")
        # May or may not have elements depending on what's detected
        assert isinstance(result.ordered_elements, list)

    @pytest.mark.integration
    @pytest.mark.reading_order
    @pytest.mark.cpu
    def test_reading_order_output_methods(self, sample_document: Image.Image):
        """Test reading order output methods."""
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig
        from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor

        # Initialize components
        layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu"))
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        predictor = RuleBasedReadingOrderPredictor()

        # Extract and predict
        layout_result = layout_extractor.extract(sample_document)
        ocr_result = ocr.extract(sample_document)
        result = predictor.predict(layout_result, ocr_result)

        # Test get_full_text method
        full_text = result.get_full_text()
        assert full_text is not None
        assert isinstance(full_text, str)


class TestReadingOrderWithDifferentLayouts:
    """Tests for reading order with various document layouts."""

    @pytest.mark.skipif(not tesseract_available, reason="tesseract not installed")
    @pytest.mark.integration
    @pytest.mark.reading_order
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_reading_order_with_table(self, sample_document: Image.Image):
        """Test reading order on document with table."""
        from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
        from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig
        from omnidocs.tasks.reading_order import RuleBasedReadingOrderPredictor

        # Initialize components with Tesseract (alternative OCR)
        layout_extractor = DocLayoutYOLO(config=DocLayoutYOLOConfig(device="cpu"))
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        predictor = RuleBasedReadingOrderPredictor()

        # Extract and predict
        layout_result = layout_extractor.extract(sample_document)
        ocr_result = ocr.extract(sample_document)
        result = predictor.predict(layout_result, ocr_result)

        assert result is not None
        assert hasattr(result, "ordered_elements")
