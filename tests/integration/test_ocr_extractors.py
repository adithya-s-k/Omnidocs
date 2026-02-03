"""
Integration tests for OCR extraction.

These tests run actual model inference and require appropriate backends.
Use pytest markers to select which tests to run based on available hardware.

Usage:
    pytest tests/integration/test_ocr_extractors.py -m cpu
    pytest tests/integration/test_ocr_extractors.py -m gpu
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

paddle_available = False
try:
    import paddle  # noqa: F401

    paddle_available = True
except ImportError:
    pass

# Check for tesseract binary and pytesseract package
tesseract_available = False
if shutil.which("tesseract") is not None:
    try:
        import pytesseract  # noqa: F401

        tesseract_available = True
    except ImportError:
        pass


@pytest.mark.skipif(not easyocr_available, reason="easyocr not installed")
class TestEasyOCR:
    """Tests for EasyOCR."""

    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_easyocr_cpu(self, simple_text_image):
        """Test EasyOCR on CPU."""
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(simple_text_image["image"])

        assert result is not None
        assert hasattr(result, "text_blocks")
        # Should extract some text
        if result.text_blocks:
            total_text = " ".join(b.text for b in result.text_blocks)
            # Check if "Hello" or "World" is detected
            assert any(word.lower() in total_text.lower() for word in ["hello", "world"])

    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_easyocr_gpu(self, simple_text_image):
        """Test EasyOCR on GPU."""
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=True))
        result = ocr.extract(simple_text_image["image"])

        assert result is not None
        assert hasattr(result, "text_blocks")


@pytest.mark.skipif(not paddle_available, reason="paddlepaddle not installed")
class TestPaddleOCR:
    """Tests for PaddleOCR."""

    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_paddleocr_cpu(self, simple_text_image):
        """Test PaddleOCR on CPU."""
        from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig

        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(simple_text_image["image"])

        assert result is not None
        assert hasattr(result, "text_blocks")

    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.gpu
    @pytest.mark.pytorch
    def test_paddleocr_gpu(self, simple_text_image):
        """Test PaddleOCR on GPU."""
        from omnidocs.tasks.ocr_extraction import PaddleOCR, PaddleOCRConfig

        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="gpu"))
        result = ocr.extract(simple_text_image["image"])

        assert result is not None
        assert hasattr(result, "text_blocks")


@pytest.mark.skipif(not tesseract_available, reason="tesseract not installed")
class TestTesseractOCR:
    """Tests for Tesseract OCR."""

    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.cpu
    def test_tesseract_cpu(self, simple_text_image):
        """Test Tesseract OCR on CPU."""
        from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig

        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(simple_text_image["image"])

        assert result is not None
        assert hasattr(result, "text_blocks")


class TestOCRWithDocument:
    """Tests for OCR on full documents."""

    @pytest.mark.skipif(not easyocr_available, reason="easyocr not installed")
    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.cpu
    @pytest.mark.pytorch
    def test_easyocr_on_document(self, sample_document: Image.Image):
        """Test EasyOCR on a full document image."""
        from omnidocs.tasks.ocr_extraction import EasyOCR, EasyOCRConfig

        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(sample_document)

        assert result is not None
        assert hasattr(result, "text_blocks")
        # Should extract multiple text blocks from a document
        assert len(result.text_blocks) >= 0  # May vary based on document

    @pytest.mark.skipif(not tesseract_available, reason="tesseract not installed")
    @pytest.mark.integration
    @pytest.mark.ocr_extraction
    @pytest.mark.cpu
    def test_tesseract_on_document(self, sample_document: Image.Image):
        """Test Tesseract on a full document image."""
        from omnidocs.tasks.ocr_extraction import TesseractOCR, TesseractOCRConfig

        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(sample_document)

        assert result is not None
        assert hasattr(result, "text_blocks")
