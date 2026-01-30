"""
Integration tests for OCR extractors.

Tests actual OCR functionality with synthetic images.
Requires OCR engines to be installed (tesseract, easyocr, paddleocr).
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from omnidocs.tasks.ocr_extraction import (
    EasyOCR,
    EasyOCRConfig,
    OCROutput,
    PaddleOCR,
    PaddleOCRConfig,
    TesseractOCR,
    TesseractOCRConfig,
)
from omnidocs.tasks.ocr_extraction.models import OCRGranularity

# ============= Fixtures =============


@pytest.fixture
def simple_text_image() -> Image.Image:
    """Create a simple image with clear text for OCR testing."""
    img = Image.new("RGB", (400, 150), color="white")
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except (OSError, IOError):
            font = ImageFont.load_default()

    draw.text((30, 30), "Hello World", fill="black", font=font)
    draw.text((30, 80), "Test 12345", fill="black", font=font)

    return img


@pytest.fixture
def document_image() -> tuple[Image.Image, list[str]]:
    """Create a more complex document-like image."""
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()

    expected_texts = []

    # Title
    title = "Document Title"
    draw.text((50, 40), title, fill="black", font=font_title)
    expected_texts.append(title)

    # Body text
    body_lines = [
        "This is a test document.",
        "It contains multiple lines.",
        "Numbers: 12345 67890",
    ]
    y = 120
    for line in body_lines:
        draw.text((50, y), line, fill="black", font=font_body)
        expected_texts.append(line)
        y += 35

    return img, expected_texts


@pytest.fixture
def numpy_array_image(simple_text_image: Image.Image) -> np.ndarray:
    """Convert test image to numpy array."""
    return np.array(simple_text_image)


# ============= Helper Functions =============


def check_ocr_result_structure(result: OCROutput, expected_min_blocks: int = 1):
    """Verify OCROutput structure is valid."""
    assert isinstance(result, OCROutput)
    assert result.image_width > 0
    assert result.image_height > 0
    assert result.block_count >= expected_min_blocks
    assert len(result.full_text) > 0

    for block in result.text_blocks:
        assert len(block.text) > 0
        assert 0 <= block.confidence <= 1
        assert block.bbox.x1 >= 0
        assert block.bbox.y1 >= 0
        assert block.bbox.x2 > block.bbox.x1
        assert block.bbox.y2 > block.bbox.y1


def text_found_in_result(result: OCROutput, expected_text: str, case_sensitive: bool = False) -> bool:
    """Check if expected text is found in OCR result."""
    full_text = result.full_text
    if not case_sensitive:
        full_text = full_text.lower()
        expected_text = expected_text.lower()
    return expected_text in full_text


# ============= Tesseract OCR Tests =============


class TestTesseractOCR:
    """Integration tests for TesseractOCR."""

    @pytest.fixture(autouse=True)
    def skip_if_tesseract_unavailable(self):
        """Skip tests if Tesseract is not installed."""
        import shutil

        if not shutil.which("tesseract"):
            pytest.skip("Tesseract not installed on system")

    def test_tesseract_initialization(self):
        """Test TesseractOCR can be initialized."""
        config = TesseractOCRConfig(languages=["eng"])
        ocr = TesseractOCR(config=config)
        assert ocr.MODEL_NAME == "tesseract"
        assert ocr.config.languages == ["eng"]

    def test_tesseract_extract_pil_image(self, simple_text_image: Image.Image):
        """Test TesseractOCR with PIL Image input."""
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(simple_text_image)

        check_ocr_result_structure(result, expected_min_blocks=2)
        assert result.model_name == "tesseract"
        assert text_found_in_result(result, "hello")
        assert text_found_in_result(result, "world")

    def test_tesseract_extract_numpy_array(self, numpy_array_image: np.ndarray):
        """Test TesseractOCR with numpy array input."""
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(numpy_array_image)

        check_ocr_result_structure(result, expected_min_blocks=2)

    def test_tesseract_extract_document(self, document_image: tuple[Image.Image, list[str]]):
        """Test TesseractOCR with document-like image."""
        img, expected_texts = document_image
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(img)

        check_ocr_result_structure(result, expected_min_blocks=5)
        assert result.image_width == 800
        assert result.image_height == 600

        # Check that key content is found
        assert text_found_in_result(result, "document")
        assert text_found_in_result(result, "title")
        assert text_found_in_result(result, "12345")

    def test_tesseract_extract_lines(self, document_image: tuple[Image.Image, list[str]]):
        """Test TesseractOCR line-level extraction."""
        img, _ = document_image
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract_lines(img)

        check_ocr_result_structure(result, expected_min_blocks=1)

        # Line-level extraction should have fewer blocks than word-level
        word_result = ocr.extract(img)
        assert result.block_count <= word_result.block_count

        # All blocks should be LINE granularity
        for block in result.text_blocks:
            assert block.granularity == OCRGranularity.LINE

    def test_tesseract_word_granularity(self, simple_text_image: Image.Image):
        """Test that default extraction returns word-level blocks."""
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(simple_text_image)

        for block in result.text_blocks:
            assert block.granularity == OCRGranularity.WORD

    def test_tesseract_confidence_scores(self, simple_text_image: Image.Image):
        """Test that confidence scores are reasonable."""
        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(simple_text_image)

        # Clear text should have high confidence
        assert result.average_confidence > 0.5

    def test_tesseract_custom_psm(self, simple_text_image: Image.Image):
        """Test TesseractOCR with custom page segmentation mode."""
        config = TesseractOCRConfig(languages=["eng"], psm=6)  # Uniform block
        ocr = TesseractOCR(config=config)
        result = ocr.extract(simple_text_image)

        check_ocr_result_structure(result, expected_min_blocks=1)

    def test_tesseract_custom_oem(self, simple_text_image: Image.Image):
        """Test TesseractOCR with custom OCR engine mode."""
        config = TesseractOCRConfig(languages=["eng"], oem=1)  # LSTM only
        ocr = TesseractOCR(config=config)
        result = ocr.extract(simple_text_image)

        check_ocr_result_structure(result, expected_min_blocks=1)


# ============= EasyOCR Tests =============


class TestEasyOCR:
    """Integration tests for EasyOCR."""

    @pytest.fixture(autouse=True)
    def skip_if_easyocr_unavailable(self):
        """Skip tests if EasyOCR is not installed."""
        try:
            import easyocr  # noqa: F401
        except ImportError:
            pytest.skip("EasyOCR not installed")

    def test_easyocr_initialization(self):
        """Test EasyOCR can be initialized."""
        config = EasyOCRConfig(languages=["en"], gpu=False)
        ocr = EasyOCR(config=config)
        assert ocr.MODEL_NAME == "easyocr"
        assert ocr.config.languages == ["en"]

    def test_easyocr_extract_pil_image(self, simple_text_image: Image.Image):
        """Test EasyOCR with PIL Image input."""
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(simple_text_image)

        check_ocr_result_structure(result, expected_min_blocks=1)
        assert result.model_name == "easyocr"
        assert text_found_in_result(result, "hello")

    def test_easyocr_extract_numpy_array(self, numpy_array_image: np.ndarray):
        """Test EasyOCR with numpy array input."""
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(numpy_array_image)

        check_ocr_result_structure(result, expected_min_blocks=1)

    def test_easyocr_extract_document(self, document_image: tuple[Image.Image, list[str]]):
        """Test EasyOCR with document-like image."""
        img, _ = document_image
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(img)

        check_ocr_result_structure(result, expected_min_blocks=1)
        assert result.image_width == 800
        assert result.image_height == 600
        assert text_found_in_result(result, "document")

    def test_easyocr_default_word_granularity(self, simple_text_image: Image.Image):
        """Test that EasyOCR returns word-level blocks by default."""
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(simple_text_image)

        # EasyOCR returns word-level by default (paragraph=False)
        for block in result.text_blocks:
            assert block.granularity == OCRGranularity.WORD

    def test_easyocr_polygon_coordinates(self, simple_text_image: Image.Image):
        """Test that EasyOCR provides polygon coordinates."""
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(simple_text_image)

        # EasyOCR provides polygon coordinates
        for block in result.text_blocks:
            assert block.polygon is not None
            assert len(block.polygon) == 4  # Quadrilateral

    def test_easyocr_confidence_scores(self, simple_text_image: Image.Image):
        """Test that confidence scores are reasonable."""
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
        result = ocr.extract(simple_text_image)

        assert result.average_confidence > 0.5

    def test_easyocr_invalid_detail_parameter(self, simple_text_image: Image.Image):
        """Test that invalid detail parameter raises ValueError."""
        ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))

        with pytest.raises(ValueError, match="detail must be 0 or 1"):
            ocr.extract(simple_text_image, detail=2)

        with pytest.raises(ValueError, match="detail must be 0 or 1"):
            ocr.extract(simple_text_image, detail=-1)


# ============= PaddleOCR Tests =============


class TestPaddleOCR:
    """Integration tests for PaddleOCR."""

    @pytest.fixture(autouse=True)
    def skip_if_paddleocr_unavailable(self):
        """Skip tests if PaddleOCR is not installed."""
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            pytest.skip("PaddleOCR not installed")

    def test_paddleocr_initialization(self):
        """Test PaddleOCR can be initialized."""
        config = PaddleOCRConfig(lang="en", device="cpu")
        ocr = PaddleOCR(config=config)
        assert ocr.MODEL_NAME == "paddleocr"
        assert ocr.config.lang == "en"

    def test_paddleocr_extract_pil_image(self, simple_text_image: Image.Image):
        """Test PaddleOCR with PIL Image input."""
        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(simple_text_image)

        check_ocr_result_structure(result, expected_min_blocks=1)
        assert result.model_name == "paddleocr"
        assert text_found_in_result(result, "hello")

    def test_paddleocr_extract_numpy_array(self, numpy_array_image: np.ndarray):
        """Test PaddleOCR with numpy array input."""
        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(numpy_array_image)

        check_ocr_result_structure(result, expected_min_blocks=1)

    def test_paddleocr_extract_document(self, document_image: tuple[Image.Image, list[str]]):
        """Test PaddleOCR with document-like image."""
        img, _ = document_image
        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(img)

        check_ocr_result_structure(result, expected_min_blocks=1)
        assert result.image_width == 800
        assert result.image_height == 600
        assert text_found_in_result(result, "document")

    def test_paddleocr_line_granularity(self, simple_text_image: Image.Image):
        """Test that PaddleOCR returns line-level blocks."""
        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(simple_text_image)

        for block in result.text_blocks:
            assert block.granularity == OCRGranularity.LINE

    def test_paddleocr_polygon_coordinates(self, simple_text_image: Image.Image):
        """Test that PaddleOCR provides polygon coordinates."""
        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(simple_text_image)

        for block in result.text_blocks:
            assert block.polygon is not None
            assert len(block.polygon) == 4

    def test_paddleocr_confidence_scores(self, simple_text_image: Image.Image):
        """Test that confidence scores are reasonable."""
        ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
        result = ocr.extract(simple_text_image)

        assert result.average_confidence > 0.5


# ============= Cross-Engine Comparison Tests =============


class TestOCRComparison:
    """Tests comparing results across OCR engines."""

    @pytest.fixture
    def available_engines(self, simple_text_image: Image.Image) -> dict:
        """Get available OCR engines and their results."""
        engines = {}

        # Try Tesseract
        import shutil

        if shutil.which("tesseract"):
            try:
                ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
                engines["tesseract"] = ocr.extract(simple_text_image)
            except Exception:
                pass

        # Try EasyOCR
        try:
            import easyocr  # noqa: F401

            ocr = EasyOCR(config=EasyOCRConfig(languages=["en"], gpu=False))
            engines["easyocr"] = ocr.extract(simple_text_image)
        except ImportError:
            pass

        # Try PaddleOCR
        try:
            import paddleocr  # noqa: F401

            ocr = PaddleOCR(config=PaddleOCRConfig(lang="en", device="cpu"))
            engines["paddleocr"] = ocr.extract(simple_text_image)
        except ImportError:
            pass

        if len(engines) < 2:
            pytest.skip("Need at least 2 OCR engines for comparison tests")

        return engines

    def test_all_engines_find_hello(self, available_engines: dict):
        """Test that all engines find 'Hello' in the image."""
        for name, result in available_engines.items():
            assert text_found_in_result(result, "hello"), f"{name} failed to find 'Hello'"

    def test_all_engines_return_valid_output(self, available_engines: dict):
        """Test that all engines return valid OCROutput."""
        for name, result in available_engines.items():
            check_ocr_result_structure(result, expected_min_blocks=1)

    def test_engines_have_consistent_image_size(self, available_engines: dict):
        """Test that all engines report the same image size."""
        sizes = [(name, (r.image_width, r.image_height)) for name, r in available_engines.items()]

        first_size = sizes[0][1]
        for name, size in sizes[1:]:
            assert size == first_size, f"{name} has different size: {size} vs {first_size}"


# ============= Edge Case Tests =============


class TestOCREdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ocr(self):
        """Skip if no OCR engine is available."""
        import shutil

        has_tesseract = shutil.which("tesseract") is not None
        has_easyocr = False
        has_paddleocr = False

        try:
            import easyocr  # noqa: F401

            has_easyocr = True
        except ImportError:
            pass

        try:
            import paddleocr  # noqa: F401

            has_paddleocr = True
        except ImportError:
            pass

        if not (has_tesseract or has_easyocr or has_paddleocr):
            pytest.skip("No OCR engine available")

    def test_blank_image(self):
        """Test OCR on blank white image."""
        img = Image.new("RGB", (200, 100), color="white")

        import shutil

        if shutil.which("tesseract"):
            ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
            result = ocr.extract(img)
            # Should return empty or near-empty result
            assert isinstance(result, OCROutput)

    def test_very_small_image(self):
        """Test OCR on very small image."""
        img = Image.new("RGB", (50, 20), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((5, 2), "Hi", fill="black")

        import shutil

        if shutil.which("tesseract"):
            ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
            result = ocr.extract(img)
            assert isinstance(result, OCROutput)

    def test_grayscale_image(self):
        """Test OCR on grayscale image."""
        img = Image.new("L", (200, 100), color=255)  # L mode is grayscale
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "Gray", fill=0)

        import shutil

        if shutil.which("tesseract"):
            ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
            result = ocr.extract(img)
            assert isinstance(result, OCROutput)

    def test_rgba_image(self):
        """Test OCR on RGBA image with alpha channel."""
        img = Image.new("RGBA", (200, 100), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "Alpha", fill=(0, 0, 0, 255))

        import shutil

        if shutil.which("tesseract"):
            ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
            result = ocr.extract(img)
            assert isinstance(result, OCROutput)

    def test_invalid_image_path(self):
        """Test that invalid image path raises error."""
        import shutil

        if shutil.which("tesseract"):
            ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
            with pytest.raises(FileNotFoundError):
                ocr.extract("/nonexistent/path/to/image.png")

    def test_output_visualization(self, simple_text_image: Image.Image, tmp_path):
        """Test OCR output visualization."""
        import shutil

        if not shutil.which("tesseract"):
            pytest.skip("Tesseract not installed")

        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(simple_text_image)

        # Test visualization
        viz_path = tmp_path / "viz.png"
        viz_image = result.visualize(simple_text_image, output_path=viz_path)

        assert viz_path.exists()
        assert isinstance(viz_image, Image.Image)
        assert viz_image.size == simple_text_image.size

    def test_output_json_roundtrip(self, simple_text_image: Image.Image, tmp_path):
        """Test saving and loading OCR output as JSON."""
        import shutil

        if not shutil.which("tesseract"):
            pytest.skip("Tesseract not installed")

        ocr = TesseractOCR(config=TesseractOCRConfig(languages=["eng"]))
        result = ocr.extract(simple_text_image)

        json_path = tmp_path / "ocr_result.json"
        result.save_json(json_path)

        loaded = OCROutput.load_json(json_path)

        assert loaded.image_width == result.image_width
        assert loaded.image_height == result.image_height
        assert loaded.block_count == result.block_count
        assert loaded.full_text == result.full_text
