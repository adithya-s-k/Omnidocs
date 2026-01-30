"""
Tests for the OCR extraction base class.

Tests the abstract base class and shared functionality across all OCR extractors.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from omnidocs.tasks.ocr_extraction.base import BaseOCRExtractor
from omnidocs.tasks.ocr_extraction.models import OCROutput


class ConcreteOCRExtractor(BaseOCRExtractor):
    """Concrete implementation of BaseOCRExtractor for testing."""

    def __init__(self):
        self._load_model()

    def _load_model(self) -> None:
        """Mock model loading."""
        self._loaded = True

    def extract(self, image) -> OCROutput:
        """Mock extraction returning empty result."""
        pil_image = self._prepare_image(image)
        return OCROutput(
            text_blocks=[],
            full_text="",
            image_width=pil_image.size[0],
            image_height=pil_image.size[1],
        )


class TestBaseOCRExtractor:
    """Tests for BaseOCRExtractor abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseOCRExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseOCRExtractor()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""
        extractor = ConcreteOCRExtractor()
        assert extractor._loaded is True


class TestPrepareImage:
    """Tests for _prepare_image method."""

    @pytest.fixture
    def extractor(self):
        """Create a concrete extractor for testing."""
        return ConcreteOCRExtractor()

    def test_prepare_pil_image_rgb(self, extractor):
        """Test preparing RGB PIL Image."""
        img = Image.new("RGB", (100, 100), color="red")
        result = extractor._prepare_image(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_prepare_pil_image_rgba(self, extractor):
        """Test preparing RGBA PIL Image - converts to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
        result = extractor._prepare_image(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_prepare_pil_image_grayscale(self, extractor):
        """Test preparing grayscale PIL Image - converts to RGB."""
        img = Image.new("L", (100, 100), color=128)
        result = extractor._prepare_image(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_prepare_pil_image_palette(self, extractor):
        """Test preparing palette mode PIL Image - converts to RGB."""
        img = Image.new("P", (100, 100))
        result = extractor._prepare_image(img)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_prepare_numpy_array_rgb(self, extractor):
        """Test preparing RGB numpy array."""
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = extractor._prepare_image(arr)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_prepare_numpy_array_grayscale(self, extractor):
        """Test preparing grayscale numpy array."""
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = extractor._prepare_image(arr)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"  # Should be converted to RGB

    def test_prepare_numpy_array_rgba(self, extractor):
        """Test preparing RGBA numpy array."""
        arr = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        result = extractor._prepare_image(arr)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"  # Should be converted to RGB

    def test_prepare_string_path(self, extractor, tmp_path):
        """Test preparing image from string path."""
        img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "test_image.png"
        img.save(img_path)

        result = extractor._prepare_image(str(img_path))

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_prepare_path_object(self, extractor, tmp_path):
        """Test preparing image from Path object."""
        img = Image.new("RGB", (100, 100), color="green")
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        result = extractor._prepare_image(Path(img_path))

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == (100, 100)

    def test_prepare_nonexistent_path(self, extractor):
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            extractor._prepare_image("/nonexistent/path/image.png")

    def test_prepare_nonexistent_pathlib_path(self, extractor):
        """Test that nonexistent Path object raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            extractor._prepare_image(Path("/nonexistent/path/image.png"))

    def test_prepare_unsupported_type(self, extractor):
        """Test that unsupported input type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            extractor._prepare_image(12345)  # int is not supported

        with pytest.raises(ValueError, match="Unsupported image type"):
            extractor._prepare_image([1, 2, 3])  # list is not supported

        with pytest.raises(ValueError, match="Unsupported image type"):
            extractor._prepare_image({"image": "data"})  # dict is not supported

    def test_prepare_various_image_formats(self, extractor, tmp_path):
        """Test preparing images from various file formats."""
        img = Image.new("RGB", (50, 50), color="yellow")

        formats = [
            ("test.png", "PNG"),
            ("test.jpg", "JPEG"),
            ("test.bmp", "BMP"),
            ("test.gif", "GIF"),
        ]

        for filename, fmt in formats:
            path = tmp_path / filename
            if fmt == "JPEG":
                # JPEG doesn't support alpha, ensure RGB
                img.convert("RGB").save(path, format=fmt)
            else:
                img.save(path, format=fmt)

            result = extractor._prepare_image(path)
            assert isinstance(result, Image.Image), f"Failed for format: {fmt}"
            assert result.mode == "RGB", f"Wrong mode for format: {fmt}"


class TestExtractMethod:
    """Tests for the extract method interface."""

    @pytest.fixture
    def extractor(self):
        """Create a concrete extractor for testing."""
        return ConcreteOCRExtractor()

    def test_extract_returns_ocr_output(self, extractor):
        """Test that extract returns OCROutput."""
        img = Image.new("RGB", (100, 100))
        result = extractor.extract(img)

        assert isinstance(result, OCROutput)

    def test_extract_preserves_image_dimensions(self, extractor):
        """Test that extract correctly reports image dimensions."""
        sizes = [(100, 200), (500, 300), (1920, 1080)]

        for width, height in sizes:
            img = Image.new("RGB", (width, height))
            result = extractor.extract(img)

            assert result.image_width == width
            assert result.image_height == height

    def test_extract_accepts_all_input_types(self, extractor, tmp_path):
        """Test that extract accepts all supported input types."""
        # PIL Image
        pil_img = Image.new("RGB", (100, 100))
        result = extractor.extract(pil_img)
        assert isinstance(result, OCROutput)

        # Numpy array
        np_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = extractor.extract(np_img)
        assert isinstance(result, OCROutput)

        # String path
        img_path = tmp_path / "test.png"
        pil_img.save(img_path)
        result = extractor.extract(str(img_path))
        assert isinstance(result, OCROutput)

        # Path object
        result = extractor.extract(img_path)
        assert isinstance(result, OCROutput)
