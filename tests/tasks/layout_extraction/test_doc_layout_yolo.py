"""
Tests for DocLayoutYOLO layout extractor.
"""

import os
import tempfile

import pytest
from pydantic import ValidationError

from omnidocs.tasks.layout_extraction import DocLayoutYOLO, DocLayoutYOLOConfig
from omnidocs.tasks.layout_extraction.models import LayoutLabel, LayoutOutput


class TestDocLayoutYOLOConfig:
    """Tests for DocLayoutYOLOConfig."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = DocLayoutYOLOConfig()
        assert config.device == "cuda"
        assert config.model_path is None
        assert config.img_size == 1024
        assert config.confidence == 0.25

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = DocLayoutYOLOConfig(
            device="cpu",
            model_path="/custom/path",
            img_size=640,
            confidence=0.5,
        )
        assert config.device == "cpu"
        assert config.model_path == "/custom/path"
        assert config.img_size == 640
        assert config.confidence == 0.5

    def test_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            DocLayoutYOLOConfig(confidence=1.5)

        with pytest.raises(ValidationError):
            DocLayoutYOLOConfig(confidence=-0.1)

    def test_img_size_validation(self):
        """Test that img_size must be within valid range."""
        with pytest.raises(ValidationError):
            DocLayoutYOLOConfig(img_size=100)  # Too small

        with pytest.raises(ValidationError):
            DocLayoutYOLOConfig(img_size=3000)  # Too large

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            DocLayoutYOLOConfig(unknown_param="value")


class TestDocLayoutYOLOModelPath:
    """Tests for model path resolution."""

    def test_model_path_from_env(self):
        """Test model path resolution from environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["OMNIDOCS_MODELS_DIR"] = tmpdir

            config = DocLayoutYOLOConfig(device="cpu")
            # Note: We're not initializing the model here to avoid download
            # Just testing that config is created properly

            assert config.model_path is None  # Config stores None
            # Path resolution happens in the extractor

            del os.environ["OMNIDOCS_MODELS_DIR"]

    def test_explicit_model_path(self):
        """Test explicit model path in config."""
        config = DocLayoutYOLOConfig(
            device="cpu",
            model_path="/my/custom/path/to/model",
        )
        assert config.model_path == "/my/custom/path/to/model"


@pytest.mark.slow
class TestDocLayoutYOLOExtractor:
    """Integration tests for DocLayoutYOLO extractor.

    These tests require the model to be downloaded, so they're marked as slow.
    """

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        config = DocLayoutYOLOConfig(
            device="cpu",  # Use CPU for testing
            confidence=0.25,
        )
        return DocLayoutYOLO(config=config)

    def test_extract_returns_layout_output(self, extractor, sample_image):
        """Test that extract returns LayoutOutput."""
        result = extractor.extract(sample_image)

        assert isinstance(result, LayoutOutput)
        assert result.image_width == sample_image.width
        assert result.image_height == sample_image.height
        assert result.model_name == "DocLayout-YOLO"

    def test_extract_from_path(self, extractor, sample_image, tmp_path):
        """Test extracting from file path."""
        img_path = tmp_path / "test_image.png"
        sample_image.save(img_path)

        result = extractor.extract(str(img_path))

        assert isinstance(result, LayoutOutput)
        assert result.image_width == sample_image.width

    def test_extract_from_numpy(self, extractor, sample_image):
        """Test extracting from numpy array."""
        import numpy as np

        np_image = np.array(sample_image)
        result = extractor.extract(np_image)

        assert isinstance(result, LayoutOutput)

    def test_boxes_have_valid_labels(self, extractor, sample_document_image):
        """Test that detected boxes have valid standardized labels."""
        result = extractor.extract(sample_document_image)

        for box in result.bboxes:
            assert isinstance(box.label, LayoutLabel)
            assert box.confidence >= 0.0
            assert box.confidence <= 1.0

    def test_boxes_have_valid_coordinates(self, extractor, sample_document_image):
        """Test that detected boxes have valid coordinates."""
        result = extractor.extract(sample_document_image)

        for box in result.bboxes:
            # Coordinates should be within image bounds
            assert box.bbox.x1 >= 0
            assert box.bbox.y1 >= 0
            assert box.bbox.x2 <= result.image_width
            assert box.bbox.y2 <= result.image_height
            # x2 > x1 and y2 > y1
            assert box.bbox.x2 > box.bbox.x1
            assert box.bbox.y2 > box.bbox.y1

    def test_boxes_sorted_by_position(self, extractor, sample_document_image):
        """Test that boxes are sorted by y-coordinate (reading order)."""
        result = extractor.extract(sample_document_image)

        if len(result.bboxes) > 1:
            for i in range(len(result.bboxes) - 1):
                current_y = result.bboxes[i].bbox.y1
                next_y = result.bboxes[i + 1].bbox.y1
                # Current box should be above or at same level as next
                assert current_y <= next_y or abs(current_y - next_y) < 50  # Allow some tolerance


class TestDocLayoutYOLOEdgeCases:
    """Edge case tests for DocLayoutYOLO."""

    def test_invalid_image_path(self):
        """Test that invalid image path raises error."""
        config = DocLayoutYOLOConfig(device="cpu")

        # Skip if model not available
        try:
            extractor = DocLayoutYOLO(config=config)
        except Exception:
            pytest.skip("Model not available for testing")

        with pytest.raises(FileNotFoundError):
            extractor.extract("/nonexistent/path/to/image.png")

    def test_invalid_image_type(self):
        """Test that invalid image type raises error."""
        config = DocLayoutYOLOConfig(device="cpu")

        # Skip if model not available
        try:
            extractor = DocLayoutYOLO(config=config)
        except Exception:
            pytest.skip("Model not available for testing")

        with pytest.raises(ValueError):
            extractor.extract({"invalid": "type"})
