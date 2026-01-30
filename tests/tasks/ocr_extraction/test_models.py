"""
Tests for OCR extraction Pydantic models.
"""

import pytest
from pydantic import ValidationError

from omnidocs.tasks.ocr_extraction.models import (
    NORMALIZED_SIZE,
    BoundingBox,
    OCRGranularity,
    OCROutput,
    TextBlock,
)


class TestOCRGranularity:
    """Tests for OCRGranularity enum."""

    def test_granularity_values(self):
        """Test that OCRGranularity has expected values."""
        assert OCRGranularity.CHARACTER.value == "character"
        assert OCRGranularity.WORD.value == "word"
        assert OCRGranularity.LINE.value == "line"
        assert OCRGranularity.BLOCK.value == "block"

    def test_granularity_is_string_enum(self):
        """Test that OCRGranularity can be used as string."""
        assert OCRGranularity.WORD.value == "word"


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_create_bounding_box(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200

    def test_bounding_box_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
        assert bbox.center == (50, 25)

    def test_bounding_box_to_list(self):
        """Test converting bounding box to list."""
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        assert bbox.to_list() == [10, 20, 30, 40]

    def test_bounding_box_to_xyxy(self):
        """Test converting bounding box to xyxy tuple."""
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        assert bbox.to_xyxy() == (10, 20, 30, 40)

    def test_bounding_box_to_xywh(self):
        """Test converting bounding box to xywh format."""
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=50)
        assert bbox.to_xywh() == (10, 20, 20, 30)

    def test_bounding_box_from_list(self):
        """Test creating bounding box from list."""
        bbox = BoundingBox.from_list([10, 20, 30, 40])
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 30
        assert bbox.y2 == 40

    def test_bounding_box_from_list_invalid_length(self):
        """Test that invalid list length raises error."""
        with pytest.raises(ValueError, match="Expected 4 coordinates"):
            BoundingBox.from_list([10, 20, 30])

    def test_bounding_box_from_polygon(self):
        """Test creating bounding box from polygon."""
        polygon = [[10, 20], [100, 20], [100, 80], [10, 80]]
        bbox = BoundingBox.from_polygon(polygon)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 80

    def test_bounding_box_from_polygon_rotated(self):
        """Test creating bounding box from rotated polygon."""
        # Rotated rectangle - should get axis-aligned bounding box
        polygon = [[50, 10], [90, 50], [50, 90], [10, 50]]
        bbox = BoundingBox.from_polygon(polygon)
        assert bbox.x1 == 10
        assert bbox.y1 == 10
        assert bbox.x2 == 90
        assert bbox.y2 == 90

    def test_bounding_box_from_polygon_empty(self):
        """Test that empty polygon raises error."""
        with pytest.raises(ValueError, match="Polygon cannot be empty"):
            BoundingBox.from_polygon([])

    def test_bounding_box_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=20, x2=30, y2=40, extra_field=100)

    def test_to_normalized(self):
        """Test converting to normalized coordinates (0-1024)."""
        bbox = BoundingBox(x1=100, y1=50, x2=500, y2=300)
        normalized = bbox.to_normalized(image_width=1000, image_height=800)

        # x: 100/1000*1024 = 102.4, y: 50/800*1024 = 64
        assert normalized.x1 == pytest.approx(102.4)
        assert normalized.y1 == pytest.approx(64.0)
        # x: 500/1000*1024 = 512, y: 300/800*1024 = 384
        assert normalized.x2 == pytest.approx(512.0)
        assert normalized.y2 == pytest.approx(384.0)

    def test_to_absolute(self):
        """Test converting from normalized back to absolute coordinates."""
        normalized_bbox = BoundingBox(x1=102.4, y1=64.0, x2=512.0, y2=384.0)
        absolute = normalized_bbox.to_absolute(image_width=1000, image_height=800)

        assert absolute.x1 == pytest.approx(100.0)
        assert absolute.y1 == pytest.approx(50.0)
        assert absolute.x2 == pytest.approx(500.0)
        assert absolute.y2 == pytest.approx(300.0)

    def test_roundtrip_normalization(self):
        """Test that normalizing and then converting back gives original values."""
        original = BoundingBox(x1=150, y1=200, x2=450, y2=600)
        width, height = 800, 1000

        normalized = original.to_normalized(width, height)
        back_to_absolute = normalized.to_absolute(width, height)

        assert back_to_absolute.x1 == pytest.approx(original.x1)
        assert back_to_absolute.y1 == pytest.approx(original.y1)
        assert back_to_absolute.x2 == pytest.approx(original.x2)
        assert back_to_absolute.y2 == pytest.approx(original.y2)


class TestTextBlock:
    """Tests for TextBlock model."""

    def test_create_text_block(self):
        """Test creating a text block."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=40)
        block = TextBlock(
            text="Hello World",
            bbox=bbox,
            confidence=0.95,
        )
        assert block.text == "Hello World"
        assert block.confidence == 0.95
        assert block.granularity == OCRGranularity.WORD  # Default

    def test_text_block_with_all_fields(self):
        """Test text block with all optional fields."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=40)
        polygon = [[10, 20], [100, 20], [100, 40], [10, 40]]
        block = TextBlock(
            text="Test",
            bbox=bbox,
            confidence=0.9,
            granularity=OCRGranularity.LINE,
            polygon=polygon,
            language="en",
        )
        assert block.granularity == OCRGranularity.LINE
        assert block.polygon == polygon
        assert block.language == "en"

    def test_text_block_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)

        with pytest.raises(ValidationError):
            TextBlock(text="Test", bbox=bbox, confidence=1.5)

        with pytest.raises(ValidationError):
            TextBlock(text="Test", bbox=bbox, confidence=-0.1)

    def test_text_block_to_dict(self):
        """Test converting text block to dict."""
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        block = TextBlock(
            text="Hello",
            bbox=bbox,
            confidence=0.9,
            granularity=OCRGranularity.WORD,
            language="en",
        )
        d = block.to_dict()
        assert d["text"] == "Hello"
        assert d["bbox"] == [10, 20, 30, 40]
        assert d["confidence"] == 0.9
        assert d["granularity"] == "word"
        assert d["language"] == "en"

    def test_text_block_get_normalized_bbox(self):
        """Test getting normalized bbox from TextBlock."""
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        block = TextBlock(text="Test", bbox=bbox, confidence=0.9)

        normalized = block.get_normalized_bbox(image_width=1000, image_height=1000)

        assert normalized.x1 == pytest.approx(102.4)
        assert normalized.y1 == pytest.approx(102.4)


class TestOCROutput:
    """Tests for OCROutput model."""

    def test_create_ocr_output(self):
        """Test creating OCR output."""
        output = OCROutput(
            text_blocks=[],
            full_text="",
            image_width=1024,
            image_height=768,
        )
        assert output.image_width == 1024
        assert output.image_height == 768
        assert output.block_count == 0

    def test_ocr_output_with_blocks(self):
        """Test OCR output with text blocks."""
        blocks = [
            TextBlock(
                text="Hello",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
            TextBlock(
                text="World",
                bbox=BoundingBox(x1=110, y1=0, x2=200, y2=30),
                confidence=0.85,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Hello World",
            image_width=800,
            image_height=600,
            model_name="TestOCR",
        )
        assert output.block_count == 2
        assert output.model_name == "TestOCR"
        assert output.full_text == "Hello World"

    def test_ocr_output_properties(self):
        """Test OCROutput computed properties."""
        blocks = [
            TextBlock(
                text="Hello",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
            TextBlock(
                text="World",
                bbox=BoundingBox(x1=0, y1=40, x2=100, y2=70),
                confidence=0.8,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Hello World",
            image_width=800,
            image_height=600,
        )

        assert output.block_count == 2
        assert output.word_count == 2
        assert output.average_confidence == pytest.approx(0.85)

    def test_ocr_output_filter_by_confidence(self):
        """Test filtering blocks by confidence."""
        blocks = [
            TextBlock(
                text="High",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
            TextBlock(
                text="Low",
                bbox=BoundingBox(x1=0, y1=40, x2=100, y2=70),
                confidence=0.5,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="High Low",
            image_width=800,
            image_height=600,
        )

        high_conf_blocks = output.filter_by_confidence(0.7)
        assert len(high_conf_blocks) == 1
        assert high_conf_blocks[0].text == "High"

    def test_ocr_output_filter_by_granularity(self):
        """Test filtering blocks by granularity."""
        blocks = [
            TextBlock(
                text="Word",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
                granularity=OCRGranularity.WORD,
            ),
            TextBlock(
                text="A full line of text",
                bbox=BoundingBox(x1=0, y1=40, x2=300, y2=70),
                confidence=0.85,
                granularity=OCRGranularity.LINE,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Word A full line of text",
            image_width=800,
            image_height=600,
        )

        word_blocks = output.filter_by_granularity(OCRGranularity.WORD)
        assert len(word_blocks) == 1
        assert word_blocks[0].text == "Word"

        line_blocks = output.filter_by_granularity(OCRGranularity.LINE)
        assert len(line_blocks) == 1
        assert line_blocks[0].text == "A full line of text"

    def test_ocr_output_sort_by_position(self):
        """Test sorting blocks by position."""
        blocks = [
            TextBlock(
                text="Second",
                bbox=BoundingBox(x1=0, y1=100, x2=100, y2=130),
                confidence=0.8,
            ),
            TextBlock(
                text="First",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Second First",
            image_width=800,
            image_height=600,
        )

        sorted_output = output.sort_by_position()
        assert sorted_output.text_blocks[0].text == "First"
        assert sorted_output.text_blocks[1].text == "Second"
        assert sorted_output.full_text == "First Second"

    def test_ocr_output_to_dict(self):
        """Test converting OCR output to dict."""
        blocks = [
            TextBlock(
                text="Hello",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Hello",
            image_width=800,
            image_height=600,
            model_name="TestOCR",
            languages_detected=["en"],
        )
        d = output.to_dict()

        assert d["image_width"] == 800
        assert d["image_height"] == 600
        assert d["model_name"] == "TestOCR"
        assert d["languages_detected"] == ["en"]
        assert d["block_count"] == 1
        assert d["word_count"] == 1
        assert len(d["text_blocks"]) == 1

    def test_ocr_output_get_normalized_blocks(self):
        """Test getting all normalized blocks."""
        blocks = [
            TextBlock(
                text="Test",
                bbox=BoundingBox(x1=0, y1=0, x2=500, y2=100),
                confidence=0.9,
                granularity=OCRGranularity.WORD,
                language="en",
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Test",
            image_width=1000,
            image_height=800,
        )

        normalized = output.get_normalized_blocks()

        assert len(normalized) == 1
        assert normalized[0]["text"] == "Test"
        assert normalized[0]["bbox"][2] == pytest.approx(512.0)  # 500/1000*1024
        assert normalized[0]["bbox"][3] == pytest.approx(128.0)  # 100/800*1024

    def test_ocr_output_save_json(self, tmp_path):
        """Test saving OCROutput to JSON file."""
        blocks = [
            TextBlock(
                text="Hello",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Hello",
            image_width=800,
            image_height=600,
            model_name="TestOCR",
        )
        json_path = tmp_path / "test_ocr_output.json"
        output.save_json(json_path)

        assert json_path.exists()
        with open(json_path, "r") as f:
            content = f.read()
            assert "image_width" in content
            assert "text_blocks" in content

    def test_ocr_output_load_json(self, tmp_path):
        """Test loading OCROutput from JSON file."""
        blocks = [
            TextBlock(
                text="Hello",
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=30),
                confidence=0.9,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Hello",
            image_width=800,
            image_height=600,
            model_name="TestOCR",
        )
        json_path = tmp_path / "test_ocr_output.json"
        output.save_json(json_path)

        loaded = OCROutput.load_json(json_path)

        assert loaded.image_width == output.image_width
        assert loaded.image_height == output.image_height
        assert loaded.model_name == output.model_name
        assert loaded.block_count == output.block_count
        assert loaded.text_blocks[0].text == output.text_blocks[0].text


class TestOCROutputVisualize:
    """Tests for OCROutput visualization."""

    def test_visualize_returns_image(self):
        """Test that visualize returns a PIL Image."""
        from PIL import Image

        test_image = Image.new("RGB", (800, 600), color="white")

        blocks = [
            TextBlock(
                text="Hello World",
                bbox=BoundingBox(x1=50, y1=30, x2=200, y2=60),
                confidence=0.95,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Hello World",
            image_width=800,
            image_height=600,
        )

        result = output.visualize(test_image)

        assert isinstance(result, Image.Image)
        assert result.size == (800, 600)

    def test_visualize_does_not_modify_original(self):
        """Test that visualize doesn't modify the original image."""
        import numpy as np
        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="white")
        original_pixels = np.array(test_image).copy()

        blocks = [
            TextBlock(
                text="Test",
                bbox=BoundingBox(x1=10, y1=10, x2=90, y2=90),
                confidence=0.9,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Test",
            image_width=100,
            image_height=100,
        )

        _ = output.visualize(test_image)

        assert np.array_equal(np.array(test_image), original_pixels)

    def test_visualize_saves_to_path(self, tmp_path):
        """Test that visualize can save to a file path."""
        from PIL import Image

        test_image = Image.new("RGB", (200, 200), color="white")
        output_path = tmp_path / "viz_output.png"

        blocks = [
            TextBlock(
                text="Test",
                bbox=BoundingBox(x1=20, y1=20, x2=180, y2=60),
                confidence=0.92,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Test",
            image_width=200,
            image_height=200,
        )

        output.visualize(test_image, output_path=output_path)

        assert output_path.exists()
        saved_image = Image.open(output_path)
        assert saved_image.size == (200, 200)

    def test_visualize_with_polygon(self):
        """Test visualize with polygon coordinates."""
        from PIL import Image

        test_image = Image.new("RGB", (200, 200), color="white")

        polygon = [[20, 20], [180, 20], [180, 60], [20, 60]]
        blocks = [
            TextBlock(
                text="Test",
                bbox=BoundingBox.from_polygon(polygon),
                confidence=0.9,
                polygon=polygon,
            ),
        ]
        output = OCROutput(
            text_blocks=blocks,
            full_text="Test",
            image_width=200,
            image_height=200,
        )

        result = output.visualize(test_image)
        assert isinstance(result, Image.Image)

    def test_visualize_empty_blocks(self):
        """Test visualize with no text blocks."""
        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="white")
        output = OCROutput(
            text_blocks=[],
            full_text="",
            image_width=100,
            image_height=100,
        )

        result = output.visualize(test_image)
        assert isinstance(result, Image.Image)


class TestNormalizedSizeConstant:
    """Tests for NORMALIZED_SIZE constant."""

    def test_normalized_size_is_1024(self):
        """Test that NORMALIZED_SIZE is 1024."""
        assert NORMALIZED_SIZE == 1024
