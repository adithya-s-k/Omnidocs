"""
Tests for layout extraction Pydantic models.
"""

import pytest
from pydantic import ValidationError

from omnidocs.tasks.layout_extraction.models import (
    DOCLAYOUT_YOLO_CLASS_NAMES,
    DOCLAYOUT_YOLO_MAPPING,
    RTDETR_MAPPING,
    BoundingBox,
    LayoutBox,
    LayoutLabel,
    LayoutOutput,
)


class TestLayoutLabel:
    """Tests for LayoutLabel enum."""

    def test_layout_label_values(self):
        """Test that LayoutLabel has expected values."""
        assert LayoutLabel.TITLE.value == "title"
        assert LayoutLabel.TEXT.value == "text"
        assert LayoutLabel.FIGURE.value == "figure"
        assert LayoutLabel.TABLE.value == "table"
        assert LayoutLabel.FORMULA.value == "formula"
        assert LayoutLabel.CAPTION.value == "caption"

    def test_layout_label_is_string_enum(self):
        """Test that LayoutLabel can be used as string."""
        assert str(LayoutLabel.TITLE) == "LayoutLabel.TITLE"
        assert LayoutLabel.TITLE.value == "title"


class TestLabelMapping:
    """Tests for LabelMapping class."""

    def test_doclayout_yolo_mapping(self):
        """Test DocLayout-YOLO label mapping."""
        mapping = DOCLAYOUT_YOLO_MAPPING

        assert mapping.to_standard("title") == LayoutLabel.TITLE
        assert mapping.to_standard("plain_text") == LayoutLabel.TEXT
        assert mapping.to_standard("figure") == LayoutLabel.FIGURE
        assert mapping.to_standard("table") == LayoutLabel.TABLE
        assert mapping.to_standard("isolate_formula") == LayoutLabel.FORMULA
        assert mapping.to_standard("figure_caption") == LayoutLabel.CAPTION

    def test_rtdetr_mapping(self):
        """Test RT-DETR label mapping."""
        mapping = RTDETR_MAPPING

        assert mapping.to_standard("title") == LayoutLabel.TITLE
        assert mapping.to_standard("text") == LayoutLabel.TEXT
        assert mapping.to_standard("picture") == LayoutLabel.FIGURE
        assert mapping.to_standard("table") == LayoutLabel.TABLE
        assert mapping.to_standard("formula") == LayoutLabel.FORMULA
        assert mapping.to_standard("list-item") == LayoutLabel.LIST

    def test_unknown_label_returns_unknown(self):
        """Test that unknown labels return UNKNOWN."""
        mapping = DOCLAYOUT_YOLO_MAPPING
        assert mapping.to_standard("nonexistent_label") == LayoutLabel.UNKNOWN

    def test_case_insensitive_mapping(self):
        """Test that mapping is case insensitive."""
        mapping = DOCLAYOUT_YOLO_MAPPING
        assert mapping.to_standard("TITLE") == LayoutLabel.TITLE
        assert mapping.to_standard("Title") == LayoutLabel.TITLE
        assert mapping.to_standard("title") == LayoutLabel.TITLE

    def test_supported_labels(self):
        """Test getting supported labels."""
        mapping = DOCLAYOUT_YOLO_MAPPING
        labels = mapping.supported_labels
        assert "title" in labels
        assert "plain_text" in labels


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

    def test_bounding_box_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=20, x2=30, y2=40, extra_field=100)


class TestLayoutBox:
    """Tests for LayoutBox model."""

    def test_create_layout_box(self):
        """Test creating a layout box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        box = LayoutBox(
            label=LayoutLabel.TITLE,
            bbox=bbox,
            confidence=0.95,
        )
        assert box.label == LayoutLabel.TITLE
        assert box.confidence == 0.95
        assert box.bbox.x1 == 10

    def test_layout_box_with_class_id(self):
        """Test layout box with optional class_id."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        box = LayoutBox(
            label=LayoutLabel.TEXT,
            bbox=bbox,
            confidence=0.8,
            class_id=1,
            original_label="plain_text",
        )
        assert box.class_id == 1
        assert box.original_label == "plain_text"

    def test_layout_box_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)

        with pytest.raises(ValidationError):
            LayoutBox(label=LayoutLabel.TEXT, bbox=bbox, confidence=1.5)

        with pytest.raises(ValidationError):
            LayoutBox(label=LayoutLabel.TEXT, bbox=bbox, confidence=-0.1)

    def test_layout_box_to_dict(self):
        """Test converting layout box to dict."""
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        box = LayoutBox(
            label=LayoutLabel.TITLE,
            bbox=bbox,
            confidence=0.9,
            class_id=0,
            original_label="title",
        )
        d = box.to_dict()
        assert d["label"] == "title"
        assert d["bbox"] == [10, 20, 30, 40]
        assert d["confidence"] == 0.9
        assert d["class_id"] == 0
        assert d["original_label"] == "title"


class TestLayoutOutput:
    """Tests for LayoutOutput model."""

    def test_create_layout_output(self):
        """Test creating layout output."""
        output = LayoutOutput(
            bboxes=[],
            image_width=1024,
            image_height=768,
        )
        assert output.image_width == 1024
        assert output.image_height == 768
        assert output.element_count == 0

    def test_layout_output_with_boxes(self):
        """Test layout output with detected boxes."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=200),
                confidence=0.85,
            ),
        ]
        output = LayoutOutput(
            bboxes=boxes,
            image_width=800,
            image_height=600,
            model_name="TestModel",
        )
        assert output.element_count == 2
        assert output.model_name == "TestModel"

    def test_layout_output_labels_found(self):
        """Test getting unique labels found."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=200),
                confidence=0.85,
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=210, x2=100, y2=300),
                confidence=0.8,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=800, image_height=600)
        labels = output.labels_found
        assert "title" in labels
        assert "text" in labels
        assert len(labels) == 2

    def test_layout_output_filter_by_label(self):
        """Test filtering boxes by label."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=200),
                confidence=0.85,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=800, image_height=600)

        title_boxes = output.filter_by_label(LayoutLabel.TITLE)
        assert len(title_boxes) == 1
        assert title_boxes[0].label == LayoutLabel.TITLE

    def test_layout_output_filter_by_confidence(self):
        """Test filtering boxes by confidence."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=200),
                confidence=0.5,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=800, image_height=600)

        high_conf_boxes = output.filter_by_confidence(0.7)
        assert len(high_conf_boxes) == 1
        assert high_conf_boxes[0].confidence >= 0.7

    def test_layout_output_sort_by_position(self):
        """Test sorting boxes by position."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=200, x2=100, y2=300),
                confidence=0.8,
            ),
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=800, image_height=600)
        sorted_output = output.sort_by_position()

        # First box should be the title (y1=0)
        assert sorted_output.bboxes[0].label == LayoutLabel.TITLE
        assert sorted_output.bboxes[1].label == LayoutLabel.TEXT

    def test_layout_output_to_dict(self):
        """Test converting layout output to dict."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
        ]
        output = LayoutOutput(
            bboxes=boxes,
            image_width=800,
            image_height=600,
            model_name="TestModel",
        )
        d = output.to_dict()

        assert d["image_width"] == 800
        assert d["image_height"] == 600
        assert d["model_name"] == "TestModel"
        assert d["element_count"] == 1
        assert len(d["bboxes"]) == 1
        assert d["labels_found"] == ["title"]

    def test_layout_save_json(self, tmp_path):
        """Test saving LayoutOutput to JSON file."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
        ]
        output = LayoutOutput(
            bboxes=boxes,
            image_width=800,
            image_height=600,
            model_name="TestModel",
        )
        json_path = tmp_path / "test_layout_output.json"
        output.save_json(json_path)

        # Read back the file and check contents
        with open(json_path, "r") as f:
            content = f.read()
            assert "image_width" in content
            assert "image_height" in content
            assert "model_name" in content
            assert "bboxes" in content

    def test_layout_load_json(self, tmp_path):
        """Test loading LayoutOutput from JSON file."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                confidence=0.9,
            ),
        ]
        output = LayoutOutput(
            bboxes=boxes,
            image_width=800,
            image_height=600,
            model_name="TestModel",
        )
        json_path = tmp_path / "test_layout_output.json"
        output.save_json(json_path)

        new_output = LayoutOutput.load_json(json_path)

        assert new_output.image_width == output.image_width
        assert new_output.image_height == output.image_height
        assert new_output.model_name == output.model_name
        assert new_output.element_count == output.element_count
        assert len(new_output.bboxes) == len(output.bboxes)


class TestDocLayoutYOLOClassNames:
    """Tests for DocLayout-YOLO class names mapping."""

    def test_class_names_mapping(self):
        """Test that class IDs map to correct names."""
        assert DOCLAYOUT_YOLO_CLASS_NAMES[0] == "title"
        assert DOCLAYOUT_YOLO_CLASS_NAMES[1] == "plain_text"
        assert DOCLAYOUT_YOLO_CLASS_NAMES[3] == "figure"
        assert DOCLAYOUT_YOLO_CLASS_NAMES[5] == "table"
        assert DOCLAYOUT_YOLO_CLASS_NAMES[8] == "isolate_formula"

    def test_all_class_ids_present(self):
        """Test that all expected class IDs are present."""
        expected_ids = range(10)
        for i in expected_ids:
            assert i in DOCLAYOUT_YOLO_CLASS_NAMES


class TestBoundingBoxNormalization:
    """Tests for BoundingBox normalization methods."""

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

        # Start with normalized coords (0-1024 range)
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

    def test_normalized_size_constant(self):
        """Test that NORMALIZED_SIZE is 1024."""
        from omnidocs.tasks.layout_extraction.models import NORMALIZED_SIZE

        assert NORMALIZED_SIZE == 1024


class TestLayoutBoxNormalization:
    """Tests for LayoutBox normalization methods."""

    def test_get_normalized_bbox(self):
        """Test getting normalized bbox from LayoutBox."""
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        box = LayoutBox(label=LayoutLabel.TITLE, bbox=bbox, confidence=0.9)

        normalized = box.get_normalized_bbox(image_width=1000, image_height=1000)

        # 100/1000*1024 = 102.4
        assert normalized.x1 == pytest.approx(102.4)
        assert normalized.y1 == pytest.approx(102.4)
        # 200/1000*1024 = 204.8
        assert normalized.x2 == pytest.approx(204.8)
        assert normalized.y2 == pytest.approx(204.8)


class TestLayoutOutputNormalization:
    """Tests for LayoutOutput normalization methods."""

    def test_get_normalized_bboxes(self):
        """Test getting all normalized bboxes from LayoutOutput."""
        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=500, y2=100),
                confidence=0.9,
                class_id=0,
                original_label="title",
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=0, y1=150, x2=500, y2=400),
                confidence=0.85,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=1000, image_height=800)

        normalized = output.get_normalized_bboxes()

        assert len(normalized) == 2

        # First box: title
        assert normalized[0]["label"] == "title"
        assert normalized[0]["bbox"][0] == pytest.approx(0.0)  # x1
        assert normalized[0]["bbox"][1] == pytest.approx(0.0)  # y1
        assert normalized[0]["bbox"][2] == pytest.approx(512.0)  # x2: 500/1000*1024
        assert normalized[0]["bbox"][3] == pytest.approx(128.0)  # y2: 100/800*1024
        assert normalized[0]["confidence"] == 0.9
        assert normalized[0]["class_id"] == 0
        assert normalized[0]["original_label"] == "title"

        # Second box: text
        assert normalized[1]["label"] == "text"
        assert normalized[1]["bbox"][1] == pytest.approx(192.0)  # y1: 150/800*1024


class TestLabelColors:
    """Tests for label color constants."""

    def test_label_colors_defined(self):
        """Test that colors are defined for all standard labels."""
        from omnidocs.tasks.layout_extraction.models import LABEL_COLORS

        expected_labels = [
            LayoutLabel.TITLE,
            LayoutLabel.TEXT,
            LayoutLabel.LIST,
            LayoutLabel.FIGURE,
            LayoutLabel.TABLE,
            LayoutLabel.CAPTION,
            LayoutLabel.FORMULA,
            LayoutLabel.FOOTNOTE,
            LayoutLabel.PAGE_HEADER,
            LayoutLabel.PAGE_FOOTER,
            LayoutLabel.ABANDON,
            LayoutLabel.UNKNOWN,
        ]

        for label in expected_labels:
            assert label in LABEL_COLORS
            assert LABEL_COLORS[label].startswith("#")  # Valid hex color

    def test_label_colors_are_hex(self):
        """Test that all colors are valid hex format."""
        import re

        from omnidocs.tasks.layout_extraction.models import LABEL_COLORS

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for label, color in LABEL_COLORS.items():
            assert hex_pattern.match(color), f"Invalid color {color} for {label}"


class TestLayoutOutputVisualize:
    """Tests for LayoutOutput visualization."""

    def test_visualize_returns_image(self):
        """Test that visualize returns a PIL Image."""
        from PIL import Image

        # Create test image
        test_image = Image.new("RGB", (800, 600), color="white")

        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=50, y1=30, x2=400, y2=70),
                confidence=0.95,
            ),
            LayoutBox(
                label=LayoutLabel.TEXT,
                bbox=BoundingBox(x1=50, y1=100, x2=400, y2=300),
                confidence=0.88,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=800, image_height=600)

        result = output.visualize(test_image)

        assert isinstance(result, Image.Image)
        assert result.size == (800, 600)

    def test_visualize_does_not_modify_original(self):
        """Test that visualize doesn't modify the original image."""
        import numpy as np
        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="white")
        original_pixels = np.array(test_image).copy()

        boxes = [
            LayoutBox(
                label=LayoutLabel.TITLE,
                bbox=BoundingBox(x1=10, y1=10, x2=90, y2=90),
                confidence=0.9,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=100, image_height=100)

        _ = output.visualize(test_image)

        # Original should be unchanged
        assert np.array_equal(np.array(test_image), original_pixels)

    def test_visualize_saves_to_path(self, tmp_path):
        """Test that visualize can save to a file path."""
        from PIL import Image

        test_image = Image.new("RGB", (200, 200), color="white")
        output_path = tmp_path / "viz_output.png"

        boxes = [
            LayoutBox(
                label=LayoutLabel.TABLE,
                bbox=BoundingBox(x1=20, y1=20, x2=180, y2=180),
                confidence=0.92,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=200, image_height=200)

        output.visualize(test_image, output_path=output_path)

        assert output_path.exists()
        saved_image = Image.open(output_path)
        assert saved_image.size == (200, 200)

    def test_visualize_with_options(self):
        """Test visualize with different options."""
        from PIL import Image

        test_image = Image.new("RGB", (400, 300), color="white")

        boxes = [
            LayoutBox(
                label=LayoutLabel.FIGURE,
                bbox=BoundingBox(x1=50, y1=50, x2=350, y2=250),
                confidence=0.85,
            ),
        ]
        output = LayoutOutput(bboxes=boxes, image_width=400, image_height=300)

        # Test with different options
        result = output.visualize(
            test_image,
            show_labels=False,
            show_confidence=False,
            line_width=5,
        )

        assert isinstance(result, Image.Image)

    def test_visualize_empty_boxes(self):
        """Test visualize with no bounding boxes."""
        from PIL import Image

        test_image = Image.new("RGB", (100, 100), color="white")
        output = LayoutOutput(bboxes=[], image_width=100, image_height=100)

        result = output.visualize(test_image)

        assert isinstance(result, Image.Image)
