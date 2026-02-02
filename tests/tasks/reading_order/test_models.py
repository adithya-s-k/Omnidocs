"""
Tests for reading order Pydantic models.
"""

import pytest
from pydantic import ValidationError

from omnidocs.tasks.reading_order.models import (
    NORMALIZED_SIZE,
    BoundingBox,
    ElementType,
    OrderedElement,
    ReadingOrderOutput,
)


class TestElementType:
    """Tests for ElementType enum."""

    def test_element_type_values(self):
        """Test that ElementType has expected values."""
        assert ElementType.TITLE.value == "title"
        assert ElementType.TEXT.value == "text"
        assert ElementType.LIST.value == "list"
        assert ElementType.FIGURE.value == "figure"
        assert ElementType.TABLE.value == "table"
        assert ElementType.CAPTION.value == "caption"
        assert ElementType.FORMULA.value == "formula"
        assert ElementType.FOOTNOTE.value == "footnote"
        assert ElementType.PAGE_HEADER.value == "page_header"
        assert ElementType.PAGE_FOOTER.value == "page_footer"
        assert ElementType.CODE.value == "code"
        assert ElementType.OTHER.value == "other"

    def test_element_type_is_string_enum(self):
        """Test that ElementType can be used as string."""
        assert ElementType.TITLE.value == "title"


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
        assert bbox.center == (50, 25)

    def test_bounding_box_to_list(self):
        """Test converting bounding box to list."""
        bbox = BoundingBox(x1=10, y1=20, x2=30, y2=40)
        assert bbox.to_list() == [10, 20, 30, 40]

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

    def test_to_normalized(self):
        """Test converting to normalized coordinates (0-1024)."""
        bbox = BoundingBox(x1=100, y1=50, x2=500, y2=300)
        normalized = bbox.to_normalized(image_width=1000, image_height=800)

        assert normalized.x1 == pytest.approx(102.4)
        assert normalized.y1 == pytest.approx(64.0)
        assert normalized.x2 == pytest.approx(512.0)
        assert normalized.y2 == pytest.approx(384.0)


class TestOrderedElement:
    """Tests for OrderedElement model."""

    def test_create_ordered_element(self):
        """Test creating an ordered element."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        elem = OrderedElement(
            index=0,
            element_type=ElementType.TITLE,
            bbox=bbox,
            text="Document Title",
        )
        assert elem.index == 0
        assert elem.element_type == ElementType.TITLE
        assert elem.text == "Document Title"
        assert elem.confidence == 1.0  # Default
        assert elem.page_no == 0  # Default

    def test_ordered_element_with_all_fields(self):
        """Test ordered element with all fields."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        elem = OrderedElement(
            index=5,
            element_type=ElementType.TEXT,
            bbox=bbox,
            text="Paragraph text",
            confidence=0.95,
            page_no=2,
            original_id=42,
        )
        assert elem.index == 5
        assert elem.confidence == 0.95
        assert elem.page_no == 2
        assert elem.original_id == 42

    def test_ordered_element_index_validation(self):
        """Test that index must be non-negative."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        with pytest.raises(ValidationError):
            OrderedElement(
                index=-1,
                element_type=ElementType.TEXT,
                bbox=bbox,
            )

    def test_ordered_element_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)

        with pytest.raises(ValidationError):
            OrderedElement(
                index=0,
                element_type=ElementType.TEXT,
                bbox=bbox,
                confidence=1.5,
            )

        with pytest.raises(ValidationError):
            OrderedElement(
                index=0,
                element_type=ElementType.TEXT,
                bbox=bbox,
                confidence=-0.1,
            )

    def test_ordered_element_to_dict(self):
        """Test converting ordered element to dict."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        elem = OrderedElement(
            index=3,
            element_type=ElementType.FIGURE,
            bbox=bbox,
            text="Figure description",
            confidence=0.9,
            page_no=1,
            original_id=7,
        )
        d = elem.to_dict()
        assert d["index"] == 3
        assert d["element_type"] == "figure"
        assert d["bbox"] == [10, 20, 100, 50]
        assert d["text"] == "Figure description"
        assert d["confidence"] == 0.9
        assert d["page_no"] == 1
        assert d["original_id"] == 7


class TestReadingOrderOutput:
    """Tests for ReadingOrderOutput model."""

    def test_create_reading_order_output(self):
        """Test creating reading order output."""
        output = ReadingOrderOutput(
            ordered_elements=[],
            image_width=800,
            image_height=600,
        )
        assert output.image_width == 800
        assert output.image_height == 600
        assert output.element_count == 0

    def test_reading_order_output_with_elements(self):
        """Test reading order output with elements."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.TITLE,
                bbox=BoundingBox(x1=50, y1=50, x2=550, y2=100),
                text="Title",
            ),
            OrderedElement(
                index=1,
                element_type=ElementType.TEXT,
                bbox=BoundingBox(x1=50, y1=120, x2=550, y2=300),
                text="Main content paragraph.",
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            image_width=600,
            image_height=400,
            model_name="TestPredictor",
        )
        assert output.element_count == 2
        assert output.model_name == "TestPredictor"

    def test_reading_order_output_get_full_text(self):
        """Test getting full text in reading order."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                text="Document Title",
            ),
            OrderedElement(
                index=1,
                element_type=ElementType.TEXT,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=100),
                text="First paragraph.",
            ),
            OrderedElement(
                index=2,
                element_type=ElementType.PAGE_HEADER,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=20),
                text="Page Header",
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            image_width=100,
            image_height=200,
        )

        full_text = output.get_full_text()
        # Page header should be excluded
        assert "Document Title" in full_text
        assert "First paragraph." in full_text
        assert "Page Header" not in full_text

    def test_reading_order_output_get_elements_by_type(self):
        """Test filtering elements by type."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                text="Title",
            ),
            OrderedElement(
                index=1,
                element_type=ElementType.TABLE,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=200),
                text="Table content",
                original_id=1,
            ),
            OrderedElement(
                index=2,
                element_type=ElementType.CAPTION,
                bbox=BoundingBox(x1=0, y1=210, x2=100, y2=230),
                text="Table 1: Description",
                original_id=2,
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            image_width=100,
            image_height=300,
        )

        tables = output.get_elements_by_type(ElementType.TABLE)
        assert len(tables) == 1
        assert tables[0].text == "Table content"

        captions = output.get_elements_by_type(ElementType.CAPTION)
        assert len(captions) == 1

    def test_reading_order_output_caption_map(self):
        """Test caption map functionality."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.FIGURE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                text="Figure",
                original_id=0,
            ),
            OrderedElement(
                index=1,
                element_type=ElementType.CAPTION,
                bbox=BoundingBox(x1=0, y1=110, x2=100, y2=130),
                text="Figure 1: Description",
                original_id=1,
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            caption_map={0: [1]},  # Figure 0 has caption 1
            image_width=100,
            image_height=150,
        )

        captions = output.get_captions_for(0)
        assert len(captions) == 1
        assert captions[0].text == "Figure 1: Description"

    def test_reading_order_output_footnote_map(self):
        """Test footnote map functionality."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.TEXT,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                text="Main text",
                original_id=0,
            ),
            OrderedElement(
                index=1,
                element_type=ElementType.FOOTNOTE,
                bbox=BoundingBox(x1=0, y1=500, x2=100, y2=520),
                text="1. Footnote text",
                original_id=1,
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            footnote_map={0: [1]},  # Text 0 has footnote 1
            image_width=100,
            image_height=600,
        )

        footnotes = output.get_footnotes_for(0)
        assert len(footnotes) == 1
        assert footnotes[0].text == "1. Footnote text"

    def test_reading_order_output_to_dict(self):
        """Test converting reading order output to dict."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                text="Title",
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            caption_map={1: [2]},
            footnote_map={3: [4]},
            merge_map={5: [6, 7]},
            image_width=800,
            image_height=600,
        )
        d = output.to_dict()

        assert d["image_width"] == 800
        assert d["image_height"] == 600
        assert d["element_count"] == 1
        assert len(d["ordered_elements"]) == 1
        assert d["caption_map"] == {1: [2]}
        assert d["footnote_map"] == {3: [4]}
        assert d["merge_map"] == {5: [6, 7]}

    def test_reading_order_output_save_load_json(self, tmp_path):
        """Test saving and loading reading order output from JSON."""
        elements = [
            OrderedElement(
                index=0,
                element_type=ElementType.TITLE,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=50),
                text="Title",
                original_id=0,
            ),
            OrderedElement(
                index=1,
                element_type=ElementType.TEXT,
                bbox=BoundingBox(x1=0, y1=60, x2=100, y2=150),
                text="Content",
                original_id=1,
            ),
        ]
        output = ReadingOrderOutput(
            ordered_elements=elements,
            caption_map={2: [3]},
            image_width=100,
            image_height=200,
            model_name="TestPredictor",
        )

        json_path = tmp_path / "reading_order.json"
        output.save_json(json_path)

        assert json_path.exists()

        loaded = ReadingOrderOutput.load_json(json_path)
        assert loaded.element_count == output.element_count
        assert loaded.image_width == output.image_width
        assert loaded.image_height == output.image_height
        assert loaded.model_name == output.model_name
        assert loaded.ordered_elements[0].text == "Title"
        assert loaded.ordered_elements[1].text == "Content"


class TestNormalizedSizeConstant:
    """Tests for NORMALIZED_SIZE constant."""

    def test_normalized_size_is_1024(self):
        """Test that NORMALIZED_SIZE is 1024."""
        assert NORMALIZED_SIZE == 1024
