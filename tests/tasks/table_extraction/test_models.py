"""
Tests for table extraction Pydantic models.
"""

import pytest
from pydantic import ValidationError

from omnidocs.tasks.table_extraction.models import (
    NORMALIZED_SIZE,
    BoundingBox,
    CellType,
    TableCell,
    TableOutput,
)


class TestCellType:
    """Tests for CellType enum."""

    def test_cell_type_values(self):
        """Test that CellType has expected values."""
        assert CellType.DATA.value == "data"
        assert CellType.COLUMN_HEADER.value == "column_header"
        assert CellType.ROW_HEADER.value == "row_header"
        assert CellType.SECTION_ROW.value == "section_row"

    def test_cell_type_is_string_enum(self):
        """Test that CellType can be used as string."""
        assert CellType.DATA.value == "data"


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

    def test_bounding_box_from_ltrb(self):
        """Test creating bounding box from ltrb."""
        bbox = BoundingBox.from_ltrb(left=10, top=20, right=100, bottom=80)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 80

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


class TestTableCell:
    """Tests for TableCell model."""

    def test_create_table_cell(self):
        """Test creating a table cell."""
        cell = TableCell(row=0, col=0, text="Hello")
        assert cell.row == 0
        assert cell.col == 0
        assert cell.text == "Hello"
        assert cell.row_span == 1
        assert cell.col_span == 1
        assert cell.cell_type == CellType.DATA

    def test_table_cell_with_spans(self):
        """Test table cell with row and column spans."""
        cell = TableCell(row=0, col=0, row_span=2, col_span=3, text="Header")
        assert cell.row_span == 2
        assert cell.col_span == 3
        assert cell.end_row == 2
        assert cell.end_col == 3

    def test_table_cell_with_bbox(self):
        """Test table cell with bounding box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        cell = TableCell(row=0, col=0, text="Cell", bbox=bbox)
        assert cell.bbox is not None
        assert cell.bbox.x1 == 10

    def test_table_cell_header_types(self):
        """Test table cell header type detection."""
        data_cell = TableCell(row=0, col=0, cell_type=CellType.DATA)
        assert not data_cell.is_header

        col_header = TableCell(row=0, col=0, cell_type=CellType.COLUMN_HEADER)
        assert col_header.is_header

        row_header = TableCell(row=0, col=0, cell_type=CellType.ROW_HEADER)
        assert row_header.is_header

    def test_table_cell_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            TableCell(row=0, col=0, confidence=1.5)

        with pytest.raises(ValidationError):
            TableCell(row=0, col=0, confidence=-0.1)

    def test_table_cell_row_col_validation(self):
        """Test that row and col must be non-negative."""
        with pytest.raises(ValidationError):
            TableCell(row=-1, col=0)

        with pytest.raises(ValidationError):
            TableCell(row=0, col=-1)

    def test_table_cell_to_dict(self):
        """Test converting table cell to dict."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        cell = TableCell(
            row=1,
            col=2,
            row_span=1,
            col_span=2,
            text="Test",
            cell_type=CellType.COLUMN_HEADER,
            bbox=bbox,
            confidence=0.95,
        )
        d = cell.to_dict()
        assert d["row"] == 1
        assert d["col"] == 2
        assert d["row_span"] == 1
        assert d["col_span"] == 2
        assert d["text"] == "Test"
        assert d["cell_type"] == "column_header"
        assert d["bbox"] == [10, 20, 100, 50]
        assert d["confidence"] == 0.95


class TestTableOutput:
    """Tests for TableOutput model."""

    def test_create_table_output(self):
        """Test creating table output."""
        output = TableOutput(
            cells=[],
            num_rows=0,
            num_cols=0,
        )
        assert output.num_rows == 0
        assert output.num_cols == 0
        assert output.cell_count == 0

    def test_table_output_with_cells(self):
        """Test table output with cells."""
        cells = [
            TableCell(row=0, col=0, text="A", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=0, col=1, text="B", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=1, col=0, text="1"),
            TableCell(row=1, col=1, text="2"),
        ]
        output = TableOutput(
            cells=cells,
            num_rows=2,
            num_cols=2,
            image_width=800,
            image_height=600,
            model_name="TestModel",
        )
        assert output.cell_count == 4
        assert output.has_headers

    def test_table_output_get_cell(self):
        """Test getting cell at specific position."""
        cells = [
            TableCell(row=0, col=0, text="A"),
            TableCell(row=0, col=1, text="B"),
            TableCell(row=1, col=0, text="C"),
            TableCell(row=1, col=1, text="D"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        cell = output.get_cell(0, 0)
        assert cell is not None
        assert cell.text == "A"

        cell = output.get_cell(1, 1)
        assert cell is not None
        assert cell.text == "D"

        cell = output.get_cell(5, 5)
        assert cell is None

    def test_table_output_get_cell_with_span(self):
        """Test getting cell that spans multiple positions."""
        cells = [
            TableCell(row=0, col=0, row_span=2, col_span=2, text="Merged"),
            TableCell(row=0, col=2, text="Right"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=3)

        # All positions in merged cell should return same cell
        cell = output.get_cell(0, 0)
        assert cell is not None
        assert cell.text == "Merged"

        cell = output.get_cell(0, 1)
        assert cell is not None
        assert cell.text == "Merged"

        cell = output.get_cell(1, 0)
        assert cell is not None
        assert cell.text == "Merged"

        cell = output.get_cell(1, 1)
        assert cell is not None
        assert cell.text == "Merged"

    def test_table_output_get_row(self):
        """Test getting all cells in a row."""
        cells = [
            TableCell(row=0, col=0, text="A"),
            TableCell(row=0, col=1, text="B"),
            TableCell(row=1, col=0, text="C"),
            TableCell(row=1, col=1, text="D"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        row_0 = output.get_row(0)
        assert len(row_0) == 2
        assert {c.text for c in row_0} == {"A", "B"}

    def test_table_output_get_column(self):
        """Test getting all cells in a column."""
        cells = [
            TableCell(row=0, col=0, text="A"),
            TableCell(row=0, col=1, text="B"),
            TableCell(row=1, col=0, text="C"),
            TableCell(row=1, col=1, text="D"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        col_1 = output.get_column(1)
        assert len(col_1) == 2
        assert {c.text for c in col_1} == {"B", "D"}

    def test_table_output_to_html(self):
        """Test converting table to HTML."""
        cells = [
            TableCell(row=0, col=0, text="Header", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=1, col=0, text="Data"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=1)

        html = output.to_html(include_styles=False)
        assert "<table>" in html
        assert "<th>Header</th>" in html
        assert "<td>Data</td>" in html
        assert "</table>" in html

    def test_table_output_to_html_with_spans(self):
        """Test converting table with spans to HTML."""
        cells = [
            TableCell(row=0, col=0, col_span=2, text="Wide Header", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=1, col=0, text="Left"),
            TableCell(row=1, col=1, text="Right"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        html = output.to_html(include_styles=False)
        assert 'colspan="2"' in html

    def test_table_output_to_html_escapes_special_chars(self):
        """Test that HTML special characters are escaped."""
        cells = [
            TableCell(row=0, col=0, text="<script>alert('xss')</script>"),
        ]
        output = TableOutput(cells=cells, num_rows=1, num_cols=1)

        html = output.to_html()
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_table_output_to_markdown(self):
        """Test converting table to Markdown."""
        cells = [
            TableCell(row=0, col=0, text="A"),
            TableCell(row=0, col=1, text="B"),
            TableCell(row=1, col=0, text="1"),
            TableCell(row=1, col=1, text="2"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        md = output.to_markdown()
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_table_output_to_markdown_empty(self):
        """Test converting empty table to Markdown."""
        output = TableOutput(cells=[], num_rows=0, num_cols=0)
        md = output.to_markdown()
        assert md == ""

    def test_table_output_to_dict(self):
        """Test converting table output to dict."""
        cells = [
            TableCell(row=0, col=0, text="Test"),
        ]
        output = TableOutput(
            cells=cells,
            num_rows=1,
            num_cols=1,
            image_width=800,
            image_height=600,
            model_name="TestModel",
        )
        d = output.to_dict()

        assert d["num_rows"] == 1
        assert d["num_cols"] == 1
        assert d["image_width"] == 800
        assert d["image_height"] == 600
        assert d["model_name"] == "TestModel"
        assert len(d["cells"]) == 1
        assert "html" in d

    def test_table_output_save_load_json(self, tmp_path):
        """Test saving and loading table output from JSON."""
        cells = [
            TableCell(row=0, col=0, text="A", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=1, col=0, text="1"),
        ]
        output = TableOutput(
            cells=cells,
            num_rows=2,
            num_cols=1,
            image_width=400,
            image_height=300,
            model_name="Test",
        )

        json_path = tmp_path / "table_output.json"
        output.save_json(json_path)

        assert json_path.exists()

        loaded = TableOutput.load_json(json_path)
        assert loaded.num_rows == output.num_rows
        assert loaded.num_cols == output.num_cols
        assert loaded.cell_count == output.cell_count
        assert loaded.cells[0].text == "A"
        assert loaded.cells[0].cell_type == CellType.COLUMN_HEADER


class TestTableOutputDataFrame:
    """Tests for TableOutput.to_dataframe()."""

    def test_to_dataframe_basic(self):
        """Test basic DataFrame conversion."""
        pytest.importorskip("pandas")

        cells = [
            TableCell(row=0, col=0, text="A"),
            TableCell(row=0, col=1, text="B"),
            TableCell(row=1, col=0, text="1"),
            TableCell(row=1, col=1, text="2"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        df = output.to_dataframe()
        assert df.shape == (2, 2)
        assert df.iloc[0, 0] == "A"
        assert df.iloc[1, 1] == "2"

    def test_to_dataframe_with_headers(self):
        """Test DataFrame conversion with header detection."""
        pytest.importorskip("pandas")

        cells = [
            TableCell(row=0, col=0, text="Name", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=0, col=1, text="Age", cell_type=CellType.COLUMN_HEADER),
            TableCell(row=1, col=0, text="Alice"),
            TableCell(row=1, col=1, text="30"),
        ]
        output = TableOutput(cells=cells, num_rows=2, num_cols=2)

        df = output.to_dataframe()
        assert list(df.columns) == ["Name", "Age"]
        assert df.shape == (1, 2)  # One data row
        assert df.iloc[0, 0] == "Alice"


class TestNormalizedSizeConstant:
    """Tests for NORMALIZED_SIZE constant."""

    def test_normalized_size_is_1024(self):
        """Test that NORMALIZED_SIZE is 1024."""
        assert NORMALIZED_SIZE == 1024
