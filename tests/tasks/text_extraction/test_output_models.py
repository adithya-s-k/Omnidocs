"""
Tests for text extraction output models.
"""

import pytest
from pydantic import ValidationError


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        from omnidocs.tasks.text_extraction import OutputFormat

        assert OutputFormat.HTML.value == "html"
        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_from_string(self):
        """Test creating from string."""
        from omnidocs.tasks.text_extraction import OutputFormat

        assert OutputFormat("html") == OutputFormat.HTML
        assert OutputFormat("markdown") == OutputFormat.MARKDOWN

    def test_invalid_value(self):
        """Test invalid value raises error."""
        from omnidocs.tasks.text_extraction import OutputFormat

        with pytest.raises(ValueError):
            OutputFormat("invalid")


class TestTextOutput:
    """Tests for TextOutput model."""

    def test_minimal_output(self):
        """Test creating with minimal fields."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        output = TextOutput(
            content="# Hello World",
            format=OutputFormat.MARKDOWN,
        )

        assert output.content == "# Hello World"
        assert output.format == OutputFormat.MARKDOWN
        assert output.raw_output is None
        assert output.plain_text is None
        assert output.image_width is None
        assert output.image_height is None
        assert output.model_name is None

    def test_full_output(self):
        """Test creating with all fields."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        output = TextOutput(
            content="# Hello World",
            format=OutputFormat.MARKDOWN,
            raw_output="<!-- Raw -->\\n# Hello World",
            plain_text="Hello World",
            image_width=1024,
            image_height=768,
            model_name="Qwen3-VL",
        )

        assert output.content == "# Hello World"
        assert output.raw_output == "<!-- Raw -->\\n# Hello World"
        assert output.plain_text == "Hello World"
        assert output.image_width == 1024
        assert output.image_height == 768
        assert output.model_name == "Qwen3-VL"

    def test_content_required(self):
        """Test that content is required."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        with pytest.raises(ValidationError) as exc_info:
            TextOutput(format=OutputFormat.MARKDOWN)

        assert "content" in str(exc_info.value)

    def test_format_required(self):
        """Test that format is required."""
        from omnidocs.tasks.text_extraction import TextOutput

        with pytest.raises(ValidationError) as exc_info:
            TextOutput(content="Hello")

        assert "format" in str(exc_info.value)

    def test_image_dimensions_validation(self):
        """Test image dimension validation."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        # Valid dimensions
        output = TextOutput(
            content="test",
            format=OutputFormat.HTML,
            image_width=1,
            image_height=1,
        )
        assert output.image_width == 1

        # Invalid dimensions (must be >= 1)
        with pytest.raises(ValidationError):
            TextOutput(
                content="test",
                format=OutputFormat.HTML,
                image_width=0,
            )

        with pytest.raises(ValidationError):
            TextOutput(
                content="test",
                format=OutputFormat.HTML,
                image_height=-1,
            )

    def test_content_length_property(self):
        """Test content_length property."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        output = TextOutput(
            content="Hello World",  # 11 characters
            format=OutputFormat.MARKDOWN,
        )

        assert output.content_length == 11

    def test_word_count_property(self):
        """Test word_count property."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        # Uses plain_text if available
        output = TextOutput(
            content="# Hello World\\nThis is a test",
            format=OutputFormat.MARKDOWN,
            plain_text="Hello World This is a test",
        )
        assert output.word_count == 6

        # Falls back to content if plain_text is None
        output = TextOutput(
            content="one two three",
            format=OutputFormat.MARKDOWN,
        )
        assert output.word_count == 3

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        with pytest.raises(ValidationError) as exc_info:
            TextOutput(
                content="test",
                format=OutputFormat.HTML,
                unknown_field="value",
            )

        assert "extra_forbidden" in str(exc_info.value)

    def test_format_from_string(self):
        """Test creating with format as string."""
        from omnidocs.tasks.text_extraction import OutputFormat, TextOutput

        # Pydantic should coerce string to enum
        output = TextOutput(content="test", format="html")
        assert output.format == OutputFormat.HTML
