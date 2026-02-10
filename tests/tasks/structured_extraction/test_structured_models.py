"""
Tests for structured extraction output models.
"""

import pytest
from pydantic import ValidationError


class TestStructuredOutput:
    """Tests for StructuredOutput model."""

    def test_minimal_output(self):
        """Test creating with minimal fields."""
        from pydantic import BaseModel

        from omnidocs.tasks.structured_extraction import StructuredOutput

        class SimpleSchema(BaseModel):
            name: str

        data = SimpleSchema(name="test")
        output = StructuredOutput(data=data)

        assert output.data.name == "test"
        assert output.raw_output is None
        assert output.image_width is None
        assert output.image_height is None
        assert output.model_name is None

    def test_full_output(self):
        """Test creating with all fields."""
        from pydantic import BaseModel

        from omnidocs.tasks.structured_extraction import StructuredOutput

        class Invoice(BaseModel):
            vendor: str
            total: float

        data = Invoice(vendor="ACME", total=99.99)
        output = StructuredOutput(
            data=data,
            raw_output='{"vendor": "ACME", "total": 99.99}',
            image_width=800,
            image_height=600,
            model_name="VLM (gemini/gemini-2.5-flash)",
        )

        assert output.data.vendor == "ACME"
        assert output.data.total == 99.99
        assert output.image_width == 800
        assert output.model_name == "VLM (gemini/gemini-2.5-flash)"

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from pydantic import BaseModel

        from omnidocs.tasks.structured_extraction import StructuredOutput

        class SimpleSchema(BaseModel):
            name: str

        with pytest.raises(ValidationError) as exc_info:
            StructuredOutput(data=SimpleSchema(name="test"), unknown=True)

        assert "extra_forbidden" in str(exc_info.value)

    def test_image_dimensions_validation(self):
        """Test that image dimensions must be positive."""
        from pydantic import BaseModel

        from omnidocs.tasks.structured_extraction import StructuredOutput

        class SimpleSchema(BaseModel):
            name: str

        with pytest.raises(ValidationError):
            StructuredOutput(data=SimpleSchema(name="test"), image_width=0)
