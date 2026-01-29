"""
Tests for CustomLabel type-safe custom labels.
"""

import pytest
from pydantic import ValidationError


class TestCustomLabel:
    """Tests for CustomLabel class."""

    def test_minimal_label(self):
        """Test creating label with just name."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        label = CustomLabel(name="code_block")

        assert label.name == "code_block"
        assert label.description is None
        assert label.color is None
        assert label.detection_prompt is None

    def test_full_label(self):
        """Test creating label with all fields."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        label = CustomLabel(
            name="sidebar",
            description="Secondary content panel",
            color="#9B59B6",
            detection_prompt="Look for side panels with navigation",
        )

        assert label.name == "sidebar"
        assert label.description == "Secondary content panel"
        assert label.color == "#9B59B6"
        assert label.detection_prompt == "Look for side panels with navigation"

    def test_name_required(self):
        """Test that name is required."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        with pytest.raises(ValidationError) as exc_info:
            CustomLabel()

        assert "name" in str(exc_info.value)

    def test_name_min_length(self):
        """Test name minimum length."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        # Empty string should fail
        with pytest.raises(ValidationError):
            CustomLabel(name="")

        # Single character should work
        label = CustomLabel(name="a")
        assert label.name == "a"

    def test_name_max_length(self):
        """Test name maximum length."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        # 50 characters should work
        label = CustomLabel(name="a" * 50)
        assert len(label.name) == 50

        # 51 characters should fail
        with pytest.raises(ValidationError):
            CustomLabel(name="a" * 51)

    def test_color_validation(self):
        """Test color hex pattern validation."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        # Valid hex colors
        for color in ["#FFFFFF", "#000000", "#9B59B6", "#aabbcc"]:
            label = CustomLabel(name="test", color=color)
            assert label.color == color

        # Invalid colors
        invalid_colors = [
            "red",  # Named color
            "#FFF",  # Too short
            "#GGGGGG",  # Invalid hex
            "FFFFFF",  # Missing #
            "#FFFFFF00",  # Too long (8 chars)
        ]
        for color in invalid_colors:
            with pytest.raises(ValidationError):
                CustomLabel(name="test", color=color)

    def test_str_method(self):
        """Test __str__ returns name."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        label = CustomLabel(name="code_block")
        assert str(label) == "code_block"

    def test_hash_method(self):
        """Test CustomLabel is hashable."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        label1 = CustomLabel(name="code_block")
        label2 = CustomLabel(name="sidebar")

        # Can be used in sets
        label_set = {label1, label2}
        assert len(label_set) == 2

        # Same name hashes equal
        label3 = CustomLabel(name="code_block", description="Different desc")
        assert hash(label1) == hash(label3)

    def test_equality(self):
        """Test equality comparison."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        label1 = CustomLabel(name="code_block")
        label2 = CustomLabel(name="code_block", description="Different")
        label3 = CustomLabel(name="sidebar")

        # Equal by name
        assert label1 == label2

        # Not equal with different name
        assert label1 != label3

        # Equal to string with same name
        assert label1 == "code_block"
        assert label1 != "sidebar"

        # Not equal to other types
        assert label1 != 123
        assert label1 is not None

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.layout_extraction import CustomLabel

        with pytest.raises(ValidationError) as exc_info:
            CustomLabel(name="test", unknown_field="value")

        assert "extra_forbidden" in str(exc_info.value)
