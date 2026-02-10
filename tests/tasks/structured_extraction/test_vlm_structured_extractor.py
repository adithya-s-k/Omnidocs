"""
Tests for VLMStructuredExtractor.

Uses mocked litellm to test without API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pydantic import BaseModel


class Invoice(BaseModel):
    vendor: str
    total: float
    items: list[str]


class SimpleDoc(BaseModel):
    title: str
    page_count: int


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new("RGB", (800, 600), color="white")


@pytest.fixture
def vlm_config():
    """Create a VLMAPIConfig for testing."""
    from omnidocs.vlm import VLMAPIConfig

    return VLMAPIConfig(model="gemini/gemini-2.5-flash")


def _mock_response(content: str) -> MagicMock:
    """Create a mock litellm response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestVLMStructuredExtractor:
    """Tests for VLMStructuredExtractor."""

    def test_init(self, vlm_config):
        """Test extractor initialization."""
        from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

        extractor = VLMStructuredExtractor(config=vlm_config)
        assert extractor._loaded is True
        assert extractor.config is vlm_config

    @patch("litellm.completion")
    def test_extract_returns_structured_output(self, mock_completion, vlm_config, sample_image):
        """Test extraction returns StructuredOutput with validated data."""
        from omnidocs.tasks.structured_extraction import StructuredOutput, VLMStructuredExtractor

        mock_completion.return_value = _mock_response('{"vendor": "ACME", "total": 99.99, "items": ["Widget"]}')

        extractor = VLMStructuredExtractor(config=vlm_config)
        result = extractor.extract(sample_image, schema=Invoice, prompt="Extract invoice")

        assert isinstance(result, StructuredOutput)
        assert isinstance(result.data, Invoice)
        assert result.data.vendor == "ACME"
        assert result.data.total == 99.99
        assert result.data.items == ["Widget"]
        assert result.image_width == 800
        assert result.image_height == 600
        assert "VLM" in result.model_name

    @patch("litellm.completion")
    def test_extract_different_schema(self, mock_completion, vlm_config, sample_image):
        """Test extraction with a different Pydantic schema."""
        from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

        mock_completion.return_value = _mock_response('{"title": "Report", "page_count": 5}')

        extractor = VLMStructuredExtractor(config=vlm_config)
        result = extractor.extract(sample_image, schema=SimpleDoc, prompt="Extract doc info")

        assert isinstance(result.data, SimpleDoc)
        assert result.data.title == "Report"
        assert result.data.page_count == 5

    @patch("litellm.completion")
    def test_extract_numpy_input(self, mock_completion, vlm_config):
        """Test extraction with numpy array input."""
        import numpy as np

        from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

        mock_completion.return_value = _mock_response('{"title": "Test", "page_count": 1}')

        np_image = np.zeros((600, 800, 3), dtype=np.uint8)
        extractor = VLMStructuredExtractor(config=vlm_config)
        result = extractor.extract(np_image, schema=SimpleDoc, prompt="Extract")

        assert result.data.title == "Test"
        assert result.image_width == 800
        assert result.image_height == 600


class TestVLMStructuredCompletionFallback:
    """Tests for vlm_structured_completion fallback behavior."""

    @patch("litellm.completion")
    def test_fallback_strips_markdown_fencing(self, mock_completion, vlm_config, sample_image):
        """Test that fallback strips markdown code fences."""
        from omnidocs.tasks.structured_extraction import VLMStructuredExtractor

        # First call (native response_format) fails
        # Second call (fallback) returns markdown-fenced JSON
        mock_completion.side_effect = [
            Exception("response_format not supported"),
            _mock_response('```json\n{"title": "Fallback", "page_count": 3}\n```'),
        ]

        extractor = VLMStructuredExtractor(config=vlm_config)
        result = extractor.extract(sample_image, schema=SimpleDoc, prompt="Extract doc info")

        assert result.data.title == "Fallback"
        assert result.data.page_count == 3
        assert mock_completion.call_count == 2
