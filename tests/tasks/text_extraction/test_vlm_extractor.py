"""
Tests for VLMTextExtractor.

Uses mocked litellm to test without API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


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


class TestVLMTextExtractor:
    """Tests for VLMTextExtractor."""

    def test_init(self, vlm_config):
        """Test extractor initialization."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        extractor = VLMTextExtractor(config=vlm_config)
        assert extractor._loaded is True
        assert extractor.config is vlm_config

    @patch("litellm.completion")
    def test_extract_markdown(self, mock_completion, vlm_config, sample_image):
        """Test markdown extraction."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        mock_completion.return_value = _mock_response("# Document Title\n\nSome extracted text.")

        extractor = VLMTextExtractor(config=vlm_config)
        result = extractor.extract(sample_image, output_format="markdown")

        assert result.content == "# Document Title\n\nSome extracted text."
        assert result.format.value == "markdown"
        assert result.image_width == 800
        assert result.image_height == 600
        assert "VLM" in result.model_name
        mock_completion.assert_called_once()

    @patch("litellm.completion")
    def test_extract_html(self, mock_completion, vlm_config, sample_image):
        """Test HTML extraction."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        mock_completion.return_value = _mock_response("<h1>Title</h1><p>Text</p>")

        extractor = VLMTextExtractor(config=vlm_config)
        result = extractor.extract(sample_image, output_format="html")

        assert result.content == "<h1>Title</h1><p>Text</p>"
        assert result.format.value == "html"

    @patch("litellm.completion")
    def test_extract_custom_prompt(self, mock_completion, vlm_config, sample_image):
        """Test extraction with custom prompt."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        mock_completion.return_value = _mock_response("| Col1 | Col2 |")

        extractor = VLMTextExtractor(config=vlm_config)
        result = extractor.extract(sample_image, prompt="Extract only table data")

        # Verify custom prompt was passed in the messages
        call_kwargs = mock_completion.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"][0]["text"]
        assert prompt_text == "Extract only table data"
        assert result.content == "| Col1 | Col2 |"

    def test_extract_invalid_format(self, vlm_config, sample_image):
        """Test invalid output format raises error."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        extractor = VLMTextExtractor(config=vlm_config)

        with pytest.raises(ValueError, match="Invalid output_format"):
            extractor.extract(sample_image, output_format="xml")

    @patch("litellm.completion")
    def test_extract_plain_text_property(self, mock_completion, vlm_config, sample_image):
        """Test that plain_text is extracted from content."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor

        mock_completion.return_value = _mock_response("# Title\n\nParagraph text here.")

        extractor = VLMTextExtractor(config=vlm_config)
        result = extractor.extract(sample_image)

        assert result.plain_text is not None
        assert "Title" in result.plain_text
        assert "Paragraph text here" in result.plain_text

    @patch("litellm.completion")
    def test_extract_numpy_input(self, mock_completion, vlm_config):
        """Test extraction with numpy array input."""
        import numpy as np

        from omnidocs.tasks.text_extraction import VLMTextExtractor

        mock_completion.return_value = _mock_response("Extracted text")
        np_image = np.zeros((600, 800, 3), dtype=np.uint8)

        extractor = VLMTextExtractor(config=vlm_config)
        result = extractor.extract(np_image)

        assert result.content is not None
        assert result.image_width == 800
        assert result.image_height == 600

    @patch("litellm.completion")
    def test_build_kwargs_azure(self, mock_completion):
        """Test that Azure config uses max_completion_tokens."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor
        from omnidocs.vlm import VLMAPIConfig

        config = VLMAPIConfig(model="azure/gpt-4o", api_version="2024-12-01-preview")
        mock_completion.return_value = _mock_response("text")

        extractor = VLMTextExtractor(config=config)
        sample = Image.new("RGB", (100, 100), color="white")
        extractor.extract(sample)

        call_kwargs = mock_completion.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["api_version"] == "2024-12-01-preview"

    @patch("litellm.completion")
    def test_build_kwargs_standard(self, mock_completion):
        """Test that non-Azure config uses max_tokens."""
        from omnidocs.tasks.text_extraction import VLMTextExtractor
        from omnidocs.vlm import VLMAPIConfig

        config = VLMAPIConfig(model="gemini/gemini-2.5-flash")
        mock_completion.return_value = _mock_response("text")

        extractor = VLMTextExtractor(config=config)
        sample = Image.new("RGB", (100, 100), color="white")
        extractor.extract(sample)

        call_kwargs = mock_completion.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_completion_tokens" not in call_kwargs


class TestVLMTextDefaultPrompts:
    """Tests for default prompt constants."""

    def test_default_prompts_exist(self):
        """Test that default prompts are defined."""
        from omnidocs.tasks.text_extraction.vlm import DEFAULT_PROMPTS

        assert "markdown" in DEFAULT_PROMPTS
        assert "html" in DEFAULT_PROMPTS

    def test_markdown_prompt_content(self):
        """Test markdown prompt mentions key formatting."""
        from omnidocs.tasks.text_extraction.vlm import DEFAULT_PROMPTS

        prompt = DEFAULT_PROMPTS["markdown"]
        assert "Markdown" in prompt
        assert "table" in prompt.lower()

    def test_html_prompt_content(self):
        """Test HTML prompt mentions HTML tags."""
        from omnidocs.tasks.text_extraction.vlm import DEFAULT_PROMPTS

        prompt = DEFAULT_PROMPTS["html"]
        assert "HTML" in prompt


class TestExtractPlainText:
    """Tests for _extract_plain_text helper."""

    def test_strip_markdown(self):
        """Test stripping markdown formatting."""
        from omnidocs.tasks.text_extraction.vlm import _extract_plain_text

        result = _extract_plain_text("# Title\n\n**bold** text", "markdown")
        assert "Title" in result
        assert "bold" in result
        assert "#" not in result
        assert "**" not in result

    def test_strip_html(self):
        """Test stripping HTML tags."""
        from omnidocs.tasks.text_extraction.vlm import _extract_plain_text

        result = _extract_plain_text("<h1>Title</h1><p>Text</p>", "html")
        assert "Title" in result
        assert "Text" in result
        assert "<" not in result
