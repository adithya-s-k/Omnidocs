"""
Tests for VLMLayoutDetector.

Uses mocked litellm to test without API keys.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    return Image.new("RGB", (1024, 1400), color="white")


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


class TestVLMLayoutDetector:
    """Tests for VLMLayoutDetector."""

    def test_init(self, vlm_config):
        """Test detector initialization."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector

        detector = VLMLayoutDetector(config=vlm_config)
        assert detector._loaded is True
        assert detector.config is vlm_config

    @patch("litellm.completion")
    def test_extract_returns_layout_output(self, mock_completion, vlm_config, sample_image):
        """Test extraction returns LayoutOutput."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector
        from omnidocs.tasks.layout_extraction.models import LayoutOutput

        mock_completion.return_value = _mock_response(
            '[{"bbox": [100, 50, 900, 120], "label": "title"}, {"bbox": [100, 150, 900, 400], "label": "text"}]'
        )

        detector = VLMLayoutDetector(config=vlm_config)
        result = detector.extract(sample_image)

        assert isinstance(result, LayoutOutput)
        assert len(result.bboxes) == 2
        assert result.image_width == 1024
        assert result.image_height == 1400

    @patch("litellm.completion")
    def test_extract_label_mapping(self, mock_completion, vlm_config, sample_image):
        """Test that labels are mapped to LayoutLabel enum."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector
        from omnidocs.tasks.layout_extraction.models import LayoutLabel

        mock_completion.return_value = _mock_response('[{"bbox": [10, 10, 200, 50], "label": "title"}]')

        detector = VLMLayoutDetector(config=vlm_config)
        result = detector.extract(sample_image)

        assert len(result.bboxes) == 1
        assert result.bboxes[0].label == LayoutLabel.TITLE

    @patch("litellm.completion")
    def test_extract_unknown_label(self, mock_completion, vlm_config, sample_image):
        """Test that unknown labels map to UNKNOWN."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector
        from omnidocs.tasks.layout_extraction.models import LayoutLabel

        mock_completion.return_value = _mock_response('[{"bbox": [10, 10, 200, 50], "label": "sidebar"}]')

        detector = VLMLayoutDetector(config=vlm_config)
        result = detector.extract(sample_image)

        assert len(result.bboxes) == 1
        assert result.bboxes[0].label == LayoutLabel.UNKNOWN
        assert result.bboxes[0].original_label == "sidebar"

    @patch("litellm.completion")
    def test_extract_empty_response(self, mock_completion, vlm_config, sample_image):
        """Test handling of empty response."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector

        mock_completion.return_value = _mock_response("[]")

        detector = VLMLayoutDetector(config=vlm_config)
        result = detector.extract(sample_image)

        assert len(result.bboxes) == 0

    @patch("litellm.completion")
    def test_extract_with_custom_labels(self, mock_completion, vlm_config, sample_image):
        """Test extraction with custom labels."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector

        mock_completion.return_value = _mock_response('[{"bbox": [10, 10, 200, 50], "label": "code_block"}]')

        detector = VLMLayoutDetector(config=vlm_config)
        detector.extract(sample_image, custom_labels=["code_block", "sidebar"])

        # Verify custom labels were in the prompt
        call_kwargs = mock_completion.call_args[1]
        prompt_text = call_kwargs["messages"][0]["content"][0]["text"]
        assert "code_block" in prompt_text
        assert "sidebar" in prompt_text

    @patch("litellm.completion")
    def test_bbox_clamping(self, mock_completion, vlm_config, sample_image):
        """Test that bboxes are clamped to image bounds."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector

        mock_completion.return_value = _mock_response('[{"bbox": [-10, -5, 1100, 1500], "label": "text"}]')

        detector = VLMLayoutDetector(config=vlm_config)
        result = detector.extract(sample_image)

        if len(result.bboxes) > 0:
            box = result.bboxes[0]
            assert box.bbox.x1 >= 0
            assert box.bbox.y1 >= 0
            assert box.bbox.x2 <= 1024
            assert box.bbox.y2 <= 1400

    @patch("litellm.completion")
    def test_reading_order_sort(self, mock_completion, vlm_config, sample_image):
        """Test that results are sorted by reading order (top-to-bottom, left-to-right)."""
        from omnidocs.tasks.layout_extraction import VLMLayoutDetector

        mock_completion.return_value = _mock_response(
            '[{"bbox": [100, 500, 900, 600], "label": "text"}, {"bbox": [100, 50, 900, 120], "label": "title"}]'
        )

        detector = VLMLayoutDetector(config=vlm_config)
        result = detector.extract(sample_image)

        assert len(result.bboxes) == 2
        # Title (y1=50) should come before text (y1=500)
        assert result.bboxes[0].bbox.y1 < result.bboxes[1].bbox.y1


class TestParseLayoutResponse:
    """Tests for _parse_layout_response helper."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON array."""
        from omnidocs.tasks.layout_extraction.vlm import _parse_layout_response

        raw = '[{"bbox": [10, 20, 100, 50], "label": "title"}]'
        result = _parse_layout_response(raw, (800, 600))

        assert len(result) == 1
        assert result[0]["label"] == "title"
        assert result[0]["bbox"] == [10, 20, 100, 50]

    def test_parse_markdown_fenced_json(self):
        """Test parsing JSON wrapped in markdown code fence."""
        from omnidocs.tasks.layout_extraction.vlm import _parse_layout_response

        raw = '```json\n[{"bbox": [10, 20, 100, 50], "label": "title"}]\n```'
        result = _parse_layout_response(raw, (800, 600))

        assert len(result) == 1

    def test_parse_empty_array(self):
        """Test parsing empty JSON array."""
        from omnidocs.tasks.layout_extraction.vlm import _parse_layout_response

        result = _parse_layout_response("[]", (800, 600))
        assert result == []

    def test_parse_invalid_json_fallback(self):
        """Test regex fallback for truncated/invalid JSON."""
        from omnidocs.tasks.layout_extraction.vlm import _parse_layout_response

        raw = '[{"bbox": [10, 20, 100, 50], "label": "title"}, {"bbox": [10, 60, 100'
        result = _parse_layout_response(raw, (800, 600))

        assert len(result) >= 1

    def test_invalid_bbox_passes_json_parse(self):
        """Test that JSON parse returns raw results (validation happens in _build_layout_boxes)."""
        from omnidocs.tasks.layout_extraction.vlm import _parse_layout_response

        raw = '[{"bbox": [200, 20, 100, 50], "label": "title"}]'
        result = _parse_layout_response(raw, (800, 600))

        # _parse_layout_response returns raw JSON; bbox validation is in _build_layout_boxes
        assert len(result) == 1


class TestBuildLayoutPrompt:
    """Tests for _build_layout_prompt helper."""

    def test_labels_in_prompt(self):
        """Test that labels appear in generated prompt."""
        from omnidocs.tasks.layout_extraction.vlm import _build_layout_prompt

        prompt = _build_layout_prompt(["title", "text", "table"])
        assert "title" in prompt
        assert "text" in prompt
        assert "table" in prompt
        assert "JSON" in prompt

    def test_custom_labels(self):
        """Test custom labels in prompt."""
        from omnidocs.tasks.layout_extraction.vlm import _build_layout_prompt

        prompt = _build_layout_prompt(["code_block", "sidebar"])
        assert "code_block" in prompt
        assert "sidebar" in prompt
