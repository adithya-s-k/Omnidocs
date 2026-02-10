"""
Tests for VLMAPIConfig and VLM text extractor configuration.
"""

import pytest
from pydantic import ValidationError


class TestVLMAPIConfig:
    """Tests for VLMAPIConfig."""

    def test_required_model(self):
        """Test that model is required."""
        from omnidocs.vlm import VLMAPIConfig

        with pytest.raises(ValidationError):
            VLMAPIConfig()

    def test_minimal_config(self):
        """Test creating config with only model."""
        from omnidocs.vlm import VLMAPIConfig

        config = VLMAPIConfig(model="gemini/gemini-2.5-flash")

        assert config.model == "gemini/gemini-2.5-flash"
        assert config.api_key is None
        assert config.api_base is None
        assert config.max_tokens == 8192
        assert config.temperature == 0.1
        assert config.timeout == 180
        assert config.api_version is None
        assert config.extra_headers is None

    def test_custom_values(self):
        """Test creating config with all fields."""
        from omnidocs.vlm import VLMAPIConfig

        config = VLMAPIConfig(
            model="openrouter/qwen/qwen3-vl-8b-instruct",
            api_key="sk-test-key",
            api_base="https://custom.api.com/v1",
            max_tokens=16384,
            temperature=0.5,
            timeout=300,
            api_version="2024-12-01-preview",
            extra_headers={"X-Custom": "value"},
        )

        assert config.model == "openrouter/qwen/qwen3-vl-8b-instruct"
        assert config.api_key == "sk-test-key"
        assert config.api_base == "https://custom.api.com/v1"
        assert config.max_tokens == 16384
        assert config.temperature == 0.5
        assert config.timeout == 300
        assert config.api_version == "2024-12-01-preview"
        assert config.extra_headers == {"X-Custom": "value"}

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.vlm import VLMAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            VLMAPIConfig(model="gemini/gemini-2.5-flash", unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)

    def test_max_tokens_bounds(self):
        """Test max_tokens validation bounds."""
        from omnidocs.vlm import VLMAPIConfig

        with pytest.raises(ValidationError):
            VLMAPIConfig(model="gemini/gemini-2.5-flash", max_tokens=100)

        with pytest.raises(ValidationError):
            VLMAPIConfig(model="gemini/gemini-2.5-flash", max_tokens=200000)

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        from omnidocs.vlm import VLMAPIConfig

        with pytest.raises(ValidationError):
            VLMAPIConfig(model="gemini/gemini-2.5-flash", temperature=-0.1)

        with pytest.raises(ValidationError):
            VLMAPIConfig(model="gemini/gemini-2.5-flash", temperature=2.5)

    def test_timeout_minimum(self):
        """Test timeout minimum validation."""
        from omnidocs.vlm import VLMAPIConfig

        with pytest.raises(ValidationError):
            VLMAPIConfig(model="gemini/gemini-2.5-flash", timeout=5)

    def test_azure_config(self):
        """Test Azure-specific configuration."""
        from omnidocs.vlm import VLMAPIConfig

        config = VLMAPIConfig(
            model="azure/gpt-4o",
            api_version="2024-12-01-preview",
        )

        assert config.model == "azure/gpt-4o"
        assert config.api_version == "2024-12-01-preview"

    def test_openai_compatible_config(self):
        """Test OpenAI-compatible provider config."""
        from omnidocs.vlm import VLMAPIConfig

        config = VLMAPIConfig(
            model="openai/qwen3-vl-8b-instruct",
            api_key="test-key",
            api_base="https://api.anannas.ai/v1",
        )

        assert config.model == "openai/qwen3-vl-8b-instruct"
        assert config.api_base == "https://api.anannas.ai/v1"
