"""
Tests for Qwen3-VL text extraction configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestQwenTextPyTorchConfig:
    """Tests for QwenTextPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

        config = QwenTextPyTorchConfig()

        assert config.model == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.device == "cuda"
        assert config.torch_dtype == "auto"
        assert config.max_new_tokens == 8192  # Higher for text extraction
        assert config.temperature == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

        config = QwenTextPyTorchConfig(
            model="Qwen/Qwen3-VL-32B-Instruct",
            device="mps",
            max_new_tokens=16384,
        )

        assert config.model == "Qwen/Qwen3-VL-32B-Instruct"
        assert config.device == "mps"
        assert config.max_new_tokens == 16384

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            QwenTextPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestQwenTextVLLMConfig:
    """Tests for QwenTextVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextVLLMConfig

        config = QwenTextVLLMConfig()

        assert config.model == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.max_tokens == 8192  # Higher for text extraction
        assert config.gpu_memory_utilization == 0.85


class TestQwenTextMLXConfig:
    """Tests for QwenTextMLXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextMLXConfig

        config = QwenTextMLXConfig()

        assert config.model == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
        assert config.max_tokens == 8192  # Higher for text extraction


class TestQwenTextAPIConfig:
    """Tests for QwenTextAPIConfig."""

    def test_default_values(self):
        """Test default configuration values (api_key optional with litellm)."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

        config = QwenTextAPIConfig()

        assert config.model == "openrouter/qwen/qwen3-vl-8b-instruct"
        assert config.api_key is None
        assert config.api_base is None
        assert config.max_tokens == 8192
        assert config.timeout == 180

    def test_with_api_key(self):
        """Test configuration with api_key."""
        from omnidocs.tasks.text_extraction.qwen import QwenTextAPIConfig

        config = QwenTextAPIConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.max_tokens == 8192  # Higher for text extraction
        assert config.timeout == 180  # Longer timeout for text extraction
