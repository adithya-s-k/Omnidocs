"""
Tests for Qwen3-VL layout detection configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestQwenLayoutPyTorchConfig:
    """Tests for QwenLayoutPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        config = QwenLayoutPyTorchConfig()

        assert config.model == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.device == "cuda"
        assert config.torch_dtype == "auto"
        assert config.device_map == "auto"
        assert config.trust_remote_code is True
        assert config.use_flash_attention is False
        assert config.max_new_tokens == 4096
        assert config.temperature == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        config = QwenLayoutPyTorchConfig(
            model="Qwen/Qwen3-VL-4B-Instruct",
            device="mps",
            torch_dtype="bfloat16",
            max_new_tokens=2048,
            temperature=0.5,
        )

        assert config.model == "Qwen/Qwen3-VL-4B-Instruct"
        assert config.device == "mps"
        assert config.torch_dtype == "bfloat16"
        assert config.max_new_tokens == 2048
        assert config.temperature == 0.5

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            QwenLayoutPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)

    def test_temperature_bounds(self):
        """Test temperature validation."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        # Valid bounds
        config = QwenLayoutPyTorchConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = QwenLayoutPyTorchConfig(temperature=2.0)
        assert config.temperature == 2.0

        # Invalid bounds
        with pytest.raises(ValidationError):
            QwenLayoutPyTorchConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            QwenLayoutPyTorchConfig(temperature=2.1)

    def test_max_tokens_bounds(self):
        """Test max_new_tokens validation."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        # Valid bounds
        config = QwenLayoutPyTorchConfig(max_new_tokens=256)
        assert config.max_new_tokens == 256

        config = QwenLayoutPyTorchConfig(max_new_tokens=16384)
        assert config.max_new_tokens == 16384

        # Invalid bounds
        with pytest.raises(ValidationError):
            QwenLayoutPyTorchConfig(max_new_tokens=255)

        with pytest.raises(ValidationError):
            QwenLayoutPyTorchConfig(max_new_tokens=16385)

    def test_torch_dtype_literal(self):
        """Test torch_dtype literal validation."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutPyTorchConfig

        # Valid values
        for dtype in ["float16", "bfloat16", "float32", "auto"]:
            config = QwenLayoutPyTorchConfig(torch_dtype=dtype)
            assert config.torch_dtype == dtype

        # Invalid value
        with pytest.raises(ValidationError):
            QwenLayoutPyTorchConfig(torch_dtype="invalid")


class TestQwenLayoutVLLMConfig:
    """Tests for QwenLayoutVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

        config = QwenLayoutVLLMConfig()

        assert config.model == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 32768
        assert config.trust_remote_code is True
        assert config.enforce_eager is False
        assert config.max_tokens == 4096

    def test_gpu_memory_bounds(self):
        """Test gpu_memory_utilization validation."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

        # Valid bounds
        config = QwenLayoutVLLMConfig(gpu_memory_utilization=0.1)
        assert config.gpu_memory_utilization == 0.1

        config = QwenLayoutVLLMConfig(gpu_memory_utilization=1.0)
        assert config.gpu_memory_utilization == 1.0

        # Invalid bounds
        with pytest.raises(ValidationError):
            QwenLayoutVLLMConfig(gpu_memory_utilization=0.09)

        with pytest.raises(ValidationError):
            QwenLayoutVLLMConfig(gpu_memory_utilization=1.01)

    def test_tensor_parallel_size_minimum(self):
        """Test tensor_parallel_size minimum."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutVLLMConfig

        config = QwenLayoutVLLMConfig(tensor_parallel_size=1)
        assert config.tensor_parallel_size == 1

        with pytest.raises(ValidationError):
            QwenLayoutVLLMConfig(tensor_parallel_size=0)


class TestQwenLayoutMLXConfig:
    """Tests for QwenLayoutMLXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutMLXConfig

        config = QwenLayoutMLXConfig()

        assert config.model == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
        assert config.max_tokens == 4096
        assert config.temperature == 0.1


class TestQwenLayoutAPIConfig:
    """Tests for QwenLayoutAPIConfig."""

    def test_api_key_required(self):
        """Test that api_key is required."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            QwenLayoutAPIConfig()

        assert "api_key" in str(exc_info.value)

    def test_with_api_key(self):
        """Test configuration with api_key."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

        config = QwenLayoutAPIConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.model == "qwen/qwen3-vl-8b-instruct"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.max_tokens == 4096
        assert config.timeout == 120

    def test_custom_base_url(self):
        """Test custom base_url."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

        config = QwenLayoutAPIConfig(
            api_key="test-key",
            base_url="https://api.novita.ai/v3/openai",
        )

        assert config.base_url == "https://api.novita.ai/v3/openai"

    def test_extra_headers(self):
        """Test extra_headers field."""
        from omnidocs.tasks.layout_extraction.qwen import QwenLayoutAPIConfig

        config = QwenLayoutAPIConfig(
            api_key="test-key",
            extra_headers={"X-Custom-Header": "value"},
        )

        assert config.extra_headers == {"X-Custom-Header": "value"}
