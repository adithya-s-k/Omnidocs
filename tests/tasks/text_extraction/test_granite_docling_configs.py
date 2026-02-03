"""
Tests for Granite Docling text extraction configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestGraniteDoclingTextPyTorchConfig:
    """Tests for GraniteDoclingTextPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextPyTorchConfig,
        )

        config = GraniteDoclingTextPyTorchConfig()

        assert config.model == "ibm-granite/granite-docling-258M"
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.max_new_tokens == 8192
        assert config.temperature == 0.1
        assert config.use_flash_attention is True
        assert config.do_sample is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextPyTorchConfig,
        )

        config = GraniteDoclingTextPyTorchConfig(
            device="mps",
            torch_dtype="float16",
            max_new_tokens=4096,
            use_flash_attention=False,
        )

        assert config.device == "mps"
        assert config.torch_dtype == "float16"
        assert config.max_new_tokens == 4096
        assert config.use_flash_attention is False

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextPyTorchConfig,
        )

        with pytest.raises(ValidationError) as exc_info:
            GraniteDoclingTextPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)

    def test_max_new_tokens_validation(self):
        """Test max_new_tokens validation bounds."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextPyTorchConfig,
        )

        # Below minimum
        with pytest.raises(ValidationError):
            GraniteDoclingTextPyTorchConfig(max_new_tokens=100)

        # Above maximum
        with pytest.raises(ValidationError):
            GraniteDoclingTextPyTorchConfig(max_new_tokens=100000)

    def test_temperature_validation(self):
        """Test temperature validation bounds."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextPyTorchConfig,
        )

        # Below minimum
        with pytest.raises(ValidationError):
            GraniteDoclingTextPyTorchConfig(temperature=-0.1)

        # Above maximum
        with pytest.raises(ValidationError):
            GraniteDoclingTextPyTorchConfig(temperature=2.5)


class TestGraniteDoclingTextVLLMConfig:
    """Tests for GraniteDoclingTextVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextVLLMConfig,
        )

        config = GraniteDoclingTextVLLMConfig()

        assert config.model == "ibm-granite/granite-docling-258M"
        assert config.revision == "untied"  # Critical for VLLM
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_tokens == 8192
        assert config.fast_boot is True
        assert config.limit_mm_per_prompt == 1

    def test_revision_default_is_untied(self):
        """Test that revision defaults to 'untied' for VLLM compatibility."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextVLLMConfig,
        )

        config = GraniteDoclingTextVLLMConfig()
        assert config.revision == "untied"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextVLLMConfig,
        )

        config = GraniteDoclingTextVLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_tokens=4096,
            fast_boot=False,
        )

        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.9
        assert config.max_tokens == 4096
        assert config.fast_boot is False

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextVLLMConfig,
        )

        with pytest.raises(ValidationError) as exc_info:
            GraniteDoclingTextVLLMConfig(invalid_param="test")

        assert "extra_forbidden" in str(exc_info.value)


class TestGraniteDoclingTextMLXConfig:
    """Tests for GraniteDoclingTextMLXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextMLXConfig,
        )

        config = GraniteDoclingTextMLXConfig()

        assert config.model == "ibm-granite/granite-docling-258M-mlx"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0

    def test_mlx_model_default_has_mlx_suffix(self):
        """Test MLX uses different model ID with -mlx suffix."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextMLXConfig,
        )

        config = GraniteDoclingTextMLXConfig()
        assert "-mlx" in config.model

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextMLXConfig,
        )

        config = GraniteDoclingTextMLXConfig(
            max_tokens=2048,
            temperature=0.1,
        )

        assert config.max_tokens == 2048
        assert config.temperature == 0.1

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextMLXConfig,
        )

        with pytest.raises(ValidationError) as exc_info:
            GraniteDoclingTextMLXConfig(device="cuda")

        assert "extra_forbidden" in str(exc_info.value)


class TestGraniteDoclingTextAPIConfig:
    """Tests for GraniteDoclingTextAPIConfig."""

    def test_api_key_required(self):
        """Test that api_key is required."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextAPIConfig,
        )

        with pytest.raises(ValidationError) as exc_info:
            GraniteDoclingTextAPIConfig()

        assert "api_key" in str(exc_info.value)

    def test_with_api_key(self):
        """Test configuration with api_key."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextAPIConfig,
        )

        config = GraniteDoclingTextAPIConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.max_tokens == 8192
        assert config.timeout == 180

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextAPIConfig,
        )

        config = GraniteDoclingTextAPIConfig(
            api_key="my-key",
            base_url="https://custom.api.com/v1",
            max_tokens=4096,
            timeout=300,
        )

        assert config.api_key == "my-key"
        assert config.base_url == "https://custom.api.com/v1"
        assert config.max_tokens == 4096
        assert config.timeout == 300

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextAPIConfig,
        )

        with pytest.raises(ValidationError) as exc_info:
            GraniteDoclingTextAPIConfig(api_key="key", unknown=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestGraniteDoclingConfigImports:
    """Test that all configs can be imported from the module."""

    def test_import_from_granitedocling_module(self):
        """Test imports from granitedocling submodule."""
        from omnidocs.tasks.text_extraction.granitedocling import (
            GraniteDoclingTextAPIConfig,
            GraniteDoclingTextExtractor,
            GraniteDoclingTextMLXConfig,
            GraniteDoclingTextPyTorchConfig,
            GraniteDoclingTextVLLMConfig,
        )

        assert GraniteDoclingTextExtractor is not None
        assert GraniteDoclingTextPyTorchConfig is not None
        assert GraniteDoclingTextVLLMConfig is not None
        assert GraniteDoclingTextMLXConfig is not None
        assert GraniteDoclingTextAPIConfig is not None

    def test_import_extractor_from_main_module(self):
        """Test extractor can be imported from main text_extraction module."""
        from omnidocs.tasks.text_extraction import GraniteDoclingTextExtractor

        assert GraniteDoclingTextExtractor is not None
