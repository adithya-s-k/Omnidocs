"""
Tests for MinerU VL layout detection configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestMinerUVLLayoutPyTorchConfig:
    """Tests for MinerUVLLayoutPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

        config = MinerUVLLayoutPyTorchConfig()

        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.device == "auto"
        assert config.torch_dtype == "float16"
        assert config.use_flash_attention is False  # SDPA by default
        assert config.trust_remote_code is True
        assert config.layout_image_size == (1036, 1036)
        assert config.max_new_tokens == 4096

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

        config = MinerUVLLayoutPyTorchConfig(
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
        )

        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.use_flash_attention is False

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLLayoutPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLLayoutVLLMConfig:
    """Tests for MinerUVLLayoutVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutVLLMConfig

        config = MinerUVLLayoutVLLMConfig()

        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 16384
        assert config.enforce_eager is True

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutVLLMConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLLayoutVLLMConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLLayoutMLXConfig:
    """Tests for MinerUVLLayoutMLXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutMLXConfig

        config = MinerUVLLayoutMLXConfig()

        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutMLXConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLLayoutMLXConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLLayoutAPIConfig:
    """Tests for MinerUVLLayoutAPIConfig."""

    def test_server_url_required(self):
        """Test that server_url is required."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLLayoutAPIConfig()

        assert "server_url" in str(exc_info.value)

    def test_with_server_url(self):
        """Test configuration with server_url."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutAPIConfig

        config = MinerUVLLayoutAPIConfig(server_url="https://api.example.com")

        assert config.server_url == "https://api.example.com"
        assert config.model_name == "mineru-vl"
        assert config.timeout == 300

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.layout_extraction.mineruvl import MinerUVLLayoutAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLLayoutAPIConfig(server_url="https://api.example.com", unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLLayoutDetectorImports:
    """Test that all MinerU VL layout exports are importable."""

    def test_layout_detector_import(self):
        """Test importing MinerUVLLayoutDetector from layout_extraction."""
        from omnidocs.tasks.layout_extraction import MinerUVLLayoutDetector

        assert MinerUVLLayoutDetector is not None

    def test_config_imports_from_submodule(self):
        """Test importing configs from mineruvl submodule."""
        from omnidocs.tasks.layout_extraction.mineruvl import (
            MinerUVLLayoutAPIConfig,
            MinerUVLLayoutMLXConfig,
            MinerUVLLayoutPyTorchConfig,
            MinerUVLLayoutVLLMConfig,
        )

        assert MinerUVLLayoutPyTorchConfig is not None
        assert MinerUVLLayoutVLLMConfig is not None
        assert MinerUVLLayoutMLXConfig is not None
        assert MinerUVLLayoutAPIConfig is not None
