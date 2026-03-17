"""
Tests for GLM-OCR text extraction configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestGLMOCRPyTorchConfig:
    """Tests for GLMOCRPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        config = GLMOCRPyTorchConfig()

        assert config.model == "zai-org/GLM-OCR"
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.device_map == "auto"
        assert config.use_flash_attention is False
        assert config.cache_dir is None
        assert config.max_new_tokens == 4096
        assert config.temperature == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        config = GLMOCRPyTorchConfig(
            model="zai-org/GLM-OCR",
            device="mps",
            torch_dtype="float16",
            use_flash_attention=True,
            max_new_tokens=8192,
            temperature=0.5,
        )

        assert config.device == "mps"
        assert config.torch_dtype == "float16"
        assert config.use_flash_attention is True
        assert config.max_new_tokens == 8192
        assert config.temperature == 0.5

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        with pytest.raises(ValidationError):
            GLMOCRPyTorchConfig(torch_dtype="float64")

    def test_invalid_max_new_tokens_too_low(self):
        """Test that max_new_tokens below minimum raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        with pytest.raises(ValidationError):
            GLMOCRPyTorchConfig(max_new_tokens=100)

    def test_invalid_max_new_tokens_too_high(self):
        """Test that max_new_tokens above maximum raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        with pytest.raises(ValidationError):
            GLMOCRPyTorchConfig(max_new_tokens=20000)

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        with pytest.raises(ValidationError):
            GLMOCRPyTorchConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            GLMOCRPyTorchConfig(temperature=3.0)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            GLMOCRPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestGLMOCRVLLMConfig:
    """Tests for GLMOCRVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        config = GLMOCRVLLMConfig()

        assert config.model == "zai-org/GLM-OCR"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 8192
        assert config.trust_remote_code is True
        assert config.enforce_eager is False
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.repetition_penalty == 1.05
        assert config.download_dir is None
        assert config.disable_custom_all_reduce is False

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        config = GLMOCRVLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            enforce_eager=True,
            max_tokens=2048,
            repetition_penalty=1.1,
        )

        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 4096
        assert config.enforce_eager is True
        assert config.max_tokens == 2048
        assert config.repetition_penalty == 1.1

    def test_invalid_tensor_parallel_size(self):
        """Test that tensor_parallel_size below 1 raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(tensor_parallel_size=0)

    def test_invalid_gpu_memory_utilization_too_high(self):
        """Test that gpu_memory_utilization above 1.0 raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(gpu_memory_utilization=1.5)

    def test_invalid_gpu_memory_utilization_too_low(self):
        """Test that gpu_memory_utilization below 0.1 raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(gpu_memory_utilization=0.05)

    def test_invalid_max_model_len(self):
        """Test that max_model_len below minimum raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(max_model_len=256)

    def test_invalid_max_tokens_too_low(self):
        """Test that max_tokens below minimum raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(max_tokens=100)

    def test_invalid_max_tokens_too_high(self):
        """Test that max_tokens above maximum raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(max_tokens=20000)

    def test_invalid_repetition_penalty(self):
        """Test that repetition_penalty below 1.0 raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError):
            GLMOCRVLLMConfig(repetition_penalty=0.9)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRVLLMConfig

        with pytest.raises(ValidationError) as exc_info:
            GLMOCRVLLMConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestGLMOCRAPIConfig:
    """Tests for GLMOCRAPIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

        config = GLMOCRAPIConfig(api_base="http://localhost:8192/v1")

        assert config.model == "openai/glm-ocr"
        assert config.api_key is None
        assert config.api_base == "http://localhost:8192/v1"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.timeout == 120
        assert config.api_version is None
        assert config.extra_headers is None

    def test_with_self_hosted_server(self):
        """Test configuration for a self-hosted vLLM server."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

        config = GLMOCRAPIConfig(
            api_base="http://localhost:8000/v1",
            api_key="token-abc",
        )

        assert config.api_base == "http://localhost:8000/v1"
        assert config.api_key == "token-abc"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

        config = GLMOCRAPIConfig(
            model="openai/custom-model",
            api_key="test-key",
            api_base="https://custom.api.com/v1",
            max_tokens=2048,
            temperature=0.1,
            timeout=300,
        )

        assert config.model == "openai/custom-model"
        assert config.api_key == "test-key"
        assert config.api_base == "https://custom.api.com/v1"
        assert config.max_tokens == 2048
        assert config.temperature == 0.1
        assert config.timeout == 300

    def test_invalid_timeout(self):
        """Test that timeout below minimum raises error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

        with pytest.raises(ValidationError):
            GLMOCRAPIConfig(timeout=5)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.glmocr import GLMOCRAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            GLMOCRAPIConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestGLMOCRImports:
    """Test that all GLM-OCR exports are importable."""

    def test_extractor_import_from_text_extraction(self):
        """Test importing GLMOCRTextExtractor from the top-level module."""
        from omnidocs.tasks.text_extraction import GLMOCRTextExtractor

        assert GLMOCRTextExtractor is not None

    def test_config_imports_from_submodule(self):
        """Test importing all configs from the glmocr submodule."""
        from omnidocs.tasks.text_extraction.glmocr import (
            GLMOCRAPIConfig,
            GLMOCRPyTorchConfig,
            GLMOCRTextExtractor,
            GLMOCRVLLMConfig,
        )

        assert GLMOCRTextExtractor is not None
        assert GLMOCRPyTorchConfig is not None
        assert GLMOCRVLLMConfig is not None
        assert GLMOCRAPIConfig is not None
