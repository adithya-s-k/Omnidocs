"""
Tests for Dots OCR text extraction configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestDotsOCRPyTorchConfig:
    """Tests for DotsOCRPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        config = DotsOCRPyTorchConfig()

        assert config.model == "rednote-hilab/dots.ocr"
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.trust_remote_code is True
        assert config.device_map == "auto"
        assert config.attn_implementation == "flash_attention_2"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        config = DotsOCRPyTorchConfig(
            model="custom/dots-ocr-model",
            device="cpu",
            torch_dtype="float16",
            attn_implementation="flash_attention_2",
        )

        assert config.model == "custom/dots-ocr-model"
        assert config.device == "cpu"
        assert config.torch_dtype == "float16"
        assert config.attn_implementation == "flash_attention_2"

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        with pytest.raises(ValidationError):
            DotsOCRPyTorchConfig(torch_dtype="float64")

    def test_invalid_attn_implementation(self):
        """Test that invalid attention implementation raises error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        with pytest.raises(ValidationError):
            DotsOCRPyTorchConfig(attn_implementation="invalid")

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            DotsOCRPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestDotsOCRVLLMConfig:
    """Tests for DotsOCRVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        config = DotsOCRVLLMConfig()

        assert config.model == "rednote-hilab/dots.ocr"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 32768
        assert config.trust_remote_code is True
        assert config.dtype == "bfloat16"
        assert config.enforce_eager is False

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        config = DotsOCRVLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=16384,
            enforce_eager=True,
        )

        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 16384
        assert config.enforce_eager is True

    def test_invalid_tensor_parallel_size(self):
        """Test that invalid tensor_parallel_size raises error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        with pytest.raises(ValidationError):
            DotsOCRVLLMConfig(tensor_parallel_size=0)

    def test_invalid_gpu_memory_utilization(self):
        """Test that invalid gpu_memory_utilization raises error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        with pytest.raises(ValidationError):
            DotsOCRVLLMConfig(gpu_memory_utilization=1.5)

        with pytest.raises(ValidationError):
            DotsOCRVLLMConfig(gpu_memory_utilization=0.05)

    def test_invalid_max_model_len(self):
        """Test that invalid max_model_len raises error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        with pytest.raises(ValidationError):
            DotsOCRVLLMConfig(max_model_len=512)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRVLLMConfig

        with pytest.raises(ValidationError) as exc_info:
            DotsOCRVLLMConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestDotsOCRAPIConfig:
    """Tests for DotsOCRAPIConfig."""

    def test_api_base_required(self):
        """Test that api_base is required."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            DotsOCRAPIConfig()

        assert "api_base" in str(exc_info.value)

    def test_with_api_base(self):
        """Test configuration with api_base."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

        config = DotsOCRAPIConfig(api_base="https://api.example.com/v1")

        assert config.api_base == "https://api.example.com/v1"
        assert config.model == "dotsocr"
        assert config.api_key is None
        assert config.timeout == 120
        assert config.max_retries == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

        config = DotsOCRAPIConfig(
            api_base="https://custom.api.com/v1",
            model="custom/dots-ocr",
            api_key="test-key",
            timeout=300,
            max_retries=5,
        )

        assert config.api_base == "https://custom.api.com/v1"
        assert config.model == "custom/dots-ocr"
        assert config.api_key == "test-key"
        assert config.timeout == 300
        assert config.max_retries == 5

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.dotsocr import DotsOCRAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            DotsOCRAPIConfig(api_base="https://api.example.com/v1", unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)
