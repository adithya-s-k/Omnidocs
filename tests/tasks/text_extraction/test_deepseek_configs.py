"""
Tests for DeepSeek-OCR / DeepSeek-OCR-2 text extraction configuration classes.

Models covered:
  deepseek-ai/DeepSeek-OCR-2  (default, latest — Jan 2026, Apache 2.0)
  deepseek-ai/DeepSeek-OCR    (v1 — Oct 2024, MIT)
"""

import pytest
from pydantic import ValidationError


class TestDeepSeekOCRTextPyTorchConfig:
    """Tests for DeepSeekOCRTextPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        config = DeepSeekOCRTextPyTorchConfig()

        assert config.model == "deepseek-ai/DeepSeek-OCR-2"
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.use_flash_attention is False
        assert config.trust_remote_code is True
        assert config.cache_dir is None
        assert config.base_size == 1024
        assert config.image_size == 768
        assert config.crop_mode is True

    def test_default_model_is_v2(self):
        """Test that the default model is DeepSeek-OCR-2 (latest)."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        config = DeepSeekOCRTextPyTorchConfig()
        assert config.model == "deepseek-ai/DeepSeek-OCR-2"

    def test_v1_model(self):
        """Test switching to DeepSeek-OCR v1."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        config = DeepSeekOCRTextPyTorchConfig(model="deepseek-ai/DeepSeek-OCR")
        assert config.model == "deepseek-ai/DeepSeek-OCR"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        config = DeepSeekOCRTextPyTorchConfig(
            device="cpu",
            torch_dtype="float16",
            use_flash_attention=True,
            base_size=512,
            image_size=512,
            crop_mode=False,
        )

        assert config.device == "cpu"
        assert config.torch_dtype == "float16"
        assert config.use_flash_attention is True
        assert config.base_size == 512
        assert config.image_size == 512
        assert config.crop_mode is False

    def test_invalid_torch_dtype(self):
        """Test that invalid dtype raises error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(torch_dtype="float64")

    def test_max_new_tokens_bounds(self):
        """Test max_new_tokens validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(max_new_tokens=100)  # below min 256

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(max_new_tokens=100000)  # above max 32768

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(temperature=2.5)

    def test_base_size_bounds(self):
        """Test base_size validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(base_size=256)  # below min 512

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(base_size=4096)  # above max 2048

    def test_image_size_bounds(self):
        """Test image_size validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(image_size=128)  # below min 256

        with pytest.raises(ValidationError):
            DeepSeekOCRTextPyTorchConfig(image_size=2048)  # above max 1024

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            DeepSeekOCRTextPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)

    def test_cache_dir_optional(self):
        """Test that cache_dir is optional."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextPyTorchConfig

        config = DeepSeekOCRTextPyTorchConfig(cache_dir="/tmp/models")
        assert config.cache_dir == "/tmp/models"


class TestDeepSeekOCRTextVLLMConfig:
    """Tests for DeepSeekOCRTextVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        config = DeepSeekOCRTextVLLMConfig()

        assert config.model == "deepseek-ai/DeepSeek-OCR"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 8192
        assert config.trust_remote_code is True
        assert config.enforce_eager is False
        assert config.max_tokens == 8192
        assert config.temperature == 0.0
        assert config.download_dir is None
        assert config.disable_custom_all_reduce is False
        assert config.enable_prefix_caching is False
        assert config.mm_processor_cache_gb == 0
        assert config.use_ngram_logits_processor is True
        assert config.ngram_size == 30
        assert config.ngram_window_size == 90
        assert config.skip_special_tokens is False

    def test_default_model_is_v1(self):
        """Test that the default model is DeepSeek-OCR."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        config = DeepSeekOCRTextVLLMConfig()
        assert config.model == "deepseek-ai/DeepSeek-OCR"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        config = DeepSeekOCRTextVLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.95,
            max_model_len=16384,
            enforce_eager=True,
            max_tokens=4096,
            temperature=0.1,
        )

        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.95
        assert config.max_model_len == 16384
        assert config.enforce_eager is True
        assert config.max_tokens == 4096
        assert config.temperature == 0.1

    def test_invalid_tensor_parallel_size(self):
        """Test that invalid tensor_parallel_size raises error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextVLLMConfig(tensor_parallel_size=0)

    def test_invalid_gpu_memory_utilization(self):
        """Test that invalid gpu_memory_utilization raises error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextVLLMConfig(gpu_memory_utilization=1.5)

        with pytest.raises(ValidationError):
            DeepSeekOCRTextVLLMConfig(gpu_memory_utilization=0.05)

    def test_invalid_max_model_len(self):
        """Test that invalid max_model_len raises error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextVLLMConfig(max_model_len=512)  # below min 1024

    def test_invalid_max_tokens_bounds(self):
        """Test max_tokens validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextVLLMConfig(max_tokens=100)  # below min 256

        with pytest.raises(ValidationError):
            DeepSeekOCRTextVLLMConfig(max_tokens=8193)  # above max 32768

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        with pytest.raises(ValidationError) as exc_info:
            DeepSeekOCRTextVLLMConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)

    def test_download_dir_optional(self):
        """Test that download_dir is optional."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextVLLMConfig

        config = DeepSeekOCRTextVLLMConfig(download_dir="/data/models")
        assert config.download_dir == "/data/models"


class TestDeepSeekOCRTextMLXConfig:
    """Tests for DeepSeekOCRTextMLXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        config = DeepSeekOCRTextMLXConfig()

        assert config.model == "mlx-community/DeepSeek-OCR-4bit"
        assert config.cache_dir is None
        assert config.max_tokens == 8192
        assert config.temperature == 0.0

    def test_default_model_is_4bit(self):
        """Test that the default MLX model is 4-bit quantized."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        config = DeepSeekOCRTextMLXConfig()
        assert "4bit" in config.model

    def test_8bit_variant(self):
        """Test switching to 8-bit quantized model."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        config = DeepSeekOCRTextMLXConfig(model="mlx-community/DeepSeek-OCR-8bit")
        assert config.model == "mlx-community/DeepSeek-OCR-8bit"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        config = DeepSeekOCRTextMLXConfig(
            model="mlx-community/DeepSeek-OCR-8bit",
            max_tokens=4096,
            temperature=0.1,
            cache_dir="/tmp/mlx_cache",
        )

        assert config.model == "mlx-community/DeepSeek-OCR-8bit"
        assert config.max_tokens == 4096
        assert config.temperature == 0.1
        assert config.cache_dir == "/tmp/mlx_cache"

    def test_max_tokens_bounds(self):
        """Test max_tokens validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextMLXConfig(max_tokens=100)  # below min 256

        with pytest.raises(ValidationError):
            DeepSeekOCRTextMLXConfig(max_tokens=100000)  # above max 32768

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextMLXConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            DeepSeekOCRTextMLXConfig(temperature=2.5)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextMLXConfig

        with pytest.raises(ValidationError) as exc_info:
            DeepSeekOCRTextMLXConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestDeepSeekOCRTextAPIConfig:
    """Tests for DeepSeekOCRTextAPIConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        config = DeepSeekOCRTextAPIConfig()

        assert config.model == "novita/deepseek/deepseek-ocr"
        assert config.api_key is None
        assert config.api_base is None
        assert config.max_tokens == 8192
        assert config.temperature == 0.0
        assert config.timeout == 180
        assert config.api_version is None
        assert config.extra_headers is None

    def test_greedy_decoding_default(self):
        """Test that temperature defaults to 0.0 (greedy decoding) for OCR accuracy."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        config = DeepSeekOCRTextAPIConfig()
        assert config.temperature == 0.0

    def test_with_api_key(self):
        """Test configuration with explicit api_key."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        config = DeepSeekOCRTextAPIConfig(api_key="novita-test-key")

        assert config.api_key == "novita-test-key"

    def test_with_api_base_override(self):
        """Test overriding api_base."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        config = DeepSeekOCRTextAPIConfig(
            api_base="https://api.novita.ai/v3/openai"
        )
        assert config.api_base == "https://api.novita.ai/v3/openai"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        config = DeepSeekOCRTextAPIConfig(
            model="novita/deepseek/deepseek-ocr",
            api_key="test-key",
            max_tokens=4096,
            timeout=300,
            extra_headers={"X-Custom-Header": "value"},
        )

        assert config.model == "novita/deepseek/deepseek-ocr"
        assert config.api_key == "test-key"
        assert config.max_tokens == 4096
        assert config.timeout == 300
        assert config.extra_headers == {"X-Custom-Header": "value"}

    def test_max_tokens_bounds(self):
        """Test max_tokens validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextAPIConfig(max_tokens=100)  # below min 256

    def test_timeout_minimum(self):
        """Test timeout minimum validation."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextAPIConfig(timeout=5)  # below min 10

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        with pytest.raises(ValidationError):
            DeepSeekOCRTextAPIConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            DeepSeekOCRTextAPIConfig(temperature=2.5)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.deepseek import DeepSeekOCRTextAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            DeepSeekOCRTextAPIConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestDeepSeekOCRImports:
    """Test that all DeepSeek-OCR exports are importable."""

    def test_import_extractor_from_main_module(self):
        """Test importing DeepSeekOCRTextExtractor from main text_extraction module."""
        from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor

        assert DeepSeekOCRTextExtractor is not None

    def test_import_configs_from_submodule(self):
        """Test importing all configs from deepseek submodule."""
        from omnidocs.tasks.text_extraction.deepseek import (
            DeepSeekOCRTextAPIConfig,
            DeepSeekOCRTextExtractor,
            DeepSeekOCRTextMLXConfig,
            DeepSeekOCRTextPyTorchConfig,
            DeepSeekOCRTextVLLMConfig,
        )

        assert DeepSeekOCRTextExtractor is not None
        assert DeepSeekOCRTextPyTorchConfig is not None
        assert DeepSeekOCRTextVLLMConfig is not None
        assert DeepSeekOCRTextMLXConfig is not None
        assert DeepSeekOCRTextAPIConfig is not None

    def test_extractor_is_base_subclass(self):
        """Test that extractor inherits from BaseTextExtractor."""
        from omnidocs.tasks.text_extraction import DeepSeekOCRTextExtractor
        from omnidocs.tasks.text_extraction.base import BaseTextExtractor

        assert issubclass(DeepSeekOCRTextExtractor, BaseTextExtractor)
