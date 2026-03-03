"""Unit tests for LightOn text extraction."""

import pytest


class TestLightOnTextPyTorchConfig:
    """Test PyTorch config validation."""

    def test_default_config(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        config = LightOnTextPyTorchConfig()
        assert config.model == "lightonai/LightOnOCR-2-1B"
        assert config.device == "auto"
        assert config.torch_dtype == "bfloat16"
        assert config.use_flash_attention is False
        assert config.max_new_tokens == 4096

    def test_custom_config(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        config = LightOnTextPyTorchConfig(
            device="cpu",
            torch_dtype="float32",
            max_new_tokens=2048,
        )
        assert config.device == "cpu"
        assert config.torch_dtype == "float32"
        assert config.max_new_tokens == 2048

    def test_extra_fields_forbidden(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        with pytest.raises(ValueError):
            LightOnTextPyTorchConfig(invalid_param="value")

    def test_invalid_dtype(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        with pytest.raises(ValueError):
            LightOnTextPyTorchConfig(torch_dtype="invalid")

    def test_invalid_device(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextPyTorchConfig

        with pytest.raises(ValueError):
            LightOnTextPyTorchConfig(device="invalid")


class TestLightOnTextVLLMConfig:
    """Test VLLM config validation."""

    def test_default_config(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextVLLMConfig

        config = LightOnTextVLLMConfig()
        assert config.model == "lightonai/LightOnOCR-2-1B"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_tokens == 4096

    def test_custom_config(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextVLLMConfig

        config = LightOnTextVLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            max_tokens=8192,
        )
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.90
        assert config.max_tokens == 8192

    def test_invalid_gpu_memory_utilization(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextVLLMConfig

        with pytest.raises(ValueError):
            LightOnTextVLLMConfig(gpu_memory_utilization=1.5)


class TestLightOnTextMLXConfig:
    """Test MLX config validation."""

    def test_default_config(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextMLXConfig

        config = LightOnTextMLXConfig()
        assert config.model == "lightonai/LightOnOCR-2-1B"
        assert config.max_tokens == 4096

    def test_custom_config(self):
        from omnidocs.tasks.text_extraction.lighton import LightOnTextMLXConfig

        config = LightOnTextMLXConfig(max_tokens=2048)
        assert config.max_tokens == 2048


class TestLightOnTextExtractorInit:
    """Test extractor initialization with different configs."""

    def test_pytorch_backend_config_type(self):
        from omnidocs.tasks.text_extraction.lighton import (
            LightOnTextPyTorchConfig,
        )

        config = LightOnTextPyTorchConfig(device="cpu")
        # Test that config is properly stored (no actual model loading in unit test)
        assert config.device == "cpu"
        assert isinstance(config, LightOnTextPyTorchConfig)

    def test_vllm_backend_config_type(self):
        from omnidocs.tasks.text_extraction.lighton import (
            LightOnTextVLLMConfig,
        )

        config = LightOnTextVLLMConfig(tensor_parallel_size=2)
        assert config.tensor_parallel_size == 2
        assert isinstance(config, LightOnTextVLLMConfig)

    def test_mlx_backend_config_type(self):
        from omnidocs.tasks.text_extraction.lighton import (
            LightOnTextMLXConfig,
        )

        config = LightOnTextMLXConfig()
        assert isinstance(config, LightOnTextMLXConfig)


class TestLightOnUtilFunctions:
    """Test utility functions."""

    def test_simple_post_process_removes_special_tokens(self):
        from omnidocs.tasks.text_extraction.lighton.utils import (
            simple_post_process,
        )

        text = "<|im_start|>Some text<|im_end|>"
        result = simple_post_process(text)
        assert "<|im_start|>" not in result
        assert "<|im_end|>" not in result
        assert "Some text" in result

    def test_simple_post_process_removes_extra_whitespace(self):
        from omnidocs.tasks.text_extraction.lighton.utils import (
            simple_post_process,
        )

        text = "   Some text   \n\n   More text   "
        result = simple_post_process(text)
        assert result.startswith("Some text")
        assert "More text" in result

    def test_parse_layout_output_basic(self):
        from omnidocs.tasks.text_extraction.lighton.utils import (
            parse_layout_output,
        )

        text = "First paragraph\n\nSecond paragraph"
        result = parse_layout_output(text)
        assert "text" in result
        assert "sections" in result
        # After simple_post_process, empty lines are removed, so double newlines won't exist
        # The text becomes "First paragraph\nSecond paragraph" with no separation
        assert len(result["sections"]) == 1
        assert "First paragraph" in result["text"]
        assert "Second paragraph" in result["text"]
