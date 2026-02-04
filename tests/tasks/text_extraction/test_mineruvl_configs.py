"""
Tests for MinerU VL text extraction configuration classes.
"""

import pytest
from pydantic import ValidationError


class TestMinerUVLTextPyTorchConfig:
    """Tests for MinerUVLTextPyTorchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        config = MinerUVLTextPyTorchConfig()

        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.device == "auto"
        assert config.torch_dtype == "float16"
        assert config.use_flash_attention is True
        assert config.trust_remote_code is True
        assert config.layout_image_size == (1036, 1036)
        assert config.max_new_tokens == 4096
        assert config.device_map == "auto"

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        config = MinerUVLTextPyTorchConfig(
            model="custom/mineru-model",
            device="cuda",
            torch_dtype="bfloat16",
            use_flash_attention=False,
            max_new_tokens=8192,
        )

        assert config.model == "custom/mineru-model"
        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.use_flash_attention is False
        assert config.max_new_tokens == 8192

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        with pytest.raises(ValidationError):
            MinerUVLTextPyTorchConfig(torch_dtype="float64")

    def test_invalid_device(self):
        """Test that invalid device raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        with pytest.raises(ValidationError):
            MinerUVLTextPyTorchConfig(device="tpu")

    def test_invalid_max_new_tokens(self):
        """Test that invalid max_new_tokens raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        with pytest.raises(ValidationError):
            MinerUVLTextPyTorchConfig(max_new_tokens=100)  # Below min 256

        with pytest.raises(ValidationError):
            MinerUVLTextPyTorchConfig(max_new_tokens=20000)  # Above max 16384

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextPyTorchConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLTextPyTorchConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLTextVLLMConfig:
    """Tests for MinerUVLTextVLLMConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

        config = MinerUVLTextVLLMConfig()

        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.85
        assert config.max_model_len == 16384
        assert config.trust_remote_code is True
        assert config.enforce_eager is True
        assert config.layout_image_size == (1036, 1036)
        assert config.max_tokens == 4096

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

        config = MinerUVLTextVLLMConfig(
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            enforce_eager=False,
        )

        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 8192
        assert config.enforce_eager is False

    def test_invalid_tensor_parallel_size(self):
        """Test that invalid tensor_parallel_size raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

        with pytest.raises(ValidationError):
            MinerUVLTextVLLMConfig(tensor_parallel_size=0)

    def test_invalid_gpu_memory_utilization(self):
        """Test that invalid gpu_memory_utilization raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

        with pytest.raises(ValidationError):
            MinerUVLTextVLLMConfig(gpu_memory_utilization=1.5)

        with pytest.raises(ValidationError):
            MinerUVLTextVLLMConfig(gpu_memory_utilization=0.05)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextVLLMConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLTextVLLMConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLTextMLXConfig:
    """Tests for MinerUVLTextMLXConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

        config = MinerUVLTextMLXConfig()

        assert config.model == "opendatalab/MinerU2.5-2509-1.2B"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.layout_image_size == (1036, 1036)

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

        config = MinerUVLTextMLXConfig(
            model="custom/mineru-mlx",
            max_tokens=8192,
            temperature=0.5,
        )

        assert config.model == "custom/mineru-mlx"
        assert config.max_tokens == 8192
        assert config.temperature == 0.5

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

        with pytest.raises(ValidationError):
            MinerUVLTextMLXConfig(temperature=-0.5)

        with pytest.raises(ValidationError):
            MinerUVLTextMLXConfig(temperature=3.0)

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextMLXConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLTextMLXConfig(unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLTextAPIConfig:
    """Tests for MinerUVLTextAPIConfig."""

    def test_server_url_required(self):
        """Test that server_url is required."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLTextAPIConfig()

        assert "server_url" in str(exc_info.value)

    def test_with_server_url(self):
        """Test configuration with server_url."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextAPIConfig

        config = MinerUVLTextAPIConfig(server_url="https://api.example.com")

        assert config.server_url == "https://api.example.com"
        assert config.model_name == "mineru-vl"
        assert config.timeout == 300
        assert config.max_retries == 3
        assert config.max_concurrency == 100
        assert config.api_key is None

    def test_custom_values(self):
        """Test custom configuration values."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextAPIConfig

        config = MinerUVLTextAPIConfig(
            server_url="https://custom.api.com",
            model_name="custom-mineru",
            timeout=600,
            max_retries=5,
            max_concurrency=50,
            api_key="test-key",
        )

        assert config.server_url == "https://custom.api.com"
        assert config.model_name == "custom-mineru"
        assert config.timeout == 600
        assert config.max_retries == 5
        assert config.max_concurrency == 50
        assert config.api_key == "test-key"

    def test_extra_forbid(self):
        """Test that extra parameters raise error."""
        from omnidocs.tasks.text_extraction.mineruvl import MinerUVLTextAPIConfig

        with pytest.raises(ValidationError) as exc_info:
            MinerUVLTextAPIConfig(server_url="https://api.example.com", unknown_param=True)

        assert "extra_forbidden" in str(exc_info.value)


class TestMinerUVLImports:
    """Test that all MinerU VL exports are importable."""

    def test_text_extractor_import(self):
        """Test importing MinerUVLTextExtractor from text_extraction."""
        from omnidocs.tasks.text_extraction import MinerUVLTextExtractor

        assert MinerUVLTextExtractor is not None

    def test_config_imports_from_submodule(self):
        """Test importing configs from mineruvl submodule."""
        from omnidocs.tasks.text_extraction.mineruvl import (
            MinerUVLTextAPIConfig,
            MinerUVLTextMLXConfig,
            MinerUVLTextPyTorchConfig,
            MinerUVLTextVLLMConfig,
        )

        assert MinerUVLTextPyTorchConfig is not None
        assert MinerUVLTextVLLMConfig is not None
        assert MinerUVLTextMLXConfig is not None
        assert MinerUVLTextAPIConfig is not None

    def test_utils_imports(self):
        """Test importing utilities from mineruvl."""
        from omnidocs.tasks.text_extraction.mineruvl import (
            BlockType,
            ContentBlock,
            MinerUSamplingParams,
            convert_otsl_to_html,
            parse_layout_output,
        )

        assert BlockType is not None
        assert ContentBlock is not None
        assert MinerUSamplingParams is not None
        assert parse_layout_output is not None
        assert convert_otsl_to_html is not None


class TestContentBlock:
    """Tests for ContentBlock data structure."""

    def test_valid_content_block(self):
        """Test creating a valid ContentBlock."""
        from omnidocs.tasks.text_extraction.mineruvl import BlockType, ContentBlock

        block = ContentBlock(
            type=BlockType.TEXT,
            bbox=[0.1, 0.2, 0.9, 0.8],
            angle=None,
            content="Test content",
        )

        assert block.type == BlockType.TEXT
        assert block.bbox == [0.1, 0.2, 0.9, 0.8]
        assert block.angle is None
        assert block.content == "Test content"

    def test_content_block_with_angle(self):
        """Test ContentBlock with rotation angle."""
        from omnidocs.tasks.text_extraction.mineruvl import BlockType, ContentBlock

        block = ContentBlock(
            type=BlockType.TABLE,
            bbox=[0.0, 0.0, 0.5, 0.5],
            angle=90,
        )

        assert block.angle == 90

    def test_invalid_bbox_range(self):
        """Test that invalid bbox range raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import BlockType, ContentBlock

        with pytest.raises(ValueError):
            ContentBlock(
                type=BlockType.TEXT,
                bbox=[0.1, 0.2, 1.5, 0.8],  # x2 > 1.0
            )

    def test_invalid_bbox_order(self):
        """Test that invalid bbox order raises error."""
        from omnidocs.tasks.text_extraction.mineruvl import BlockType, ContentBlock

        with pytest.raises(ValueError):
            ContentBlock(
                type=BlockType.TEXT,
                bbox=[0.9, 0.2, 0.1, 0.8],  # x1 > x2
            )

    def test_content_block_to_absolute(self):
        """Test converting bbox to absolute coordinates."""
        from omnidocs.tasks.text_extraction.mineruvl import BlockType, ContentBlock

        block = ContentBlock(
            type=BlockType.TEXT,
            bbox=[0.1, 0.2, 0.5, 0.6],
        )

        abs_coords = block.to_absolute(1000, 800)
        assert abs_coords == [100, 160, 500, 480]


class TestParseLayoutOutput:
    """Tests for layout output parsing."""

    def test_parse_valid_output(self):
        """Test parsing valid layout output."""
        from omnidocs.tasks.text_extraction.mineruvl import BlockType, parse_layout_output

        output = "<|box_start|>100 200 500 400<|box_end|><|ref_start|>text<|ref_end|>"
        blocks = parse_layout_output(output)

        assert len(blocks) == 1
        assert blocks[0].type == BlockType.TEXT
        assert blocks[0].bbox == [0.1, 0.2, 0.5, 0.4]

    def test_parse_with_rotation(self):
        """Test parsing output with rotation token."""
        from omnidocs.tasks.text_extraction.mineruvl import parse_layout_output

        output = "<|box_start|>100 200 500 400<|box_end|><|ref_start|>text<|ref_end|><|rotate_right|>"
        blocks = parse_layout_output(output)

        assert len(blocks) == 1
        assert blocks[0].angle == 90

    def test_parse_multiple_blocks(self):
        """Test parsing multiple blocks."""
        from omnidocs.tasks.text_extraction.mineruvl import parse_layout_output

        output = """<|box_start|>100 200 500 400<|box_end|><|ref_start|>text<|ref_end|>
<|box_start|>0 0 1000 100<|box_end|><|ref_start|>title<|ref_end|>"""
        blocks = parse_layout_output(output)

        assert len(blocks) == 2

    def test_parse_invalid_output(self):
        """Test that invalid output returns empty list."""
        from omnidocs.tasks.text_extraction.mineruvl import parse_layout_output

        output = "invalid output without tokens"
        blocks = parse_layout_output(output)

        assert len(blocks) == 0


class TestConvertOTSLToHTML:
    """Tests for OTSL to HTML conversion."""

    def test_simple_table(self):
        """Test converting simple OTSL table."""
        from omnidocs.tasks.text_extraction.mineruvl import convert_otsl_to_html

        otsl = "<fcel>Header1<fcel>Header2<nl><fcel>Data1<fcel>Data2"
        html = convert_otsl_to_html(otsl)

        assert "<table>" in html
        assert "</table>" in html
        assert "<tr>" in html
        assert "Header1" in html
        assert "Data1" in html

    def test_already_html(self):
        """Test that HTML input is returned as-is."""
        from omnidocs.tasks.text_extraction.mineruvl import convert_otsl_to_html

        html_input = "<table><tr><td>Test</td></tr></table>"
        result = convert_otsl_to_html(html_input)

        assert result == html_input

    def test_empty_cells(self):
        """Test handling empty cells."""
        from omnidocs.tasks.text_extraction.mineruvl import convert_otsl_to_html

        otsl = "<fcel>Data<ecel><nl><ecel><fcel>More"
        html = convert_otsl_to_html(otsl)

        assert "<table>" in html
