"""
Tests for unified model cache system.

Tests cache key normalization, cross-task sharing, runtime param exclusion,
reference counting, and LRU eviction.

Run with: pytest tests/test_cache.py -v
"""

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnidocs.cache import (
    _global_cache,
    add_reference,
    clear_cache,
    get_cache_info,
    get_cache_key,
    get_cached,
    set_cache_config,
    set_cached,
)

# ============= Fake configs for testing (no GPU/model deps) =============


class QwenTextVLLMConfig(BaseModel):
    model: str = Field(default="Qwen/Qwen3-VL-8B-Instruct")
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 32768
    max_tokens: int = 8192
    temperature: float = 0.1
    model_config = ConfigDict(extra="forbid")


class QwenLayoutVLLMConfig(BaseModel):
    model: str = Field(default="Qwen/Qwen3-VL-8B-Instruct")
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 32768
    max_tokens: int = 4096
    temperature: float = 0.1
    model_config = ConfigDict(extra="forbid")


class QwenTextMLXConfig(BaseModel):
    model: str = Field(default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    max_tokens: int = 8192
    temperature: float = 0.1
    model_config = ConfigDict(extra="forbid")


class QwenLayoutMLXConfig(BaseModel):
    model: str = Field(default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    max_tokens: int = 4096
    temperature: float = 0.1
    model_config = ConfigDict(extra="forbid")


class QwenTextAPIConfig(BaseModel):
    model: str = Field(default="Qwen/Qwen3-VL-8B-Instruct")
    api_key: str = "test"
    max_tokens: int = 8192
    timeout: int = 180
    model_config = ConfigDict(extra="forbid")


class MinerUVLTextVLLMConfig(BaseModel):
    model: str = Field(default="MinerU2.5-2509-1.2B")
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    max_tokens: int = 4096
    model_config = ConfigDict(extra="forbid")


class MinerUVLLayoutVLLMConfig(BaseModel):
    model: str = Field(default="MinerU2.5-2509-1.2B")
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    max_tokens: int = 2048
    model_config = ConfigDict(extra="forbid")


class NanonetsTextPyTorchConfig(BaseModel):
    model: str = Field(default="nanonets/Nanonets-OCR-s")
    device: str = "cuda"
    max_new_tokens: int = 4096
    temperature: float = 0.0
    model_config = ConfigDict(extra="forbid")


class DotsOCRPyTorchConfig(BaseModel):
    model: str = Field(default="rednote-hilab/dots.ocr")
    device: str = "cuda"
    max_new_tokens: int = 8192
    temperature: float = 0.0
    model_config = ConfigDict(extra="forbid")


class DotsOCRVLLMConfig(BaseModel):
    model: str = Field(default="rednote-hilab/dots.ocr")
    gpu_memory_utilization: float = 0.85
    max_tokens: int = 8192
    model_config = ConfigDict(extra="forbid")


class RTDETRConfig(BaseModel):
    device: str = "cuda"
    model_name: str = "HuggingPanda/docling-layout"
    image_size: int = 640
    confidence: float = 0.4
    model_config = ConfigDict(extra="forbid")


class DocLayoutYOLOConfig(BaseModel):
    device: str = "cuda"
    img_size: int = 1024
    confidence: float = 0.25
    model_config = ConfigDict(extra="forbid")


class TableFormerConfig(BaseModel):
    mode: str = "fast"
    device: str = "auto"
    model_config = ConfigDict(extra="forbid")


class EasyOCRConfig(BaseModel):
    languages: list = Field(default=["en"])
    gpu: bool = True
    model_config = ConfigDict(extra="forbid")


class PaddleOCRConfig(BaseModel):
    lang: str = "en"
    device: str = "cpu"
    model_config = ConfigDict(extra="forbid")


# ============= Fixtures =============


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============= Cache Key Normalization Tests =============


class TestCacheKeyNormalization:
    """Test that config class names are normalized to strip task markers."""

    def test_text_and_layout_same_model_vllm(self):
        """QwenTextVLLMConfig and QwenLayoutVLLMConfig should produce the same key."""
        text_key = get_cache_key(QwenTextVLLMConfig())
        layout_key = get_cache_key(QwenLayoutVLLMConfig())
        assert text_key == layout_key

    def test_text_and_layout_same_model_mlx(self):
        """QwenTextMLXConfig and QwenLayoutMLXConfig should produce the same key."""
        text_key = get_cache_key(QwenTextMLXConfig())
        layout_key = get_cache_key(QwenLayoutMLXConfig())
        assert text_key == layout_key

    def test_mineruvl_text_and_layout_same_key(self):
        """MinerUVL text and layout configs should produce the same key."""
        text_key = get_cache_key(MinerUVLTextVLLMConfig())
        layout_key = get_cache_key(MinerUVLLayoutVLLMConfig())
        assert text_key == layout_key

    def test_different_models_different_keys(self):
        """Different model IDs should produce different keys."""
        default = get_cache_key(QwenTextVLLMConfig())
        custom = get_cache_key(QwenTextVLLMConfig(model="Qwen/Qwen3-VL-2B-Instruct"))
        assert default != custom

    def test_different_backends_different_keys(self):
        """VLLM and MLX keys should differ (different type + different model ID)."""
        vllm_key = get_cache_key(QwenTextVLLMConfig())
        mlx_key = get_cache_key(QwenTextMLXConfig())
        assert vllm_key != mlx_key

    def test_key_contains_normalized_type(self):
        """Key should contain 'Qwen:VLLMConfig' not 'QwenTextVLLMConfig'."""
        key = get_cache_key(QwenTextVLLMConfig())
        assert "Qwen:VLLMConfig" in key
        assert "QwenText" not in key

    def test_ocr_marker_stripped(self):
        """OCR marker in DotsOCRPyTorchConfig should be stripped."""
        key = get_cache_key(DotsOCRPyTorchConfig())
        assert "Dots:PyTorchConfig" in key
        assert "DotsOCR" not in key

    def test_ocr_marker_stripped_vllm(self):
        """OCR marker in DotsOCRVLLMConfig should be stripped."""
        key = get_cache_key(DotsOCRVLLMConfig())
        assert "Dots:VLLMConfig" in key


class TestEmptyModelFamilyGuard:
    """Test that configs starting with a task marker aren't corrupted."""

    def test_tableformer_config_not_corrupted(self):
        """TableFormerConfig should NOT strip 'Table' since model_family would be empty."""
        key = get_cache_key(TableFormerConfig())
        assert "TableFormerConfig" in key
        assert key.startswith("TableFormerConfig")
        assert not key.startswith(":")

    def test_rtdetr_config_no_marker(self):
        """RTDETRConfig has no task marker, should stay as-is."""
        key = get_cache_key(RTDETRConfig())
        assert "RTDETRConfig" in key


class TestRuntimeParamExclusion:
    """Test that runtime params (max_tokens, temperature) are excluded from keys."""

    def test_max_tokens_excluded(self):
        """Different max_tokens should produce the same key."""
        key1 = get_cache_key(QwenTextVLLMConfig(max_tokens=4096))
        key2 = get_cache_key(QwenTextVLLMConfig(max_tokens=16384))
        assert key1 == key2

    def test_temperature_excluded(self):
        """Different temperature should produce the same key."""
        key1 = get_cache_key(QwenTextMLXConfig(temperature=0.0))
        key2 = get_cache_key(QwenTextMLXConfig(temperature=1.0))
        assert key1 == key2

    def test_max_new_tokens_excluded(self):
        """Different max_new_tokens should produce the same key."""
        key1 = get_cache_key(NanonetsTextPyTorchConfig(max_new_tokens=2048))
        key2 = get_cache_key(NanonetsTextPyTorchConfig(max_new_tokens=8192))
        assert key1 == key2

    def test_model_loading_params_included(self):
        """Params that affect model loading (gpu_memory, model_len) should differ."""
        key1 = get_cache_key(QwenTextVLLMConfig(gpu_memory_utilization=0.8))
        key2 = get_cache_key(QwenTextVLLMConfig(gpu_memory_utilization=0.95))
        assert key1 != key2

    def test_model_id_included(self):
        """Model ID should be in the key."""
        key = get_cache_key(QwenTextMLXConfig())
        assert "mlx-community/Qwen3-VL-8B-Instruct-4bit" in key


# ============= Cache Get/Set Tests =============


class TestCacheGetSet:
    """Test basic cache get/set operations."""

    def test_set_and_get(self):
        """Set and retrieve a value from cache."""
        key = get_cache_key(QwenTextMLXConfig())
        sentinel = ("fake_model", "fake_processor")
        set_cached(key, sentinel)

        result = get_cached(key)
        assert result is sentinel

    def test_get_missing_key(self):
        """Getting a missing key returns None."""
        result = get_cached("nonexistent:key")
        assert result is None

    def test_cross_task_sharing(self):
        """Text extractor caches model, layout detector retrieves it."""
        text_config = QwenTextMLXConfig()
        layout_config = QwenLayoutMLXConfig()

        # Text extractor stores model
        text_key = get_cache_key(text_config)
        fake_model = ("shared_backend", "shared_processor")
        set_cached(text_key, fake_model, owner=object())

        # Layout detector retrieves it
        layout_key = get_cache_key(layout_config)
        result = get_cached(layout_key)

        assert result is fake_model
        assert text_key == layout_key

    def test_api_config_not_cached(self):
        """API configs produce different keys from local configs (different fields)."""
        api_key = get_cache_key(QwenTextAPIConfig())
        vllm_key = get_cache_key(QwenTextVLLMConfig())
        assert api_key != vllm_key

    def test_cache_info(self):
        """Cache info reports correct entry count."""
        key = get_cache_key(QwenTextMLXConfig())
        set_cached(key, ("model",))

        info = get_cache_info()
        assert info["num_entries"] == 1
        assert key in info["keys"]


# ============= Reference Counting Tests =============


class TestReferenceCounting:
    """Test reference counting for cache entries."""

    def test_owner_adds_reference(self):
        """Setting cache with owner adds a reference."""
        key = get_cache_key(QwenTextMLXConfig())
        owner = object()
        set_cached(key, ("model",), owner=owner)

        info = get_cache_info()
        assert info["entries"][key]["ref_count"] == 1

    def test_add_reference_increments(self):
        """add_reference increases ref_count."""
        key = get_cache_key(QwenTextMLXConfig())
        owner1 = object()
        set_cached(key, ("model",), owner=owner1)

        owner2 = object()
        add_reference(key, owner2)

        info = get_cache_info()
        assert info["entries"][key]["ref_count"] == 2

    def test_cross_task_reference(self):
        """Both text and layout extractors track as references."""
        text_config = QwenTextMLXConfig()
        layout_config = QwenLayoutMLXConfig()

        key = get_cache_key(text_config)
        text_owner = object()
        set_cached(key, ("model", "processor"), owner=text_owner)

        layout_owner = object()
        layout_key = get_cache_key(layout_config)
        add_reference(layout_key, layout_owner)

        info = get_cache_info()
        assert info["entries"][key]["ref_count"] == 2

    def test_weak_ref_cleanup(self):
        """Deleting owner reduces ref_count after cleanup."""
        key = get_cache_key(QwenTextMLXConfig())
        owner = _DummyOwner()
        set_cached(key, ("model",), owner=owner)

        assert get_cache_info()["entries"][key]["ref_count"] == 1

        del owner
        # Force cleanup
        _global_cache._cache[key].cleanup_dead_refs()

        assert get_cache_info()["entries"][key]["ref_count"] == 0


# ============= LRU Eviction Tests =============


class TestLRUEviction:
    """Test LRU eviction behavior."""

    def test_eviction_when_over_limit(self):
        """Oldest entry gets evicted when cache is full."""
        set_cache_config(max_entries=2)

        set_cached("key1", "val1")
        set_cached("key2", "val2")
        set_cached("key3", "val3")  # Should evict key1

        assert get_cached("key1") is None
        assert get_cached("key2") == "val2"
        assert get_cached("key3") == "val3"

    def test_evict_unreferenced_first(self):
        """Unreferenced entries should be evicted before referenced ones."""
        set_cache_config(max_entries=2)

        owner = _DummyOwner()
        set_cached("referenced", "val1", owner=owner)  # ref_count=1
        set_cached("unreferenced", "val2")  # ref_count=0

        # Adding a third should evict unreferenced, not referenced
        set_cached("new", "val3")

        assert get_cached("referenced") == "val1"
        assert get_cached("unreferenced") is None
        assert get_cached("new") == "val3"


# ============= Single-Backend Config Tests =============


class TestSingleBackendKeys:
    """Test cache keys for single-backend models."""

    def test_rtdetr_key_stable(self):
        """Same RTDETRConfig produces same key."""
        key1 = get_cache_key(RTDETRConfig())
        key2 = get_cache_key(RTDETRConfig())
        assert key1 == key2

    def test_doclayout_yolo_key(self):
        """DocLayoutYOLOConfig key includes device and config."""
        key = get_cache_key(DocLayoutYOLOConfig())
        assert "confidence=0.25" in key
        assert "img_size=1024" in key

    def test_easyocr_key(self):
        """EasyOCR with same languages produces same key."""
        key1 = get_cache_key(EasyOCRConfig(languages=["en"]))
        key2 = get_cache_key(EasyOCRConfig(languages=["en"]))
        assert key1 == key2

    def test_easyocr_different_langs(self):
        """EasyOCR with different languages produces different keys."""
        key1 = get_cache_key(EasyOCRConfig(languages=["en"]))
        key2 = get_cache_key(EasyOCRConfig(languages=["en", "ch_sim"]))
        assert key1 != key2

    def test_paddleocr_key(self):
        """PaddleOCR key includes lang and device."""
        key = get_cache_key(PaddleOCRConfig())
        assert "lang=en" in key
        assert "device=cpu" in key


# ============= Integration: Real Config Classes =============


class TestRealConfigs:
    """Test with actual omnidocs config classes (no model loading)."""

    def test_qwen_vllm_cross_task(self):
        """Real Qwen VLLM configs share cache key across tasks."""
        from omnidocs.tasks.layout_extraction.qwen.vllm import QwenLayoutVLLMConfig
        from omnidocs.tasks.text_extraction.qwen.vllm import QwenTextVLLMConfig

        text_key = get_cache_key(QwenTextVLLMConfig())
        layout_key = get_cache_key(QwenLayoutVLLMConfig())
        assert text_key == layout_key

    def test_qwen_mlx_cross_task(self):
        """Real Qwen MLX configs share cache key across tasks."""
        from omnidocs.tasks.layout_extraction.qwen.mlx import QwenLayoutMLXConfig
        from omnidocs.tasks.text_extraction.qwen.mlx import QwenTextMLXConfig

        text_key = get_cache_key(QwenTextMLXConfig())
        layout_key = get_cache_key(QwenLayoutMLXConfig())
        assert text_key == layout_key

    def test_mineruvl_vllm_cross_task(self):
        """Real MinerUVL VLLM configs share cache key across tasks."""
        from omnidocs.tasks.layout_extraction.mineruvl.vllm import MinerUVLLayoutVLLMConfig
        from omnidocs.tasks.text_extraction.mineruvl.vllm import MinerUVLTextVLLMConfig

        text_key = get_cache_key(MinerUVLTextVLLMConfig())
        layout_key = get_cache_key(MinerUVLLayoutVLLMConfig())
        assert text_key == layout_key

    def test_mineruvl_pytorch_cross_task(self):
        """Real MinerUVL PyTorch configs share cache key across tasks."""
        from omnidocs.tasks.layout_extraction.mineruvl.pytorch import MinerUVLLayoutPyTorchConfig
        from omnidocs.tasks.text_extraction.mineruvl.pytorch import MinerUVLTextPyTorchConfig

        text_key = get_cache_key(MinerUVLTextPyTorchConfig())
        layout_key = get_cache_key(MinerUVLLayoutPyTorchConfig())
        assert text_key == layout_key

    def test_qwen_pytorch_cross_task(self):
        """Real Qwen PyTorch configs share cache key across tasks."""
        from omnidocs.tasks.layout_extraction.qwen.pytorch import QwenLayoutPyTorchConfig
        from omnidocs.tasks.text_extraction.qwen.pytorch import QwenTextPyTorchConfig

        text_key = get_cache_key(QwenTextPyTorchConfig())
        layout_key = get_cache_key(QwenLayoutPyTorchConfig())
        assert text_key == layout_key

    def test_different_families_no_sharing(self):
        """Qwen and MinerUVL should NOT share keys."""
        from omnidocs.tasks.text_extraction.mineruvl.vllm import MinerUVLTextVLLMConfig
        from omnidocs.tasks.text_extraction.qwen.vllm import QwenTextVLLMConfig

        qwen_key = get_cache_key(QwenTextVLLMConfig())
        mineruvl_key = get_cache_key(MinerUVLTextVLLMConfig())
        assert qwen_key != mineruvl_key


# ============= Helpers =============


class _DummyOwner:
    """Dummy class that supports weak references (for testing)."""

    pass
