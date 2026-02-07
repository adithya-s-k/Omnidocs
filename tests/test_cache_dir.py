"""
Tests for cache directory utility functions (omnidocs.utils.cache).

Verifies that OMNIDOCS_MODELS_DIR overwrites HF_HOME so every backend
(PyTorch, VLLM, MLX, snapshot_download) downloads to the same place.
"""

import os

from omnidocs.utils.cache import configure_backend_cache, get_model_cache_dir, get_storage_info


class TestGetModelCacheDir:
    """Test get_model_cache_dir function."""

    def test_custom_dir_override(self, tmp_path):
        """custom_dir parameter takes highest priority."""
        custom_dir = tmp_path / "custom_cache"
        result = get_model_cache_dir(str(custom_dir))

        assert result == custom_dir
        assert result.exists()

    def test_omnidocs_models_dir_env(self, tmp_path, monkeypatch):
        """OMNIDOCS_MODELS_DIR env var is used when set."""
        cache_dir = tmp_path / "omnidocs_cache"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))

        result = get_model_cache_dir()

        assert result == cache_dir
        assert result.exists()

    def test_hf_home_fallback(self, tmp_path, monkeypatch):
        """Falls back to HF_HOME when OMNIDOCS_MODELS_DIR not set."""
        cache_dir = tmp_path / "hf_cache"
        monkeypatch.delenv("OMNIDOCS_MODELS_DIR", raising=False)
        monkeypatch.setenv("HF_HOME", str(cache_dir))

        result = get_model_cache_dir()

        assert result == cache_dir
        assert result.exists()

    def test_default_cache_dir(self, tmp_path, monkeypatch):
        """Falls back to ~/.cache/huggingface when no env vars set."""
        monkeypatch.delenv("OMNIDOCS_MODELS_DIR", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        def mock_expanduser(path):
            if path.startswith("~"):
                return str(tmp_path) + path[1:]
            return path

        monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

        result = get_model_cache_dir()

        expected = tmp_path / ".cache" / "huggingface"
        assert result == expected
        assert result.exists()

    def test_creates_directory(self, tmp_path, monkeypatch):
        """Directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        result = get_model_cache_dir()

        assert result.exists()
        assert result.is_dir()


class TestConfigureBackendCache:
    """Test configure_backend_cache overwrites HF_HOME."""

    def test_overwrites_hf_home(self, tmp_path, monkeypatch):
        """HF_HOME is overwritten (not setdefault) so all backends use it."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        # Set a different HF_HOME first
        monkeypatch.setenv("HF_HOME", "/some/other/path")

        configure_backend_cache()

        # Must be overwritten to the OMNIDOCS_MODELS_DIR value
        assert os.environ["HF_HOME"] == str(cache_dir)

    def test_overwrites_transformers_cache(self, tmp_path, monkeypatch):
        """TRANSFORMERS_CACHE is overwritten too."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        monkeypatch.setenv("TRANSFORMERS_CACHE", "/old/path")

        configure_backend_cache()

        assert os.environ["TRANSFORMERS_CACHE"] == str(cache_dir)

    def test_custom_cache_dir_param(self, tmp_path, monkeypatch):
        """cache_dir parameter overrides environment."""
        custom_dir = tmp_path / "custom"

        configure_backend_cache(str(custom_dir))

        assert os.environ["HF_HOME"] == str(custom_dir)
        assert os.environ["TRANSFORMERS_CACHE"] == str(custom_dir)


class TestGetStorageInfo:
    """Test get_storage_info function."""

    def test_returns_dict(self):
        """Returns a dictionary with expected keys."""
        result = get_storage_info()

        assert isinstance(result, dict)
        assert "omnidocs_cache" in result
        assert "omnidocs_models_dir_env" in result
        assert "hf_home" in result
        assert "transformers_cache" in result

    def test_storage_info_accuracy(self, tmp_path, monkeypatch):
        """Returned values match actual env vars."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        monkeypatch.setenv("HF_HOME", str(cache_dir))

        result = get_storage_info()

        assert result["omnidocs_cache"] == str(cache_dir)
        assert result["omnidocs_models_dir_env"] == str(cache_dir)
        assert result["hf_home"] == str(cache_dir)

    def test_none_when_env_not_set(self, monkeypatch):
        """Returns None for unset env vars."""
        monkeypatch.delenv("OMNIDOCS_MODELS_DIR", raising=False)

        result = get_storage_info()

        assert result["omnidocs_models_dir_env"] is None


class TestHFDownloadRespectsDir:
    """Verify that HuggingFace downloads actually land in OMNIDOCS_MODELS_DIR.

    Uses a tiny model (hf-internal-testing/tiny-random-gpt2) so download is fast.
    """

    def test_hf_hub_download_uses_cache_dir(self, tmp_path, monkeypatch):
        """huggingface_hub.hf_hub_download respects HF_HOME set by configure_backend_cache."""
        cache_dir = tmp_path / "models"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        configure_backend_cache()

        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id="hf-internal-testing/tiny-random-gpt2", filename="config.json")

        # Something must have been created inside our cache dir
        files = list(cache_dir.rglob("*"))
        assert len(files) > 0, f"No files downloaded into {cache_dir}"

        # The config.json we asked for should be somewhere in there
        config_files = [f for f in files if f.name == "config.json"]
        assert len(config_files) > 0, "config.json not found in cache dir"

    def test_snapshot_download_uses_cache_dir(self, tmp_path, monkeypatch):
        """huggingface_hub.snapshot_download respects HF_HOME."""
        cache_dir = tmp_path / "models"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        configure_backend_cache()

        from huggingface_hub import snapshot_download

        snapshot_download(repo_id="hf-internal-testing/tiny-random-gpt2")

        files = list(cache_dir.rglob("*"))
        assert len(files) > 0, f"No files downloaded into {cache_dir}"

    def test_auto_processor_uses_cache_dir(self, tmp_path, monkeypatch):
        """transformers AutoProcessor.from_pretrained uses cache_dir kwarg."""
        cache_dir = tmp_path / "models"
        monkeypatch.setenv("OMNIDOCS_MODELS_DIR", str(cache_dir))
        configure_backend_cache()

        from transformers import AutoTokenizer

        # Use a tiny tokenizer (no GPU needed)
        AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2",
            cache_dir=str(cache_dir),
        )

        files = list(cache_dir.rglob("*"))
        assert len(files) > 0, f"No files downloaded into {cache_dir}"

        tokenizer_files = [f for f in files if f.name == "tokenizer_config.json"]
        assert len(tokenizer_files) > 0, "tokenizer_config.json not found in cache dir"
