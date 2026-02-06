"""
Tests for cache utility functions.
"""

import os

from omnidocs.utils.cache import configure_backend_cache, get_cache_info, get_model_cache_dir


class TestGetModelCacheDir:
    """Test get_model_cache_dir function."""

    def test_custom_dir_override(self, tmp_path):
        """Test that custom_dir parameter overrides environment variables."""
        custom_dir = tmp_path / "custom_cache"
        result = get_model_cache_dir(str(custom_dir))

        assert result == custom_dir
        assert result.exists()

    def test_omnidocs_model_cache_env(self, tmp_path, monkeypatch):
        """Test OMNIDOCS_MODEL_CACHE environment variable."""
        cache_dir = tmp_path / "omnidocs_cache"
        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(cache_dir))

        result = get_model_cache_dir()

        assert result == cache_dir
        assert result.exists()

    def test_hf_home_fallback(self, tmp_path, monkeypatch):
        """Test fallback to HF_HOME when OMNIDOCS_MODEL_CACHE not set."""
        cache_dir = tmp_path / "hf_cache"
        monkeypatch.delenv("OMNIDOCS_MODEL_CACHE", raising=False)
        monkeypatch.setenv("HF_HOME", str(cache_dir))

        result = get_model_cache_dir()

        assert result == cache_dir
        assert result.exists()

    def test_default_cache_dir(self, tmp_path, monkeypatch):
        """Test default cache directory when no env vars set."""
        monkeypatch.delenv("OMNIDOCS_MODEL_CACHE", raising=False)
        monkeypatch.delenv("HF_HOME", raising=False)

        # Mock os.path.expanduser to use tmp_path instead of real home
        def mock_expanduser(path):
            if path.startswith("~"):
                return str(tmp_path) + path[1:]
            return path

        import os
        monkeypatch.setattr(os.path, "expanduser", mock_expanduser)

        result = get_model_cache_dir()

        expected = tmp_path / ".cache" / "huggingface"
        assert result == expected
        assert result.exists()

    def test_creates_directory(self, tmp_path, monkeypatch):
        """Test that directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(cache_dir))
        result = get_model_cache_dir()

        assert result.exists()
        assert result.is_dir()

    def test_expanduser(self, tmp_path, monkeypatch):
        """Test that ~ is expanded to home directory."""
        # Use tmp_path to avoid creating directories in real home
        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(tmp_path / "test_cache"))

        result = get_model_cache_dir()

        assert "~" not in str(result)
        assert result.is_absolute()


class TestConfigureBackendCache:
    """Test configure_backend_cache function."""

    def test_sets_hf_home(self, tmp_path, monkeypatch):
        """Test that HF_HOME is set."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(cache_dir))
        monkeypatch.delenv("HF_HOME", raising=False)

        configure_backend_cache()

        assert os.environ["HF_HOME"] == str(cache_dir)

    def test_sets_transformers_cache(self, tmp_path, monkeypatch):
        """Test that TRANSFORMERS_CACHE is set."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(cache_dir))
        monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

        configure_backend_cache()

        assert os.environ["TRANSFORMERS_CACHE"] == str(cache_dir)

    def test_preserves_existing_env_vars(self, tmp_path, monkeypatch):
        """Test that existing env vars are not overridden."""
        existing_hf_home = tmp_path / "existing_hf"
        cache_dir = tmp_path / "cache"

        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(cache_dir))
        monkeypatch.setenv("HF_HOME", str(existing_hf_home))

        configure_backend_cache()

        # Should preserve existing value
        assert os.environ["HF_HOME"] == str(existing_hf_home)

    def test_custom_cache_dir_param(self, tmp_path, monkeypatch):
        """Test that cache_dir parameter overrides environment."""
        custom_dir = tmp_path / "custom"
        monkeypatch.delenv("HF_HOME", raising=False)

        configure_backend_cache(str(custom_dir))

        assert os.environ["HF_HOME"] == str(custom_dir)


class TestGetCacheInfo:
    """Test get_cache_info function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = get_cache_info()

        assert isinstance(result, dict)
        assert "omnidocs_cache" in result
        assert "omnidocs_model_cache_env" in result
        assert "hf_home" in result
        assert "transformers_cache" in result

    def test_cache_info_accuracy(self, tmp_path, monkeypatch):
        """Test that returned info is accurate."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("OMNIDOCS_MODEL_CACHE", str(cache_dir))
        monkeypatch.setenv("HF_HOME", str(cache_dir))

        result = get_cache_info()

        assert result["omnidocs_cache"] == str(cache_dir)
        assert result["omnidocs_model_cache_env"] == str(cache_dir)
        assert result["hf_home"] == str(cache_dir)

    def test_none_when_env_not_set(self, monkeypatch):
        """Test that None is returned when env vars not set."""
        monkeypatch.delenv("OMNIDOCS_MODEL_CACHE", raising=False)

        result = get_cache_info()

        assert result["omnidocs_model_cache_env"] is None
