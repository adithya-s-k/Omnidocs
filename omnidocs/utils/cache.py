"""
Unified model cache directory management for OmniDocs.

When OMNIDOCS_MODELS_DIR is set, ALL model downloads (PyTorch, VLLM, MLX,
snapshot_download) go into that directory. It overwrites HF_HOME so every
backend respects the same path.

Environment Variables:
    OMNIDOCS_MODELS_DIR: Primary cache directory for all OmniDocs models.
                         Overwrites HF_HOME when set.
    HF_HOME: HuggingFace cache directory (used as fallback).

Example:
    ```bash
    export OMNIDOCS_MODELS_DIR=/data/models
    ```

    ```python
    from omnidocs.utils.cache import get_model_cache_dir

    cache_dir = get_model_cache_dir()  # -> /data/models
    ```
"""

import os
from pathlib import Path
from typing import Optional


def get_model_cache_dir(custom_dir: Optional[str] = None) -> Path:
    """
    Get unified model cache directory.

    Priority order:
    1. custom_dir parameter (if provided)
    2. OMNIDOCS_MODELS_DIR environment variable
    3. HF_HOME environment variable
    4. Default: ~/.cache/huggingface

    Args:
        custom_dir: Optional custom cache directory path.
                   Overrides environment variables if provided.

    Returns:
        Path object pointing to the cache directory.
        Directory is created if it doesn't exist.
    """
    if custom_dir:
        cache_dir = custom_dir
    else:
        cache_dir = os.environ.get(
            "OMNIDOCS_MODELS_DIR",
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        )

    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_backend_cache(cache_dir: Optional[str] = None) -> None:
    """
    Configure cache directories for all backends.

    When OMNIDOCS_MODELS_DIR is set (or cache_dir is passed), this OVERWRITES
    HF_HOME and TRANSFORMERS_CACHE so every backend downloads to the same place.

    This is called automatically on ``import omnidocs``.

    Args:
        cache_dir: Optional cache directory path. If None, uses get_model_cache_dir().
    """
    cache_path = str(get_model_cache_dir(cache_dir))

    # Overwrite HF_HOME so PyTorch, MLX, VLLM, and snapshot_download all use it
    os.environ["HF_HOME"] = cache_path
    os.environ["TRANSFORMERS_CACHE"] = cache_path


def get_storage_info() -> dict:
    """
    Get current cache directory configuration information.

    Returns:
        Dictionary with cache paths and environment variable values.
    """
    return {
        "omnidocs_cache": str(get_model_cache_dir()),
        "omnidocs_models_dir_env": os.environ.get("OMNIDOCS_MODELS_DIR"),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
    }
