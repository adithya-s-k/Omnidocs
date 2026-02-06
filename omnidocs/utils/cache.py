"""
Unified model cache directory management for OmniDocs.

This module provides a centralized way to manage model cache directories
across all backends (PyTorch, VLLM, MLX, API).

Environment Variables:
    OMNIDOCS_MODEL_CACHE: Primary cache directory for all OmniDocs models.
                          Falls back to HF_HOME or ~/.cache/huggingface if not set.
    HF_HOME: HuggingFace cache directory (used as fallback).

Example:
    ```python
    from omnidocs.utils.cache import get_model_cache_dir, configure_backend_cache

    # Get unified cache directory
    cache_dir = get_model_cache_dir()

    # Configure all backend environment variables
    configure_backend_cache()
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
    2. OMNIDOCS_MODEL_CACHE environment variable
    3. HF_HOME environment variable
    4. Default: ~/.cache/huggingface

    Args:
        custom_dir: Optional custom cache directory path.
                   Overrides environment variables if provided.

    Returns:
        Path object pointing to the cache directory.
        Directory is created if it doesn't exist.

    Example:
        ```python
        # Use default
        cache = get_model_cache_dir()

        # Use custom directory
        cache = get_model_cache_dir("/data/models")

        # Use environment variable
        os.environ["OMNIDOCS_MODEL_CACHE"] = "/mnt/ssd/models"
        cache = get_model_cache_dir()
        ```
    """
    if custom_dir:
        cache_dir = custom_dir
    else:
        cache_dir = os.environ.get(
            "OMNIDOCS_MODEL_CACHE",
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        )

    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_backend_cache(cache_dir: Optional[str] = None) -> None:
    """
    Configure cache directories for all backends.

    Sets environment variables used by different backend libraries:
    - HF_HOME: HuggingFace Transformers cache
    - TRANSFORMERS_CACHE: Legacy Transformers cache (for compatibility)

    This should be called once on package import or before creating extractors.

    Args:
        cache_dir: Optional cache directory path. If None, uses get_model_cache_dir().

    Example:
        ```python
        # Configure with default
        configure_backend_cache()

        # Configure with custom path
        configure_backend_cache("/data/models")
        ```

    Note:
        Only sets environment variables if they are not already set.
        Existing values are preserved.
    """
    cache_path = str(get_model_cache_dir(cache_dir))

    # Set HF_HOME for HuggingFace/Transformers (used by PyTorch, MLX)
    os.environ.setdefault("HF_HOME", cache_path)

    # VLLM uses HF_HOME by default, but we can set explicit cache if needed
    # This is primarily for future extensibility
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_path)


def get_cache_info() -> dict:
    """
    Get current cache configuration information.

    Returns:
        Dictionary containing cache paths and environment variable values.

    Example:
        ```python
        info = get_cache_info()
        print(f"Using cache: {info['omnidocs_cache']}")
        print(f"HF_HOME: {info['hf_home']}")
        ```
    """
    return {
        "omnidocs_cache": str(get_model_cache_dir()),
        "omnidocs_model_cache_env": os.environ.get("OMNIDOCS_MODEL_CACHE"),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
    }
