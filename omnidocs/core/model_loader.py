"""
Model loader utilities for working with the ModelRegistry.

This module provides helper functions for loading and initializing models
based on their configurations in the registry.
"""

import os
from pathlib import Path
from typing import Optional, Any, Tuple
import torch

from omnidocs.core.model_registry import (
    ModelConfig,
    ModelRegistry,
    HuggingFaceModelConfig,
    YOLOModelConfig,
    LibraryManagedModelConfig,
    VLMModelConfig,
    ModelType,
)
from omnidocs.utils.model_config import setup_model_environment


class ModelLoader:
    """Utility class for loading models from the registry."""

    _models_dir: Optional[Path] = None

    @classmethod
    def get_models_dir(cls) -> Path:
        """Get the models directory (sets up environment if needed)."""
        if cls._models_dir is None:
            cls._models_dir = setup_model_environment()
        return cls._models_dir

    @classmethod
    def get_model_path(cls, config: ModelConfig) -> Path:
        """Get the local path for a model based on its config."""
        models_dir = cls.get_models_dir()

        if config.local_dir:
            # Use specified local directory
            model_path = models_dir / config.local_dir
        else:
            # Generate path from model ID
            safe_id = config.id.replace("/", "_").replace(":", "_")
            model_path = models_dir / safe_id

        return model_path

    @classmethod
    def get_device(cls, config: ModelConfig, override: Optional[str] = None) -> str:
        """
        Get the device to use for a model.

        Args:
            config: Model configuration
            override: Optional device override (e.g., from user parameters)

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if override:
            return override
        if config.device:
            return config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def download_huggingface_model(
        cls,
        config: HuggingFaceModelConfig,
        force: bool = False
    ) -> Path:
        """
        Download a HuggingFace model.

        Args:
            config: HuggingFace model configuration
            force: Force re-download even if exists

        Returns:
            Path to downloaded model
        """
        from huggingface_hub import snapshot_download

        model_path = cls.get_model_path(config)

        if model_path.exists() and not force:
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=config.hf_model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )

        return model_path

    @classmethod
    def download_yolo_model(
        cls,
        config: YOLOModelConfig,
        force: bool = False
    ) -> Path:
        """
        Download a YOLO model from HuggingFace Hub.

        Args:
            config: YOLO model configuration
            force: Force re-download even if exists

        Returns:
            Path to downloaded model file
        """
        from huggingface_hub import hf_hub_download

        model_path = cls.get_model_path(config)
        model_file = model_path / config.model_filename

        if model_file.exists() and not force:
            return model_file

        model_path.mkdir(parents=True, exist_ok=True)

        downloaded_file = hf_hub_download(
            repo_id=config.model_repo,
            filename=config.model_filename,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )

        return Path(downloaded_file)

    @classmethod
    def load_huggingface_model(
        cls,
        config: HuggingFaceModelConfig,
        device: Optional[str] = None,
        download_if_needed: bool = True,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Load a HuggingFace model and optional processor.

        Args:
            config: HuggingFace model configuration
            device: Device to load model on
            download_if_needed: Download model if not present

        Returns:
            Tuple of (model, processor) - processor is None if not required
        """
        from transformers import AutoModel, AutoProcessor

        if download_if_needed and config.requires_download:
            model_path = cls.download_huggingface_model(config)
        else:
            model_path = cls.get_model_path(config)

        device = cls.get_device(config, device)

        # Load model
        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=config.trust_remote_code,
        )
        model = model.to(device)
        model.eval()

        # Load processor if required
        processor = None
        if config.requires_processor:
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=config.trust_remote_code,
            )

        return model, processor

    @classmethod
    def load_yolo_model(
        cls,
        config: YOLOModelConfig,
        device: Optional[str] = None,
        download_if_needed: bool = True,
    ) -> Any:
        """
        Load a YOLO model.

        Args:
            config: YOLO model configuration
            device: Device to load model on
            download_if_needed: Download model if not present

        Returns:
            Loaded YOLO model
        """
        from ultralytics import YOLO

        if download_if_needed:
            model_file = cls.download_yolo_model(config)
        else:
            model_path = cls.get_model_path(config)
            model_file = model_path / config.model_filename

        device = cls.get_device(config, device)

        model = YOLO(str(model_file))
        model.to(device)

        return model

    @classmethod
    def get_library_model_path(cls, config: LibraryManagedModelConfig) -> Path:
        """
        Get the storage path for library-managed models.

        Note: The library itself handles downloads, this just provides
        a consistent storage location if needed.

        Args:
            config: Library-managed model configuration

        Returns:
            Path for library model storage
        """
        model_path = cls.get_model_path(config)
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path

    @classmethod
    def load_from_registry(
        cls,
        model_id: str,
        device: Optional[str] = None,
        download_if_needed: bool = True,
        **kwargs
    ) -> Any:
        """
        Load a model by its registry ID.

        Args:
            model_id: ID of model in registry
            device: Device to load model on
            download_if_needed: Download model if not present
            **kwargs: Additional arguments for specific model types

        Returns:
            Loaded model (type depends on model configuration)

        Raises:
            ValueError: If model_id not found in registry
            NotImplementedError: If model type not yet supported
        """
        config = ModelRegistry.get(model_id)
        if config is None:
            raise ValueError(f"Model '{model_id}' not found in registry")

        if isinstance(config, HuggingFaceModelConfig):
            model, processor = cls.load_huggingface_model(
                config, device, download_if_needed
            )
            return {"model": model, "processor": processor, "config": config}

        elif isinstance(config, YOLOModelConfig):
            model = cls.load_yolo_model(config, device, download_if_needed)
            return {"model": model, "config": config}

        elif isinstance(config, LibraryManagedModelConfig):
            # Library-managed models are loaded by the extractors themselves
            # Just return the config and path
            return {
                "config": config,
                "model_path": cls.get_library_model_path(config)
            }

        elif isinstance(config, VLMModelConfig):
            # VLM loading logic (to be implemented)
            raise NotImplementedError("VLM model loading not yet implemented")

        else:
            raise NotImplementedError(
                f"Loading not implemented for model type: {config.model_type}"
            )


def get_model_config(model_id: str) -> Optional[ModelConfig]:
    """
    Convenience function to get a model config from the registry.

    Args:
        model_id: ID of model in registry

    Returns:
        Model configuration or None if not found
    """
    return ModelRegistry.get(model_id)


def list_models_for_task(task_type: str) -> list:
    """
    List all available models for a specific task.

    Args:
        task_type: Type of task (e.g., 'layout_analysis', 'ocr')

    Returns:
        List of model configurations
    """
    from omnidocs.core.model_registry import TaskType

    try:
        task_enum = TaskType(task_type)
        return ModelRegistry.get_by_task(task_enum)
    except ValueError:
        return []
